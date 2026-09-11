"""Run real HTTP inference on a recorded observation, with fresh model servers.

This is an inference smoke, not a simulated episode or a task-success result.
Blocked models remain blocked: no replacement weights or fabricated actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError

import numpy as np

REPO = Path(os.environ.get('FMHB_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve()
sys.path.insert(0, str(REPO))
from fm_humanoid_bench.protocols.brain_protocol import infer, reset
from fm_humanoid_bench.environments.task_registry import TASKS
from fm_humanoid_bench.catalog import load_model_catalog


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_inventory(entry: dict) -> list[dict]:
    """Hash actual weight files and configs, including all indexed shards."""
    checkpoint = Path(entry['checkpoint'])
    if checkpoint.is_file():
        paths = {checkpoint}
    elif checkpoint.is_dir():
        index = checkpoint / 'model.safetensors.index.json'
        if index.is_file():
            paths = {index}
            mapping = json.loads(index.read_text())['weight_map']
            for name in set(mapping.values()):
                child = (checkpoint / name).resolve()
                if not child.is_relative_to(checkpoint.resolve()):
                    raise ValueError('Checkpoint shard path escapes checkpoint directory')
                paths.add(child)
        elif (checkpoint / 'model.safetensors').is_file():
            paths = {checkpoint / 'model.safetensors'}
        else:
            raise FileNotFoundError(f'No supported model weights in {checkpoint}')
    else:
        raise FileNotFoundError(checkpoint)
    for item in entry.get('evidence', []):
        path = Path(item['path'])
        if path.is_file() and path.suffix in ('.json', '.yaml', '.py', '.safetensors'):
            paths.add(path)
    # Preflight every shard before hashing large files or starting a server.
    # Files present in Git may be tiny LFS pointer text, not model tensors.
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open('rb') as source:
            if source.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise FileNotFoundError(f'Unmaterialized Git LFS pointer, not model weights: {path}')
    result = []
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append({'path': str(path), 'bytes': path.stat().st_size,
                       'sha256': sha256_file(path)})
    return result


def load_observation(path: Path) -> dict:
    observation = json.loads(path.read_text())
    if not isinstance(observation, dict) or 'state64' not in observation or 'ego_image' not in observation:
        raise ValueError('Observation JSON must contain state64 and ego_image')
    image = Path(observation['ego_image'])
    if not image.is_absolute():
        image = (path.parent / image).resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    observation['ego_image'] = str(image)
    return observation


def stop_process_group(process: subprocess.Popen) -> None:
    """Own Popen uses start_new_session; stop its children as well as parent."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # The launcher may exit while a GR00T policy/bridge child remains alive.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=15)


def run_one(model_id: str, entry: dict, observation: dict, *, root: Path,
            device: str, startup_timeout: float, request_timeout: float) -> dict:
    directory = root / model_id
    directory.mkdir()
    before = time.monotonic()
    result = {'model_id': model_id, 'kind': 'recorded_observation_inference_smoke',
              'status': 'pending', 'simulation_executed': False, 'task_success_evaluated': False,
              'task': entry.get('task'), 'family': entry.get('family')}
    process = None
    try:
        if not entry.get('available'):
            result.update(status='blocked', reason=entry.get('blocked_reason', 'Model unavailable in catalog'))
            return result
        if entry.get('kind') != 'http':
            raise ValueError('This smoke supports catalog HTTP models only')
        task = entry['task']
        if task not in TASKS:
            raise ValueError(f'Task not present in canonical registry: {task}')
        instruction = TASKS[task]['instruction']
        observed_task = observation.get('task')
        if isinstance(observed_task, dict):
            observed_task = observed_task.get('name') or observed_task.get('id')
        if observed_task in TASKS and observed_task != task:
            raise ValueError(f'Recorded observation task {observed_task} differs from checkpoint task {task}')
        weights = checkpoint_inventory(entry)
        (directory / 'checkpoint_provenance.json').write_text(json.dumps(weights, indent=2) + '\n')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        server = f'http://127.0.0.1:{port}'
        command = [part.format(port=port, device=device) for part in entry['server_command']]
        if not command:
            raise ValueError('No catalog server_command')
        environment = os.environ.copy()
        environment.update(entry.get('env', {}))
        environment['PYTHONUNBUFFERED'] = '1'
        provenance = {'catalog_entry': entry, 'command': command, 'cwd': entry['cwd'],
                      'server': server, 'instruction': instruction,
                      'fresh_server_process': True, 'inference_seed_policy': entry.get('rng_reset')}
        (directory / 'launch.json').write_text(json.dumps(provenance, indent=2) + '\n')
        with (directory / 'server.log').open('w') as log:
            process = subprocess.Popen(command, cwd=entry['cwd'], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + startup_timeout
            seed = int(observation.get('seed', 42))
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Model server exited with code {process.returncode}; see server.log')
                try:
                    receipt = reset(server, seed, timeout=min(3, request_timeout))
                    break
                except HTTPError:
                    raise
                except (URLError, TimeoutError, ConnectionError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'Server did not become ready within {startup_timeout}s')
                    time.sleep(.5)
            (directory / 'reset.json').write_text(json.dumps(receipt, indent=2) + '\n')
            actions, metadata = infer(server, observation, instruction,
                                      expected_horizon=entry['horizon'], timeout=request_timeout)
            np.save(directory / 'prediction.npy', actions, allow_pickle=False)
            (directory / 'prediction_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            result.update(status='passed', actual_action_shape=list(actions.shape),
                          schema=metadata['schema'], control_dt=metadata['control_dt'],
                          inference_seconds=metadata['latency_seconds'], finite=bool(np.isfinite(actions).all()),
                          inference_seed_policy=entry.get('rng_reset'), checkpoint_files=len(weights))
    except FileNotFoundError as error:
        result.update(status='blocked', error_type=type(error).__name__,
                      reason=f'Required local model artifact or executable is missing: {error}')
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        if process is not None:
            try:
                stop_process_group(process)
                result['owned_process_group_stopped'] = True
            except Exception as error:
                result.update(status='failed', cleanup_error=f'{type(error).__name__}: {error}')
        result['elapsed_seconds'] = time.monotonic() - before
        (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=REPO / 'fm_humanoid_bench/model_catalog.json')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--observation', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True, help='A new output directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--startup-timeout', type=float, default=300)
    parser.add_argument('--request-timeout', type=float, default=180)
    args = parser.parse_args()
    for timeout in (args.startup_timeout, args.request_timeout):
        if not np.isfinite(timeout) or timeout <= 0:
            parser.error('Timeouts must be finite positive seconds')
    catalog = load_model_catalog(args.catalog)
    if len(set(args.models)) != len(args.models):
        parser.error('Duplicate model IDs')
    for name in args.models:
        if name not in catalog['models']:
            parser.error(f'Unknown catalog model {name}')
        if Path(name).name != name or name in ('.', '..'):
            parser.error('Model IDs must be plain directory names')
    observation = load_observation(args.observation.resolve())
    args.root.mkdir(parents=True, exist_ok=False)
    protocol = {'kind': 'recorded_observation_inference_smoke',
                'catalog': str(args.catalog.resolve()), 'catalog_sha256': sha256_file(args.catalog),
                'observation': str(args.observation.resolve()), 'observation_sha256': sha256_file(args.observation),
                'ego_image_sha256': sha256_file(Path(observation['ego_image'])),
                'models': args.models, 'fresh_server_per_model': True,
                'simulation_executed': False, 'task_success_evaluated': False}
    (args.root / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows = []
    for name in args.models:
        print(f'START {name}', flush=True)
        rows.append(run_one(name, catalog['models'][name], observation, root=args.root,
                            device=args.device, startup_timeout=args.startup_timeout,
                            request_timeout=args.request_timeout))
        print(json.dumps(rows[-1]), flush=True)
        summary = {'schema': 'arena_brain_inference_smoke_v1',
                   'complete': len(rows) == len(args.models), 'models': rows,
                   'passed': sum(row['status'] == 'passed' for row in rows),
                   'blocked': sum(row['status'] == 'blocked' for row in rows),
                   'failed': sum(row['status'] == 'failed' for row in rows),
                   'simulation_executed': False, 'task_success_evaluated': False}
        (args.root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0 if all(row['status'] == 'passed' for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
