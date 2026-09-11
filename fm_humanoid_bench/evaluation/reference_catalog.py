"""Pinned, task-matched demonstration selection shared by Brain adapters.

The catalog is evaluator metadata. Only ``reference_options`` and the selected
``demonstration`` payload may be sent to a Brain. No success labels, object
poses, source paths, or unused modalities are exposed by these functions.
Existing arena_episode_reference_v1 bundles remain valid without rewriting.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

CATALOG_SCHEMA = 'arena_reference_catalog_v1'
SELECTION_SCHEMA = 'arena_reference_selection_v1'
BUNDLE_SCHEMA = 'arena_episode_reference_v1'
NUMERIC_SCHEMA = 'unitree_g1_gmt_refpose_v3_1'
CONDITIONS = ('none', 'state_action', 'images')


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _task(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('A nonempty task ID/instruction is required')
    # Prefer the registry but support legacy OpenDoor bundles in isolated tools.
    try:
        from fm_humanoid_bench.environments.task_registry import TASKS
    except ImportError:
        TASKS = {}
    key = re.sub(r'[^a-z0-9]', '', value.lower())
    for task_id, task in TASKS.items():
        names = (task_id, task.get('instruction', ''), task.get('env_id', ''))
        if key in [re.sub(r'[^a-z0-9]', '', name.lower()) for name in names]:
            return task_id
    aliases = {'openthedoor': 'open_door', 'opendoor': 'open_door',
               'sitonthesofa': 'sit_sofa', 'sitsofa': 'sit_sofa'}
    return aliases.get(key, re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_'))


def _vector(value, width, name):
    if (not isinstance(value, list) or len(value) != width or
        any(isinstance(x, bool) or not isinstance(x, (int, float)) or
            not math.isfinite(x) for x in value)):
        raise ValueError(f'{name} must contain {width} finite numbers')


def _asset(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Reference image paths must be relative to their bundle')
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError(f'Reference image escapes bundle or does not exist: {name}')
    if target.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
        raise ValueError('Reference images must be image files')
    return target


def _read_bundle(path):
    path = Path(path).resolve()
    meta = json.loads(path.read_text())
    if meta.get('schema') != BUNDLE_SCHEMA:
        raise ValueError('Unsupported reference bundle schema')
    task = _task(meta.get('task_id', meta.get('task')))
    if 'task_id' in meta and 'task' in meta and task != _task(meta['task']):
        raise ValueError('Reference task ID and instruction disagree')
    if not isinstance(meta.get('reference_split'), str) or not meta['reference_split']:
        raise ValueError('Reference split provenance is required')
    if type(meta.get('episode')) is not int or meta['episode'] < 0:
        raise ValueError('Reference episode must be a nonnegative integer')
    frames = meta.get('selected_indices')
    if (not isinstance(frames, list) or not frames or
        any(type(x) is not int or x < 0 for x in frames) or
        frames != sorted(set(frames))):
        raise ValueError('Selected frame indices must strictly increase')
    if 'num_frames' in meta and (type(meta['num_frames']) is not int or
                                 meta['num_frames'] <= frames[-1]):
        raise ValueError('Selected indices exceed the source episode')
    modalities = []
    rows = meta.get('state_action')
    if rows:
        if meta.get('numeric_schema') != NUMERIC_SCHEMA:
            raise ValueError('Unsupported reference numeric action schema')
        if not isinstance(rows, list) or len(rows) != len(frames):
            raise ValueError('State/action samples must match selected indices')
        previous_time = -1.
        for frame, row in zip(frames, rows):
            if not isinstance(row, dict) or type(row.get('frame')) is not int or row['frame'] != frame:
                raise ValueError('State/action frame does not match selected index')
            timestamp = row.get('time_seconds')
            if (isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)) or
                not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous_time):
                raise ValueError('Sample timestamps must be finite and strictly increasing')
            previous_time = timestamp
            _vector(row.get('state64'), 64, 'state64')
            _vector(row.get('action40'), 40, 'action40')
        modalities.append('state_action')
    images = meta.get('images')
    assets = {}
    if images:
        if not isinstance(images, list) or len(images) != len(frames) or len(set(images)) != len(images):
            raise ValueError('Images must match selected indices without duplicates')
        assets = {name: _hash(_asset(path.parent, name)) for name in images}
        modalities.append('images')
    if not modalities:
        raise ValueError('A reference must provide state/action or ego image samples')
    return path, meta, task, modalities, assets


def build_catalog(bundle_paths, output=None):
    """Import legacy bundles and pin bytes; importing is not success certification."""
    entries = []
    for path in bundle_paths:
        path, meta, task, modalities, assets = _read_bundle(path)
        digest = _hash(path)
        entries.append(dict(id=f'{task}_ep{meta["episode"]:06d}_{digest[:12]}',
                            task_id=task, bundle_path=str(path), sha256=digest,
                            asset_sha256=assets, modalities=modalities))
    if len({entry['id'] for entry in entries}) != len(entries):
        raise ValueError('Duplicate reference IDs')
    catalog = dict(schema=CATALOG_SCHEMA, entries=entries)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            stream.write(json.dumps(catalog, indent=2) + '\n')
    return catalog


def load_catalog(path):
    """Validate every pinned bundle/asset before exposing reference choices."""
    catalog_path = Path(path).expanduser().resolve()
    catalog = json.loads(catalog_path.read_text())
    for entry in catalog.get('entries', []):
        bundle = Path(entry.get('bundle_path', ''))
        if not bundle.is_absolute():
            entry['bundle_path'] = str((catalog_path.parent / bundle).resolve())
    _validate_catalog(catalog)
    return catalog


def _validate_catalog(catalog):
    if not isinstance(catalog, dict) or catalog.get('schema') != CATALOG_SCHEMA:
        raise ValueError('Unsupported reference catalog schema')
    if not isinstance(catalog.get('entries'), list):
        raise ValueError('Catalog entries must be a list')
    seen = set()
    for entry in catalog['entries']:
        if not isinstance(entry, dict) or not isinstance(entry.get('id'), str) or not entry['id']:
            raise ValueError('Reference ID is required')
        if entry['id'] in seen:
            raise ValueError('Duplicate reference IDs')
        seen.add(entry['id'])
        path, meta, task, modalities, assets = _read_bundle(entry['bundle_path'])
        if not Path(entry['bundle_path']).is_absolute():
            raise ValueError('Resolved catalog bundle paths must be absolute')
        if entry.get('task_id') != task or entry.get('modalities') != modalities:
            raise ValueError('Catalog task/modalities disagree with bundle')
        if entry.get('sha256') != _hash(path) or entry.get('asset_sha256') != assets:
            raise ValueError('Reference content changed after catalog registration')


def reference_options(catalog, task, condition):
    """Safe public menu for pre-episode Agent selection under a fixed condition."""
    if condition not in CONDITIONS:
        raise ValueError('Unknown demonstration condition')
    _validate_catalog(catalog)
    if condition == 'none':
        return []
    return [dict(id=entry['id'], task_id=entry['task_id'], condition=condition)
            for entry in catalog['entries']
            if entry['task_id'] == _task(task) and condition in entry['modalities']]


def select_reference(catalog, *, task, condition, reference_id=None,
                     selector='config', step=0, log_path=None):
    """Resolve exactly one modality and lock its receipt before physical actions.

    An Agent submits the ID from ``reference_options``. It cannot change the
    experiment's condition. Call once before action; ``log_path`` is created
    exclusively and refuses overwrite/reselection.
    """
    if type(step) is not int or step != 0:
        raise ValueError('Reference selection must precede the first physical action')
    if selector not in ('config', 'agent'):
        raise ValueError('Selection source must be config or agent')
    options = reference_options(catalog, task, condition)
    demonstration = None
    entry = None
    if condition == 'none':
        if reference_id is not None:
            raise ValueError('The none condition cannot select a demonstration')
    else:
        allowed = {option['id'] for option in options}
        if reference_id not in allowed:
            raise ValueError('Reference ID is absent, task-mismatched, or lacks the requested modality')
        entry = next(item for item in catalog['entries'] if item['id'] == reference_id)
        path, meta, _, _, _ = _read_bundle(entry['bundle_path'])
        demonstration = dict(reference_id=reference_id, task_id=entry['task_id'],
                             source_episode=meta['episode'], condition=condition,
                             frame_indices=deepcopy(meta['selected_indices']))
        if condition == 'state_action':
            demonstration['numeric_schema'] = NUMERIC_SCHEMA
            # Whitelist each sample: extra metadata must not leak object truth.
            demonstration['state_action'] = [
                {key: deepcopy(row[key]) for key in ('frame', 'time_seconds', 'state64', 'action40')}
                for row in meta['state_action']]
        else:
            demonstration['ego_images'] = [str(_asset(path.parent, name)) for name in meta['images']]
    receipt = dict(schema=SELECTION_SCHEMA, task_id=_task(task), condition=condition,
                   reference_id=reference_id, selector=selector, selected_at_step=step,
                   catalog_sha256=hashlib.sha256(_canonical(catalog).encode()).hexdigest(),
                   bundle_sha256=entry['sha256'] if entry else None,
                   asset_sha256=deepcopy(entry['asset_sha256']) if entry and condition == 'images' else {})
    receipt['payload_sha256'] = hashlib.sha256(_canonical(demonstration).encode()).hexdigest()
    receipt['delivered_asset_sha256'] = {
        path: _hash(path) for path in (demonstration or {}).get('ego_images', [])}
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('x') as stream:
            stream.write(json.dumps(receipt, indent=2) + '\n')
    return dict(receipt=receipt, demonstration=demonstration)


def validate_selection(selection, *, task, condition):
    """Check a resolved selection before a Brain adapter builds its prompt."""
    receipt = selection['receipt']
    if (receipt.get('schema') != SELECTION_SCHEMA or receipt.get('task_id') != _task(task)
        or receipt.get('condition') != condition or receipt.get('selected_at_step') != 0):
        raise ValueError('Selection does not match this experiment')
    demo = selection.get('demonstration')
    if condition == 'none':
        if demo is not None or receipt.get('reference_id') is not None:
            raise ValueError('No-reference condition contains a demonstration')
    elif (not isinstance(demo, dict) or demo.get('reference_id') != receipt.get('reference_id')
          or demo.get('task_id') != receipt['task_id'] or demo.get('condition') != condition):
        raise ValueError('Demonstration does not match selection receipt')
    elif condition == 'images' and ('state_action' in demo or 'ego_images' not in demo):
        raise ValueError('Image-only condition contains numeric demonstration data')
    elif condition == 'state_action' and ('ego_images' in demo or 'state_action' not in demo):
        raise ValueError('State/action-only condition contains demonstration images')
    if receipt.get('payload_sha256') != hashlib.sha256(_canonical(demo).encode()).hexdigest():
        raise ValueError('Selected demonstration payload changed')
    for path, digest in receipt.get('delivered_asset_sha256', {}).items():
        if _hash(path) != digest:
            raise ValueError('Selected demonstration image changed')
    return deepcopy(selection)
