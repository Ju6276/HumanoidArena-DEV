"""Run one benchmark Agent decision through a non-interactive Codex process."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]


def _nullable(value):
    return {'anyOf': [value, {'type': 'null'}]}


def _reference_schema(packet):
    number_array = lambda n: {'type': 'array', 'items': {'type': 'number'}, 'minItems': n, 'maxItems': n}
    joint_names = packet['user']['joint_names']
    joints = {
        'type': 'object',
        'properties': {name: _nullable({'type': 'number'}) for name in joint_names},
        'required': joint_names,
        'additionalProperties': False,
    }
    keyframe = {
        'type': 'object',
        'properties': {
            'frame': {'type': 'integer', 'minimum': 0, 'maximum': 249},
            'root_delta_xy': _nullable(number_array(2)),
            'root_z': _nullable({'type': 'number'}),
            'root_rpy': _nullable(number_array(3)),
            'joints': _nullable(joints),
            'hands': _nullable(number_array(2)),
        },
        'required': ['frame', 'root_delta_xy', 'root_z', 'root_rpy', 'joints', 'hands'],
        'additionalProperties': False,
    }
    return {
        'type': 'object',
        'properties': {'decision': {
            'type': 'object',
            'properties': {
                'rationale': {'type': 'string'},
                'horizon': {'type': 'integer', 'minimum': 1, 'maximum': 250},
                'keyframes': {'type': 'array', 'items': keyframe, 'minItems': 1},
            },
            'required': ['rationale', 'horizon', 'keyframes'],
            'additionalProperties': False,
        }},
        'required': ['decision'],
        'additionalProperties': False,
    }


def _native_schema(packet):
    native = packet['user']['native_body_schema']
    body = native['body']
    number_array = lambda n: {'type': 'array', 'items': {'type': 'number'}, 'minItems': n, 'maxItems': n}
    if body == 'BFM-Zero':
        command = {
            'type': 'object', 'properties': {'latent': {'type': 'string', 'enum': native['latent_names']}},
            'required': ['latent'], 'additionalProperties': False}
    elif body == 'SONIC':
        command = {
            'type': 'object',
            'properties': {'joints': number_array(29), 'root_quat_wxyz': number_array(4)},
            'required': ['joints', 'root_quat_wxyz'], 'additionalProperties': False}
    else:
        command = {
            'type': 'object',
            'properties': {
                'joints': number_array(29), 'velocity': number_array(2),
                'root_z': {'type': 'number'}, 'root_roll_pitch': number_array(2),
                'yaw_rate': {'type': 'number'},
            },
            'required': ['joints', 'velocity', 'root_z', 'root_roll_pitch', 'yaw_rate'],
            'additionalProperties': False,
        }
    return {
        'type': 'object',
        'properties': {'decision': {
            'type': 'object',
            'properties': {
                'rationale': {'type': 'string'},
                'command': command,
                'hands': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2},
                'frames': {'type': 'integer', 'minimum': 1, 'maximum': int(packet['user']['max_execute_frames'])},
            },
            'required': ['rationale', 'command', 'hands', 'frames'],
            'additionalProperties': False,
        }},
        'required': ['decision'],
        'additionalProperties': False,
    }


def _drop_null_updates(result):
    decision = result.get('decision', {})
    for keyframe in decision.get('keyframes', []):
        for key in list(keyframe):
            if keyframe[key] is None:
                del keyframe[key]
        if isinstance(keyframe.get('joints'), dict):
            keyframe['joints'] = {key:value for key,value in keyframe['joints'].items() if value is not None}
            if not keyframe['joints']:
                del keyframe['joints']
    return result


def main():
    packet = json.load(sys.stdin)
    supplied_hash = packet.get('prompt_sha256')
    unsigned = dict(packet);unsigned.pop('prompt_sha256', None)
    expected_hash = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
    if supplied_hash != expected_hash:
        raise ValueError('Codex worker received a packet with an invalid prompt hash')
    native = packet.get('system_prompt', {}).get('id') == 'native_body_v1'
    schema = _native_schema(packet) if native else _reference_schema(packet)
    instruction = (
        'Act only as the humanoid benchmark Brain. Do not use tools or modify files. '
        'Read the supplied versioned system prompt and observation packet, inspect the attached ego image when present, '
        'and return exactly one JSON object matching the response schema.\n\nBENCHMARK REQUEST:\n' +
        json.dumps(packet, ensure_ascii=False)
    )
    with tempfile.TemporaryDirectory(prefix='arena-codex-agent-') as directory:
        directory = Path(directory)
        schema_path = directory / 'response.schema.json'
        answer_path = directory / 'answer.json'
        schema_path.write_text(json.dumps(schema))
        command = [
            os.environ.get('CODEX_BIN', 'codex'), 'exec', '--ephemeral',
            '--ignore-user-config', '--ignore-rules', '--sandbox', 'read-only',
            '--model', os.environ.get('CODEX_BRAIN_MODEL', 'gpt-6-astra'),
            '--cd', os.environ.get('FMHB_ROOT', os.environ.get('ARENA_ROOT', str(REPO_ROOT))),
            '--output-schema', str(schema_path), '--output-last-message', str(answer_path), '-'
        ]
        image = Path(str(packet.get('user', {}).get('ego_image', '')))
        if image.is_file():
            command[2:2] = ['--image', str(image)]
        completed = subprocess.run(command, input=instruction, text=True, capture_output=True)
        if completed.returncode:
            sys.stderr.write(completed.stderr)
            raise SystemExit(completed.returncode)
        result = _drop_null_updates(json.loads(answer_path.read_text()))
    result['prompt_sha256'] = supplied_hash
    result['model'] = os.environ.get('CODEX_BRAIN_MODEL', 'gpt-6-astra')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
