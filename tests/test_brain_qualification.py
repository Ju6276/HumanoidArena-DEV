"""Orchestration tests with fixture bytes and mocked model processes only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fm_humanoid_bench.cli import qualify_brains as smoke


class BrainQualificationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.weights = self.root / 'model.safetensors'
        self.weights.write_bytes(b'fixture weight bytes, not a real model')
        self.entry = {'kind': 'http', 'family': 'psi0', 'task': 'open_door', 'horizon': 30,
                      'available': True, 'checkpoint': str(self.weights),
                      'server_command': ['python', 'fixture.py', '--port', '{port}', '--device', '{device}'],
                      'cwd': str(self.root), 'env': {}, 'evidence': []}
        self.observation = {'task': 'open_door', 'state64': [0] * 64, 'ego_image': 'unused.png', 'step': 0}

    def run_fixture(self, entry=None):
        return smoke.run_one('fixture', entry or self.entry, self.observation,
                             root=self.root, device='cpu', startup_timeout=1, request_timeout=1)

    def test_all_indexed_shards_are_hashed(self):
        directory = self.root / 'checkpoint'
        directory.mkdir()
        (directory / 'a.safetensors').write_bytes(b'AAA')
        (directory / 'b.safetensors').write_bytes(b'BBB')
        (directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'q': 'a.safetensors', 'v': 'b.safetensors'}}))
        records = smoke.checkpoint_inventory({**self.entry, 'checkpoint': str(directory)})
        self.assertEqual(len(records), 3)
        actual = {Path(row['path']).name: row['sha256'] for row in records}
        self.assertEqual(actual['b.safetensors'], hashlib.sha256(b'BBB').hexdigest())
        (directory / 'b.safetensors').unlink()
        with self.assertRaises(FileNotFoundError):
            smoke.checkpoint_inventory({**self.entry, 'checkpoint': str(directory)})

    def test_unavailable_model_is_blocked_without_starting_process(self):
        with patch.object(smoke.subprocess, 'Popen') as process:
            result = self.run_fixture({**self.entry, 'available': False, 'blocked_reason': 'HA checkpoint missing'})
        process.assert_not_called()
        self.assertEqual(result['status'], 'blocked')
        self.assertFalse(result['task_success_evaluated'])
        self.assertEqual(json.loads((self.root / 'fixture/result.json').read_text())['status'], 'blocked')

    def test_missing_actual_weights_are_blocked_despite_stale_catalog(self):
        self.weights.unlink()
        with patch.object(smoke.subprocess, 'Popen') as process:
            result = self.run_fixture()
        process.assert_not_called()
        self.assertEqual(result['status'], 'blocked')

    def test_lfs_pointer_is_blocked_before_hashing_or_launch(self):
        self.weights.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:' + 'a' * 64 + '\nsize 9354116320\n')
        with patch.object(smoke.subprocess, 'Popen') as process, \
             patch.object(smoke, 'sha256_file') as digest:
            result = self.run_fixture()
        process.assert_not_called()
        digest.assert_not_called()
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('Git LFS pointer', result['reason'])

    def test_success_saves_prediction_and_stops_only_owned_process(self):
        process = MagicMock()
        process.poll.return_value = None
        socket = MagicMock()
        socket.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 4567)
        metadata = {'schema': 'unitree_g1_gmt_refpose_v3_1', 'control_dt': .02,
                    'latency_seconds': .1, 'server_metadata': {'model': 'fixture'}}
        actions = np.ones((30, 40), dtype=np.float32)
        with patch.object(smoke.socket, 'socket', return_value=socket), \
             patch.object(smoke.subprocess, 'Popen', return_value=process) as launch, \
             patch.object(smoke, 'reset', return_value={'ok': True}), \
             patch.object(smoke, 'infer', return_value=(actions, metadata)) as infer, \
             patch.object(smoke, 'stop_process_group') as cleanup:
            result = self.run_fixture()
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(launch.call_args.kwargs['start_new_session'])
        self.assertEqual(launch.call_args.args[0][-1], 'cpu')
        cleanup.assert_called_once_with(process)
        self.assertEqual(infer.call_args.kwargs['expected_horizon'], 30)
        np.testing.assert_array_equal(np.load(self.root / 'fixture/prediction.npy'), actions)
        self.assertFalse(result['simulation_executed'])

    def test_inference_failure_always_cleans_process_group(self):
        process = MagicMock()
        process.poll.return_value = None
        socket = MagicMock()
        socket.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 4567)
        with patch.object(smoke.socket, 'socket', return_value=socket), \
             patch.object(smoke.subprocess, 'Popen', return_value=process), \
             patch.object(smoke, 'reset', return_value={'ok': True}), \
             patch.object(smoke, 'infer', side_effect=TimeoutError('fixture timeout')), \
             patch.object(smoke, 'stop_process_group') as cleanup:
            result = self.run_fixture()
        self.assertEqual(result['status'], 'failed')
        cleanup.assert_called_once_with(process)
        self.assertFalse((self.root / 'fixture/prediction.npy').exists())


if __name__ == '__main__':
    unittest.main()
