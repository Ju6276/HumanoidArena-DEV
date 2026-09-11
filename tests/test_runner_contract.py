"""Exercise common runner decisions without GPU, model calls or physics.

These tests check orchestration contracts only; the fake simulator deliberately
does not certify motor control or task success.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fm_humanoid_bench.cli import run_episode as runner
from fm_humanoid_bench.brains import agent_prompt
from fm_humanoid_bench.protocols import reference


class RunnerContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prompts = []
        self.requests = []
        self.step = 0
        self.run = None
        template_location = str(Path(reference.__file__).parent / 'agent_prompt.py')
        self.mock_file = patch.object(agent_prompt, '__file__', template_location)
        self.mock_file.start()
        self.addCleanup(self.mock_file.stop)

    def launch(self, command, **kwargs):
        self.run = Path(command[command.index('--recording_save_dir') + 1])
        (self.run / 'debug' / 'runner.log').write_text('BFM_READY\n')
        (self.run / 'body_contract.json').write_text(json.dumps({'reference_offsets': [0]}))
        np.savez(self.run / 'episode.npz', empty=np.array([]))
        class Process:
            terminated = False
            def poll(p): return 0 if p.terminated else None
            def terminate(p): p.terminated = True
            def kill(p): p.terminated = True
            def wait(p, **kw): return 0
        return Process()

    def observation(self):
        return dict(step=self.step, _runner_terminal='success' if self.step >= 40 else None,
                    pos=[0., 0., .75], quat=[1., 0., 0., 0.], q=[0.] * 29, dq=[0.] * 29,
                    state64=[1., 0., 0., 1., 0., 0.] + [0.] * 58,
                    hands=[0, 0], ego_image='/current/ego.png', compact_proprioception={},
                    native_body_schema={'command': 'fixture native command'},
                    forbidden_task_truth={'door_angle': 2.})

    def socket(self):
        owner = self
        class Socket:
            def setsockopt(s, *args): pass
            def connect(s, *args): pass
            def close(s, **kw): pass
            def send_json(s, request): s.request = request
            def recv_json(s):
                if s.request['op'] in ('reference_step', 'native_step'):
                    owner.requests.append(s.request)
                    owner.step += s.request['frames']
                if s.request['op'] == 'finish':
                    return {'success': True, 'steps': owner.step}
                return owner.observation()
        return Socket()

    def agent(self, packet, root, stem, **kwargs):
        self.prompts.append(packet)
        if stem == 'reference_selection':
            return {'reference_id': packet['options'][0]['id']}, {'prompt_sha256': 'testhash'}
        if 'native_body_schema' in packet['user']:
            decision = {'command': {'velocity': [0., 0., 0.]}, 'hands': [0, 0], 'frames': 20}
        else:
            decision = {'rationale': 'fixture', 'horizon': 20, 'keyframes': [{'frame': 0}]}
        return {'decision': decision}, {'prompt_sha256': 'testhash', 'latency_seconds': .01}

    def execute(self, *args):
        class Context:
            def socket(c, *args): return self.socket()
        argv = ['run_episode.py', '--body', 'holo', '--brain', 'gpt6',
                '--root', str(self.root / 'runs'), *args]
        with patch.object(runner.sys, 'argv', argv), \
             patch.object(runner.subprocess, 'Popen', side_effect=self.launch), \
             patch.object(runner.subprocess, 'run'), \
             patch.object(runner.zmq.Context, 'instance', return_value=Context()), \
             patch.object(runner, 'request_agent', side_effect=self.agent), \
             patch.object(runner, 'publish'), \
             patch.object(runner, 'tracking_metrics', return_value={}):
            runner.main()

    def test_native_history_contains_decisions_without_task_truth(self):
        self.execute('--track', 'native')
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.prompts[1]['user']['history'][0]['decision'],
                         {'command': {'velocity': [0., 0., 0.]}, 'hands': [0, 0], 'frames': 20})
        self.assertNotIn('start_reference40', self.prompts[1]['user'])
        self.assertNotIn('forbidden_task_truth', json.dumps(self.prompts))
        for prompt in self.prompts:
            self.assertEqual(prompt['user']['reference_selection']['condition'], 'none')
            self.assertEqual(prompt['system_prompt']['id'], 'native_body_v1')

    def test_selection_receipt_repeated_in_every_decision_record(self):
        self.execute()
        records = [json.loads(line) for line in (self.run / 'brain_predictions.jsonl').read_text().splitlines()]
        self.assertEqual(len(records), 2)
        for record, prompt in zip(records, self.prompts):
            self.assertEqual(record['reference_selection'], prompt['user']['reference_selection'])

    def test_legacy_bundle_requires_no_new_reference_id_argument(self):
        bundle = self.root / 'reference.json'
        bundle.write_text(json.dumps(dict(
            schema='arena_episode_reference_v1', task='Open the door.', episode=0,
            reference_split='training', selected_indices=[0],
            numeric_schema='unitree_g1_gmt_refpose_v3_1',
            state_action=[dict(frame=0, time_seconds=0., state64=[0.] * 64, action40=[0.] * 40)])))
        self.execute('--condition', 'state_action', '--reference', str(bundle))
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.prompts[0]['user']['reference_selection']['condition'], 'state_action')
        self.assertTrue((self.run / 'reference_selection.json').is_file())

    def test_fixed_replay_preserves_predictions_and_variable_execution_prefixes(self):
        source = self.root / 'source'
        source.mkdir()
        (source / 'experiment.json').write_text(json.dumps(dict(task='open_door', mode='Base', seed=42)))
        first = np.zeros((30, 40), dtype=np.float32)
        first[:, 2] = .75
        first[:, 3:9] = [1, 0, 0, 1, 0, 0]
        second = first.copy()
        second[:, 0] = .002
        records = [dict(decision=dict(action_chunk=chunk.tolist(), frames=frames))
                   for chunk, frames in ((first, 25), (second, 15))]
        (source / 'brain_decisions.jsonl').write_text('\n'.join(json.dumps(row) for row in records))
        self.execute('--brain', 'reference_replay', '--replay', str(source), '--execute', '20')
        self.assertEqual([request['frames'] for request in self.requests], [25, 15])
        np.testing.assert_array_equal(self.requests[0]['action_chunk'], first)
        np.testing.assert_array_equal(self.requests[1]['action_chunk'], second)
        self.assertEqual(self.prompts, [])
        metrics = json.loads((self.run / 'metrics.json').read_text())
        self.assertEqual(metrics['evaluation_kind'], 'reference_replay')

    def test_invalid_late_realized_frame_fails_before_simulator_launch(self):
        trajectory = np.zeros((300, 40), dtype=np.float32)
        trajectory[:, 2] = .75
        trajectory[:, 3:9] = [1, 0, 0, 1, 0, 0]
        trajectory[-1, 10] = np.nan
        path = self.root / 'trajectory.npy'
        np.save(path, trajectory)
        with self.assertRaises(ValueError):
            self.execute('--brain', 'realized_replay', '--trajectory', str(path))
        self.assertIsNone(self.run)


if __name__ == '__main__':
    unittest.main()
