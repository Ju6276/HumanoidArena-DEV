"""Prompt integration: task budgets, selected reference identity and modality."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fm_humanoid_bench.brains import agent_prompt
from fm_humanoid_bench.protocols import reference
from fm_humanoid_bench.evaluation.reference_catalog import build_catalog, select_reference


class ReferencePromptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.path = root / 'reference.json'
        (root / 'ego.png').write_bytes(b'image fixture')
        self.path.write_text(json.dumps(dict(
            schema='arena_episode_reference_v1', task='Open the door.', episode=0,
            reference_split='training', selected_indices=[0], images=['ego.png'],
            numeric_schema='unitree_g1_gmt_refpose_v3_1',
            state_action=[dict(frame=0, time_seconds=0., state64=[0.] * 64, action40=[0.] * 40)])))
        self.catalog = build_catalog([self.path])
        self.ref_id = self.catalog['entries'][0]['id']
        self.obs = dict(pos=[0, 0, .75], state64=[0.] * 64, hands=[0, 0], step=0,
                        ego_image='/current/ego.png', compact_proprioception={})
        # Also works when this test and prompt module are staged beside a repo.
        template_location = str(Path(reference.__file__).parent / 'agent_prompt.py')
        self.mock_file = patch.object(agent_prompt, '__file__', template_location)
        self.mock_file.start()
        self.addCleanup(self.mock_file.stop)

    def test_non_door_task_and_explicit_budget(self):
        self.obs['step'] = 123
        packet, _ = agent_prompt.packet(self.obs, [], None, 'none',
                                            task='Sit on the sofa.', remaining_steps=1877)
        self.assertEqual(packet['user']['task'], 'Sit on the sofa.')
        self.assertEqual(packet['user']['task_id'], 'sit_sofa')
        self.assertEqual(packet['user']['task_instruction'], 'Sit on the sofa.')
        self.assertIn('seated pose', packet['user']['task_description'])
        self.assertEqual(packet['user']['remaining_steps'], 1877)
        self.assertNotIn('demonstration', packet['user'])
        self.assertEqual(packet['system_prompt']['id'], 'reference_v31')

    def test_native_track_loads_versioned_native_prompt(self):
        self.obs['native_body_schema'] = {'command': {'velocity': '[vx, vy, yaw_rate]'}}
        packet, _ = agent_prompt.packet(
            self.obs, [], None, 'none', task='Open the door.',
            remaining_steps=1800, track='native')
        self.assertEqual(packet['system_prompt']['id'], 'native_body_v1')
        self.assertEqual(len(packet['system_prompt']['sha256']), 64)
        self.assertIn('native_body_schema', packet['user'])
        self.assertNotIn('start_reference40', packet['user'])

    def test_selected_image_reference_is_bound_to_prompt(self):
        selection = select_reference(self.catalog, task='open_door', condition='images',
                                     reference_id=self.ref_id, selector='agent')
        packet, _ = agent_prompt.packet(self.obs, [], None, 'images',
                                            remaining_steps=1800, selection=selection)
        self.assertEqual(packet['user']['reference_selection']['reference_id'], self.ref_id)
        self.assertEqual(packet['user']['reference_selection']['selector'], 'agent')
        self.assertNotIn('state_action', packet['user']['demonstration'])
        self.assertEqual(packet['user']['ego_image'], '/current/ego.png')

    def test_legacy_reference_path_passes_through_same_validation(self):
        packet, _ = agent_prompt.packet(self.obs, [], self.path, 'state_action', remaining_steps=1800)
        self.assertEqual(packet['user']['reference_selection']['reference_id'], self.ref_id)
        self.assertNotIn('ego_images', packet['user']['demonstration'])
        with self.assertRaises(ValueError):
            agent_prompt.packet(self.obs, [], self.path, 'state_action',
                                    task='Sit on the sofa.', remaining_steps=2000)

    def test_unused_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            agent_prompt.packet(self.obs, [], self.path, 'none', remaining_steps=1800)


if __name__ == '__main__':
    unittest.main()
