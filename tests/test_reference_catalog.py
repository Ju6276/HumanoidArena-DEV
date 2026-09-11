"""Reference isolation, provenance and pre-action selection regressions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from fm_humanoid_bench.evaluation.reference_catalog import (build_catalog, load_catalog, reference_options,
                                         select_reference, validate_selection)


class ReferenceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'ego_000000.png').write_bytes(b'fixture image bytes')
        self.path = self.root / 'reference.json'
        self.meta = dict(schema='arena_episode_reference_v1', task='Open the door.', episode=0,
                         reference_split='training demonstration', source_dataset='/private/dataset',
                         source_video='/private/video', success_label='must not reach Brain',
                         task_object_truth={'door_angle': 1.5}, num_frames=1, selected_indices=[0],
                         images=['ego_000000.png'], numeric_schema='unitree_g1_gmt_refpose_v3_1',
                         state_action=[dict(frame=0, time_seconds=0., state64=[0.] * 64,
                                            action40=[0.] * 40, task_object_truth={'x': 99})])
        self.write()
        self.catalog = build_catalog([self.path])
        self.ref_id = self.catalog['entries'][0]['id']

    def write(self):
        self.path.write_text(json.dumps(self.meta))

    def select(self, condition, **kw):
        return select_reference(self.catalog, task='open_door', condition=condition,
                                reference_id=self.ref_id, **kw)

    def test_legacy_bundle_and_public_options(self):
        output = self.root / 'catalog.json'
        self.assertEqual(build_catalog([self.path], output), load_catalog(output))
        options = reference_options(self.catalog, 'Open the door.', 'images')
        self.assertEqual(options, [dict(id=self.ref_id, task_id='open_door', condition='images')])
        self.assertEqual(reference_options(self.catalog, 'sit_sofa', 'images'), [])

    def test_modality_whitelists_remove_truth_and_source_metadata(self):
        images = self.select('images')['demonstration']
        numeric = self.select('state_action')['demonstration']
        self.assertNotIn('state_action', images)
        self.assertNotIn('ego_images', numeric)
        for demo in (images, numeric):
            for forbidden in ('task_object_truth', 'success_label', '/private', 'source_dataset'):
                self.assertNotIn(forbidden, json.dumps(demo))
        self.assertEqual(numeric['state_action'][0]['action40'], [0.] * 40)

    def test_task_and_unknown_id_rejected(self):
        with self.assertRaises(ValueError):
            select_reference(self.catalog, task='sit_sofa', condition='images', reference_id=self.ref_id)
        with self.assertRaises(ValueError):
            select_reference(self.catalog, task='open_door', condition='images', reference_id='/tmp/other.json')

    def test_agent_selection_receipt_before_action_is_locked(self):
        path = self.root / 'reference_selection.json'
        result = self.select('images', selector='agent', log_path=path)
        self.assertEqual(json.loads(path.read_text()), result['receipt'])
        self.assertEqual(result['receipt']['selector'], 'agent')
        self.assertEqual(len(result['receipt']['bundle_sha256']), 64)
        with self.assertRaises(FileExistsError):
            self.select('state_action', selector='agent', log_path=path)
        with self.assertRaises(ValueError):
            self.select('images', selector='agent', step=1)

    def test_none_never_delivers_reference(self):
        result = select_reference(self.catalog, task='open_door', condition='none')
        self.assertIsNone(result['demonstration'])
        validate_selection(result, task='open_door', condition='none')
        with self.assertRaises(ValueError):
            self.select('none')

    def test_modified_bundle_or_image_rejected(self):
        self.meta['episode'] = 1
        self.write()
        with self.assertRaises(ValueError):
            self.select('images')
        self.meta['episode'] = 0
        self.write()
        (self.root / 'ego_000000.png').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.select('images')

    def test_selected_image_hash_checked_before_prompt(self):
        selection = self.select('images')
        validate_selection(selection, task='open_door', condition='images')
        (self.root / 'ego_000000.png').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            validate_selection(selection, task='open_door', condition='images')

    def test_selection_cannot_change_condition_or_payload(self):
        selection = self.select('state_action')
        with self.assertRaises(ValueError):
            validate_selection(selection, task='open_door', condition='images')
        selection['demonstration']['state_action'][0]['action40'][0] = 9
        with self.assertRaises(ValueError):
            validate_selection(selection, task='open_door', condition='state_action')

    def test_paths_cannot_escape_bundle(self):
        outside = self.root.parent / (self.root.name + '_outside.png')
        outside.write_bytes(b'outside')
        self.addCleanup(outside.unlink)
        self.meta['images'] = ['../' + outside.name]
        self.write()
        with self.assertRaises(ValueError):
            build_catalog([self.path])
        link = self.root / 'escape.png'
        link.symlink_to(outside)
        self.meta['images'] = ['escape.png']
        self.write()
        with self.assertRaises(ValueError):
            build_catalog([self.path])

    def test_bad_numeric_or_frame_alignment_rejected(self):
        self.meta['state_action'][0]['action40'][2] = float('nan')
        self.write()
        with self.assertRaises(ValueError):
            build_catalog([self.path])
        self.meta['state_action'][0]['action40'][2] = 0
        self.meta['state_action'][0]['frame'] = 2
        self.write()
        with self.assertRaises(ValueError):
            build_catalog([self.path])

    def test_image_only_bundle_cannot_be_selected_as_numeric(self):
        self.meta.pop('state_action')
        self.write()
        catalog = build_catalog([self.path])
        with self.assertRaises(ValueError):
            select_reference(catalog, task='open_door', condition='state_action',
                             reference_id=catalog['entries'][0]['id'])


if __name__ == '__main__':
    unittest.main()
