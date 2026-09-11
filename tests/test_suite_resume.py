"""Resume must preserve a frozen plan and never archive beyond an attempt limit."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fm_humanoid_bench.cli import run_suite as suite


class SuiteResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'results'
        self.config = self.base / 'suite.json'
        self.catalog = self.base / 'catalog.json'
        self.spec = dict(schema='arena_benchmark_suite_v1', name='resume_fixture',
                         models=['fixture_agent'], bodies=['holo'], tasks=['open_door'],
                         modes=['Base'], tracks=['reference'], conditions=['none'],
                         group_seeds=[0], repeats=1, execute_frames=20)
        self.model_catalog = dict(schema='arena_model_catalog_v1', models={
            'fixture_agent': dict(kind='file', family='gpt6', available=True,
                                  tasks=['open_door'], horizon=30)})
        self.config.write_text(json.dumps(self.spec))
        self.catalog.write_text(json.dumps(self.model_catalog))

    def main(self, *options):
        argv = ['run_suite.py', str(self.config), '--catalog', str(self.catalog),
                '--root', str(self.root), *options]
        with patch.object(suite.sys, 'argv', argv):
            suite.main()

    def test_changed_resolved_environment_cannot_reuse_same_frozen_spec(self):
        self.main()
        path = self.root / 'plan.json'
        original_bytes = path.read_bytes()
        changed = deepcopy(json.loads(original_bytes))
        changed['jobs'][0]['environment_config_sha256'] = '0' * 64
        self.assertEqual(changed['spec_digest'], json.loads(original_bytes)['spec_digest'])
        with patch.object(suite, 'plan', return_value=changed):
            with self.assertRaises(ValueError):
                self.main()
        self.assertEqual(path.read_bytes(), original_bytes)

    def test_zero_attempt_budget_does_not_archive_failed_run(self):
        self.main()
        planned = json.loads((self.root / 'plan.json').read_text())
        job = planned['jobs'][0]
        run = self.root / job['directory']
        run.mkdir(parents=True)
        artifact = run / 'partial_recording.txt'
        artifact.write_text('preserve previous attempt')
        error = self.root / 'jobs' / (job['job_id'] + '.error.json')
        error.write_text(json.dumps({'error': 'previous infrastructure failure'}))
        with patch.object(suite.subprocess, 'Popen') as launch:
            self.main('--run', '--retry-errors', '--max-jobs', '0')
        launch.assert_not_called()
        self.assertEqual(artifact.read_text(), 'preserve previous attempt')
        self.assertTrue(error.is_file())
        self.assertFalse((self.root / 'infrastructure_attempts').exists())


if __name__ == '__main__':
    unittest.main()
