import copy
import json
from pathlib import Path
import tempfile
import unittest
import yaml
from fm_humanoid_bench.evaluation.experiment_suite import plan,summarize,episode_seed,episode_index,publish_video_gallery
from fm_humanoid_bench.environments.task_registry import TASKS,MODES,original_budget,environment_config

class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.spec=dict(schema='arena_benchmark_suite_v1',models=['a','b'],bodies=['sonic','twist2'],tasks=['sit_sofa'],modes=['Base','Visual'],tracks=['reference'],conditions=['none'],group_seeds=[0,1],repeats=2,execute_frames=20)
        self.catalog={'models':{key:dict(kind='http',family='psi0',tasks=['sit_sofa'],available=True) for key in ['a','b']}}

    def test_pairing_and_original_seed_formula(self):
        result=plan(self.spec,self.catalog)
        self.assertEqual(len(result['jobs']),32)
        self.assertEqual(episode_seed('sit_sofa',0,0),1737204259)
        matched=[j for j in result['jobs'] if j['group_seed']==0 and j['repeat_index']==0]
        self.assertEqual(len({j['seed'] for j in matched}),1)
        self.assertEqual(len({j['directory'] for j in result['jobs']}),32)
        self.assertTrue(all(j['max_steps']==2000 for j in result['jobs']))

    def test_all_56_routes_and_original_budgets(self):
        expected={'open_door':1800,'sit_sofa':2000,'boxing':900,'football':2000,'pp_box':1450,'doubledesk':2000,'vision_navi':1800}
        for task in TASKS:
            for backend in ['sonic','twist2']:
                self.assertEqual(original_budget(task,backend)[0],expected[task])
                for mode in MODES:
                    raw=yaml.safe_load(Path(environment_config(task,mode,backend)).read_text())
                    self.assertEqual(raw['task_name'],TASKS[task]['env_id'])
                    self.assertEqual(bool(raw.get('test_defaults',{}).get('vision_randomization',{}).get('enabled')),mode=='Visual')

    def test_blocked_and_unsupported_not_silently_counted(self):
        self.catalog['models']['b']['available']=False
        self.spec['tracks']=['reference','native']
        result=plan(self.spec,self.catalog)
        self.assertTrue(result['excluded_combinations'])
        with tempfile.TemporaryDirectory() as d:
            summary=summarize(result,d)
        self.assertFalse(summary['complete'])
        self.assertTrue(all(g['success_rate'] is None for g in summary['groups']))
        self.assertEqual(sum(g['blocked'] for g in summary['groups']),16)

    def test_infrastructure_error_overrides_leftover_metrics(self):
        result=plan(self.spec,self.catalog);job=result['jobs'][0]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/job['directory'];run.mkdir(parents=True)
            (root/'jobs').mkdir()
            (run/'metrics.json').write_text(json.dumps(dict(success=True,termination='success',recording_verified=True,evaluation_kind='brain_closed_loop')))
            (run/'interface_audit.json').write_text('{"passed":true}')
            (root/'jobs'/f"{job['job_id']}.error.json").write_text('{}')
            summary=summarize(result,root)
            self.assertEqual(sum(g['successes'] for g in summary['groups']),0)
            self.assertEqual(sum(g['infrastructure_errors'] for g in summary['groups']),1)

    def test_file_agent_without_worker_is_not_scheduled_as_ready(self):
        self.catalog['models']['a'] = dict(kind='file', family='gpt6', tasks=['sit_sofa'], available=True)
        result = plan({**self.spec, 'models':['a']}, self.catalog)
        self.assertTrue(all(job['availability']=='requires_agent_worker' for job in result['jobs']))

    def test_episode_index_and_video_gallery_keep_one_canonical_copy(self):
        result=plan({**self.spec,'models':['a'],'bodies':['sonic'],'modes':['Base'],'group_seeds':[0],'repeats':1},self.catalog)
        job=result['jobs'][0]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/job['directory'];(run/'videos').mkdir(parents=True);(root/'jobs').mkdir()
            (run/'videos/ego.mp4').write_bytes(b'ego');(run/'videos/g1.mp4').write_bytes(b'g1')
            (run/'metrics.json').write_text(json.dumps(dict(success=False,termination='timeout',steps=2000,recording_verified=True)))
            (run/'interface_audit.json').write_text('{"passed":true}')
            rows=episode_index(result,root);links=publish_video_gallery(rows,root)
            self.assertEqual(rows[0]['status'],'complete');self.assertEqual(rows[0]['termination'],'timeout')
            self.assertEqual(len(links),2)
            for link in links:
                path=root/link['link'];self.assertTrue(path.is_symlink());self.assertEqual(path.resolve(),root/link['source'])

    def test_frozen_psi0_sonic40_population_is_1680(self):
        root=Path(__file__).resolve().parents[1]
        spec=json.loads((root/'fm_humanoid_bench/configs/reproduction/psi0_sonic40_table_s7.json').read_text())
        catalog=json.loads((root/'fm_humanoid_bench/model_catalog.json').read_text())
        result=plan(spec,catalog)
        self.assertEqual(len(result['jobs']),1680)
        self.assertTrue(all(job['body']=='sonic' for job in result['jobs']))
        self.assertTrue(all(catalog['models'][job['model_id']]['training_body']=='sonic' for job in result['jobs']))
        self.assertEqual({job['mode'] for job in result['jobs']},set(MODES))

    def test_duplicate_or_unknown_inputs_rejected(self):
        for change in [{'models':['a','a']},{'tasks':['unknown']},{'group_seeds':[True]},{'repeats':0},{'hidden_override':1}]:
            with self.assertRaises((ValueError,KeyError)):plan({**self.spec,**change},self.catalog)

if __name__=='__main__':unittest.main()
