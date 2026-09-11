import unittest
from fm_humanoid_bench.environments.task_registry import TASKS, original_budget
from fm_humanoid_bench.environments.sim_protocol import evaluator_source


class TaskBudgetTests(unittest.TestCase):
    def test_original_evaluator_sources_exist_after_repository_layout_changes(self):
        for backend in ('sonic','twist2'):
            self.assertTrue(evaluator_source(backend).is_file())

    def test_original_budgets_agree_across_ha_backends(self):
        for task, expected in [('open_door', 1800), ('sit_sofa', 2000)]:
            for backend in ('sonic', 'twist2'):
                with self.subTest(task=task, backend=backend):
                    ticks, source = original_budget(task, backend)
                    self.assertEqual(ticks, expected)
                    self.assertIn(TASKS[task]['launch'], str(source))

    def test_unregistered_task_is_rejected(self):
        with self.assertRaises(KeyError):
            original_budget('unknown_task')

    def test_registered_profiles_resolve_in_original_runtime(self):
        from task_runtime_profiles import resolve_task_runtime_profile, RUNTIME_CONTEXT_LIVE_INFERENCE
        for task in TASKS.values():
            resolve_task_runtime_profile(task['env_id'], task['runtime_profile'], RUNTIME_CONTEXT_LIVE_INFERENCE)

    def test_all_tasks_have_agent_facing_instruction_and_description(self):
        self.assertEqual(len(TASKS),7)
        for task_id,task in TASKS.items():
            self.assertTrue(task['instruction'].strip(),task_id)
            self.assertTrue(task['description'].strip(),task_id)


if __name__ == '__main__':
    unittest.main()
