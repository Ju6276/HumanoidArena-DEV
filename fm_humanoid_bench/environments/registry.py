"""Explicit mapping onto the existing 7 x 4 Arena configurations."""
from pathlib import Path
import importlib
from functools import lru_cache
import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
SIMULATOR = REPOSITORY / 'simulator'
MODES = {'Base': 'base_test', 'Visual': 'vision', 'Semantic': 'semantic', 'Execution': 'execution'}
TASKS = {
    'open_door': ('open_door', 'move_open_door_g1_29dof_dex3_wholebody'),
    'boxing': ('boxing', 'move_boxing_bag_g1_29dof_dex3_wholebody'),
    'football': ('football_single', 'move_football_g1_29dof_dex3_wholebody'),
    'pp_box': ('pp_box', 'move_pickplace_box_g1_29dof_dex3_wholedoby'),
    'doubledesk': ('doubledesk', 'move_pickplace_doubledesk_g1_29dof_dex3_wholebody'),
    'sit_sofa': ('sit_sofa', 'move_sit_sofa_g1_29dof_dex3_wholebody'),
    'vision_navi': ('vision_navi', 'move_small_warehouse_vision_navigation_g1_29dof_dex3_wholebody'),
}

@lru_cache(maxsize=140)
def resolve(task, mode, body='sonic'):
    from fm_humanoid_bench.evaluation.artifact_contract import BODY_NAMES
    body=BODY_NAMES[body.lower()]
    # Environment configuration is independent of the native Body runtime.
    config_route='sonic' if body=='sonic' else 'twist2'
    stem, module = TASKS[task]
    path = SIMULATOR/'tasks/common_test_config'/MODES[mode]/f'{stem}_{config_route}_test.yaml'
    cfg = yaml.safe_load(path.read_text())
    return dict(task=task, mode=mode, body=body, environment_config_route=config_route, task_id=cfg['task_name'], config=str(path), reward_module=module)

def evaluate(env, task_id):
    # Privileged evaluation only. Never serialize this into Brain observations.
    matches = [resolve(k, 'Base') for k in TASKS]
    entry = next((x for x in matches if x['task_id'] == task_id), None)
    if entry is None:
        raise ValueError(f'Unregistered task {task_id}')
    module = importlib.import_module(f"tasks.g1_tasks.{entry['reward_module']}.mdp.rewards")
    if hasattr(module, 'compute_success_mask'):
        success = bool(module.compute_success_mask(env)[0].item())
        source = 'compute_success_mask'
    else:
        # Football and DoubleDesk native raw reward is binary task completion.
        params = dict(env.reward_manager.get_term_cfg('reward').params)
        success = bool(module.compute_reward(env, **params)[0].item() >= 1.0)
        source = 'compute_reward>=1'
    return dict(task=task_id, success=success, success_source=source)
