"""All seven Arena tasks, original instructions and native budget sources."""
from pathlib import Path
import re
from fm_humanoid_bench.environments.registry import MODES, resolve

_DESCRIPTIONS = {
    'open_door': ('Open the door.', 'Approach the door using ego vision, interact with it, and open it until the environment declares success.', 'HSI_open_door'),
    'sit_sofa': ('Sit on the sofa.', 'Approach the sofa and place the humanoid in a seated pose on it.', 'HSI_sit_sofa'),
    'boxing': ('Strike the green markers on the punching bag.', 'Use whole-body motion to strike the visible green target markers on the punching bag.', 'HSI_boxing'),
    'football': ('Kick the soccer ball into the goal.', 'Approach and kick the soccer ball so that it enters the visible goal.', 'HOI_football'),
    'pp_box': ('Move the box from the table onto the shelf.', 'Pick up the box from the table, transport it, and place it on the shelf.', 'HOI_pp_box'),
    'doubledesk': ('Put the hammer from the right table into the basket on the left table.', 'Pick up the hammer from the right table and place it in the basket on the left table.', 'HOI_double_desk'),
    'vision_navi': ('Avoid obstacles and move to the yellow marked area.', 'Navigate using ego vision, avoid obstacles, and enter the yellow target area.', 'HSI_vision_navi'),
}
TASKS = {key: dict(env_id=resolve(key, 'Base', 'twist2')['task_id'], instruction=value[0],description=value[1],
                   launch=value[2]+'_run_vla_eval_parallel.sh',
                   runtime_profile='inference' if key=='open_door' else 'auto')
         for key,value in _DESCRIPTIONS.items()}


def original_budget(task, backend='twist2'):
    if backend not in ('twist2', 'sonic'):
        raise ValueError(f'Unknown Arena evaluator backend: {backend}')
    source = Path(__file__).resolve().parents[2]/'simulator/script/eval_scripts'/backend/TASKS[task]['launch']
    match = re.search(r'MAX_STEPS:-([0-9]+)', source.read_text())
    if match is None:
        raise ValueError(f'No original step budget in {source}')
    return int(match.group(1)), source


def environment_config(task, mode='Base', backend='twist2'):
    if backend not in ('sonic','twist2'):
        raise ValueError('Environment configuration route must be sonic or twist2')
    return resolve(task, mode, backend)['config']
