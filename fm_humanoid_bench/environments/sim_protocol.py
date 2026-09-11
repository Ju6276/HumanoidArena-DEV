"""Shared simulator services, independent of Brain and Body tensor schemas."""
from pathlib import Path
import importlib.util
import numpy as np
from fm_humanoid_bench.environments.task_registry import TASKS, original_budget

def evaluator_source(source_backend='twist2'):
    if source_backend not in ('sonic','twist2'):
        raise ValueError(f'Unknown Arena evaluator backend: {source_backend}')
    return Path(__file__).resolve().parents[2]/'simulator/script/eval_scripts'/source_backend/'sim_eval_vla.py'

def original_task_protocol(env,args,source_backend='twist2'):
    task=next((key for key,value in TASKS.items() if value['env_id']==args.task),None)
    if task is None:
        raise ValueError('Register the original evaluator and budget before running another task')
    max_steps,launch=original_budget(task,source_backend)
    source=evaluator_source(source_backend)
    spec=importlib.util.spec_from_file_location('arena_original_'+source_backend,source)
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    detector=evaluator._build_fall_detector(env,args)
    metadata=dict(source=str(source),budget_source=str(launch),max_steps=max_steps,control_dt=.02,timeout_sim_seconds=max_steps*.02,success='original _extract_reward_info(env)[raw_total] >= 1.0',fall='original _build_fall_detector / _detect_robot_fall',priority=['success','fall','timeout'])
    return evaluator,detector,max_steps,metadata

def original_open_door_protocol(env,args,source_backend='twist2'):
    if args.task!=TASKS['open_door']['env_id']:
        raise ValueError('OpenDoor-only caller cannot evaluate another task')
    return original_task_protocol(env,args,source_backend)

def check_control_tick(before,after,dt=.02):
    if not np.isclose(after-before,dt,atol=1e-6):
        raise RuntimeError(f'Incorrect physics step: {after-before}, expected {dt}')

def refresh_cameras(env):
    """Render only; invalidate camera caches without moving simulation time."""
    before=float(env.sim.current_time)
    if 'world_camera' in env.scene.keys():
        p=env.scene['robot'].data.root_pos_w
        env.scene['world_camera'].set_world_poses_from_view(p+p.new_tensor([0.,-1.8,1.1]),p+p.new_tensor([0.,0.,.15]))
    env.sim.render();env.sim.render()
    for name in ('front_camera','world_camera','left_wrist_camera','right_wrist_camera'):
        if name in env.scene.keys():
            c=env.scene[name];c._is_outdated[:]=True;c.update(0.,force_recompute=True)
    if float(env.sim.current_time)!=before:raise RuntimeError('Render advanced physics')

def evaluate_tick(env,evaluator,detector,steps,max_steps,fall_streak):
    reward=evaluator._extract_reward_info(env)
    if reward['raw_total']>=1.0:return reward,'success',fall_streak
    fallen,_=evaluator._detect_robot_fall(env,detector)
    fall_streak=fall_streak+1 if fallen else 0
    terminal='fall' if fallen and fall_streak>=detector['confirm_steps'] else ('timeout' if steps>=max_steps else None)
    return reward,terminal,fall_streak
