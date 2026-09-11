"""Synchronous ego-only RPC control over Arena's native Sonic provider.

No physics, policy history, or recording advances while waiting for a request.
Actions are references to the Sonic encoder, never direct simulator targets.
"""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import zmq
from PIL import Image
from isaaclab.utils.math import quat_mul


def run_control(env, controller, provider, args, app):
    twist2 = getattr(args, 'gmt_backend', '') == 'twist2'
    if not twist2 and (not provider._sonic_joint29_mode or provider._use_lerobot_vla):
        raise ValueError('Requires a native SONIC/TWIST2 provider without VLA')
    root = Path(args.recording_save_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ['ASTRA_CONTROL_ENDPOINT']
    socket = zmq.Context.instance().socket(zmq.REP)
    socket.bind(endpoint)
    action = (np.r_[provider._default_mimic_obs[0].cpu().numpy(), 0., 0.] if twist2 else np.r_[0., 0., .78, 1., 0., 0., 1., 0., 0., provider._sonic_default_np, 0., 0.]).astype(np.float32)
    if not twist2:
        provider._fetch_redis_pose = lambda: provider._apply_lerobot_semantic_action(action)
        # Hand references are supplied by Astra, without a live DDS override.
        provider._dex3_dds = None
        provider._pose_source = 'astra_semantic40'
        provider._fetch_zmq_pose = provider._fetch_redis_pose
    steps = 0
    if twist2:
        from action_provider.action_provider_wh_twist2 import DEFAULT_HAND_POSE
        def fetch_twist2():
            provider._twist2_hand_valid = True
            for i, side in enumerate(('left', 'right')):
                pose = DEFAULT_HAND_POSE['unitree_g1_with_hands'][side]['close' if action[35+i] >= .5 else 'open']
                getattr(provider, '_twist2_action_hand_'+side).copy_(torch.tensor(pose, device=env.device).unsqueeze(0))
            return torch.tensor(action[:35].copy(), device=env.device).unsqueeze(0)
        provider._twist2_fetch_actions = fetch_twist2
        provider._should_start_recording_on_first_call = True
        provider.gripper_dds = None
    terminal = None
    native_eval = os.environ.get('ASTRA_NATIVE_EVAL') == '1'
    native_evaluator = None
    native_reward = None
    fall_streak = 0
    if native_eval:
        from fm_humanoid_bench.environments.sim_protocol import original_open_door_protocol,evaluate_tick
        native_evaluator,fall_detector,max_steps,protocol = original_open_door_protocol(env,args,'twist2' if twist2 else 'sonic')
        (root/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2))
    log = (root / 'astra_actions.jsonl').open('a', buffering=1)
    timeline = (root / 'simulation_timeline.jsonl').open('a', buffering=1)

    def render_recording_cameras():
        from fm_humanoid_bench.environments.sim_protocol import refresh_cameras
        refresh_cameras(env)

    def observe():
        cam = env.scene['front_camera']
        rgb = cam.data.output['rgb'][0].cpu().numpy()[..., :3]
        path = root / f'ego_{steps:06d}.png'
        Image.fromarray(rgb.astype(np.uint8)).save(path)
        if os.environ.get('ASTRA_RECORDING_REVIEW') == '1' and 'world_camera' in env.scene.keys():
            # Recording QA only. Never expose this image through task observations.
            review_rgb = env.scene['world_camera'].data.output['rgb'][0].cpu().numpy()[..., :3]
            Image.fromarray(review_rgb.astype(np.uint8)).save(root/'recording_preview.png')
        robot = env.scene['robot']
        d = robot.data
        # CameraCfg caches its world pose by default. Use current robot FK and
        # the authored fixed mount, independent of scene-object state.
        mount = d.body_state_w[0, robot.body_names.index('d435_link'), :7]
        offset = cam.cfg.offset
        if offset.convention != 'ros' or tuple(offset.pos) != (0, 0, 0):
            raise ValueError('Unsupported ego mount calibration')
        camera_quat = quat_mul(mount[3:7], torch.tensor(offset.rot, device=mount.device))
        def arr(x):
            return x.detach().cpu().numpy().tolist()
        return dict(paused=True, step=steps, sim_time=float(env.sim.current_time),
                    ego_image=str(path), _runner_terminal=terminal, joint_names=robot.joint_names,
                    action_joint_indices=(list(provider.twist2_action_indices) if twist2 else arr(provider._sonic_idx)),
                    action_format=('twist2_mimic35_grippers2' if twist2 else 'sonic_semantic40'),
                    joint_pos=arr(d.joint_pos[0]), joint_vel=arr(d.joint_vel[0]),
                    root_state=arr(d.root_state_w[0]), body_names=robot.body_names,
                    body_poses=arr(d.body_state_w[0, :, :7]),
                    camera_position=arr(mount[:3]),
                    camera_quat_wxyz=arr(camera_quat),
                    camera_intrinsics=arr(cam.data.intrinsic_matrices[0]),
                    last_action=action.tolist())

    for _ in range(5):
        render_recording_cameras()
    env.observation_manager.compute()
    print(f'ASTRA_READY {endpoint}', flush=True)
    try:
        with torch.inference_mode():
            while app.is_running() and controller.is_running:
                if not socket.poll(1000):
                    continue
                request = socket.recv_json()
                try:
                    op = request.get('op', 'observe')
                    if op == 'step':
                        if terminal:
                            raise ValueError('Episode has terminated; call finish')
                        actions = request.get('actions')
                        if actions is None:
                            frames = int(request.get('frames', 1))
                            actions = [request.get('action', action.tolist())] * frames
                        actions = np.asarray(actions, dtype=np.float32)
                        if actions.ndim != 2 or actions.shape[1] != len(action) or not 1 <= len(actions) <= 250 or not np.isfinite(actions).all():
                            raise ValueError(f'Expected 1..250 finite native actions of width {len(action)}')
                        log.write(json.dumps(dict(step=steps, request=request))+'\n')
                        for next_action in actions:
                            action[:] = next_action
                            before = float(env.sim.current_time)
                            controller.step()
                            after = float(env.sim.current_time)
                            if not np.isclose(after-before, .02, atol=1e-6):
                                raise RuntimeError(f'Incorrect physics step: {after-before}')
                            render_recording_cameras()
                            if float(env.sim.current_time) != after:
                                raise RuntimeError('Recording render advanced physics')
                            timeline.write(json.dumps(dict(step=steps, sim_time=before, next_sim_time=after))+'\n')
                            steps += 1
                            if native_eval:
                                native_reward,terminal,fall_streak = evaluate_tick(env,native_evaluator,fall_detector,steps,max_steps,fall_streak)
                                if terminal:
                                    break
                    elif op == 'finish':
                        from fm_humanoid_bench.environments.registry import evaluate
                        evaluation = evaluate(env, args.task)
                        evaluation.update(steps=steps, sim_time=float(env.sim.current_time))
                        if native_eval:
                            evaluation.update(success=terminal == 'success',
                                success_source=f'{native_evaluator.__file__}:_extract_reward_info.raw_total>=1',
                                termination=terminal or 'manual_finish', max_steps=max_steps,
                                timeout_sim_seconds=max_steps*.02, final_reward=native_reward)

                        if terminal == 'success':
                            evaluation['success'] = True
                        (root / 'task_evaluation.json').write_text(json.dumps(evaluation, indent=2))
                        result = []
                        if not (hasattr(provider.recording_manager, 'retry_failed_saves') and provider.recording_manager.retry_failed_saves(result.append)):
                            provider.recording_manager.save_recording(completion_callback=result.append)
                        provider.recording_manager.save_queue.join()
                        if result != [True]:
                            raise RuntimeError(f'Recording save failed: {result}')
                        if native_eval and 'Open-Door' in args.task:
                            # Native providers keep their tensor schema; publication is shared.
                            log.flush();timeline.flush()
                            from benchmark_runtime.verify_astra_recording import verify
                            from fm_humanoid_bench.evaluation.artifact_contract import publish
                            if not (root/'provenance.json').exists():
                                (root/'provenance.json').write_text(json.dumps(dict(
                                    brain=os.environ.get('ASTRA_BRAIN_ID','gpt6'),
                                    body='twist2' if twist2 else 'sonic',runner_pid=os.getpid(),
                                    observations='ego RGB and robot proprioception',
                                    source=str(Path(__file__).resolve())),indent=2))
                            verify(root)
                            root=publish(root,'twist2' if twist2 else 'sonic',
                                         brain=os.environ.get('ASTRA_BRAIN_ID','gpt6'),
                                         mode=os.environ.get('ASTRA_MODE','Base'),seed=args.seed)
                        socket.send_json(dict(saved=True, steps=steps, directory=str(root)))
                        break
                    elif op != 'observe':
                        raise ValueError(f'Unknown operation {op}')
                    socket.send_json(observe())
                except Exception as exc:
                    socket.send_json(dict(error=str(exc), paused=True, step=steps))
    finally:
        log.close()
        timeline.close()
        socket.close(linger=0)
