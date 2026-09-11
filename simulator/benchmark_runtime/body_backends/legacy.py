# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache-2.0
"""Simulator-independent adapters preserving the installed HA SONIC/TWIST2 paths.

These are learned policies, not direct reference-q motor playback. SONIC uses
HA's current-reference joint29 mode, including its hold window and reset history.
TWIST2 uses HA's 35-D mimic observation and pre-update ten-step observation history.
No position-feedback servo or task logic is added. Separate native commands expose
exactly the fields these selected modes can consume.
"""
from pathlib import Path
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from action_provider.vla_smpl_runtime import (
    TWIST2_G1_JOINT_NAMES_29, compute_yaw_rate_from_quats, quat_to_roll_pitch_wxyz,
)
from benchmark_runtime.body_backends.native import ReferenceBody
from benchmark_runtime.body_backends.legacy_parameters import (
    SONIC_ISAACLAB_JOINT_ORDER, SONIC_DEFAULT_POS, G1_ACTION_SCALE_ISAACLAB,
    SONIC_PD_KP, SONIC_PD_KD, SONIC_EFFORT_LIMIT,
    TWIST2_DEFAULT_POS, TWIST2_PD_KP, TWIST2_PD_KD, TWIST2_EFFORT_LIMIT,
)
from benchmark_runtime.body_backends.legacy_math import (
    compute_anchor_rot6d_wxyz, gravity_dir_from_base_quat_wxyz,
)

REPO = Path(__file__).resolve().parents[2]
ARENA = Path(os.environ.get('FMHB_ROOT', os.environ.get('HUMANOIDARENA_ROOT', REPO.parent)))


def native_session(path, device):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    providers = ['CPUExecutionProvider']
    if str(device).startswith('cuda'):
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('CUDA requested but ONNX CUDA provider unavailable')
        index = int(str(device).split(':')[1]) if ':' in str(device) else 0
        providers.insert(0, ('CUDAExecutionProvider', {'device_id': index}))
    policy = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    if str(device).startswith('cuda') and 'CUDAExecutionProvider' not in policy.get_providers():
        raise RuntimeError('ONNX failed to load requested CUDA provider')
    return policy


def vector(value, n, name):
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (n,) or not np.isfinite(arr).all():
        raise ValueError(f'{name} must contain {n} finite values')
    return arr.copy()


def joint_vector(value, names, label='joints'):
    if isinstance(value, dict):
        if set(value) != set(names):
            raise ValueError(f'{label} map must specify all 29 declared native joints')
        value = [value[name] for name in names]
    return vector(value, 29, label)


def validate_native_state(body, state):
    from fm_humanoid_bench.protocols.body_contract import validate_body_input
    pose = np.r_[state['pos'], state['quat'], state['q']]
    validate_body_input(state, np.tile(pose, (len(body.reference_offsets), 1)), body.names, body.reference_offsets)


class Sonic(ReferenceBody):
    name = 'SONIC'
    reference_offsets = np.array([-1, 0])
    native_mode = 'ha_joint29_current_hold'
    native_command_schema = {
        'required': ['joints', 'root_quat_wxyz'],
        'optional': ['joint_velocities'],
        'units': 'absolute joint radians, joint radians/s, world quaternion wxyz',
        'joint_order': 'native_joint_names; alternatively a complete name-value map',
        'absent_joint_velocities': 'backward difference of native commands; zero on first tick',
    }

    def __init__(self, device='cuda'):
        self.device = device
        self.names = list(SONIC_ISAACLAB_JOINT_ORDER)
        self.default = SONIC_DEFAULT_POS.copy()
        self.kp, self.kd, self.limits = SONIC_PD_KP.copy(), SONIC_PD_KD.copy(), SONIC_EFFORT_LIMIT.copy()
        base = Path(os.environ.get('SONIC_POLICY_DIR', os.environ.get('SONIC_POLICY_ROOT', ARENA/'models/body/sonic')))
        self.encoder_path = base / 'model_encoder.onnx'
        self.path = base / 'model_decoder.onnx'
        self.checkpoint_paths = {'encoder': self.encoder_path, 'decoder': self.path}
        self.source_dir = ARENA / 'simulator'
        self.encoder = native_session(self.encoder_path, device)
        self.policy = native_session(self.path, device)
        self.reset_history()

    def reset_history(self):
        self.steps = 0
        self.q_history = None
        self.dq_history = None
        self.omega_history = np.zeros((10, 3), np.float32)
        self.gravity_history = np.zeros((10, 3), np.float32)
        self.action_history = np.zeros((10, 29), np.float32)
        self.previous_native_q = None

    def infer(self, state, reference):
        previous, current = np.asarray(reference, dtype=np.float32)
        dq = np.zeros(29, np.float32) if self.steps == 0 else (current[7:] - previous[7:]) / np.float32(.02)
        return self._infer_condition(state, current[7:], dq, current[3:7])

    def infer_native(self, state, command):
        from fm_humanoid_bench.protocols.body_contract import validate_body_output
        validate_native_state(self, state)
        if not isinstance(command, dict) or not {'joints', 'root_quat_wxyz'} <= set(command) or set(command) - {'joints', 'root_quat_wxyz', 'joint_velocities'}:
            raise ValueError('Unknown SONIC joint29 command fields')
        q = joint_vector(command['joints'], self.names)
        quat = vector(command['root_quat_wxyz'], 4, 'root_quat_wxyz')
        if not np.isclose(np.linalg.norm(quat), 1., atol=1e-4):
            raise ValueError('Native target quaternion must be unit wxyz')
        if 'joint_velocities' in command:
            dq = joint_vector(command['joint_velocities'], self.names, 'joint_velocities')
        else:
            dq = np.zeros(29, np.float32) if self.previous_native_q is None else (q - self.previous_native_q) / np.float32(.02)
        result = validate_body_output(*self._infer_condition(state, q, dq, quat))
        self.previous_native_q = q.copy()
        return result

    def _infer_condition(self, state, ref_q, ref_dq, ref_quat):
        anchor = compute_anchor_rot6d_wxyz(state['quat'], ref_quat, np.array([1, 0, 0, 0], np.float32), False)
        encoder = np.concatenate([
            np.zeros(4, np.float32), np.tile(ref_q, 10), np.tile(ref_dq, 10),
            np.zeros(10, np.float32), np.zeros(1, np.float32), anchor, np.tile(anchor, 10),
            np.zeros(120 + 120 + 9 + 12 + 720 + 60 + 60, np.float32),
        ])[None].astype(np.float32)
        latent = self.encoder.run(None, {self.encoder.get_inputs()[0].name: encoder})[0]
        q = np.asarray(state['q'], np.float32) - self.default
        dq = np.asarray(state['dq'], np.float32)
        if self.q_history is None:
            # Matches HA on_env_reset at the first measured state, without an
            # unrecorded warmup or physical settling phase.
            self.q_history = np.tile(q, (10, 1))
            self.dq_history = np.tile(dq, (10, 1))
        self.q_history = np.roll(self.q_history, -1, axis=0)
        self.q_history[-1] = q
        self.dq_history = np.roll(self.dq_history, -1, axis=0)
        self.dq_history[-1] = dq
        self.omega_history = np.roll(self.omega_history, -1, axis=0)
        self.omega_history[-1] = state['omega']
        self.gravity_history = np.roll(self.gravity_history, -1, axis=0)
        self.gravity_history[-1] = gravity_dir_from_base_quat_wxyz(state['quat'])
        obs = np.concatenate([
            latent.reshape(-1), self.omega_history.reshape(-1), self.q_history.reshape(-1),
            self.dq_history.reshape(-1), self.action_history.reshape(-1), self.gravity_history.reshape(-1),
        ])[None].astype(np.float32)
        raw = self.policy.run(None, {self.policy.get_inputs()[0].name: obs})[0]
        self.action_history = np.roll(self.action_history, -1, axis=0)
        self.action_history[-1] = raw.reshape(-1)[:29]
        self.steps += 1
        # The installed HA provider deliberately retains un-clipped decoder output.
        target = self.default + raw.reshape(-1)[:29] * G1_ACTION_SCALE_ISAACLAB
        return target.astype(np.float32), {'obs': obs, 'raw_action': raw, 'conditioning': encoder.reshape(-1)}


class Twist2(ReferenceBody):
    name = 'TWIST2'
    reference_offsets = np.array([-1, 0])
    native_mode = 'ha_mimic35_current_hold'
    native_command_schema = {
        'required': ['joints', 'velocity', 'root_z', 'root_roll_pitch', 'yaw_rate'],
        'optional': [],
        'units': 'reference-base-local vx/vy m/s; world height m; roll/pitch rad; yaw rate rad/s; absolute joint rad',
        'joint_order': 'native_joint_names; alternatively a complete name-value map',
    }

    def __init__(self, device='cuda'):
        self.device = device
        self.names = list(TWIST2_G1_JOINT_NAMES_29)
        self.default = TWIST2_DEFAULT_POS.copy()
        self.kp, self.kd, self.limits = TWIST2_PD_KP.copy(), TWIST2_PD_KD.copy(), TWIST2_EFFORT_LIMIT.copy()
        self.path = Path(os.environ.get('TWIST2_MODEL_PATH', str(ARENA / 'external/TWIST2/assets/ckpts/twist2_1017_20k.onnx')))
        self.checkpoint_paths = {'actor': self.path}
        self.source_dir = ARENA / 'external/TWIST2'
        self.policy = native_session(self.path, device)
        self.reset_history()

    def reset_history(self):
        self.steps = 0
        self.history = np.zeros((10, 127), np.float32)
        self.last = np.zeros(29, np.float32)

    def reference_condition(self, reference):
        previous, current = np.asarray(reference, dtype=np.float32)
        delta = current[:3] - previous[:3]
        if self.steps == 0:
            # HA sets previous reference height to first target height; initial
            # measured height is not part of its first action's XY transformation.
            delta[2] = 0.
        rotation = Rotation.from_quat(current[3:7][[1, 2, 3, 0]])
        local_velocity = rotation.inv().apply(delta)[:2] / .02
        yaw = compute_yaw_rate_from_quats(previous[3:7], current[3:7], .02)
        return np.r_[local_velocity, current[2], quat_to_roll_pitch_wxyz(current[3:7]), yaw, current[7:]].astype(np.float32)

    def infer(self, state, reference):
        return self._infer_condition(state, self.reference_condition(reference))

    def infer_native(self, state, command):
        from fm_humanoid_bench.protocols.body_contract import validate_body_output
        validate_native_state(self, state)
        fields = {'joints', 'velocity', 'root_z', 'root_roll_pitch', 'yaw_rate'}
        if not isinstance(command, dict) or set(command) != fields:
            raise ValueError(f'TWIST2 mimic35 command requires exactly {sorted(fields)}')
        q = joint_vector(command['joints'], self.names)
        velocity = vector(command['velocity'], 2, 'velocity')
        rp = vector(command['root_roll_pitch'], 2, 'root_roll_pitch')
        scalar = vector([command['root_z'], command['yaw_rate']], 2, 'root_z/yaw_rate')
        condition = np.r_[velocity, scalar[0], rp, scalar[1], q].astype(np.float32)
        return validate_body_output(*self._infer_condition(state, condition))

    def _infer_condition(self, state, mimic):
        # Use Torch float32 for the exact installed HA roll/pitch convention.
        quat = torch.as_tensor(state['quat'], dtype=torch.float32, device=self.device)
        w, x, y, z = quat.unbind()
        rp = torch.stack([torch.atan2(2 * (w*x+y*z), 1-2*(x*x+y*y)), torch.asin(torch.clamp(2*(w*y-z*x), -1., 1.))]).cpu().numpy()
        dq = np.asarray(state['dq'], np.float32).copy()
        dq[[4, 5, 10, 11]] = 0.
        proprio = np.r_[np.asarray(state['omega'], np.float32)*.25, rp, np.asarray(state['q'], np.float32)-self.default, dq*.05, self.last]
        full = np.r_[mimic, proprio].astype(np.float32)
        # History uses previous ticks, while the last 35 dims repeat current mimic.
        obs = np.clip(np.r_[full, self.history.reshape(-1), mimic], -100., 100.).astype(np.float32)[None]
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = full
        raw = self.policy.run(None, {self.policy.get_inputs()[0].name: obs})[0]
        self.last = raw.reshape(-1).astype(np.float32).copy()
        self.steps += 1
        target = self.default + np.clip(self.last, -10., 10.) * np.float32(.5)
        return target, {'obs': obs, 'raw_action': raw, 'conditioning': mimic.copy()}
