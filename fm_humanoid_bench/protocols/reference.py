"""Brain-independent G1 reference trajectory, using HA's physical-unit v3.1 wire format.

Only executed frames commit the reference anchor. Predicted lookahead never
advances it. A new chunk replaces unexecuted predictions; the episode anchor,
past reference and body history survive replanning. No simulator/task imports.
"""
from copy import deepcopy
import numpy as np
from action_provider.vla_robot_current_local_runtime_v3 import (
    CANONICAL_JOINT_NAMES_29, UNITREE_G1_REFPOSE_V31_SCHEMA_VERSION,
    UnifiedRobotCurrentLocalActionRuntimeV3,
    build_vla_rotlocal_v3_observation_state,
)

SCHEMA = UNITREE_G1_REFPOSE_V31_SCHEMA_VERSION
DT = .02


def tracking_metrics(recording, current_index):
    """Pre-control measured pose versus that tick's desired reference, no lag shift."""
    actual=recording['root'];reference=recording['reference'][:,current_index]
    qa=actual[:,3:7];qr=reference[:,3:7]
    qa=qa/np.linalg.norm(qa,axis=1,keepdims=True);qr=qr/np.linalg.norm(qr,axis=1,keepdims=True)
    angle=2*np.arccos(np.clip(np.abs(np.sum(qa*qr,axis=1)),0,1))
    return dict(root_position_rmse_m=float(np.sqrt(np.mean(np.sum((actual[:,:3]-reference[:,:3])**2,axis=1)))),root_rotation_rmse_rad=float(np.sqrt(np.mean(angle**2))),joint_reference_mae_rad=float(np.mean(np.abs(recording['q']-reference[:,7:]))),alignment='pre_control_measured_pose_vs_current_reference_no_lag_shift')


def validate_chunk(value):
    a = np.asarray(value, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 40 or not 1 <= len(a) <= 250:
        raise ValueError('Expected physical-unit reference action_chunk [1..250,40]')
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite reference')
    # Gram-Schmidt must have two independent axes; never silently decode zero.
    axes = a[:, 3:9].reshape(-1, 3, 2)
    if np.any(np.linalg.norm(np.cross(axes[:, :, 0], axes[:, :, 1]), axis=-1) < 1e-6):
        raise ValueError('Degenerate row-layout root rotation 6D')
    return a.copy()


class ReferenceTrajectory:
    def __init__(self, state, joint_names):
        self.names = list(joint_names)
        self.canonical_indices = [self.names.index(n) for n in CANONICAL_JOINT_NAMES_29]
        self.body_indices = [list(CANONICAL_JOINT_NAMES_29).index(n) for n in self.names]
        self.initial_quat = np.asarray(state['quat']).copy()
        self.runtime = UnifiedRobotCurrentLocalActionRuntimeV3(root_rot6d_layout='row')
        self.runtime.reset(body_xy_world=np.asarray(state['pos'])[:2], target_root_quat_wxyz=self.initial_quat)
        self.previous = np.r_[state['pos'], state['quat'], np.asarray(state['q'])[self.canonical_indices]].astype(np.float32)
        self.actions = None
        self.cursor = 0

    def observation(self, state):
        return build_vla_rotlocal_v3_observation_state(
            initial_robot_orientation_wxyz=self.initial_quat,
            root_orientation_wxyz=state['quat'],
            joint_pos_canonical_29=np.asarray(state['q'])[self.canonical_indices],
            joint_vel_canonical_29=np.asarray(state['dq'])[self.canonical_indices])

    @staticmethod
    def pose(frame):
        return np.r_[frame.body_pos_world, frame.target_root_quat_wxyz, frame.joint_pos_canonical_29].astype(np.float32)

    def submit(self, actions, state):
        actions = validate_chunk(actions)
        preview = deepcopy(self.runtime)
        self.poses = np.stack([self.pose(preview.step(a, current_robot_quat_wxyz=state['quat'], current_robot_xy_world=np.asarray(state['pos'])[:2])) for a in actions])
        self.actions, self.cursor = actions, 0

    def window(self, offsets):
        if self.actions is None or self.cursor >= len(self.actions):
            raise ValueError('Reference exhausted: request a new Brain prediction')
        indices = self.cursor + np.asarray(offsets, dtype=int)
        rows = [self.previous if i < self.cursor else self.poses[min(i, len(self.poses)-1)] for i in indices]
        # Explicit hold-last-pose padding (no extrapolated ground-truth future).
        a = np.stack(rows)
        return np.c_[a[:, :7], a[:, 7:][:, self.body_indices]].astype(np.float32)

    def commit(self, state):
        action = self.actions[self.cursor].copy()
        frame = self.runtime.step(action, current_robot_quat_wxyz=state['quat'], current_robot_xy_world=np.asarray(state['pos'])[:2])
        np.testing.assert_allclose(self.pose(frame), self.poses[self.cursor], atol=1e-5)
        self.previous = self.pose(frame)
        self.cursor += 1
        return action


def compile_keyframes(keyframes, start_action, horizon=30):
    """GPT's compact trajectory spelling -> same dense action40 as a VLA.

    Knot fields: frame, root_delta_xy (per tick, reference-base local), root_z,
    root_rpy (episode reference radians), joints (absolute named radians), hands.
    Missing pose fields hold the previous knot. Hands are piecewise constant.
    """
    from scipy.spatial.transform import Rotation, Slerp
    from action_provider.vla_smpl_runtime import quat_to_rot6d_wxyz, rot6d_to_quat_wxyz_with_layout
    if not isinstance(horizon, int) or not 1 <= horizon <= 250:
        raise ValueError('Invalid horizon')
    base = validate_chunk(np.asarray(start_action)[None])[0]
    points, values = [-1], [base.copy()]
    for knot in keyframes:
        allowed = {'frame','root_delta_xy','root_z','root_rpy','joints','hands'}
        if set(knot)-allowed: raise ValueError('Unknown trajectory knot fields')
        tick = knot['frame']
        if not isinstance(tick,int) or not points[-1] < tick < horizon: raise ValueError('Knot frames must increase within horizon')
        a = values[-1].copy()
        if 'root_delta_xy' in knot: a[:2] = knot['root_delta_xy']
        if 'root_z' in knot: a[2] = knot['root_z']
        if 'root_rpy' in knot:
            q = Rotation.from_euler('xyz',knot['root_rpy']).as_quat()[[3,0,1,2]]
            a[3:9] = quat_to_rot6d_wxyz(q)
        for name, value in knot.get('joints',{}).items(): a[9+list(CANONICAL_JOINT_NAMES_29).index(name)] = value
        if 'hands' in knot: a[38:40] = knot['hands']
        validate_chunk(a[None]); points.append(tick); values.append(a)
    if len(points) == 1: raise ValueError('At least one knot required')
    if points[-1] != horizon-1: points.append(horizon-1); values.append(values[-1].copy())
    values = np.stack(values); times = np.arange(horizon)
    out = np.stack([np.interp(times,points,values[:,j]) for j in range(40)],axis=1)
    qs = np.stack([rot6d_to_quat_wxyz_with_layout(v[3:9],layout='row') for v in values])
    rotations = Slerp(points,Rotation.from_quat(qs[:,[1,2,3,0]]))(times).as_quat()[:,[3,0,1,2]]
    out[:,3:9] = np.stack([quat_to_rot6d_wxyz(q) for q in rotations])
    out[:,38:40] = values[np.searchsorted(points,times,side='right')-1,38:40]
    return validate_chunk(out)
