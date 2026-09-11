"""Deterministic controller probes, explicitly not a task-solving Brain."""
import numpy as np
from fm_humanoid_bench.protocols.reference import compile_keyframes


def probe_actions(observation, case):
    start=np.r_[0.,0.,observation['pos'][2],observation['state64'][:6],observation['state64'][6:35],[0.,0.]].astype(np.float32)
    names=observation['joint_names']
    # Canonical vocabulary, not simulator order, for named keyframes.
    from fm_humanoid_bench.protocols.reference import CANONICAL_JOINT_NAMES_29
    q=dict(zip(CANONICAL_JOINT_NAMES_29,start[9:38]))
    hold=np.tile(start,(100,1))
    if case=='pose':
        raised={n:q[n]+delta for n,delta in [('right_shoulder_pitch_joint',-.35),('right_elbow_joint',.3),('left_shoulder_pitch_joint',-.25),('left_elbow_joint',.25),('right_wrist_pitch_joint',.15),('left_wrist_pitch_joint',.15)]}
        segment=compile_keyframes([dict(frame=99,joints=raised),dict(frame=199)],start,200)
        hands=compile_keyframes([dict(frame=0,hands=[1,1]),dict(frame=99)],segment[-1],100)
        recovery=compile_keyframes([dict(frame=99,joints={n:q[n] for n in raised},hands=[0,0]),dict(frame=199)],hands[-1],200)
        return np.concatenate([hold,segment,hands,recovery])
    if case=='motion':
        moving=compile_keyframes([dict(frame=49,root_delta_xy=[.0015,0]),dict(frame=149),dict(frame=199,root_delta_xy=[0,0])],start,200)
        turning=compile_keyframes([dict(frame=99,root_rpy=[0,0,.25]),dict(frame=199),dict(frame=249,root_rpy=[0,0,0])],moving[-1],250)
        return np.concatenate([hold,moving,turning,np.tile(turning[-1],(50,1))])
    raise ValueError(case)
