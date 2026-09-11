"""CPU policy checks against extracted original HA provider methods.

The oracle executes the original methods from source without importing Isaac,
Redis or DDS. Actual released ONNX weights run on both sides. This validates the
controller boundary, not robot dynamics or task success.
"""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import time
import re
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'simulator'))

from fm_humanoid_bench.protocols.reference import ReferenceTrajectory
from action_provider.vla_robot_current_local_runtime_v3 import (
    UnifiedRobotCurrentLocalActionRuntimeV3, build_twist2_mimic_obs_v3,
    build_sonic_joint29_payload_v3, CANONICAL_JOINT_NAMES_29,
)
from action_provider.vla_smpl_runtime import quat_to_rot6d_wxyz
from benchmark_runtime.body_backends.legacy import Sonic, Twist2, ARENA

REPO = ARENA / 'simulator'


def source_class(filename, classname, methods, constants=(), functions=()):
    tree=ast.parse((REPO/'action_provider'/filename).read_text())
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets):
            nodes.append(node)
        elif isinstance(node,ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node,ast.ClassDef) and node.name==classname:
            node.bases=[];node.decorator_list=[];node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in methods]
            nodes.append(node)
    ns={'np':np,'torch':torch,'time':time}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,'exec'),ns)
    return ns[classname]


def make_sonic_oracle(body,state):
    cls=source_class('action_provider_sonic.py','SonicActionProvider',{'_setup_buffers','_run_gear_sonic'},
        constants={'SONIC_ISAACLAB_JOINT_ORDER','SONIC_DEFAULT_POS','G1_ACTION_SCALE_ISAACLAB','_STEP1_FRAMES','_STEP5_FRAMES','_STEP5_STRIDE','_STEP5_HISTORY_LEN','_N_SMPL_JOINTS','_N_SMPL_POSES','OFFICIAL_WRIST_INDICES'},
        functions={'quat_to_rotation_6d','quat_normalize_wxyz','quat_conjugate_wxyz','quat_mul_wxyz','compute_anchor_rot6d_wxyz','gravity_dir_from_base_quat_wxyz','build_latest_hold_window'})
    obj=cls();obj._sonic_default_np=body.default.copy();obj._use_lerobot_vla=False;obj._setup_buffers()
    obj._robot_joint_pos_hist[:]=state['q']-body.default;obj._robot_joint_vel_hist[:]=state['dq'];obj._last_action_hist.fill(0)
    obj._sonic_debug=False;obj._sonic_joint29_mode=True;obj._smpl_data_valid=True;obj._latest_consumed_new_this_step=True
    obj._encoder=body.encoder;obj._decoder=body.policy;obj._sonic_idx=list(range(29))
    return obj


def make_twist_oracle(body):
    cls=source_class('action_provider_wh_twist2.py','TWIST2ActionProvider',{'_twist2_roll_pitch_from_quaternion','compute_current_observations','compute_observations','run_policy'})
    obj=cls();obj.twist2_action_indices=list(range(29));obj.twist2_default_pos=torch.from_numpy(body.default[None])
    obj._twist2_ankle_idx=[4,5,10,11];obj._twist2_last_action=torch.zeros(1,29);obj._twist2_history=torch.zeros(10,127)
    obj._use_lerobot_vla=False;obj.enable_dex3=False;obj._twist2_hand_dim=7;obj._twist2_neck_dim=2;obj.clip_obs=100
    obj._twist2_publish_state=lambda *args:None
    obj.policy=lambda x:torch.from_numpy(body.policy.run(None,{body.policy.get_inputs()[0].name:x.numpy()})[0])
    return obj


def fake_env(state):
    d=SimpleNamespace(root_state_w=torch.from_numpy(np.r_[state['pos'],state['quat'],np.zeros(6,np.float32)][None]),root_ang_vel_b=torch.from_numpy(state['omega'][None]),joint_pos=torch.from_numpy(state['q'][None]),joint_vel=torch.from_numpy(state['dq'][None]))
    return SimpleNamespace(scene={'robot':SimpleNamespace(data=d)},device='cpu')


def states_and_actions(body,n=15):
    rng=np.random.default_rng(811)
    states=[];actions=[]
    indices=[body.names.index(name) for name in CANONICAL_JOINT_NAMES_29]
    for tick in range(n):
        quat=Rotation.from_euler('xyz',[.04*np.sin(tick),.03*np.cos(tick),.7+.001*tick]).as_quat()[[3,0,1,2]].astype(np.float32)
        state=dict(pos=np.array([.2+.001*tick,-.3,.793],np.float32),quat=quat,q=(body.default+rng.normal(0,.03,29)).astype(np.float32),dq=rng.normal(0,.1,29).astype(np.float32),omega=rng.normal(0,.1,3).astype(np.float32))
        target_quat=Rotation.from_euler('xyz',[.12,.08,.02*tick]).as_quat()[[3,0,1,2]].astype(np.float32)
        action=np.r_[.001,-.0003,.78+.001*tick,quat_to_rot6d_wxyz(target_quat),body.default[indices]+rng.normal(0,.04,29),0,0].astype(np.float32)
        states.append(state);actions.append(action)
    return states,np.stack(actions)


class LegacyNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.bodies=[Sonic('cpu'),Twist2('cpu')]

    def test_pd_parameters_match_installed_ha_source(self):
        # Read data-only AST from the HA robot asset; importing the module would
        # start Isaac dependencies. Ensures the common runner preserves tuning.
        tree = ast.parse((REPO/'robots/unitree.py').read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'G129_CFG_WITH_DEX3_WHOLEBODY' for t in n.targets))
        actuators = next(k.value for k in assignment.value.keywords if k.arg == 'actuators')
        values = {key:{} for key in ('stiffness','damping','effort_limit_sim')}
        for call in actuators.values:
            expressions = ast.literal_eval(next(k.value for k in call.keywords if k.arg == 'joint_names_expr'))
            if not any(re.fullmatch(pattern,name) for pattern in expressions for name in self.bodies[1].names):
                continue
            for k in call.keywords:
                if k.arg in values:
                    values[k.arg].update(ast.literal_eval(k.value))
        body = self.bodies[1]
        for key, actual in [('stiffness',body.kp),('damping',body.kd),('effort_limit_sim',body.limits)]:
            expected = []
            for name in body.names:
                matches = [v for pattern,v in values[key].items() if re.fullmatch(pattern,name)]
                self.assertEqual(len(matches),1)
                expected.append(matches[0])
            np.testing.assert_array_equal(actual,np.asarray(expected,np.float32))

    def test_reference_matches_original_provider_every_tick(self):
        for body in self.bodies:
            with self.subTest(body=body.name):
                body.reset_history();states,actions=states_and_actions(body)
                trajectory=ReferenceTrajectory(states[0],body.names);trajectory.submit(actions,states[0])
                original_runtime=UnifiedRobotCurrentLocalActionRuntimeV3(root_rot6d_layout='row')
                original_runtime.reset(body_xy_world=states[0]['pos'][:2],target_root_quat_wxyz=states[0]['quat'])
                oracle=make_sonic_oracle(body,states[0]) if body.name=='SONIC' else make_twist_oracle(body)
                for tick,(state,action) in enumerate(zip(states,actions)):
                    frame=original_runtime.step(action,current_robot_quat_wxyz=state['quat'],current_robot_xy_world=state['pos'][:2])
                    oracle.env=fake_env(state)
                    if body.name=='SONIC':
                        payload=build_sonic_joint29_payload_v3(runtime_frame=frame,control_dt=.02)
                        oracle._motion_joint_pos_hist[-1]=payload['joint_pos'];oracle._motion_joint_vel_hist[-1]=payload['joint_vel'];oracle._ref_body_quat_window[-1]=payload['body_quat_w']
                        expected_target=oracle._run_gear_sonic();expected_obs=oracle._latest_decoder_obs;expected_raw=oracle._latest_decoder_raw_action;expected_condition=oracle._latest_encoder_input
                    else:
                        mimic=build_twist2_mimic_obs_v3(runtime_frame=frame,control_dt=.02)
                        oracle._twist2_fetch_actions=lambda:torch.from_numpy(mimic[None])
                        expected_raw,expected_obs=oracle.run_policy();expected_raw=expected_raw.numpy().reshape(-1);expected_obs=expected_obs.numpy().reshape(-1)
                        expected_target=body.default+np.clip(expected_raw,-10,10)*.5;expected_condition=mimic
                    target,data=body.infer_reference(state,trajectory.window(body.reference_offsets))
                    for label,actual,expected in [('target',target,expected_target),('obs',data['obs'].reshape(-1),expected_obs),('raw',data['raw_action'].reshape(-1),expected_raw),('condition',data['conditioning'],expected_condition)]:
                        np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=2e-5,err_msg=f'{body.name} {label} tick {tick}')
                    trajectory.commit(state)

    def test_reset_reproducible_and_native_equivalent(self):
        for body in self.bodies:
            with self.subTest(body=body.name):
                states,actions=states_and_actions(body,2);state=states[0]
                trajectory=ReferenceTrajectory(state,body.names);trajectory.submit(actions,state);ref=trajectory.window(body.reference_offsets)
                body.reset_history();target,data=body.infer_reference(state,ref)
                body.reset_history()
                if body.name=='SONIC':command=dict(joints=dict(zip(body.names,ref[1,7:])),root_quat_wxyz=ref[1,3:7],joint_velocities=np.zeros(29,np.float32))
                else:
                    c=body.reference_condition(ref);command=dict(velocity=c[:2],root_z=c[2],root_roll_pitch=c[3:5],yaw_rate=c[5],joints=dict(zip(body.names,c[6:])))
                nt,nd=body.infer_native(state,command)
                np.testing.assert_array_equal(target,nt);np.testing.assert_array_equal(data['obs'],nd['obs'])
                body.infer_native(state,command);body.reset_history()
                rt,rd=body.infer_reference(state,ref);np.testing.assert_array_equal(target,rt);np.testing.assert_array_equal(data['obs'],rd['obs'])

    def test_native_invalid_request_does_not_advance_history(self):
        for body in self.bodies:
            body.reset_history();states,_=states_and_actions(body,1)
            for command in [dict(joints={}),dict(joints=[0]*29,velocity=[float('nan'),0]),dict(joints=[0]*29,wrong=1)]:
                with self.assertRaises((ValueError,KeyError)):
                    body.infer_native(states[0],command)
                self.assertEqual(body.steps,0)

    def test_declared_missing_position_conditions_are_invariant(self):
        for body in self.bodies:
            states,actions=states_and_actions(body,2);state=states[0];trajectory=ReferenceTrajectory(state,body.names);trajectory.submit(actions,state);ref=trajectory.window(body.reference_offsets)
            body.reset_history();a,da=body.infer_reference(state,ref)
            # Translate reference origin and measured state together: no world XY
            # origin should leak into any native input. Shift measured XY alone:
            # neither selected legacy path has root position error conditioning.
            translated=dict(state,pos=state['pos']+np.array([7,-8,0],np.float32))
            body.reset_history();b,db=body.infer_reference(translated,ref)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(da['obs'],db['obs'])
            if body.name=='SONIC':
                ref2=ref.copy();ref2[:,:3]+=np.array([3,-4,.12],np.float32)
                body.reset_history();c,dc=body.infer_reference(state,ref2)
                np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(da['conditioning'],dc['conditioning'])


if __name__=='__main__':unittest.main()
