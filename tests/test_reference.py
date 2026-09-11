import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from fm_humanoid_bench.protocols.reference import ReferenceTrajectory,validate_chunk,compile_keyframes,CANONICAL_JOINT_NAMES_29
from action_provider.vla_smpl_runtime import quat_to_rot6d_wxyz


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.names=list(CANONICAL_JOINT_NAMES_29)
        self.state=dict(pos=np.array([1.,2.,.78]),quat=np.array([1.,0,0,0]),q=np.arange(29)*.01,dq=np.zeros(29))
        self.a=np.zeros((30,40),np.float32);self.a[:,:2]=[.01,0];self.a[:,2]=.78
        self.a[:,3:9]=quat_to_rot6d_wxyz(self.state['quat']);self.a[:,9:38]=self.state['q']

    def test_preview_does_not_commit_future(self):
        t=ReferenceTrajectory(self.state,self.names);t.submit(self.a,self.state)
        np.testing.assert_allclose(t.window([0,5])[:,0],[1.01,1.06],atol=1e-6)
        t.commit(self.state);t.submit(self.a,self.state)
        np.testing.assert_allclose(t.window([0])[0,0],1.02,atol=1e-6)

    def test_anchor_survives_tracking_error(self):
        t=ReferenceTrajectory(self.state,self.names);t.submit(self.a,self.state);t.commit(self.state)
        moved={**self.state,'pos':np.array([10.,10.,.5])}
        t.submit(self.a,moved)
        np.testing.assert_allclose(t.window([0])[0,:3],[1.02,2,.78],atol=1e-6)

    def test_episode_heading_and_joint_order(self):
        state={**self.state,'quat':Rotation.from_euler('z',np.pi/2).as_quat()[[3,0,1,2]],'q':self.state['q'][::-1],'dq':self.state['dq'][::-1]}
        t=ReferenceTrajectory(state,self.names[::-1]);t.submit(self.a,state)
        np.testing.assert_allclose(t.window([0])[0,:3],[1,2.01,.78],atol=1e-6)
        np.testing.assert_allclose(t.window([0])[0,7:],self.state['q'][::-1])
        np.testing.assert_allclose(t.observation(state)[6:35],self.state['q'])

    def test_past_and_padding_hold_pose(self):
        t=ReferenceTrajectory(self.state,self.names);t.submit(self.a[:2],self.state)
        w=t.window([-1,0,1,20]);np.testing.assert_allclose(w[0,:3],self.state['pos'])
        np.testing.assert_allclose(w[2],w[3]);t.commit(self.state)
        np.testing.assert_allclose(t.window([-1])[0],w[1])

    def test_rotation_and_hand_knots(self):
        a=compile_keyframes([dict(frame=9,root_rpy=[0,0,1.],hands=[1,0])],self.a[0],10)
        np.testing.assert_allclose(a[-1,3:9],quat_to_rot6d_wxyz(Rotation.from_euler('z',1).as_quat()[[3,0,1,2]]),atol=1e-6)
        self.assertTrue((a[:9,38]==0).all());self.assertEqual(a[9,38],1)

    def test_bad_reference_rejected(self):
        for bad in (np.zeros((30,39)),np.zeros((30,40)),self.a*np.nan):
            with self.assertRaises(ValueError):validate_chunk(bad)

    def test_prompt_reference_conditions_do_not_mix_modalities(self):
        import tempfile,json
        from pathlib import Path
        from fm_humanoid_bench.brains.agent_prompt import packet
        t=ReferenceTrajectory(self.state,self.names)
        obs=dict(step=0,pos=self.state['pos'].tolist(),state64=t.observation(self.state).tolist(),hands=[0,0],ego_image='current.png',compact_proprioception={})
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'reference.json'
            (Path(d)/'demo.png').write_bytes(b'fixture')
            p.write_text(json.dumps(dict(schema='arena_episode_reference_v1',numeric_schema='unitree_g1_gmt_refpose_v3_1',task='Open the door.',episode=0,source_dataset='train',reference_split='train',state_action=[{'frame':0,'time_seconds':0.,'state64':obs['state64'],'action40':self.a[0].tolist()}],images=['demo.png'],selected_indices=[0])))
            none,_=packet(obs,[],None,'none');numeric,_=packet(obs,[],p,'state_action');visual,_=packet(obs,[],p,'images')
            self.assertNotIn('demonstration',none['user'])
            self.assertNotIn('ego_images',numeric['user']['demonstration'])
            self.assertNotIn('state_action',visual['user']['demonstration'])
            self.assertEqual(len({x['prompt_sha256'] for x in [none,numeric,visual]}),3)

if __name__=='__main__':unittest.main()
