import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from fm_humanoid_bench.protocols.body_contract import accept_reference, validate_body_input, validate_reference_request
from fm_humanoid_bench.protocols.reference import ReferenceTrajectory, CANONICAL_JOINT_NAMES_29


class BodyContractTests(unittest.TestCase):
    def test_complete_request_rejects_ambiguous_and_malformed_commands(self):
        from fm_humanoid_bench.protocols.reference import SCHEMA
        action=np.zeros((2,40));action[:,3:9]=[1,0,0,1,0,0]
        request=dict(op='reference_step',schema=SCHEMA,control_dt=.02,action_chunk=action.tolist(),frames=1)
        validate_reference_request(request,'ScaleBFM',range(6))
        for change in ({'hands':[1,1]},{'joints':{}},{'frames':1.2},{'frames':True},{'frames':3},{'allow_native_approximation':'false'}):
            with self.assertRaises(ValueError):validate_reference_request({**request,**change},'ScaleBFM',range(6))

    def test_unsupported_semantics_rejected_even_with_approximation(self):
        for body in ('HoloMotion','BFM-Zero'):
            with self.assertRaises(ValueError):
                accept_reference(body,[-1,0,1],['root_position_error'],True)
        with self.assertRaises(ValueError):
            accept_reference('BFM-Zero',[-1,0,1],['root_heading_error'],True)

    def test_semantic_losses_require_explicit_acknowledgement(self):
        with self.assertRaises(ValueError):accept_reference('HoloMotion',[-1,0,1])
        self.assertTrue(accept_reference('HoloMotion',[-1,0,1],['joint_pose'],True)['accepted'])
        self.assertEqual(accept_reference('ScaleBFM',range(6),['root_position_error'])['semantic_losses'],[])

    def test_reconstructed_poses_match_independent_se3_oracle_across_chunks(self):
        rng=np.random.default_rng(84);names=list(CANONICAL_JOINT_NAMES_29)[::-1]
        initial=Rotation.from_euler('z',1.2)
        state=dict(pos=np.array([2.,-3.,.8]),quat=initial.as_quat()[[3,0,1,2]],q=np.zeros(29),dq=np.zeros(29),omega=np.zeros(3))
        trajectory=ReferenceTrajectory(state,names)
        actions=np.zeros((67,40),np.float32)
        actions[:,:2]=rng.uniform(-.005,.005,(67,2))
        actions[:,2]=.8+.02*np.sin(np.arange(67)*.08)
        rotations=Rotation.from_euler('xyz',rng.uniform(-.15,.15,(67,3)))
        actions[:,3:9]=rotations.as_matrix()[:,:,:2].reshape(-1,6)
        actions[:,9:38]=rng.uniform(-.1,.1,(67,29))
        xy=state['pos'][:2].copy();previous_z=actions[0,2];expected=[]
        for a,rot in zip(actions,rotations):
            world=initial*rot;matrix=world.as_matrix()
            delta=np.r_[a[:2],0.]
            delta[2]=(a[2]-previous_z-matrix[2,:2]@a[:2])/matrix[2,2]
            xy=xy+(matrix@delta)[:2];previous_z=a[2]
            expected.append(np.r_[xy,a[2],world.as_quat()[[3,0,1,2]],a[9:38][::-1]])
        for start,end in ((0,13),(13,31),(31,67)):
            trajectory.submit(actions[start:end],state)
            for k in range(start,end):
                w=trajectory.window([-1,0,1,11])
                np.testing.assert_allclose(w[1],expected[k],atol=3e-6)
                np.testing.assert_allclose(w[-1],expected[min(k+11,end-1)],atol=3e-6)
                if k:np.testing.assert_allclose(w[0],expected[k-1],atol=3e-6)
                trajectory.commit(state)

    def test_native_boundary_rejects_bad_order_shape_and_quaternion(self):
        names=list(CANONICAL_JOINT_NAMES_29)
        state=dict(pos=np.zeros(3),quat=np.array([1.,0,0,0]),q=np.zeros(29),dq=np.zeros(29),omega=np.zeros(3))
        ref=np.zeros((3,36));ref[:,3]=1
        validate_body_input(state,ref,names,[-1,0,1])
        for invalid in (ref[:2],ref*np.nan,ref*np.array([1]*3+[0]*4+[1]*29)):
            with self.assertRaises(ValueError):validate_body_input(state,invalid,names,[-1,0,1])
        with self.assertRaises(ValueError):validate_body_input(state,ref,names[:-1]+names[:1],[-1,0,1])


if __name__=='__main__':unittest.main()
