from types import SimpleNamespace
import unittest
import numpy as np
from fm_humanoid_bench.protocols.native_commands import validate_request,validate_command

class NativeCommandTests(unittest.TestCase):
    def test_zero_rejects_joint_commands_or_unknown_latents(self):
        body=SimpleNamespace(name='BFM-Zero',latents={'walk':np.zeros(256)})
        self.assertEqual(validate_command(body,{'latent':'walk'}),{'latent':'walk'})
        for c in [{'latent':'unknown'},{'latent':'walk','joints':[0]*29}]:
            with self.assertRaises(ValueError):validate_command(body,c)

    def test_strict_envelope_and_finite_named_pose(self):
        body=SimpleNamespace(name='TWIST2',names=[f'joint{i}' for i in range(29)])
        command=dict(joints={n:0. for n in body.names},velocity=[0,0],root_z=.8,root_roll_pitch=[0,0],yaw_rate=0.)
        request=dict(op='native_step',schema='arena_native_body_v1',frames=20,command=command,hands=[0,1])
        c,n,h=validate_request(body,request);self.assertEqual(n,20);self.assertEqual(len(c['joints']),29)
        for change in [{'frames':True},{'hands':[0,2]},{'unexpected':1}]:
            with self.assertRaises(ValueError):validate_request(body,{**request,**change})
        with self.assertRaises(ValueError):validate_command(body,{**command,'velocity':[float('nan'),0]})

if __name__=='__main__':unittest.main()
