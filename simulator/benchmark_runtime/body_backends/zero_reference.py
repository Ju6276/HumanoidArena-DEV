"""Predicted reference -> official BFM-Zero backward encoder (no task lookup).

FK uses the training MJCF and its configured virtual head. Reference velocities
are finite differences at 50 Hz. The official observation function is loaded
without importing its simulator class; model weights and normalization are
loaded strictly from the released full checkpoint.
"""
import ast
from collections import OrderedDict
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import yaml
from safetensors import safe_open


class ZeroReferenceEncoder:
    def __init__(self, source, checkpoint, joint_names, default, device):
        self.device=device;self.names=list(joint_names);self.default=default
        sys.path.insert(0,str(source))
        from humanoidverse.agents.nn_models import BackwardArchiConfig
        from humanoidverse.agents.nn_filters import DictInputFilterConfig
        from humanoidverse.agents.normalizers import ObsNormalizerConfig, BatchNormNormalizerConfig
        from humanoidverse.utils import torch_utils as tu
        from gymnasium import spaces
        cfg=json.loads((checkpoint/'config.json').read_text())
        dims=json.loads((checkpoint/'init_kwargs.json').read_text())['obs_space']['spaces']
        space=spaces.Dict({k:spaces.Box(-np.inf,np.inf,shape=tuple(v['shape'])) for k,v in dims.items()})
        b=cfg['archi']['b'].copy();b['input_filter']=DictInputFilterConfig(**b['input_filter'])
        self.net=BackwardArchiConfig(**b).build(space,cfg['archi']['z_dim']).to(device).eval()
        self.norm=ObsNormalizerConfig(normalizers={k:BatchNormNormalizerConfig() for k in ('state','privileged_state')},allow_mismatching_keys=True).build(space).to(device).eval()
        with safe_open(checkpoint/'model.safetensors',framework='pt',device='cpu') as f:
            self.net.load_state_dict({k.removeprefix('_backward_map.'):f.get_tensor(k) for k in f.keys() if k.startswith('_backward_map.')},strict=True)
            prefix='_obs_normalizer.'
            self.norm.load_state_dict({k.removeprefix(prefix):f.get_tensor(k) for k in f.keys() if k.startswith(prefix) and any('.'+term+'.' in k for term in ('state','privileged_state'))},strict=True)
        path=source/'humanoidverse/envs/legged_robot_motions/legged_robot_motions.py'
        tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_humanoid_observations_max')
        node.decorator_list=[]
        namespace=dict(torch=torch,Tensor=torch.Tensor,OrderedDict=OrderedDict,**{k:getattr(tu,k) for k in ('calc_heading_quat_inv','my_quat_rotate','quat_mul','quat_to_tan_norm')})
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        self.observation=namespace[node.name]
        cfg=yaml.safe_load((source/'humanoidverse/config/robot/g1/g1_29dof.yaml').read_text())['robot']
        xml=ET.parse(source/'humanoidverse/data/robots/g1/g1_29dof.xml')
        self.nodes=[]
        def visit(node,parent):
            i=len(self.nodes)
            joints=[j for j in node.findall('joint') if j.get('type')!='free']
            self.nodes.append((node.get('name'),parent,np.fromstring(node.get('pos','0 0 0'),sep=' '),R.from_quat(np.fromstring(node.get('quat','1 0 0 0'),sep=' ')[[1,2,3,0]]),[(self.names.index(j.get('name')),np.fromstring(j.get('axis','0 0 1'),sep=' ')) for j in joints]))
            for child in node.findall('body'):visit(child,i)
        visit(xml.find('worldbody/body'),-1)
        self.body_names=[x[0] for x in self.nodes]
        for ext in cfg['motion']['extend_config']:
            self.nodes.append((ext['joint_name'],self.body_names.index(ext['parent_name']),np.array(ext['pos']),R.from_quat(np.array(ext['rot'])[[1,2,3,0]]),[]))
            self.body_names.append(ext['joint_name'])
        if len(self.nodes)!=31:raise ValueError(f'Expected 31 reference bodies, got {len(self.nodes)}')

    def fk(self,ref):
        positions=[];rotations=[]
        for i,(_,parent,offset,rest,joints) in enumerate(self.nodes):
            if parent<0:
                positions.append(ref[:,:3]);rotations.append(R.from_quat(ref[:,3:7][:,[1,2,3,0]]));continue
            local=rest
            for index,axis in joints:local=local*R.from_rotvec(ref[:,7+index,None]*axis)
            positions.append(positions[parent]+rotations[parent].apply(offset))
            rotations.append(rotations[parent]*local)
        return np.stack(positions,axis=1),np.stack([r.as_quat() for r in rotations],axis=1)

    @torch.inference_mode()
    def encode(self,reference):
        ref=np.asarray(reference);p,q=self.fk(ref)
        # Center frame is the current predicted target, adjacent frames are
        # reference past/future, never a future simulator observation.
        vel=(p[2]-p[0])/.04
        omega=(R.from_quat(q[2])*R.from_quat(q[0]).inv()).as_rotvec()/.04
        dq=(ref[2,7:]-ref[0,7:])/.04
        root=R.from_quat(q[1,0]);gravity=root.inv().apply([0,0,-1])
        state=np.r_[ref[1,7:]-self.default,dq,gravity,omega[0]]
        t=lambda a:torch.as_tensor(a,dtype=torch.float32,device=self.device)[None]
        full=self.observation(t(p[1]),t(q[1]),t(vel),t(omega),True,True)
        privileged=torch.cat(list(full.values()),dim=-1)
        if privileged.shape!=(1,463):raise ValueError(privileged.shape)
        inputs={'state':t(state),'privileged_state':privileged}
        z=self.net(self.norm(inputs))
        z=16*torch.nn.functional.normalize(z,dim=-1)
        return z.cpu().numpy(),np.concatenate([state,privileged.cpu().numpy()[0]]).astype(np.float32)
