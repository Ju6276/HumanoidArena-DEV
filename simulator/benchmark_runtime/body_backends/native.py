"""Task-independent native G1 policy runtimes. No task state or simulator actions as brain inputs."""
from pathlib import Path
import os
import importlib.util
import json
import re
import sys
from types import SimpleNamespace
import numpy as np
import torch
import yaml

ARENA = Path(os.environ.get('FMHB_ROOT', Path(__file__).resolve().parents[3]))
WEIGHTS = Path(os.environ.get('FMHB_BODY_MODEL_ROOT', '/home/d024/HumanoidArena_models/body_models'))
SOURCES = Path(os.environ.get('FMHB_EXTERNAL_ROOT', ARENA / 'external'))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def match_values(patterns, names):
    result = []
    for name in names:
        matches = [v for k,v in patterns.items() if re.fullmatch(k,name)]
        if len(matches) != 1:
            raise ValueError((name,matches))
        result.append(matches[0])
    return np.asarray(result, np.float32)

def session(path):
    import onnxruntime as ort
    # Torch loads CUDA/cuDNN shared libraries before ORT creates its provider.
    if hasattr(ort,'preload_dlls'): ort.preload_dlls()
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    obj = ort.InferenceSession(str(path),sess_options=options,providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    return obj

class ReferenceBody:
    """Common controller surface: native-order state + root/quaternion/q window.

    `reference_offsets` declares the required temporal window. The Brain never
    sees native model latents. Construct a fresh Body for each episode to reset
    policy histories/caches; ReferenceTrajectory owns the reference timeline.
    """
    def infer_reference(self, state, reference):
        from fm_humanoid_bench.protocols.body_contract import validate_body_input, validate_body_output
        validate_body_input(state,reference,self.names,self.reference_offsets)
        return validate_body_output(*self.infer(state, reference))

    def reset_history(self):
        """Reset policy memory without reloading weights; no physics reset."""
        raise NotImplementedError


class Zero(ReferenceBody):
    reference_offsets = np.array([-1,0,1])
    name = 'BFM-Zero'
    def infer_reference(self, state, reference):
        from fm_humanoid_bench.protocols.body_contract import validate_body_input, validate_body_output
        validate_body_input(state,reference,self.names,self.reference_offsets)
        return validate_body_output(*self.infer(state, reference, canonical=True))

    def reset_history(self):
        import copy
        for key, term in self.obs.funcs.items():
            for field, value in self._initial_observation_memory[key].items():
                setattr(term,field,copy.deepcopy(value))
        self.last=np.zeros(29,np.float32)

    def __init__(self, device='cuda'):
        self.device=device
        cfg=yaml.safe_load((SOURCES/'BFM-Zero/config/policy/motivo_newG1.yaml').read_text())
        self.names=cfg['isaac_joint_names']; self.default=np.array([next((v for k,v in cfg['default_joint_pos'].items() if re.fullmatch(k,n)),0.) for n in self.names],np.float32)
        self.kp=match_values(cfg['joint_kp'],self.names);self.kd=match_values(cfg['joint_kd'],self.names)
        self.scale=match_values(cfg['action_scale'],self.names);self.rescale=cfg['action_rescale']
        self.path=WEIGHTS/'BFM-Zero/model/exported/FBcprAuxModel.onnx';self.policy=session(self.path)
        sys.path.insert(0,str(SOURCES/'BFM-Zero'))
        from rl_policy.observations import Observation,ObsGroup
        self.state_processor=SimpleNamespace(num_dof=29)
        self.num_actions=29
        self.obs=ObsGroup('policy',{k:Observation.registry[k](env=self,**v) for k,v in cfg['observation']['policy'].items()})
        import copy
        self._initial_observation_memory={k:{field:copy.deepcopy(value) for field,value in vars(term).items() if isinstance(value,np.ndarray)} for k,term in self.obs.funcs.items()}
        import joblib
        rewards=joblib.load(WEIGHTS/'BFM-Zero/model/reward_inference/reward_locomotion.pkl')
        goals=joblib.load(WEIGHTS/'BFM-Zero/model/goal_inference/goal_reaching.pkl')
        self.latents={k:np.asarray(v,np.float32).reshape(-1,256)[0:1] for k,v in {**rewards,**goals}.items()}
        self.latents.update({f'{k}#{i}':np.asarray(z,np.float32).reshape(1,256) for k,v in rewards.items() for i,z in enumerate(v)})
        self.command='move-ego-0-0';self.last=np.zeros(29,np.float32)
        robot_cfg=yaml.safe_load((SOURCES/'BFM-Zero/config/robot/g1.yaml').read_text())
        self.lower=match_values(robot_cfg['joint_pos_lower_limit'],self.names)
        self.upper=match_values(robot_cfg['joint_pos_upper_limit'],self.names)
        import xml.etree.ElementTree as ET
        xml=ET.parse(SOURCES/'BFM-Zero/data/robots/g1/g1_29dof_freebase.xml')
        motor_limits={x.attrib['joint']:float(x.attrib['ctrlrange'].split()[1]) for x in xml.findall('.//actuator/motor') if 'ctrlrange' in x.attrib}
        self.limits=match_values(motor_limits,self.names)
    def infer(self,state,reference=None,canonical=False):
        bridge_obs=None
        if canonical:
            if not hasattr(self,'reference_encoder'):
                from benchmark_runtime.body_backends.zero_reference import ZeroReferenceEncoder
                self.reference_encoder=ZeroReferenceEncoder(SOURCES/'BFM-Zero-inference',WEIGHTS/'BFM-Zero/model/checkpoint/model',self.names,self.default,self.device)
            z,bridge_obs=self.reference_encoder.encode(reference)
        else:z=self.latents[self.command]
        sp=self.state_processor
        sp.joint_pos=state['q'];sp.joint_vel=state['dq'];sp.root_quat_b=state['quat'];sp.root_ang_vel_b=state['omega']
        for term in self.obs.funcs.values():term.update({'action':self.last})
        obs=np.concatenate([self.obs.compute(),z.reshape(-1)]).astype(np.float32)[None]
        raw=self.policy.run(None,{'actor_obs':obs})[0]
        self.last=np.clip(raw[0],-1,1)*self.rescale
        data={'obs':obs,'raw_action':raw}
        if canonical:data.update(conditioning=z.reshape(-1),backward_observation=bridge_obs)
        return np.clip(self.default+self.last*self.scale,self.lower,self.upper), data

class Scale(ReferenceBody):
    reference_offsets = np.arange(6)
    name='ScaleBFM'
    def reset_history(self):
        self.history=None
        self.last=torch.zeros(1,29,device=self.device)
    def __init__(self,device='cuda',size='m'):
        self.device=device
        base=WEIGHTS/f'ScaleBFM/compiled_checkpoint/humanoid_transformer_{size}/linux'
        self.meta=json.loads((base/'model_22200_tensorrt_metadata.json').read_text());m=self.meta
        self.names=m['joint_names'];self.default=np.array(m['default_dof_pos'],np.float32)
        self.kp=np.array(m['stiffness'],np.float32);self.kd=np.array(m['damping'],np.float32)
        self.limits=np.array(m['torque_limit'],np.float32)
        self.path=WEIGHTS/f'ScaleBFM/checkpoint/humanoid_transformer_{size}/model_22200.pt'
        net=module('scale_official_network',SOURCES/'ScaleBFM/ScaleTrack/source/my_rsl_rl/my_rsl_rl/networks/humanoid_transformer.py')
        from benchmark_runtime.body_backends import scale_export_runtime as w
        a=m['policy_architecture']
        actor=net.HumanoidTransformer(a['prop_obs_dim'],29,29,a['embedding_dim'],a['num_heads'],a['ff_dim'],a['num_layers'])
        task=net.TaskEmbedder(a['task_obs_dim'],a['embedding_dim'],a['reduced_task_dim'],a['task_embedder_hidden_dims'])
        weights=torch.load(self.path,map_location='cpu',weights_only=False)['model_state_dict']
        actor.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith('actor.')},strict=True)
        task.load_state_dict({k[len('actor_task_embedder.'):]:v for k,v in weights.items() if k.startswith('actor_task_embedder.')},strict=True)
        modes=torch.load(base/'mode_table.pt',map_location=device,weights_only=True)
        self.xml=w.parse_xml(SOURCES/'ScaleBFM/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml',device)
        bodies,joints,parents,axis,translation,rotation=self.xml
        selected=torch.tensor([bodies.index(x) for x in m['selected_body_names']],device=device)
        order=torch.tensor([self.names.index(x) for x in joints],device=device)
        self.wrapper=w.HumanoidTransformerPolicyWrapperWithMode(SimpleNamespace(actor=actor,actor_task_embedder=task),w.build_mode_mappings(modes,m['mode_feature_dims']),modes,torch.tensor(self.default,device=device)[None],torch.tensor(m['action_scale'],device=device)[None],m['history_buffer_size'],len(m['future_idx']),translation,rotation,parents,axis,selected,order).to(device).eval()
        self.w=w;self.history=None;self.last=torch.zeros(1,29,device=device)
    def fk(self,q):
        w=self.w;bodies,joints,parents,axis,translation,rotation=self.xml
        q=q[:,self.wrapper.lab_to_xml_joint_indices];half=q[...,None]/2
        jr=torch.cat([torch.cos(half),axis[None]*torch.sin(half)],-1)
        pos=torch.zeros(len(q),len(bodies),3,device=self.device);rot=torch.zeros(len(q),len(bodies),4,device=self.device);rot[...,0]=1
        for j in range(1,len(bodies)):
            p=int(parents[j]);pos[:,j]=pos[:,p]+w.quat_apply(rot[:,p],translation[j-1].expand(len(q),-1))
            rot[:,j]=w.quat_mul(rot[:,p],w.quat_mul(rotation[j-1].expand(len(q),-1),jr[:,j-1]))
        return pos[:,self.wrapper.selected_link_indices],rot[:,self.wrapper.selected_link_indices]
    def infer(self,state,reference):
        t=lambda x:torch.as_tensor(x,dtype=torch.float32,device=self.device)
        curr=[t(state['quat'])[None],t(state['omega'])[None],t(state['q'])[None],t(state['dq'])[None],self.last]
        n=self.meta['history_buffer_size']
        if self.history is None:self.history=[x[:,None].repeat(1,n,1) for x in curr]
        else:self.history=[torch.cat([h[:,1:],x[:,None]],1) for h,x in zip(self.history,curr)]
        ref=t(reference[:6]);p,r=self.fk(ref[:,7:])
        rq=ref[:,3:7,None].transpose(1,2).expand(-1,len(self.meta['selected_body_names']),-1)
        world_p=self.w.quat_apply(rq,p)+ref[:,:3,None].transpose(1,2)
        world_r=self.w.quat_mul(rq,r)
        currentq=t(state['quat'])[None,None].expand_as(world_r)
        p=self.w.quat_apply_inverse(currentq,world_p-t(state['pos'])[None,None])
        r=self.w.quat_mul_inverse_left(currentq,world_r)
        inputs=self.history+[p[None],r[None],torch.tensor([7],device=self.device),t(self.meta['future_idx'])[None,:,None]]
        target,raw=self.wrapper(*inputs);self.last=raw.detach()
        return target[0].cpu().numpy(),{'obs':torch.cat([x.flatten() for x in inputs]).cpu().numpy()[None], 'raw_action':raw.cpu().numpy(), 'conditioning':torch.cat([x.flatten() for x in inputs[5:]]).cpu().numpy()}

class Holo(ReferenceBody):
    reference_offsets = np.arange(-1,12)
    name='HoloMotion'
    def reset_history(self):
        self.kv.fill(0);self.step=0;self.last=np.zeros(29,np.float32)
    def __init__(self,device='cuda'):
        self.device=device
        base=WEIGHTS/'HoloMotion/HoloMotion_motion_tracking_model_v1.4.1'
        cfg=yaml.safe_load((base/'config.yaml').read_text())['robot']
        self.names=cfg['dof_names'];self.default=match_values(cfg['init_state']['default_joint_angles'],self.names)
        pd=cfg['actuators']['all_joints'];self.kp=match_values(pd['stiffness'],self.names);self.kd=match_values(pd['damping'],self.names)
        self.scale=match_values(cfg['actuators']['action_scale'],self.names)
        self.limits=match_values(pd['effort_limit_sim'],self.names)
        self.path=base/'exported/model_16200.onnx';self.policy=session(self.path)
        meta=self.policy.get_modelmeta().custom_metadata_map
        values=lambda k:np.asarray([float(x) for x in meta[k].split(',') if x],np.float32)
        self.names=[x for x in meta['joint_names'].split(',') if x]
        self.default=values('default_joint_pos');self.scale=values('action_scale')
        self.kp=values('joint_stiffness');self.kd=values('joint_damping')
        self.limits=match_values(pd['effort_limit_sim'],self.names)
        self.rope_max_seq_len=int(meta.get('rope_max_seq_len','8192'))
        sys.path.insert(0,str(SOURCES/'HoloMotion'))
        from holomotion.src.motion_tracking.actor_observation import MotionActorObservationInput,build_motion_actor_observation_torch
        self.Input=MotionActorObservationInput;self.build=build_motion_actor_observation_torch
        self.kv=np.zeros((1,2,1,32,4,64),np.float32);self.step=0;self.last=np.zeros(29,np.float32)
    def infer(self,state,reference):
        t=lambda x:torch.as_tensor(x,dtype=torch.float32,device=self.device)
        inp=self.Input(reference_qpos=t(reference),robot_root_quat_wxyz=t(state['quat']),robot_root_angvel_local=t(state['omega']),robot_dof_pos=t(state['q']),robot_dof_vel=t(state['dq']),last_action=t(self.last),default_dof_pos=t(self.default))
        obs=self.build(inp,fps=50,current_index=1,num_future_frames=10).cpu().numpy()
        raw,kv,*_=self.policy.run(None,{'obs':obs,'past_key_values':self.kv,'step_idx':np.array([self.step],np.int64)})
        self.kv=kv;self.step+=1;self.last=raw[0]
        return self.default+self.last*self.scale,{'obs':obs,'raw_action':raw,'conditioning':np.r_[obs.reshape(-1)[:41],obs.reshape(-1)[134:]]}

def create(name,device='cuda'):
    if name in ('sonic','twist2'):
        from benchmark_runtime.body_backends.legacy import Sonic,Twist2
        return {'sonic':Sonic,'twist2':Twist2}[name](device)
    return {'scale':Scale,'zero':Zero,'holo':Holo}[name](device)
