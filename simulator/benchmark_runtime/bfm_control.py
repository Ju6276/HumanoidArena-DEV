"""Paused GPT brain RPC with native ScaleBFM / BFM-Zero / HoloMotion inference.
The Arena environment is initialized before creating any legacy policy provider.
"""
import os,json,re,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
import zmq
import cv2
from PIL import Image
from isaaclab.utils.math import quat_apply_inverse,quat_mul,quat_conjugate
from benchmark_runtime.body_backends.native import create,module,SOURCES,WEIGHTS
from tools.get_reward import sync_reward_after_physics_step

def run_control(env,controller,provider,args,app):
    root=Path(args.recording_save_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    observation_dir=root/'debug'/'observations';observation_dir.mkdir(parents=True,exist_ok=True)
    if (root/'task_evaluation.json').exists() or ((root/'brain_decisions.jsonl').exists() and (root/'brain_decisions.jsonl').stat().st_size):raise FileExistsError(root)
    body=create(os.environ['BFM_BACKEND'],str(env.device));robot=env.scene['robot']
    idx=[robot.joint_names.index(n) for n in body.names]
    ti=torch.tensor(idx,device=env.device)
    t=lambda x:torch.as_tensor(x,dtype=torch.float32,device=env.device)
    # Install native PD gains into either implicit or explicit Isaac actuators.
    kp=robot.data.joint_stiffness.clone();kd=robot.data.joint_damping.clone()
    kp[:,ti]=t(body.kp);kd[:,ti]=t(body.kd)
    limits=robot.data.joint_effort_limits.clone()
    if hasattr(body,'limits'):limits[:,ti]=t(body.limits)
    for actuator in robot.actuators.values():
        js=actuator.joint_indices
        actuator.stiffness[:]=kp[:,js];actuator.damping[:]=kd[:,js]
        if hasattr(body,'limits'):
            actuator.effort_limit[:]=limits[:,js]
            robot.write_joint_effort_limit_to_sim(limits[:,js],joint_ids=js)
        if actuator.is_implicit_model:
            robot.write_joint_stiffness_to_sim(kp[:,js],joint_ids=js)
            robot.write_joint_damping_to_sim(kd[:,js],joint_ids=js)
    # Arena reset pose is preserved; no hidden settling or unrecorded physics.
    from fm_humanoid_bench.environments.sim_protocol import original_task_protocol,refresh_cameras,check_control_tick,evaluate_tick
    evaluator,detector,max_steps,protocol=original_task_protocol(env,args,os.environ.get("ARENA_EVALUATOR_BACKEND","twist2"))
    (root/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2))
    provenance=dict(brain='GPT-6 current Codex session',body=body.name,checkpoint=str(body.path),checkpoint_sha256=hashlib.sha256(body.path.read_bytes()).hexdigest(),native_joint_names=body.names,kp=body.kp.tolist(),kd=body.kd.tolist(),five_point_interface=False,task_reference_replay=False,legacy_twist2_policy_called=body.name=='TWIST2',legacy_provider_used=False,observation='front ego + robot proprioception',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=getattr(body,'source_dir',SOURCES/body.name),text=True).strip(),schema='native_bfm_episode_v1_multicam')
    provenance['runner_pid']=os.getpid()
    provenance['inference_providers']=body.policy.get_providers() if hasattr(getattr(body,'policy',None),'get_providers') else ['PyTorch CUDA']
    provenance['checkpoint_paths']={k:str(v) for k,v in getattr(body,'checkpoint_paths',{'actor':body.path}).items()}
    provenance['checkpoint_hashes']={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in provenance['checkpoint_paths'].items()}
    provenance['native_mode']=getattr(body,'native_mode',body.name+' released reference mode')
    provenance['native_target_limits']=bool(hasattr(body,'lower'))
    provenance['runtime_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).parent/'body_backends/native.py']}
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2))
    brain_name=os.environ.get('ARENA_BRAIN','gpt6')
    provenance['brain']=brain_name
    provenance['task_reference_replay']=brain_name in ('reference_replay','realized_replay')
    provenance['reference_source_kind']=brain_name
    provenance['reference_protocol']=os.environ.get('ARENA_REFERENCE_PROTOCOL','native')
    if provenance['reference_protocol']=='unitree_g1_gmt_refpose_v3_1':
        provenance['reference_runtime_sha256']=hashlib.sha256((Path(__file__).resolve().parents[2]/'fm_humanoid_bench/protocols/reference.py').read_bytes()).hexdigest()
        provenance['reference_window_offsets']=body.reference_offsets.tolist()
        provenance['reference_padding']='hold_last_pose'
        if body.name=='BFM-Zero':
            checkpoint=WEIGHTS/'BFM-Zero/model/checkpoint/model/model.safetensors'
            provenance['backward_checkpoint']=str(checkpoint)
            provenance['backward_checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            provenance['backward_source_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCES/'BFM-Zero-inference',text=True).strip()
            provenance['backward_bridge_sha256']=hashlib.sha256((Path(__file__).parent/'body_backends/zero_reference.py').read_bytes()).hexdigest()
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2))
    steps=0;terminal=None;streak=0;reward=None;frames=[];writers={}
    log=(root/'brain_decisions.jsonl').open('a',buffering=1);timeline=(root/'simulation_timeline.jsonl').open('a',buffering=1)
    socket=zmq.Context.instance().socket(zmq.REP);socket.bind(os.environ['ASTRA_CONTROL_ENDPOINT'])
    joint_ref=body.default.copy();velocity=np.zeros(3);hands=[0,0]
    def state():
        d=robot.data
        return dict(q=d.joint_pos[0,ti].cpu().numpy(),dq=d.joint_vel[0,ti].cpu().numpy(),quat=d.root_quat_w[0].cpu().numpy(),omega=quat_apply_inverse(d.root_quat_w,d.root_ang_vel_w)[0].cpu().numpy(),pos=d.root_pos_w[0].cpu().numpy())
    def render():
        refresh_cameras(env)
        return {n:env.scene[n+'_camera'].data.output['rgb'][0,:,:,:3].cpu().numpy().astype(np.uint8) for n in ('front','world')}
    from fm_humanoid_bench.protocols.reference import ReferenceTrajectory,SCHEMA
    from fm_humanoid_bench.protocols.body_contract import describe_body,validate_reference_request
    trajectory=ReferenceTrajectory(state(),body.names)
    from fm_humanoid_bench.protocols.native_commands import schema_for,validate_request as validate_native_request,infer_native
    body_contract=describe_body(body.name,body.reference_offsets)
    (root/'body_contract.json').write_text(json.dumps(body_contract,indent=2))
    images=render()
    def observe():
        path=observation_dir/f'ego_{steps:06d}.png';Image.fromarray(images['front']).save(path)
        st=state();o=dict(paused=True,step=steps,ego_image=str(path),_runner_terminal=terminal,body=body.name,joint_names=body.names,joint_reference=joint_ref.tolist(),hands=hands,**{k:v.tolist() for k,v in st.items()})
        local_hands={}
        for side in ('left','right'):
            bi=robot.body_names.index(side+'_wrist_yaw_link')
            bp=robot.data.body_pos_w[:,bi];bq=robot.data.body_quat_w[:,bi]
            local_hands[side]=dict(position=quat_apply_inverse(robot.data.root_quat_w,bp-robot.data.root_pos_w)[0].cpu().tolist(),orientation_wxyz=quat_mul(quat_conjugate(robot.data.root_quat_w),bq)[0].cpu().tolist())
        o['state64']=trajectory.observation(st).tolist()
        o['reference_schema']=SCHEMA
        o['body_contract']=body_contract
        o['native_body_schema']=schema_for(body)
        o['max_steps']=max_steps
        o['compact_proprioception']=dict(root_orientation_wxyz=st['quat'].tolist(),root_height=float(st['pos'][2]),hand_frame='pelvis',hands=local_hands,commanded_grippers=hands,foot_contact=None)
        if hasattr(body,'latents'):o.update(command=body.command,commands=list(body.latents))
        (root/'debug'/'latest_observation.json').write_text(json.dumps(o));return o
    def reference(st,q,vel):
        # Native full-body reference: future root trajectory and named joint pose.
        # Neither IK nor task-specific motion is embedded in this adapter.
        dt=np.arange(-1,12,dtype=np.float32)*.02 if body.name=='HoloMotion' else np.arange(13,dtype=np.float32)*.02
        quat=st['quat'];w,x,y,z=quat;yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        cy,sy=np.cos(yaw),np.sin(yaw);worldv=np.array([cy*vel[0]-sy*vel[1],sy*vel[0]+cy*vel[1],0])
        pos=st['pos'][None]+dt[:,None]*worldv;pos[:,2]=.78
        rot=np.zeros((13,4),np.float32);rot[:,0]=np.cos((yaw+dt*vel[2])/2);rot[:,3]=np.sin((yaw+dt*vel[2])/2)
        return np.concatenate([pos,rot,np.tile(q,(13,1))],axis=1).astype(np.float32)
    from pico_server.data_utils.params import DEFAULT_HAND_POSE
    def hand_target(full):
        for side,value in zip(('left','right'),hands):
            pose=DEFAULT_HAND_POSE['unitree_g1_with_hands'][side]['close' if value>=.5 else 'open']
            # Reuse native Dex3 name mapping, never body policy outputs.
            suffixes=['thumb_0','thumb_1','thumb_2','middle_0','middle_1','index_0','index_1']
            dst=[robot.joint_names.index(f'{side}_hand_{x}_joint') for x in suffixes]
            full[dst]=t(pose)
    print('BFM_READY '+body.name,flush=True)
    try:
        with torch.inference_mode():
            while app.is_running():
                if not socket.poll(1000):continue
                request=socket.recv_json()
                try:
                    op=request.get('op','observe')
                    if op in ('step','reference_step','native_step'):
                        if terminal:raise ValueError('Episode terminated')
                        canonical=op=='reference_step'
                        native=op=='native_step'
                        if native:native_command,native_count,native_hands=validate_native_request(body,request)
                        count=request.get('frames',50)
                        if type(count) is not int or not 1<=count<=250:raise ValueError('frames must be an integer in 1..250')
                        if canonical:
                            actions,count,receipt=validate_reference_request(request,body.name,body.reference_offsets)
                        newq=joint_ref.copy()
                        for n,v in request.get('joints',{}).items():newq[body.names.index(n)]=float(v)
                        newvel=np.array(request.get('velocity',[0,0,0]),np.float32)
                        if newvel.shape!=(3,) or not np.isfinite(np.r_[newq,newvel]).all():raise ValueError('Invalid reference')
                        if hasattr(body,'latents') and not canonical and not native:
                            cmd=request.get('command',body.command)
                            if cmd not in body.latents:raise ValueError('Unknown native latent')
                            body.command=cmd
                        newhands=native_hands if native else list(request.get('hands',hands))
                        if len(newhands)!=2 or not np.isfinite(newhands).all():raise ValueError('Expected two finite gripper commands')
                        hands=newhands
                        if canonical:trajectory.submit(actions,state())
                        log.write(json.dumps(dict(step=steps,ego_image=str(observation_dir/f'ego_{steps:06d}.png'),decision=request,paused=True,acceptance=receipt if canonical else (dict(accepted=True,schema='arena_native_body_v1',native_mode=provenance['native_mode']) if native else None)))+'\n')
                        start=joint_ref.copy()
                        for i in range(count):
                            blend=(1-np.cos(np.pi*(i+1)/count))/2;joint_ref=start+(newq-start)*blend
                            st=state()
                            if canonical:
                                offsets=body.reference_offsets
                                ref=trajectory.window(offsets)
                                canonical_action=trajectory.actions[trajectory.cursor].copy()
                                hands=canonical_action[38:40].tolist()
                                joint_ref=ref[list(offsets).index(0),7:].copy()
                            elif native:
                                ref=np.empty((0,36),np.float32)
                            else:
                                ref=reference(st,joint_ref,newvel)
                            if native:target,data=infer_native(body,st,native_command)
                            else:target,data=body.infer_reference(st,ref) if canonical else body.infer(st,ref)
                            if not np.isfinite(target).all():raise RuntimeError('Nonfinite model output')
                            full=robot.data.default_joint_pos[0].clone();full[ti]=t(target);hand_target(full)
                            before=float(env.sim.current_time)
                            # Record current image, observation and applied target on the same tick.
                            for n,im in images.items():
                                if n not in writers:
                                    (root/'videos').mkdir(exist_ok=True)
                                    writers[n]=cv2.VideoWriter(str(root/'videos'/({'front':'ego','world':'g1'}[n]+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),50,(im.shape[1],im.shape[0]))
                                    if not writers[n].isOpened():raise RuntimeError('Video writer failed')
                                writers[n].write(cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
                            frames.append(dict(q=st['q'],dq=st['dq'],root=np.r_[st['pos'],st['quat']],obs=data['obs'].reshape(-1),raw_action=data['raw_action'].reshape(-1),target=target.copy(),reference=ref,sim_time=before))
                            if canonical or native:
                                hand_ids=[robot.joint_names.index(f'{side}_hand_{suffix}_joint') for side in ('left','right') for suffix in ('thumb_0','thumb_1','thumb_2','middle_0','middle_1','index_0','index_1')]
                                frames[-1].update(state64=trajectory.observation(st),omega=st['omega'],hand_q=robot.data.joint_pos[0,hand_ids].cpu().numpy(),hand_target=full[hand_ids].cpu().numpy())
                                if canonical:frames[-1]['action40']=canonical_action
                                for key in ('conditioning','backward_observation'):
                                    if key in data:frames[-1]['adapter_'+key]=np.asarray(data[key]).copy()
                            for _ in range(round(.02/env.physics_dt)):
                                robot.set_joint_position_target(full);env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.physics_dt)
                            sync_reward_after_physics_step(env)
                            after=float(env.sim.current_time)
                            check_control_tick(before,after)
                            timeline.write(json.dumps(dict(step=steps,sim_time=before,next_sim_time=after))+'\n')
                            if canonical:trajectory.commit(st)
                            steps+=1;images=render()
                            reward,terminal,streak=evaluate_tick(env,evaluator,detector,steps,max_steps,streak)
                            if terminal:break
                        # Checkpoint the completed chunk before returning its observation.
                        # This preserves numerical data if the next RPC or Kit shutdown stalls.
                        snapshot={k:np.stack([f[k] for f in frames]) for k in frames[0]}
                        np.savez_compressed(root/'episode_checkpoint.tmp.npz',**snapshot,joint_names=np.array(body.names),frame_index=np.arange(steps),meta_control_dt=np.array(.02),schema=np.array('native_bfm_episode_v1_multicam'))
                        (root/'episode_checkpoint.tmp.npz').replace(root/'episode_checkpoint.npz')
                    elif op=='finish':
                        for w in writers.values():w.release()
                        arr={k:np.stack([f[k] for f in frames]) for k in frames[0]} if frames else {}
                        np.savez_compressed(root/'episode.npz',**arr,joint_names=np.array(body.names),frame_index=np.arange(steps),meta_control_dt=np.array(.02),schema=np.array('native_bfm_episode_v1_multicam'))
                        result=dict(body=body.name,brain=brain_name,task=args.task,seed=args.seed,success=terminal=='success',termination=terminal or 'manual_finish',steps=steps,sim_seconds=steps*.02,timeout_sim_seconds=.02*max_steps,final_reward=reward)
                        (root/'task_evaluation.json').write_text(json.dumps(result,indent=2));(root/'episode_checkpoint.npz').unlink(missing_ok=True);socket.send_json(dict(saved=True,**result));break
                    elif op!='observe':raise ValueError('Unknown operation')
                    socket.send_json(observe())
                except Exception as e:
                    import traceback;traceback.print_exc();socket.send_json(dict(error=str(e),step=steps,paused=True))
    finally:
        for w in writers.values():w.release()
        log.close();timeline.close();socket.close(linger=0)
