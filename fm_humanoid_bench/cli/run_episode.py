"""One paused simulator runner for Agent/VLA/WAM x native G1 Body models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError, HTTPError
import numpy as np
import zmq

REPO=Path(os.environ.get('FMHB_ROOT',Path(__file__).resolve().parents[2])).expanduser().resolve()
SIMULATOR=REPO/'simulator'
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(SIMULATOR))
from fm_humanoid_bench.protocols.reference import SCHEMA,validate_chunk,tracking_metrics,compile_keyframes
from fm_humanoid_bench.evaluation.artifact_contract import run_name,publish
from fm_humanoid_bench.environments.task_registry import TASKS,MODES,environment_config,original_budget
from fm_humanoid_bench.protocols.brain_protocol import infer as infer_http,reset as reset_http,HTTP_BRAINS
from fm_humanoid_bench.brains.agent_transport import request_agent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--body',choices=['scale','holo','zero','sonic','twist2'],required=True)
    p.add_argument('--task',choices=list(TASKS),default='open_door')
    p.add_argument('--mode',choices=list(MODES),default='Base')
    p.add_argument('--brain',choices=[*HTTP_BRAINS,'gpt6','agent','reference_probe','reference_replay','realized_replay'],default='psi0')
    p.add_argument('--brain-label',help='Pinned model/catalog ID used in artifact naming')
    p.add_argument('--track',choices=['reference','native'],default='reference')
    p.add_argument('--probe',choices=['pose','motion'])
    p.add_argument('--replay',type=Path)
    p.add_argument('--trajectory',type=Path)
    p.add_argument('--server',default='http://127.0.0.1:18080')
    p.add_argument('--horizon',type=int,default=30)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--execute',type=int,default=20)
    p.add_argument('--root',type=Path,default=REPO/'eval_results/reference_v31')
    p.add_argument('--environment-backend',choices=['sonic','twist2'],default='twist2')
    p.add_argument('--reference',type=Path)
    p.add_argument('--reference-catalog',type=Path)
    p.add_argument('--reference-id')
    p.add_argument('--reference-selection',choices=['config','agent'],default='config')
    p.add_argument('--condition',choices=['none','state_action','images'],default='none')
    p.add_argument('--agent-command-json',help='JSON argv list; packet via stdin, hashed response via stdout')
    p.add_argument('--agent-timeout',type=float,default=600)
    p.add_argument('--require-capability',action='append')
    a=p.parse_args()
    agent=a.brain in ('gpt6','agent');http_brain=a.brain in HTTP_BRAINS
    label=a.brain_label or a.brain
    if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):p.error('Invalid Brain label')
    if not 1<=a.execute<=250 or not 1<=a.horizon<=250:p.error('execute/horizon must be 1..250')
    if a.track=='native' and not agent:p.error('Native track currently requires an Agent with native command output; VLA/WAM reference outputs cannot be relabeled native')
    if not agent and a.condition!='none':p.error('Frozen HTTP models do not accept in-context demonstrations; use condition none')
    if not agent and a.reference_selection=='agent':p.error('Reference selection requires Agent transport')
    if a.condition=='none' and a.reference_selection=='agent':p.error('Agent reference selection requires a demonstration condition')
    if a.reference and a.reference_catalog:p.error('Use one reference source')
    if a.condition=='none' and (a.reference or a.reference_catalog or a.reference_id):p.error('No-reference condition must not attach demonstrations')
    if a.brain=='reference_probe' and (not a.probe or a.task!='open_door' or a.mode!='Base'):p.error('Diagnostic fixtures remain OpenDoor Base only')
    if a.brain=='reference_replay' and not a.replay:p.error('reference_replay requires --replay')
    if a.brain=='realized_replay' and not a.trajectory:p.error('realized_replay requires --trajectory')
    agent_command=json.loads(a.agent_command_json) if a.agent_command_json else None
    if agent_command is not None and (not isinstance(agent_command,list) or not agent_command or not all(isinstance(x,str) for x in agent_command)):p.error('Agent command must be an argv list')
    if a.replay and a.brain!='reference_replay':p.error('--replay is only for reference_replay')
    if a.trajectory and a.brain!='realized_replay':p.error('--trajectory is only for realized_replay')
    if a.probe and a.brain!='reference_probe':p.error('--probe is only for reference_probe')
    realized=None
    if a.trajectory:
        realized=np.load(a.trajectory,allow_pickle=False)
        if realized.ndim!=2 or realized.shape[1]!=40 or len(realized)==0:raise ValueError('Invalid realized trajectory')
        for i in range(0,len(realized),250):validate_chunk(realized[i:i+250])
    task=TASKS[a.task];max_steps,budget_source=original_budget(a.task,a.environment_backend)
    config=environment_config(a.task,a.mode,a.environment_backend)
    catalog=None
    if a.condition!='none':
        from fm_humanoid_bench.evaluation.reference_catalog import load_catalog,build_catalog
        if not (a.reference or a.reference_catalog):p.error('Demonstration condition requires a reference bundle/catalog')
        catalog=load_catalog(a.reference_catalog) if a.reference_catalog else build_catalog([a.reference])
    root=a.root.expanduser().resolve()/a.condition/run_name(label,a.body,a.task,a.mode,a.seed)
    root.mkdir(parents=True,exist_ok=False)
    debug=root/'debug';debug.mkdir()
    agent_debug=debug/'agent';agent_debug.mkdir()
    prediction_debug=debug/'predictions';prediction_debug.mkdir()
    endpoint=f'ipc:///tmp/arena-ref-{os.getpid()}.sock'
    settings=dict(brain=label,brain_family=a.brain,body=a.body,task=a.task,mode=a.mode,seed=a.seed,track=a.track,
                  execute_prefix=a.execute,prediction_horizon=a.horizon,schema=SCHEMA if a.track=='reference' else 'arena_native_body_v1',control_dt=.02,
                  condition=a.condition,reference=str(a.reference) if a.reference else None,reference_catalog=str(a.reference_catalog) if a.reference_catalog else None,
                  reference_selection=a.reference_selection,tail_policy='hold_last_pose',brain_server=a.server if http_brain else None,
                  probe=a.probe,replay=str(a.replay) if a.replay else None,trajectory=str(a.trajectory) if a.trajectory else None,
                  allow_native_approximation=True,required_capabilities=a.require_capability or ['joint_pose','root_tilt','hands'],
                  env_id=task['env_id'],instruction=task['instruction'],runtime_profile=task['runtime_profile'],environment_backend=a.environment_backend,
                  environment_config=config,environment_config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
                  max_steps=max_steps,budget_source=str(budget_source),reset_policy='new simulator per episode; no in-episode retries',
                  observation='ego RGB + robot proprioception',recorded_cameras=['ego','g1'],position_feedback_compensation=False,
                  live_video_preview=False,brain_visual_input='current ego frame at replanning only; continuous MP4 is recording output',
                  execution_policy='source_request_frames' if a.replay else 'bounded_prefix')
    if a.replay:settings['execute_prefix']=None
    if a.trajectory:settings['trajectory_sha256']=hashlib.sha256(a.trajectory.read_bytes()).hexdigest()
    replay_requests=[]
    if a.replay:
        previous=json.loads((a.replay/'experiment.json').read_text())
        if previous.get('task','open_door')!=a.task or previous.get('mode','Base')!=a.mode or previous.get('track','reference')!='reference' or previous.get('seed',a.seed)!=a.seed:raise ValueError('Replay source task/mode mismatch')
        for key,expected in [('environment_backend',a.environment_backend),('runtime_profile',task['runtime_profile'])]:
            if key in previous and previous[key]!=expected:raise ValueError('Replay source environment configuration mismatch')
        source=a.replay/'brain_decisions.jsonl';replay_requests=[json.loads(line)['decision'] for line in source.read_text().splitlines()]
        settings['replay_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    (root/'experiment.json').write_text(json.dumps(settings,indent=2))
    if http_brain:
        deadline=time.monotonic()+300
        while True:
            try:
                reset_receipt=reset_http(a.server,a.seed,timeout=15);break
            except HTTPError:raise
            except URLError:
                if time.monotonic()>deadline:raise TimeoutError('Brain server not ready')
                time.sleep(1)
        (root/'brain_reset.json').write_text(json.dumps(reset_receipt,indent=2))
    env=os.environ.copy();env.update(BFM_BACKEND=a.body,ARENA_BRAIN=label,ARENA_REFERENCE_PROTOCOL=settings['schema'],
        ARENA_EVALUATOR_BACKEND=a.environment_backend,ASTRA_CONTROL_ENDPOINT=endpoint,ASTRA_NATIVE_EVAL='1',PYTHONPATH=str(REPO),
        ROBOT_USD_OVERRIDE=str(SIMULATOR/'assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2_thumd.usd'))
    env['PYTHONPATH']=os.pathsep.join([str(REPO),str(SIMULATOR),env.get('PYTHONPATH','')])
    cmd=[sys.executable,str(SIMULATOR/'benchmark_runtime/run_astra.py'),'--task',task['env_id'],'--action_source','twist2_wholebody','--gmt_backend','twist2',
         '--robot_type','unitree_g1_with_hands','--env_config_yaml',config,'--enable_dex3_dds','--enable_cameras','--enable_world_camera',
         '--disable_wrist_cameras','--headless','--device','cuda:0','--seed',str(a.seed),'--task_runtime_profile',task['runtime_profile'],'--recording_save_dir',str(root)]
    runner_log=debug/'runner.log'
    with runner_log.open('w') as output:
        proc=subprocess.Popen(cmd,cwd=SIMULATOR,env=env,stdout=output,stderr=subprocess.STDOUT)
        sock=zmq.Context.instance().socket(zmq.REQ);sock.setsockopt(zmq.RCVTIMEO,300000);sock.connect(endpoint)
        def rpc(request):
            sock.send_json(request);reply=sock.recv_json()
            if 'error' in reply:raise RuntimeError(reply)
            return reply
        try:
            deadline=time.monotonic()+300
            while 'BFM_READY' not in runner_log.read_text():
                if proc.poll() is not None:raise RuntimeError('Simulator exited; see debug/runner.log')
                if time.monotonic()>deadline:raise TimeoutError('Simulator startup timeout')
                time.sleep(1)
            if http_brain:reset_http(a.server,a.seed)
            observation=rpc({'op':'observe'})
            selection=None;selection_transport=None
            if catalog is not None:
                from fm_humanoid_bench.evaluation.reference_catalog import reference_options,select_reference
                options=reference_options(catalog,a.task,a.condition);reference_id=a.reference_id
                if reference_id is None and a.reference_selection=='config' and len(options)==1:reference_id=options[0]['id']
                if a.reference_selection=='agent':
                    response,selection_transport=request_agent(dict(task_id=a.task,task_instruction=task['instruction'],task_description=task['description'],condition=a.condition,ego_image=observation['ego_image'],
                        options=options,response_schema={'prompt_sha256':'copy request hash','reference_id':'one listed ID'}),agent_debug,'reference_selection',command=agent_command,timeout=a.agent_timeout)
                    reference_id=response['reference_id']
                    (root/'reference_selection_transport.json').write_text(json.dumps(selection_transport,indent=2))
                selection=select_reference(catalog,task=a.task,condition=a.condition,reference_id=reference_id,
                                           selector=a.reference_selection,step=0,log_path=root/'reference_selection.json')
                assert rpc({'op':'observe'})['step']==0,'Physics advanced during reference selection'
            fixture=None
            if a.brain=='reference_probe':
                from fm_humanoid_bench.evaluation.reference_fixtures import probe_actions
                fixture=probe_actions(observation,a.probe);np.save(root/'probe_action40.npy',fixture)
                settings['fixture_sha256']=hashlib.sha256(fixture.tobytes()).hexdigest()
                (root/'experiment.json').write_text(json.dumps(settings,indent=2))
                rejected=[]
                valid=dict(op='reference_step',schema=SCHEMA,control_dt=.02,action_chunk=fixture[:2].tolist(),frames=1,allow_native_approximation=True)
                for name,change in [('wrong_dt',{'control_dt':.04}),('wrong_schema',{'schema':'unknown'}),('unknown_capability',{'required_capabilities':['teleport']}),('excess_prefix',{'frames':3}),('malformed_chunk',{'action_chunk':[[0]*39]})]:
                    sock.send_json({**valid,**change});reply=sock.recv_json()
                    if 'error' not in reply:raise AssertionError(f'Invalid request accepted: {name}')
                    after=rpc({'op':'observe'});assert after['step']==0
                    for key in ('q','dq','pos','quat'):np.testing.assert_array_equal(after[key],observation[key])
                    rejected.append(dict(case=name,error=reply['error'],physics_unchanged=True))
                (root/'rejection_checks.json').write_text(json.dumps(rejected,indent=2))
            if a.brain=='realized_replay':fixture=realized
            history=[]
            while not observation['_runner_terminal']:
                tick=observation['step'];execute=a.execute
                if fixture is not None and tick>=len(fixture):break
                if http_brain:
                    actions,record=infer_http(a.server,observation,task['instruction'],expected_horizon=a.horizon)
                    np.save(prediction_debug/f'prediction_{tick:06d}.npy',actions)
                elif agent:
                    from fm_humanoid_bench.brains.agent_prompt import packet
                    prompt,start=packet(observation,history,None,a.condition,task=task['instruction'],task_id=a.task,remaining_steps=max_steps-tick,selection=selection,track=a.track)
                    if a.track=='native':
                        prompt['user']['max_execute_frames']=a.execute
                    prompt.pop('prompt_sha256',None)
                    response,meta=request_agent(prompt,agent_debug,f'agent_{tick:06d}',command=agent_command,timeout=a.agent_timeout)
                    decision=response['decision'];record=dict(step=tick,decision=decision,reference_selection=prompt['user']['reference_selection'],**meta)
                    if a.track=='native':
                        request=dict(op='native_step',schema='arena_native_body_v1',command=decision['command'],hands=decision['hands'],frames=decision['frames'],prompt_sha256=meta['prompt_sha256'])
                        if type(request['frames']) is not int or not 1<=request['frames']<=a.execute:raise ValueError('Native Agent exceeded execute budget')
                        record['executed_last_action']=start.tolist()
                    else:
                        actions=compile_keyframes(decision['keyframes'],start,decision['horizon'])
                        record.update(action_chunk=actions.tolist(),executed_last_action=actions[min(execute,len(actions))-1].tolist())
                elif fixture is not None:
                    actions=validate_chunk(fixture[tick:tick+min(250,a.execute+12)])
                    record=dict(step=tick,source='realized_robot_trajectory' if a.trajectory else 'deterministic_controller_probe',case=a.probe,action_chunk=actions.tolist())
                else:
                    if len(history)>=len(replay_requests):break
                    source=replay_requests[len(history)];actions=validate_chunk(source['action_chunk']);execute=source['frames']
                    record=dict(step=tick,source='recorded_reference_predictions',source_sha256=settings['replay_source_sha256'],action_chunk=actions.tolist())
                with (root/'brain_predictions.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                assert rpc({'op':'observe'})['step']==tick,'Physics advanced during Brain inference'
                if a.track=='reference':request=dict(op='reference_step',schema=SCHEMA,control_dt=.02,action_chunk=actions.tolist(),frames=min(execute,len(actions)),brain_protocol='reference_v31',prediction_step=tick,required_capabilities=settings['required_capabilities'],allow_native_approximation=True)
                observation=rpc(request);history.append(record)
                print(json.dumps(dict(body=a.body,brain=label,step=observation['step'],terminal=observation['_runner_terminal'])),flush=True)
            result=rpc({'op':'finish'})
            subprocess.run([sys.executable,str(SIMULATOR/'benchmark_runtime/verify_bfm_recording.py'),str(root)],check=True)
            tracking=None
            if a.track=='reference':
                offsets=json.loads((root/'body_contract.json').read_text())['reference_offsets']
                with np.load(root/'episode.npz') as recording:tracking=tracking_metrics(recording,offsets.index(0))
            metrics=dict(reference_tracking=tracking,condition=a.condition,track=a.track,decisions=len(history),execute_prefix=settings['execute_prefix'],
                         execution_policy=settings['execution_policy'],evaluation_kind='realized_reference_replay' if a.trajectory else ('controller_probe' if fixture is not None else ('reference_replay' if a.replay else 'brain_closed_loop')),
                         probe_complete=bool(a.probe and observation['step']==len(fixture) and not observation['_runner_terminal']),
                         brain_latency_seconds=sum(float(h.get('latency_seconds',0)) for h in history)+float((selection_transport or {}).get('latency_seconds',0)),brain_calls=len(history)+int(selection_transport is not None),usage_available=bool(history) and all(h.get('usage_available',False) for h in history))
            (root/'metrics.json').write_text(json.dumps(metrics,indent=2));publish(root,a.body,brain=label,task=a.task,mode=a.mode,seed=a.seed)
        except Exception as exc:
            (root/'infrastructure_error.json').write_text(json.dumps(dict(error=repr(exc)),indent=2));raise
        finally:
            sock.close(linger=0)
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()

if __name__=='__main__':main()
