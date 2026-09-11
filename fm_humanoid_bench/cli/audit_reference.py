"""Recompute recorded reference decoding and native inference, tick for tick."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
REPO=Path(os.environ.get('FMHB_ROOT',Path(__file__).resolve().parents[2])).expanduser().resolve();SIMULATOR=REPO/'simulator';sys.path.insert(0,str(REPO));sys.path.insert(0,str(SIMULATOR))
from fm_humanoid_bench.protocols.reference import ReferenceTrajectory
from fm_humanoid_bench.evaluation.qualification_metrics import probe_metrics
from benchmark_runtime.body_backends.native import create


@torch.inference_mode()
def audit(root, body_name):
    root=Path(root);body=create(body_name);body.reset_history()
    # NpzFile decompresses on every __getitem__; materialize once before the
    # per-tick loop, otherwise audit time grows quadratically with episode size.
    with np.load(root/'episode.npz') as archive:z={key:archive[key] for key in archive.files}
    n=len(z['q'])
    provenance=json.loads((root/'provenance.json').read_text())
    assert hashlib.sha256(body.path.read_bytes()).hexdigest()==provenance['checkpoint_sha256']
    assert z['joint_names'].tolist()==body.names
    def state(i):return dict(q=z['q'][i],dq=z['dq'][i],pos=z['root'][i,:3],quat=z['root'][i,3:7],omega=z['omega'][i])
    trajectory=ReferenceTrajectory(state(0),body.names)
    records=[json.loads(line) for line in (root/'brain_decisions.jsonl').read_text().splitlines()]
    errors={k:0. for k in ('reference','state64','target','raw_action','obs','adapter_conditioning','adapter_backward_observation')}
    tick=0
    for record in records:
        assert record['step']==tick and record['acceptance']['accepted']
        request=record['decision'];trajectory.submit(request['action_chunk'],state(tick))
        for _ in range(min(request['frames'],n-tick)):
            st=state(tick);ref=trajectory.window(body.reference_offsets)
            target,data=body.infer_reference(st,ref)
            expected=dict(reference=ref,state64=trajectory.observation(st),target=target,raw_action=data['raw_action'].reshape(-1),obs=data['obs'].reshape(-1))
            for field in ('conditioning','backward_observation'):
                if field in data:expected['adapter_'+field]=data[field]
            for key,value in expected.items():
                delta=float(np.max(np.abs(z[key][tick]-value)));errors[key]=max(errors[key],delta)
                # GPU inference numerical tolerance. Decoding/state must be tighter.
                np.testing.assert_allclose(z[key][tick],value,atol=2e-6 if key in ('reference','state64') else 2e-4,rtol=2e-5,err_msg=f'{key} tick {tick}')
            trajectory.commit(st);tick+=1
    assert tick==n
    # Common gripper branch must match actual mapped Dex3 target values.
    from pico_server.data_utils.params import DEFAULT_HAND_POSE
    for side,offset in [('left',0),('right',7)]:
        commands=z['action40'][:,38+(side=='right')]>=.5
        poses=DEFAULT_HAND_POSE['unitree_g1_with_hands'][side]
        expected=np.stack([poses['close' if c else 'open'] for c in commands])
        np.testing.assert_allclose(z['hand_target'][:,offset:offset+7],expected,atol=1e-6)
    result=dict(passed=True,steps=n,max_absolute_recomputation_errors=errors,
                reference_reconstructed_every_tick=True,native_inference_recomputed_every_tick=True,dex3_mapping_checked=True,
                semantic_losses=json.loads((root/'body_contract.json').read_text())['semantic_losses'])
    experiment=json.loads((root/'experiment.json').read_text());metrics=json.loads((root/'metrics.json').read_text())
    if experiment.get('replay'):
        source=Path(experiment['replay'])
        original=[json.loads(line)['decision'] for line in (source/'brain_decisions.jsonl').read_text().splitlines()]
        for actual,expected in zip(records,original):
            np.testing.assert_array_equal(actual['decision']['action_chunk'],expected['action_chunk'])
            assert actual['decision']['frames']==expected['frames']
        assert len(records)<=len(original)
        result['source_prediction_chunks_identical']=True
        if body.name=='ScaleBFM':
            with np.load(source/'episode.npz') as old:
                errors={key:float(np.max(np.abs(old[key]-z[key]))) for key in ('action40','reference','q','root','target','raw_action','obs')} if len(old['q'])==n else None
            reproduction=dict(source=str(source),replay_success=metrics['success'],steps=n,max_absolute_errors=errors,exact_reproduction=errors is not None and all(v==0 for v in errors.values()))
            (root/'source_reproduction.json').write_text(json.dumps(reproduction,indent=2));result['source_exact_reproduction']=reproduction['exact_reproduction']
    if experiment.get('trajectory'):
        source=Path(experiment['trajectory']);assert hashlib.sha256(source.read_bytes()).hexdigest()==experiment['trajectory_sha256']
        np.testing.assert_array_equal(z['action40'],np.load(source)[:n]);result['realized_reference_prefix_identical']=True
    if experiment.get('replay') or experiment.get('trajectory'):
        if not provenance.get('task_reference_replay'):
            provenance.setdefault('metadata_corrections',[]).append(dict(field='task_reference_replay',old=False,new=True,reason='Legacy metadata default; experiment and recorded input identify reference-assisted replay'))
            provenance['task_reference_replay']=True
            (root/'provenance.json').write_text(json.dumps(provenance,indent=2))
        if experiment.get('replay'):
            metrics.update(execute_prefix=None,execution_policy='source_request_frames',requested_chunk_lengths=[v['decision']['frames'] for v in records])
            experiment.update(execute_prefix=None,execution_policy='source_request_frames')
            (root/'metrics.json').write_text(json.dumps(metrics,indent=2));(root/'experiment.json').write_text(json.dumps(experiment,indent=2))
        text=(root/'README.md').read_text().replace('Frozen Brain;','Controller diagnostic; no new Brain inference;')
        if experiment.get('replay'):text=text.replace('Executed prefix: 20.','Execution follows the original per-chunk lengths.')
        (root/'README.md').write_text(text)
    if experiment.get('probe'):
        result['controller_probe']=probe_metrics(z,experiment['probe'],metrics['probe_complete'],list(body.reference_offsets).index(0))
        result['passed']=bool(result['passed'] and result['controller_probe']['passed'])
    (root/'interface_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--body',required=True);a=p.parse_args()
    if not audit(a.directory,a.body)['passed']:raise SystemExit(1)
