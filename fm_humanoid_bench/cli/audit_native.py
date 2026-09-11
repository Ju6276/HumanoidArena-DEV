"""Recompute each native-condition policy call, without advancing physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
REPO=Path(os.environ.get('FMHB_ROOT',Path(__file__).resolve().parents[2])).expanduser().resolve();SIMULATOR=REPO/'simulator';sys.path.insert(0,str(REPO));sys.path.insert(0,str(SIMULATOR))
from benchmark_runtime.body_backends.native import create
from fm_humanoid_bench.protocols.native_commands import validate_request,infer_native

@torch.inference_mode()
def audit(root,body_name):
    root=Path(root);body=create(body_name);body.reset_history()
    provenance=json.loads((root/'provenance.json').read_text())
    assert hashlib.sha256(body.path.read_bytes()).hexdigest()==provenance['checkpoint_sha256']
    with np.load(root/'episode.npz') as z:data={k:z[k] for k in z.files}
    assert data['reference'].shape[1:]==(0,36) and 'action40' not in data
    errors={k:0. for k in ['obs','raw_action','target']};tick=0
    for line in (root/'brain_decisions.jsonl').read_text().splitlines():
        row=json.loads(line);assert row['step']==tick
        command,count,hands=validate_request(body,row['decision'])
        for _ in range(min(count,len(data['q'])-tick)):
            state=dict(q=data['q'][tick],dq=data['dq'][tick],pos=data['root'][tick,:3],quat=data['root'][tick,3:7],omega=data['omega'][tick])
            target,info=infer_native(body,state,command)
            for key,value in dict(target=target,obs=info['obs'].reshape(-1),raw_action=info['raw_action'].reshape(-1)).items():
                errors[key]=max(errors[key],float(np.max(np.abs(data[key][tick]-value))))
                np.testing.assert_allclose(data[key][tick],value,atol=2e-4,rtol=2e-5)
            tick+=1
    assert tick==len(data['q'])
    report=dict(passed=True,track='native',steps=tick,max_absolute_recomputation_errors=errors,native_inference_recomputed_every_tick=True)
    (root/'interface_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--body',required=True);a=p.parse_args();audit(a.directory,a.body)
