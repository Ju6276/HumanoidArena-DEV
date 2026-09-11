"""Verify actual decoded camera frames and model/physics alignment."""
import json,sys
from pathlib import Path
import numpy as np
import cv2
r=Path(sys.argv[1]);e=json.loads((r/'task_evaluation.json').read_text());n=e['steps']
z=np.load(r/'episode.npz',allow_pickle=False)
assert n>0 and len(z['q'])==n
assert np.array_equal(z['frame_index'],np.arange(n))
for k in ('q','dq','root','obs','raw_action','target','reference','sim_time'):
 assert len(z[k])==n and np.isfinite(z[k]).all(),k
provenance=json.loads((r/'provenance.json').read_text())
if provenance.get('reference_protocol')=='unitree_g1_gmt_refpose_v3_1':
 assert z['state64'].shape==(n,64) and np.isfinite(z['state64']).all()
 assert z['action40'].shape==(n,40) and np.isfinite(z['action40']).all()
 requests=[json.loads(x)['decision'] for x in (r/'brain_decisions.jsonl').read_text().splitlines()]
 expected=np.concatenate([np.asarray(x['action_chunk'])[:x['frames']] for x in requests])[:n]
 np.testing.assert_allclose(z['action40'],expected,atol=1e-6)
 center=json.loads((r/'body_contract.json').read_text())['reference_offsets'].index(0)
 np.testing.assert_allclose(z['reference'][:,center,2],z['action40'][:,2],atol=1e-6)
rows=[json.loads(x) for x in (r/'simulation_timeline.jsonl').read_text().splitlines()]
assert len(rows)==n
for i,x in enumerate(rows):
 assert x['step']==i and np.isclose(x['next_sim_time']-x['sim_time'],.02,atol=1e-6)
 assert np.isclose(z['sim_time'][i],x['sim_time'])
 if i:assert np.isclose(rows[i-1]['next_sim_time'],x['sim_time'])
videos={}
for view in ('ego','g1'):
 p=r/'videos'/f'{view}.mp4';cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS);count=0
 while True:
  ok,im=cap.read()
  if not ok:break
  count+=1
 cap.release();assert count==n and np.isclose(fps,50)
 videos[view]=dict(file=str(p),decoded_frames=count,fps=fps,seconds=count/fps)
result=dict(passed=True,steps=n,sim_seconds=n*.02,continuous=True,model_observation_width=int(z['obs'].shape[1]),videos=videos)
(r/'recording_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
