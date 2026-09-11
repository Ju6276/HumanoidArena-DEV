"""One training episode -> reproducible numeric / V-JEPA 2.1 image references."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import cv2
import numpy as np
import pandas as pd
import torch

REPO = Path(os.environ.get('FMHB_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=Path('/home/d024/Psi0/data/humanoidarena_sonic_v31_opendoor'))
    p.add_argument('--episode',type=int,default=0)
    p.add_argument('--output',type=Path,default=REPO/'eval_results/reference_v31/references/opendoor_episode000000')
    p.add_argument('--checkpoint',type=Path,default=Path('/home/d024/models/vjepa2_1_vitl_dist_vitG_384.pt'))
    p.add_argument('--count',type=int,default=8)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    parquet=next(a.dataset.glob(f'data/*/episode_{a.episode:06d}.parquet'))
    video=next(a.dataset.glob(f'videos/*/observation.images.front/episode_{a.episode:06d}.mp4'))
    table=pd.read_parquet(parquet);state=np.stack(table['observation.state']);action=np.stack(table['action']);timestamps=np.array(table['timestamp'])
    candidates=np.unique(np.r_[np.arange(0,len(table),25),len(table)-1]).astype(int)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,r_frame_rate,nb_frames','-of','json',str(video)]))['streams'][0]
    width,height=probe['width'],probe['height']
    decoded=subprocess.check_output(['ffmpeg','-v','error','-threads','2','-c:v','libdav1d','-i',str(video),'-vf','select='+ '+'.join(f'eq(n\\,{i})' for i in candidates),'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-'])
    images=list(np.frombuffer(decoded,dtype=np.uint8).reshape(-1,height,width,3))
    if len(images)!=len(candidates):raise RuntimeError('Demonstration video/table frame mismatch')
    vendor=Path('/home/d024/VLA-JEPA-ONE-STAGE-DELTA/temp/starVLA/model/modules/world_model/vjepa21_vendor')
    sys.path.insert(0,str(vendor))
    from app.vjepa_2_1.models import vision_transformer as vit
    model=vit.vit_large(patch_size=16,img_size=(384,384),num_frames=64,tubelet_size=2,use_sdpa=True,use_SiLU=False,wide_SiLU=True,uniform_power=False,use_rope=True,img_temporal_dim_size=1,interpolate_rope=True)
    weights=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['ema_encoder']
    weights={k.replace('module.','').replace('backbone.',''):v for k,v in weights.items()}
    incompatible=model.load_state_dict(weights,strict=False)
    if [k for k in incompatible.missing_keys if 'pos_embed' not in k] or incompatible.unexpected_keys:raise RuntimeError(str(incompatible))
    model=model.cuda().eval();features=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for im in images:
            h,w=im.shape[:2];scale=384/min(h,w);x=cv2.resize(im,(round(w*scale),round(h*scale)))
            h,w=x.shape[:2];x=x[(h-384)//2:(h+384)//2,(w-384)//2:(w+384)//2]
            x=torch.as_tensor(x.copy(),device='cuda').permute(2,0,1).float()/255
            x=(x-torch.tensor([.485,.456,.406],device='cuda')[:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[:,None,None]
            y=model(x[None,:,None],training=False)
            features.append(y.mean(dim=1)[0].float().cpu().numpy())
    features=np.stack(features);features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-9)
    # Select diverse states from learned visual features; endpoints are fixed.
    # This is feature-space sampling, not a learned task-success detector.
    chosen=[0,len(candidates)-1]
    while len(chosen)<min(a.count,len(candidates)):
        distances=1-features@features[chosen].T
        scores=distances.min(axis=1);scores[chosen]=-np.inf
        chosen.append(int(scores.argmax()))
    selected=candidates[sorted(chosen)]
    paths=[]
    for idx in sorted(chosen):
        filename=f'ego_{candidates[idx]:06d}.png';cv2.imwrite(str(a.output/filename),cv2.cvtColor(images[idx],cv2.COLOR_RGB2BGR));paths.append(filename)
    np.savez_compressed(a.output/'episode_reference.npz',state64=state,action40=action,timestamps=timestamps,selected_indices=selected,candidate_indices=candidates,vjepa_features=features)
    metadata=dict(schema='arena_episode_reference_v1',task='Open the door.',episode=a.episode,source_dataset=str(a.dataset),source_parquet=str(parquet),source_video=str(video),source_parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),source_video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),selection='V-JEPA 2.1 ViT-L image embeddings, cosine farthest-point sampling with endpoints',checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),num_frames=len(table),selected_indices=selected.tolist(),images=paths,success_label='not independently verified; source training demonstration',reference_split='training demonstration, distinct from evaluation rollout',numeric_schema='unitree_g1_gmt_refpose_v3_1',state_action=[dict(frame=int(i),time_seconds=float(timestamps[i]),state64=state[i].tolist(),action40=action[i].tolist()) for i in selected])
    (a.output/'reference.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(dict(output=str(a.output),selected_indices=selected.tolist())),flush=True)

if __name__=='__main__':main()
