"""One public artifact layout, with two real, canonical camera files."""
from pathlib import Path
import json,csv

BODY_NAMES={'scale':'scalebfm','scalebfm':'scalebfm','zero':'bfm_zero','bfm-zero':'bfm_zero','bfm_zero':'bfm_zero','holo':'holomotion','holomotion':'holomotion','sonic':'sonic','twist2':'twist2'}
def run_name(brain,body,task,mode,seed):
    return f'{brain.lower()}_{BODY_NAMES[body.lower()]}_{task}_{mode}_s{int(seed)}'

def publish(root,body,brain='gpt6',task='open_door',mode='Base',seed=42):
    root=Path(root).resolve()
    e=json.loads((root/'task_evaluation.json').read_text())
    verify=json.loads((root/'recording_verification.json').read_text())
    if not verify.get('passed',verify.get('verified',False)):raise ValueError('Recording verification must pass before publication')
    provenance=json.loads((root/'provenance.json').read_text())
    replacements={}
    episode=root/'episode.npz'
    native=[p for p in root.glob('*.npz') if p.name!='episode.npz' and not p.is_symlink()]
    if episode.is_symlink():
        source=episode.resolve()
        if source.parent!=root or not source.is_file():raise ValueError('Episode alias must point inside the run directory')
        episode.unlink();source.rename(episode)
        replacements[source.name]=episode.name
    elif not episode.exists():
        if len(native)!=1:raise ValueError('Ambiguous native episode')
        source=native[0];source.rename(episode)
        replacements[source.name]=episode.name
    for canonical,legacy in [('ego','front'),('g1','world')]:
        target=root/'videos'/f'{canonical}.mp4'
        matches=list((root/'videos').glob(f'*{legacy}_rgb.mp4'))
        if len(matches)>1:raise ValueError((root,legacy,matches))
        if matches:
            source=matches[0]
            if target.is_symlink():
                if target.resolve()!=source.resolve():raise ValueError(target)
                target.unlink()
            if target.exists():raise FileExistsError(target)
            replacements[str(source.relative_to(root))]=str(target.relative_to(root))
            source.rename(target)
        if not target.is_file() or target.is_symlink():raise ValueError(target)
    # Rewrite only NPZ path metadata, retaining all numeric tensors unchanged.
    if replacements:
        import numpy as np
        episode_path=(root/'episode.npz').resolve()
        with np.load(episode_path,allow_pickle=False) as z:
            data={k:z[k] for k in z.files}
        changed=False
        for key,value in data.items():
            if value.shape==() and value.dtype.kind in ('U','S'):
                old=str(value)
                if old in replacements:data[key]=np.asarray(replacements[old]);changed=True
        if changed:
            temporary=episode_path.with_suffix('.migration.npz')
            np.savez_compressed(temporary,**data)
            temporary.replace(episode_path)
        # Keep saved reports and README links resolvable after the migration.
        for report in root.rglob('*'):
            if report.is_file() and not report.name.startswith('prompt_') and report.suffix in ('.json','.jsonl'):
                old=report.read_text();new=old
                for before,after in replacements.items():new=new.replace(before,after)
                if new!=old:report.write_text(new)
        migration=root/'artifact_migration.json'
        prior=json.loads(migration.read_text()) if migration.exists() else {}
        prior.update(replacements)
        migration.write_text(json.dumps(prior,indent=2))
    name=run_name(brain,body,task,mode,seed);destination=root.with_name(name)
    if destination!=root:
        if destination.exists():raise FileExistsError(destination)
        root.rename(destination);root=destination
    steps=e['steps'];metrics=json.loads((root/'metrics.json').read_text()) if (root/'metrics.json').exists() else {};metrics.update(e)
    metrics.update(schema_version='arena_metrics_v1',run_id=name,brain=brain,body=BODY_NAMES[body.lower()],task=task,mode=mode,seed=seed,steps=steps,sim_seconds=steps*.02,success=bool(e['success']),recording_verified=True)
    (root/'metrics.json').write_text(json.dumps(metrics,indent=2))
    fields=['run_id','brain','body','task','mode','seed','success','termination','steps','sim_seconds','recording_verified']
    with (root/'task_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow({k:metrics.get(k) for k in fields})
    import numpy as np
    with np.load(root/'episode.npz',allow_pickle=False) as episode:
        native_schema=str(episode['schema_version' if 'schema_version' in episode else 'schema'])
    manifest=dict(schema_version='arena_artifacts_v1',run_id=name,brain=brain,body=metrics['body'],task=task,mode=mode,seed=seed,files=dict(metrics='metrics.json',table='task_metrics.csv',evaluation='task_evaluation.json',protocol='evaluation_protocol.json',provenance='provenance.json',verification='recording_verification.json',decisions='brain_decisions.jsonl',timeline='simulation_timeline.jsonl',episode='episode.npz',ego_video='videos/ego.mp4',g1_video='videos/g1.mp4',debug='debug/'),native_tensor_schema=native_schema,video_fps=50,control_dt=.02)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return root

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--body',required=True);a=p.parse_args();print(publish(a.directory,a.body))
