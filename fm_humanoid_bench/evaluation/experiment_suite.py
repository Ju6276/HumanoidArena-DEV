"""Paired experiment plans and aggregation with explicit missing/error coverage."""
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from fm_humanoid_bench.environments.task_registry import TASKS,MODES,original_budget,environment_config
from fm_humanoid_bench.evaluation.artifact_contract import BODY_NAMES,run_name
from fm_humanoid_bench.adapters import require_body, require_brain

SCHEMA='arena_benchmark_suite_v1'
BODY_ALIASES={'sonic':'sonic','twist2':'twist2','scalebfm':'scale','scale':'scale','holomotion':'holo','holo':'holo','bfm_zero':'zero','zero':'zero'}

def canonical_task(task):
    return {'double_desk':'doubledesk','opendoor':'open_door'}.get(task,task)


def episode_seed(task,group_seed,repeat):
    payload=f'{TASKS[task]["env_id"]}|{int(group_seed)}|{int(repeat)}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[-4:],'big') & 0x7fffffff


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_frozen_plan(previous,planned):
    if digest(previous)!=digest(planned):raise ValueError('Frozen plan or its environment/reference/model inputs changed; use a new root')


def plan(spec,catalog):
    allowed={'schema','name','models','bodies','tasks','modes','tracks','conditions','group_seeds','repeats','execute_frames','environment_backend','reference_catalog','reference_selection','reference_ids','agent_timeout','notes'}
    if spec.get('schema')!=SCHEMA or set(spec)-allowed:raise ValueError('Unknown suite schema/fields')
    tasks=[canonical_task(x) for x in spec['tasks']]
    if set(tasks)-TASKS.keys() or set(spec['modes'])-MODES.keys():raise ValueError('Unknown task/mode')
    seeds=spec['group_seeds'];repeats=spec['repeats'];execute=spec.get('execute_frames',20)
    if not seeds or any(type(s) is not int or s<0 for s in seeds) or len(set(seeds))!=len(seeds):raise ValueError('Unique nonnegative group seeds required')
    if type(repeats) is not int or repeats<1 or type(execute) is not int or not 1<=execute<=250:raise ValueError('Invalid repeat/execute budget')
    tracks=spec.get('tracks',['reference']);conditions=spec.get('conditions',['none'])
    if set(tracks)-{'reference','native'} or set(conditions)-{'none','state_action','images'}:raise ValueError('Unknown track/condition')
    for key,values in [('models',spec['models']),('bodies',spec['bodies']),('tasks',tasks),('modes',spec['modes']),('tracks',tracks),('conditions',conditions)]:
        if not values or len(values)!=len(set(values)):raise ValueError(f'Empty or duplicated {key}')
    reference_catalog=None
    if spec.get('reference_catalog'):
        from fm_humanoid_bench.evaluation.reference_catalog import load_catalog
        reference_catalog=load_catalog(spec['reference_catalog'])
    entries=catalog['models'];jobs=[];excluded=[]
    for model_id in spec['models']:
        if model_id not in entries:raise ValueError(f'Model absent from catalog: {model_id}')
        model=entries[model_id]
        require_brain(model)
        supported={canonical_task(t) for t in model.get('tasks',tasks)}
        for task,body,mode,track,condition in itertools.product(tasks,spec['bodies'],spec['modes'],tracks,conditions):
            backend=BODY_ALIASES[body]
            require_body(backend, track)
            combo=dict(model_id=model_id,task=task,body=backend,mode=mode,track=track,condition=condition)
            reason=None
            if task not in supported:reason='checkpoint not trained/configured for this task'
            elif track=='native' and model['kind']=='http':reason='reference-only VLA/WAM output; no native-condition head'
            elif condition!='none' and model['kind']=='http':reason='frozen Brain does not accept in-context references'
            elif condition!='none' and not spec.get('reference_catalog'):reason='reference catalog not configured'
            if reason is None and condition!='none':
                from fm_humanoid_bench.evaluation.reference_catalog import reference_options
                options=reference_options(reference_catalog,task,condition)
                if not options:reason='no matching task/modality demonstration'
                configured_id=spec.get('reference_ids',{}).get(task)
                if configured_id and configured_id not in {x['id'] for x in options}:raise ValueError('Configured reference ID does not match task/modality')
                if options and not configured_id and spec.get('reference_selection','config')=='config' and len(options)>1:raise ValueError('Multiple demonstrations require explicit ID or Agent selection')
            if reason:excluded.append(dict(**combo,reason=reason));continue
            config=environment_config(task,mode,spec.get('environment_backend','twist2'))
            budget,source=original_budget(task,spec.get('environment_backend','twist2'))
            for seed,repeat in itertools.product(seeds,range(repeats)):
                actual=episode_seed(task,seed,repeat)
                job=dict(**combo,group_seed=seed,repeat_index=repeat,seed=actual,max_steps=budget,execute_frames=execute,
                         environment_backend=spec.get('environment_backend','twist2'),environment_config=config,
                         environment_config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),budget_source=str(source),
                         model_digest=digest(model),runtime_profile=TASKS[task]['runtime_profile'])
                job['job_id']=digest(job)[:20]
                job['directory']=str(Path(track)/condition/run_name(model_id,backend,task,mode,actual))
                if model.get('kind') == 'file' and not model.get('agent_command'):
                    job['availability'] = 'requires_agent_worker'
                    job['blocked_reason'] = 'File transport is ready, but this suite has no configured autonomous Agent command worker'
                else:
                    job['availability']='ready' if model.get('available') else 'blocked'
                    if not model.get('available'):job['blocked_reason']=model.get('unavailable_reason',model.get('reason','No compatible local checkpoint'))
                jobs.append(job)
    if len({j['directory'] for j in jobs})!=len(jobs):raise ValueError('Artifact identity collision')
    return dict(schema=SCHEMA,spec_digest=digest(spec),catalog_digest=digest(catalog),reference_catalog_digest=digest(reference_catalog),spec=spec,jobs=jobs,excluded_combinations=excluded,
                seed_policy='original HA sha256(task_name|group_seed|repeat_idx)_positive_int32',
                server_policy='fresh process per episode; reset acknowledgement is not proof of RNG reseeding',
                scoring='original HA success/fall/timeout only; infrastructure errors are separate')


def wilson(successes,n):
    if not n:return None
    z=1.959963984540054;p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-half),min(1,center+half)]


def summarize(plan_data,root):
    root=Path(root);groups={}
    for job in plan_data['jobs']:
        key=tuple(job[k] for k in ('model_id','body','task','mode','track','condition'))
        g=groups.setdefault(key,dict(zip(('model_id','body','task','mode','track','condition'),key)),)
        g.setdefault('planned',0);g['planned']+=1
        for name in ('completed','successes','falls','timeouts','infrastructure_errors','blocked','pending'):g.setdefault(name,0)
        result=root/job['directory'];metrics=result/'metrics.json'
        error=root/'jobs'/f"{job['job_id']}.error.json"
        if error.exists():g['infrastructure_errors']+=1;continue
        if metrics.exists():
            m=json.loads(metrics.read_text())
            audit=result/'interface_audit.json'
            if not audit.exists() or not json.loads(audit.read_text()).get('passed') or not m.get('recording_verified') or m.get('evaluation_kind')!='brain_closed_loop' or m.get('termination') not in ('success','fall','timeout'):
                g['infrastructure_errors']+=1;continue
            g['completed']+=1;g['successes']+=int(m['success']);g['falls']+=int(m['termination']=='fall');g['timeouts']+=int(m['termination']=='timeout')
        elif (root/'jobs'/f"{job['job_id']}.error.json").exists():g['infrastructure_errors']+=1
        elif job['availability']!='ready':g['blocked']+=1
        else:g['pending']+=1
    rows=[]
    for g in groups.values():
        n=g['completed'];g.update(success_rate=g['successes']/n if n else None,success_rate_wilson95=wilson(g['successes'],n),coverage=n/g['planned'],complete=n==g['planned'])
        rows.append(g)
    return dict(schema='arena_benchmark_summary_v1',groups=rows,complete=all(g['complete'] for g in rows) and bool(rows),
                excluded_combinations=plan_data['excluded_combinations'],success_rule='original HA only',
                note='Partial success rates use completed task episodes only; coverage and infrastructure errors remain explicit. Do not rank incomplete groups as completed benchmark cells.')


def episode_index(plan_data, root):
    """Build one queryable row per planned episode without copying artifacts."""
    root=Path(root);rows=[]
    for job in plan_data['jobs']:
        run=root/job['directory'];metrics_path=run/'metrics.json';error=root/'jobs'/f"{job['job_id']}.error.json"
        row={k:job[k] for k in ('job_id','model_id','body','task','mode','track','condition','group_seed','repeat_index','seed')}
        row.update(status='pending',success=None,termination=None,steps=None,run_dir=job['directory'],ego_video=None,g1_video=None,
                   recording_verified=None,interface_audit_passed=None)
        if error.exists():row['status']='infrastructure_error'
        elif metrics_path.exists():
            metrics=json.loads(metrics_path.read_text());audit=run/'interface_audit.json'
            row.update(status='complete',success=bool(metrics.get('success')),termination=metrics.get('termination'),steps=metrics.get('steps'),
                       recording_verified=bool(metrics.get('recording_verified')),
                       interface_audit_passed=bool(audit.exists() and json.loads(audit.read_text()).get('passed')),
                       ego_video=str(Path(job['directory'])/'videos/ego.mp4'),g1_video=str(Path(job['directory'])/'videos/g1.mp4'))
        elif job['availability']!='ready':row['status']='blocked'
        rows.append(row)
    return rows


def publish_video_gallery(rows, root):
    """Create status/view links for browsing; canonical videos stay in episodes."""
    root=Path(root);video_root=root/'videos';video_root.mkdir(exist_ok=True)
    links=[]
    for row in rows:
        if row['status']!='complete' or not row['recording_verified']:continue
        outcome='success' if row['success'] else f"failure/{row['termination'] or 'unknown'}"
        slug=(f"{row['model_id']}__{row['body']}__{row['task']}__{row['mode']}__"
              f"gs{row['group_seed']}__r{row['repeat_index']}__s{row['seed']}")
        for view,key in (('ego','ego_video'),('g1','g1_video')):
            source=root/row[key]
            if not source.is_file():continue
            destination=video_root/outcome/view/f'{slug}.mp4';destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.is_symlink() or destination.exists():destination.unlink()
            destination.symlink_to(Path(os.path.relpath(source,destination.parent)))
            links.append(dict(job_id=row['job_id'],outcome=outcome,view=view,link=str(destination.relative_to(root)),source=row[key]))
    (video_root/'index.json').write_text(json.dumps(links,indent=2))
    return links
