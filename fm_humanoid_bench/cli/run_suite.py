"""Plan/run/resume a paired Brain x Body x task x mode experiment matrix."""
import argparse
import csv
import json
import os
from pathlib import Path
import socket
import signal
import hashlib
import fcntl
import subprocess
import sys
REPO=Path(os.environ.get('FMHB_ROOT',Path(__file__).resolve().parents[2])).expanduser().resolve();sys.path.insert(0,str(REPO))
from fm_humanoid_bench.evaluation.experiment_suite import plan,summarize,digest,validate_frozen_plan,episode_index,publish_video_gallery
from fm_humanoid_bench.catalog import load_model_catalog


def stop(process):
    if process is not None:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()


def owned_run(command, **kwargs):
    process=subprocess.Popen(command,start_new_session=True,**kwargs)
    try:
        code=process.wait(timeout=3600)
        if code:raise subprocess.CalledProcessError(code,command)
    finally:stop(process)


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('--catalog',type=Path,default=REPO/'fm_humanoid_bench/model_catalog.json')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--run',action='store_true',help='Without --run, only produce immutable plan and coverage summary')
    p.add_argument('--max-jobs',type=int);p.add_argument('--job-id',action='append');p.add_argument('--retry-errors',action='store_true')
    a=p.parse_args();a.root=a.root.expanduser().resolve();spec=json.loads(a.config.read_text());catalog=load_model_catalog(a.catalog);planned=plan(spec,catalog)
    a.root.mkdir(parents=True,exist_ok=True)
    with (a.root/'.suite.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        planfile=a.root/'plan.json'
        if planfile.exists():
            previous=json.loads(planfile.read_text())
            validate_frozen_plan(previous,planned)
        else:planfile.write_text(json.dumps(planned,indent=2))
        (a.root/'catalog.json').write_text(json.dumps(catalog,indent=2));(a.root/'jobs').mkdir(exist_ok=True)
        completed=0
        if a.run:
            for job in planned['jobs']:
                if a.job_id and job['job_id'] not in a.job_id:continue
                if job['availability']!='ready':continue
                run=a.root/job['directory'];error=a.root/'jobs'/f"{job['job_id']}.error.json"
                if (run/'metrics.json').exists() and (run/'interface_audit.json').exists() and not error.exists():continue
                if error.exists() and not a.retry_errors:continue
                if a.max_jobs is not None and completed>=a.max_jobs:break
                if run.exists():
                    if not a.retry_errors:continue
                    archive=a.root/'infrastructure_attempts'/job['job_id'];archive.parent.mkdir(exist_ok=True)
                    n=0
                    while archive.with_name(archive.name+f'_{n}').exists():n+=1
                    run.rename(archive.with_name(archive.name+f'_{n}'))
                if a.max_jobs is not None and completed>=a.max_jobs:break
                model=catalog['models'][job['model_id']];server=None
                with (a.root/'jobs'/f"{job['job_id']}.server.log").open('w') as server_log,(a.root/'jobs'/f"{job['job_id']}.run.log").open('w') as run_log:
                    try:
                        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
                        cmd=[sys.executable,str(REPO/'fm_humanoid_bench/cli/run_episode.py'),'--body',job['body'],'--brain',model['family'],
                             '--brain-label',job['model_id'],'--task',job['task'],'--mode',job['mode'],'--track',job['track'],
                             '--seed',str(job['seed']),'--execute',str(job['execute_frames']),'--horizon',str(model.get('horizon',30)),
                             '--condition',job['condition'],'--root',str(a.root/job['track']),'--environment-backend',job['environment_backend']]
                        if model['kind']=='http':
                            server_cmd=[x.replace('{port}',str(port)).replace('{device}','cuda:0') for x in model['server_command']]
                            if any('{' in x or '}' in x for x in server_cmd):raise ValueError('Unresolved server command placeholder')
                            env=os.environ.copy();env.update(model.get('env',{}))
                            server=subprocess.Popen(server_cmd,cwd=model['cwd'],env=env,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
                            cmd+=['--server',f'http://127.0.0.1:{port}']
                        elif model.get('agent_command'):cmd+=['--agent-command-json',json.dumps(model['agent_command'])]
                        if job['condition']!='none':
                            cmd+=['--reference-catalog',spec['reference_catalog'],'--reference-selection',spec.get('reference_selection','config')]
                            reference_id=spec.get('reference_ids',{}).get(job['task'])
                            if reference_id:cmd+=['--reference-id',reference_id]
                        print('START '+job['job_id']+' '+job['directory'],flush=True)
                        owned_run(cmd,cwd=REPO,stdout=run_log,stderr=subprocess.STDOUT)
                        (run/'job.json').write_text(json.dumps(job,indent=2));(run/'brain_provenance.json').write_text(json.dumps(model,indent=2))
                        checkpoint=Path(model['checkpoint']) if model.get('checkpoint') else None
                        weight_files=([checkpoint] if checkpoint and checkpoint.is_file() else sorted(checkpoint.glob('*.safetensors')) if checkpoint and checkpoint.is_dir() else [])
                        hashes={}
                        for weight in weight_files:
                            with weight.open('rb') as source:hashes[str(weight)]=hashlib.file_digest(source,'sha256').hexdigest()
                        (run/'brain_checkpoint_hashes.json').write_text(json.dumps(hashes,indent=2))
                        audit='audit_native.py' if job['track']=='native' else 'audit_reference.py'
                        owned_run([sys.executable,str(REPO/'fm_humanoid_bench/cli'/audit),str(run),'--body',job['body']],cwd=REPO,stdout=run_log,stderr=subprocess.STDOUT)
                        error.unlink(missing_ok=True)
                        print('DONE '+job['directory'],flush=True)
                    except Exception as exc:
                        error.write_text(json.dumps(dict(job=job,error=repr(exc),classification='infrastructure_error_not_task_failure'),indent=2))
                        print('ERROR '+job['directory']+' '+repr(exc),flush=True)
                    finally:stop(server)
                completed+=1
        report=summarize(planned,a.root);(a.root/'summary.json').write_text(json.dumps(report,indent=2))
        rows=report['groups']
        if rows:
            with (a.root/'task_metrics.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        episodes=episode_index(planned,a.root)
        with (a.root/'episodes.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(episodes[0]) if episodes else []);writer.writeheader()
            if episodes:writer.writerows(episodes)
        with (a.root/'episodes.jsonl').open('w') as f:
            for episode in episodes:f.write(json.dumps(episode)+'\n')
        publish_video_gallery(episodes,a.root)
        (a.root/'README.md').write_text('# Foundation-model humanoid benchmark\n\nOriginal HA success/fall/timeout only. Paired episode seeds, fresh Brain/simulator per episode, no reset retries in episodes.\n\n[Plan](plan.json) · [Summary and coverage](summary.json) · [Aggregate metrics](task_metrics.csv) · [Episode index](episodes.csv) · [Video index](videos/index.json)\n\nCanonical ego/G1 videos remain inside each episode. `videos/success/` and `videos/failure/<reason>/` are relative links for browsing and do not duplicate data. Debug observations, Agent transport files and runner logs stay under each episode\'s `debug/` directory.\n')
        print(json.dumps(dict(planned_jobs=len(planned['jobs']),attempted_now=completed,complete=report['complete'])),flush=True)

if __name__=='__main__':main()
