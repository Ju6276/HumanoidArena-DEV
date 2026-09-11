"""Versioned Agent prompts with optional one-episode context."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from fm_humanoid_bench.protocols.reference import CANONICAL_JOINT_NAMES_29,compile_keyframes
from fm_humanoid_bench.evaluation.reference_catalog import build_catalog, select_reference, validate_selection


PROMPT_FILES = {
    'reference': 'reference_v31.txt',
    'native': 'native_body_v1.txt',
}


def load_system_prompt(track='reference'):
    try:
        path = Path(__file__).parent.parent / 'prompts' / PROMPT_FILES[track]
    except KeyError as exc:
        raise ValueError(f'Unknown Agent track: {track}') from exc
    text = path.read_text()
    return text, {'id': path.stem, 'sha256': hashlib.sha256(text.encode()).hexdigest()}


def packet(observation,history,reference,condition,*,task=None,task_id=None,remaining_steps=None,selection=None,track='reference'):
    system,system_prompt=load_system_prompt(track)
    start=np.r_[0.,0.,observation['pos'][2],observation['state64'][:6],observation['state64'][6:35],observation['hands']].astype(np.float32)
    if history:start=np.asarray(history[-1]['executed_last_action'],np.float32)
    task=task or observation.get('instruction') or 'Open the door.'
    from fm_humanoid_bench.environments.task_registry import TASKS
    if task_id is None:
        task_id=next((key for key,value in TASKS.items() if value['instruction']==task),None)
    if task_id not in TASKS:raise ValueError('A registered task_id is required for Agent prompting')
    if task!=TASKS[task_id]['instruction']:raise ValueError('task_id and task instruction disagree')
    if remaining_steps is None:
        max_steps=observation.get('max_steps')
        if max_steps is None:
            from fm_humanoid_bench.evaluation.reference_catalog import _task
            from fm_humanoid_bench.environments.task_registry import original_budget
            max_steps=original_budget(_task(task))[0]
        remaining_steps=max(0,max_steps-observation['step'])
    user=dict(task=task,task_id=task_id,task_instruction=task,task_description=TASKS[task_id]['description'],step=observation['step'],remaining_steps=remaining_steps,ego_image=observation['ego_image'],proprioception=observation['compact_proprioception'],joint_names=list(CANONICAL_JOINT_NAMES_29),history=[dict(step=x['step'],decision=x['decision']) for x in history[-6:]],reference_condition=condition)
    if track=='reference':user['start_reference40']=start.tolist()
    else:
        user['native_body_schema']=observation['native_body_schema']
        user['max_execute_frames']=observation.get('max_execute_frames',20)
    if 'body_contract' in observation:user['body_contract']=observation['body_contract']
    if selection is None:
        if condition!='none' and reference is None:raise ValueError('Reference condition requires a reference.json')
        if condition=='none' and reference is not None:raise ValueError('none condition cannot receive a reference')
        catalog=build_catalog([reference] if reference is not None else [])
        reference_id=catalog['entries'][0]['id'] if catalog['entries'] else None
        selection=select_reference(catalog,task=task,condition=condition,reference_id=reference_id)
    selection=validate_selection(selection,task=task,condition=condition)
    receipt=selection['receipt']
    user['reference_selection']={key:receipt[key] for key in ('reference_id','condition','selector','bundle_sha256','payload_sha256')}
    if selection['demonstration'] is not None:user['demonstration']=selection['demonstration']
    result=dict(system=system,system_prompt=system_prompt,user=user)
    result['prompt_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result,start


def decide(observation,history,reference,condition,root,execute=20,*,task=None,remaining_steps=None,selection=None):
    tick=observation['step'];prompt,start=packet(observation,history,reference,condition,task=task,remaining_steps=remaining_steps,selection=selection)
    path=root/f'prompt_{tick:06d}.json';path.write_text(json.dumps(prompt,indent=2))
    decision_path=root/f'decision_{tick:06d}.json'
    print(json.dumps(dict(waiting_for_gpt6=True,prompt=str(path),decision_file=str(decision_path))),flush=True)
    # The active GPT-6 agent reads this packet + referenced images and writes a
    # decision. Saving a prompt alone is never reported as model inference.
    while not decision_path.exists():time.sleep(.25)
    raw=json.loads(decision_path.read_text())
    if raw.get('prompt_sha256')!=prompt['prompt_sha256']:raise ValueError('Decision must bind to the exact prompt')
    decision=raw['decision']
    actions=compile_keyframes(decision['keyframes'],start,decision['horizon'])
    record=dict(step=tick,prompt_sha256=prompt['prompt_sha256'],reference_selection=prompt['user']['reference_selection'],decision=decision,action_chunk=actions.tolist(),executed_last_action=actions[min(execute,len(actions))-1].tolist(),delivery='GPT-6 active agent, file transport; no separate API usage bill')
    return actions,record
