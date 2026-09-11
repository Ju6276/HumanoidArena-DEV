"""OpenAI-compatible multimodal Brain subprocess; reads JSON stdin.

Configure BRAIN_BASE_URL, BRAIN_MODEL and BRAIN_API_KEY_ENV (name, not secret).
Only ego image bytes and public observation fields are sent to the model.
"""
import base64
import copy
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen


def messages(request):
    result = [dict(role='system', content=request['system'] + '\nReturn ONLY the action JSON. Example: '+json.dumps(request['action_example']))]
    for item in request['history']:
        if 'action' in item:
            result.append(dict(role='assistant', content=json.dumps(item['action'])))
        else:
            public = copy.deepcopy(item)
            image = public['observation'].pop('ego_image')
            data = base64.b64encode(Path(image).read_bytes()).decode('ascii')
            result.append(dict(role='user', content=[dict(type='text', text=json.dumps(public)),
                dict(type='image_url', image_url=dict(url='data:image/png;base64,'+data))]))
    return result


def complete(request):
    model = os.environ['BRAIN_MODEL']
    key_name = os.environ.get('BRAIN_API_KEY_ENV', 'OPENAI_API_KEY')
    key = os.environ.get(key_name)
    if not key:
        raise RuntimeError(f'Missing credential environment variable: {key_name}')
    base = os.environ.get('BRAIN_BASE_URL', 'https://api.openai.com/v1').rstrip('/')
    body = dict(model=model, messages=messages(request), response_format={'type':'json_object'})
    # Provider-specific generation settings must be explicit and reproducible.
    options = json.loads(os.environ.get('BRAIN_GENERATION_OPTIONS', '{}'))
    if any(k in options for k in ('model','messages','stream')):
        raise ValueError('Generation options cannot override model/messages/stream')
    body.update(options)
    req = Request(base+'/chat/completions', data=json.dumps(body).encode(), headers={
        'Authorization':'Bearer '+key, 'Content-Type':'application/json'})
    with urlopen(req, timeout=110) as response:
        reply = json.load(response)
    usage = reply.get('usage') or {}
    content = reply['choices'][0]['message'].get('content')
    try:
        action = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        # Preserve usage even for invalid model output; runner handles validation.
        action = content
    return dict(action=action, model=reply.get('model',model), usage={
        'input_tokens':usage.get('prompt_tokens',0),
        'cached_input_tokens':(usage.get('prompt_tokens_details') or {}).get('cached_tokens',0),
        'output_tokens':usage.get('completion_tokens',0),
        'reasoning_output_tokens':(usage.get('completion_tokens_details') or {}).get('reasoning_tokens',0)},
        usage_available=bool(usage), service_tier=reply.get('service_tier'))

if __name__ == '__main__':
    print(json.dumps(complete(json.load(sys.stdin))))
