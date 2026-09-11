"""Agent packets over files or JSON stdin/stdout; no human keypress path."""
import hashlib
import json
from pathlib import Path
import subprocess
import time


def request_agent(packet, root, stem, *, command=None, timeout=600):
    root=Path(root)
    prompt=root/(stem+'.request.json');answer=root/(stem+'.response.json')
    packet=dict(packet)
    packet.pop('prompt_sha256',None)
    packet['prompt_sha256']=hashlib.sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest()
    prompt.write_text(json.dumps(packet,indent=2))
    before=time.monotonic()
    if command:
        process=subprocess.run(command,input=json.dumps(packet),text=True,capture_output=True,timeout=timeout)
        if process.stderr:(root/(stem+'.stderr.log')).write_text(process.stderr)
        if process.returncode:
            if process.stdout:(root/(stem+'.stdout.log')).write_text(process.stdout)
            raise subprocess.CalledProcessError(process.returncode,command,output=process.stdout,stderr=process.stderr)
        result=json.loads(process.stdout)
        answer.write_text(json.dumps(result,indent=2))
    else:
        print(json.dumps(dict(waiting_for_agent=True,prompt=str(prompt),response=str(answer))),flush=True)
        while not answer.exists():
            if time.monotonic()-before>timeout:raise TimeoutError('Agent response timeout; physics remains paused')
            time.sleep(.2)
        result=json.loads(answer.read_text())
    if result.get('prompt_sha256')!=packet['prompt_sha256']:raise ValueError('Agent response must bind to exact request hash')
    return result,dict(prompt_sha256=packet['prompt_sha256'],latency_seconds=time.monotonic()-before,
                       usage=result.get('usage'),usage_available=isinstance(result.get('usage'),dict),
                       model=result.get('model'),transport='command' if command else 'file')
