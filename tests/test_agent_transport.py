import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess

from fm_humanoid_bench.brains.agent_prompt import packet
from fm_humanoid_bench.brains.agent_transport import request_agent


class AgentTransportTests(unittest.TestCase):
    def test_command_agent_receives_versioned_prompt_without_human_handoff(self):
        observation = dict(
            pos=[0, 0, .75], state64=[0.] * 64, hands=[0, 0], step=0,
            ego_image='/current/ego.png', compact_proprioception={})
        request, _ = packet(
            observation, [], None, 'none', task='Open the door.',
            remaining_steps=1800)
        program = (
            "import json,sys; p=json.load(sys.stdin); "
            "assert p['system_prompt']['id']=='reference_v31'; "
            "print(json.dumps({'prompt_sha256':p['prompt_sha256'],"
            "'decision':{'rationale':'smoke','horizon':1,'keyframes':[{'frame':0}]}}))"
        )
        with tempfile.TemporaryDirectory() as directory:
            response, metadata = request_agent(
                request, Path(directory), 'agent_000000',
                command=[sys.executable, '-c', program], timeout=10)
            saved = json.loads((Path(directory) / 'agent_000000.request.json').read_text())
            self.assertEqual(saved['system_prompt']['id'], 'reference_v31')
            self.assertEqual(response['prompt_sha256'], metadata['prompt_sha256'])
            self.assertEqual(metadata['transport'], 'command')

    def test_failed_agent_command_preserves_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(subprocess.CalledProcessError):
                request_agent({'value':1},root,'failed',command=[
                    sys.executable,'-c','import sys;print("diagnostic",file=sys.stderr);sys.exit(7)'])
            self.assertEqual((root/'failed.stderr.log').read_text().strip(),'diagnostic')


if __name__ == '__main__':
    unittest.main()
