"""CPU boundary checks; no model weights, simulator, or external service."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from fm_humanoid_bench.protocols import brain_protocol as bp


class BrainProtocolTest(unittest.TestCase):
    def setUp(self):
        self.observation = {"state64": np.arange(64, dtype=np.float32),
                            "front_rgb": np.full((480, 640, 3), 73, dtype=np.uint8),
                            "step": 20, "task_success": True, "door_position": [99, 99]}
        self.action = np.zeros((30, 40), dtype=np.float32)
        self.action[:, 3:9] = [1, 0, 0, 1, 0, 0]
        self.action[:, 0] = .02
        self.reply = {"action_chunk": self.action.tolist(), "model": "test", "usage": {"calls": 1}}

    def test_observation_whitelist_and_exact_wire_image(self):
        request = bp.build_request(self.observation, "Open the door.")
        self.assertEqual(set(request), {"task", "observation", "return_chunk"})
        self.assertIs(request["return_chunk"], True)
        self.assertEqual(set(request["observation"]), {"state", "images"})
        self.assertEqual(request["task"], "Open the door.")
        image = request["observation"]["images"]["front"]
        self.assertEqual(base64.b64decode(image["data_b64"]), self.observation["front_rgb"].tobytes())

    def test_source_image_path_matches_previous_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ego.png"
            source = Image.fromarray(np.full((240, 320, 3), 29, dtype=np.uint8))
            source.save(path)
            observation = {"state64": self.observation["state64"], "ego_image": path}
            spec = bp.build_request(observation, "Sit on the sofa.")["observation"]["images"]["front"]
            self.assertEqual(spec["shape"], [480, 640, 3])
            self.assertEqual(base64.b64decode(spec["data_b64"]), source.resize((640, 480)).tobytes())

    def test_invalid_observation_rejected_before_transport(self):
        cases = ({"state64": [0] * 63}, {"state64": [float("nan")] * 64},
                 {"front_rgb": np.zeros((480, 640, 3), dtype=np.float32)},
                 {"front_rgb": np.zeros((640, 480, 3), dtype=np.uint8)})
        with patch.object(bp, "http") as transport:
            for change in cases:
                with self.subTest(change=list(change)), self.assertRaises(ValueError):
                    bp.infer("http://localhost:8000", {**self.observation, **change}, "Open the door.")
            transport.assert_not_called()

    def test_physical_values_not_renormalized(self):
        self.reply["action_chunk"][0][9] = -1.25
        actions = bp.validate_response(self.reply)
        self.assertEqual(actions[0, 9], -1.25)
        self.assertEqual(actions[0, 0], np.float32(.02))

    def test_bad_action_or_semantic_declaration_rejected(self):
        cases = ({"schema": "sonic_latent"}, {"action_kind": "latent"}, {"control_dt": .04},
                 {"action_chunk": [[0] * 40] * 30}, {"action_chunk": self.action[:20].tolist()},
                 {"action_chunk": np.zeros((30, 64)).tolist()}, {"usage": {"cost": float("nan")}})
        for change in cases:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                bp.validate_response({**self.reply, **change})
        invalid = deepcopy(self.reply)
        invalid["action_chunk"][0][9] = float("inf")
        with self.assertRaises(ValueError):
            bp.validate_response(invalid)

    def test_prediction_metadata_provenance_preserved(self):
        with patch.object(bp, "http", return_value=self.reply) as transport:
            actions, record = bp.infer("http://localhost:8000", self.observation, "Open the door.")
        np.testing.assert_array_equal(actions, self.action)
        self.assertEqual(record["server_metadata"], {"model": "test", "usage": {"calls": 1}})
        self.assertEqual(record["schema_basis"], "configured_legacy_server_contract")
        self.assertGreaterEqual(record["latency_seconds"], 0)
        self.assertEqual(len(record["front_sha256"]), 64)
        self.assertNotIn("door_position", json.dumps(record))
        self.assertEqual(transport.call_args.args[1], "infer")

    def test_reset_does_not_claim_model_rng_reseed(self):
        with patch.object(bp, "http", return_value={"ok": True, "seed": 42}):
            result = bp.reset("http://localhost:8000", 42)
        self.assertFalse(result["rng_reseed_verified"])
        with self.assertRaises(ValueError):
            bp.reset("http://localhost:8000", True)
        with patch.object(bp, "http", return_value={"ok": False}), self.assertRaises(ValueError):
            bp.reset("http://localhost:8000", 42)

    def test_transport_builds_real_http_json_request(self):
        class Reply:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self): return b'{"ok":true}'
        with patch.object(bp, "urlopen", return_value=Reply()) as opener:
            self.assertEqual(bp.http("http://localhost:8000", "reset", {"seed": 42}), {"ok": True})
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:8000/reset")
        self.assertEqual(json.loads(request.data), {"seed": 42})
        self.assertEqual(request.get_method(), "POST")


if __name__ == "__main__":
    unittest.main()
