"""Shared HA HTTP Brain boundary for PSI0, GR00T, VLA-JEPA and DiT4DiT.

The legacy servers already return denormalized semantic_v3 G1 references.
Do not normalize again, interpret these as latent tokens, or pass evaluator
state to the policy. Transport agreement alone is not checkpoint qualification.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from fm_humanoid_bench.protocols.reference import DT, SCHEMA, validate_chunk

PROTOCOL = "arena_http_brain_v1"
HTTP_BRAINS = ("psi0", "groot", "vla_jepa", "dit4dit", "pi05", "http")
STATE_SCHEMA = "heading_canonical_rot6d_row_q29_dq29"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def build_request(observation: dict, instruction: str) -> dict:
    """Whitelist ego RGB + measured state64; preserve the canonical instruction.

    A path is resized exactly as the previous PSI0 runner did. An in-memory
    front_rgb array must already match the external HA image contract.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty canonical task instruction is required")
    state = np.asarray(observation["state64"], dtype=np.float32)
    if state.shape != (64,) or not np.isfinite(state).all():
        raise ValueError("Expected finite measured state64 [64]")
    if "front_rgb" in observation:
        image = np.asarray(observation["front_rgb"])
    else:
        with Image.open(Path(observation["ego_image"])) as source:
            image = np.asarray(source.convert("RGB").resize((640, 480)), dtype=np.uint8)
    if image.dtype != np.uint8 or image.shape != (480, 640, 3):
        raise ValueError("Expected uint8 front RGB [480,640,3]")
    image = np.ascontiguousarray(image)
    return {
        "task": instruction,
        "return_chunk": True,
        "observation": {
            "state": state.tolist(),
            "images": {"front": {
                "shape": list(image.shape), "dtype": "uint8",
                "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
            }},
        },
    }


def validate_response(reply: dict, expected_horizon: int | None = 30) -> np.ndarray:
    """Validate physical reference structure without changing its values.

    Legacy servers omit schema and dt: their versioned catalog declaration is
    the contract. Explicit incompatible response declarations are rejected.
    Numeric values alone cannot prove units or training-time semantics.
    """
    if not isinstance(reply, dict) or "error" in reply:
        raise ValueError("Brain server returned an error or non-object response")
    if reply.get("schema", SCHEMA) != SCHEMA:
        raise ValueError("Brain returned an incompatible reference schema")
    if reply.get("action_kind", "semantic_v3") not in ("semantic_v3", SCHEMA):
        raise ValueError("Brain returned latent or unsupported action semantics")
    if reply.get("control_dt", DT) != DT:
        raise ValueError("Brain control_dt differs from the 50 Hz reference contract")
    actions = validate_chunk(reply.get("action_chunk"))
    if expected_horizon is not None and len(actions) != expected_horizon:
        raise ValueError(f"Expected {expected_horizon} predicted frames, got {len(actions)}")
    # Validate all metadata as serializable finite JSON before physics changes.
    _json_bytes(reply)
    return actions


def http(server: str, endpoint: str, payload: dict, *, timeout: float = 180) -> dict:
    parts = urlsplit(server)
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError("Expected a credential-free HTTP(S) Brain server URL")
    if endpoint not in ("reset", "infer"):
        raise ValueError("Unsupported Brain endpoint")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    request = Request(server.rstrip("/") + "/" + endpoint,
                      data=_json_bytes(payload),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        reply = json.load(response)
    if not isinstance(reply, dict):
        raise ValueError("Brain server response must be a JSON object")
    return reply


def reset(server: str, seed: int, *, timeout: float = 180) -> dict:
    """Request episode reset; acknowledgement is not proof of RNG reseeding."""
    if type(seed) is not int:
        raise ValueError("Episode seed must be an integer")
    before = time.monotonic()
    reply = http(server, "reset", {"seed": seed}, timeout=timeout)
    if reply.get("ok") is not True or "error" in reply:
        raise ValueError("Brain did not acknowledge episode reset")
    return {"protocol": PROTOCOL, "requested_seed": seed,
            "latency_seconds": time.monotonic() - before,
            "server_response": reply, "rng_reseed_verified": False}


def infer(server: str, observation: dict, instruction: str, *,
          expected_horizon: int | None = 30, timeout: float = 180) -> tuple[np.ndarray, dict]:
    """Return a validated trajectory and auditable prediction record."""
    request = build_request(observation, instruction)
    before = time.monotonic()
    reply = http(server, "infer", request, timeout=timeout)
    latency = time.monotonic() - before
    actions = validate_response(reply, expected_horizon)
    image = request["observation"]["images"]["front"]
    record = {
        "protocol": PROTOCOL, "schema": SCHEMA, "control_dt": DT,
        "state_schema": STATE_SCHEMA, "step": observation.get("step"),
        "instruction": instruction, "state64": request["observation"]["state"],
        "ego_image": str(observation["ego_image"]) if "ego_image" in observation else None,
        "front_shape": image["shape"],
        "front_sha256": hashlib.sha256(base64.b64decode(image["data_b64"])).hexdigest(),
        "request_sha256": hashlib.sha256(_json_bytes(request)).hexdigest(),
        "response_sha256": hashlib.sha256(_json_bytes(reply)).hexdigest(),
        "latency_seconds": latency, "action_chunk": actions.tolist(),
        "server_metadata": {key: value for key, value in reply.items() if key != "action_chunk"},
        "response_declared_schema": reply.get("schema"),
        "schema_basis": "server_response" if "schema" in reply else "configured_legacy_server_contract",
    }
    return actions, record
