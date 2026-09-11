"""Canonical Brain and Body adapter registry for the benchmark.

This module describes transport and command compatibility.  Checkpoint
availability and task success remain runtime facts recorded by the suite.
"""
from types import MappingProxyType


REFERENCE_SCHEMA = "unitree_g1_gmt_refpose_v3_1"
NATIVE_SCHEMA = "arena_native_body_v1"

BRAIN_ADAPTERS = MappingProxyType({
    "gpt6": {"class": "agent", "transport": "json_command_or_file", "output": REFERENCE_SCHEMA},
    "psi0": {"class": "vla", "transport": "arena_http_brain_v1", "output": REFERENCE_SCHEMA},
    "groot": {"class": "vla", "transport": "arena_http_brain_v1", "output": REFERENCE_SCHEMA},
    "pi05": {"class": "vla", "transport": "arena_http_brain_v1", "output": REFERENCE_SCHEMA},
    "vla_jepa": {"class": "vla", "transport": "arena_http_brain_v1", "output": REFERENCE_SCHEMA},
    "dit4dit": {"class": "wam", "transport": "arena_http_brain_v1", "output": REFERENCE_SCHEMA},
})

BODY_ADAPTERS = MappingProxyType({
    "sonic": {"display_name": "SONIC", "reference": True, "native": True},
    "twist2": {"display_name": "TWIST2", "reference": True, "native": True},
    "scale": {"display_name": "ScaleBFM", "reference": True, "native": True},
    "holo": {"display_name": "HoloMotion", "reference": True, "native": True},
    "zero": {"display_name": "BFM-Zero", "reference": True, "native": True},
})


def require_brain(entry):
    """Return the registered adapter and reject a mismatched catalog entry."""
    family = entry.get("family")
    if family not in BRAIN_ADAPTERS:
        raise ValueError(f"No Brain adapter registered for {family!r}")
    adapter = BRAIN_ADAPTERS[family]
    expected_kind = "file" if adapter["class"] == "agent" else "http"
    if entry.get("kind") != expected_kind:
        raise ValueError(f"Brain {family} requires catalog kind {expected_kind}")
    if entry.get("action_dim", 40) != 40:
        raise ValueError(f"Brain {family} must produce action40")
    if entry.get("action_schema", REFERENCE_SCHEMA) != REFERENCE_SCHEMA:
        raise ValueError(f"Brain {family} declares an incompatible reference schema")
    return adapter


def require_body(name, track="reference"):
    """Return a Body adapter declaration for one benchmark track."""
    if name not in BODY_ADAPTERS:
        raise ValueError(f"No Body adapter registered for {name!r}")
    if track not in ("reference", "native"):
        raise ValueError(f"Unknown benchmark track {track!r}")
    if not BODY_ADAPTERS[name][track]:
        raise ValueError(f"Body {name} does not support the {track} track")
    return BODY_ADAPTERS[name]


def support_matrix():
    return {
        "reference_schema": REFERENCE_SCHEMA,
        "native_schema": NATIVE_SCHEMA,
        "brains": {name: dict(value) for name, value in BRAIN_ADAPTERS.items()},
        "bodies": {name: dict(value) for name, value in BODY_ADAPTERS.items()},
    }
