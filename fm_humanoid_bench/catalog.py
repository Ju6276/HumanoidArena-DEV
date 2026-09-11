"""Portable loading for the versioned Brain model catalog."""
from __future__ import annotations

import json
import os
from pathlib import Path


REPO_ROOT = Path(os.environ.get('FMHB_ROOT', Path(__file__).resolve().parents[1])).expanduser().resolve()


def _expand(value):
    if isinstance(value, str):
        value = value.replace('${FMHB_ROOT}', str(REPO_ROOT))
        value = value.replace(
            '${FMHB_MODEL_ROOT}',
            os.environ.get('FMHB_MODEL_ROOT', '/home/d024/HumanoidArena_models'),
        )
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_model_catalog(path: str | Path) -> dict:
    """Load the catalog and resolve documented environment placeholders."""
    catalog = _expand(json.loads(Path(path).expanduser().read_text()))
    if catalog.get('schema') != 'arena_model_catalog_v1':
        raise ValueError('Unsupported model catalog schema')
    return catalog
