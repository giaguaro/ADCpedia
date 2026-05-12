import copy
from pathlib import Path

import yaml


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    path = Path(path)
    with path.open('r') as f:
        cfg = yaml.safe_load(f) or {}
    cfg['_config_path'] = str(path)
    cfg['_repo_root'] = str(path.resolve().parents[1])
    return cfg


def repo_path(cfg, value):
    p = Path(value)
    if p.is_absolute():
        return p
    return Path(cfg.get('_repo_root', '.')) / p


def apply_overrides(cfg, overrides):
    data = _merge(cfg, overrides or {})
    data['_config_path'] = cfg.get('_config_path')
    data['_repo_root'] = cfg.get('_repo_root')
    return data
