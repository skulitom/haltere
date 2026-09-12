"""YAML configuration loading for the graph, brain, simulator and training runs."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar('T')


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def dataclass_from_dict(cls: type[T], d: dict | None) -> T:
    """Instantiate a (flat) dataclass from a dict, converting lists to tuples where the default is a tuple."""
    d = d or {}
    obj = cls()
    names = {f.name for f in fields(cls)}
    for k, v in d.items():
        if k not in names:
            raise KeyError(f'{cls.__name__}: unknown key {k!r} (valid: {sorted(names)})')
        cur = getattr(obj, k)
        if isinstance(cur, tuple) and isinstance(v, list):
            v = tuple(v)
        setattr(obj, k, v)
    return obj


def dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    return obj
