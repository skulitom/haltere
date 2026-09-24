"""CLI: python -m haltere.obstacles.labels {build, teacher} (offline only)."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..store import DEFAULT_STORE


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m haltere.obstacles.labels')
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--store', type=Path, default=DEFAULT_STORE)
    b.add_argument('--colliders', type=Path, required=True)
    b.add_argument('--events', type=Path, required=True, help='lateral study manifest.json')
    b.add_argument('--extra-events', type=Path, default=None, help='configs/obstacles/events_f12.json')
    b.add_argument('--stages', nargs='*', default=None)
    b.add_argument('--flight-lock', default=None)
    t = sub.add_parser('teacher')
    t.add_argument('--store', type=Path, default=DEFAULT_STORE)
    t.add_argument('--flight-lock', default=None)
    args = p.parse_args(argv)
    from . import build as _build
    return getattr(_build, args.cmd)(args)


if __name__ == '__main__':
    main()
