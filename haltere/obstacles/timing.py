"""Per-video clock refinement (offline). DELIVERED BY THE TIMING AGENT.

``python -m haltere.obstacles.timing refine --store runs/obstacle-store-v1 --search-ms 60 --step-ms 5
--flight-lock PATH``

For every RUN_VIDEO run graded good or fair: search delta in [-60, +60] ms (5 ms steps) added to the
run's align_offset_s, minimising the median bidirectional-track reprojection residual (px at 640 px
width) of corner tracks triangulated with the re-interpolated telemetry poses; frames with
|omega_z| > 2 rad/s (Flag.HIGH_YAW_RATE) are down-weighted. Output
``<store>/timing/refined.json``::

    {"schema": "haltere.obstacles.timing.v1", "search_ms": 60, "step_ms": 5,
     "runs": {"<run_id>": {"source_id": str, "delta_s": float, "residual_px_before": float,
                            "residual_px_after": float, "n_tracks": int, "accepted": bool}}}

Then ``python -m haltere.obstacles.store repose`` applies the accepted deltas (store_build.repose).
K0a: median residual <= 1.0 px at 640 px on >= 70 % of good/fair runs; otherwise geometry labels are
built from exact PNG and capture sources only (~105k targets) and the report says so.
"""
from __future__ import annotations

TIMING_SCHEMA = 'haltere.obstacles.timing.v1'
SEARCH_MS = 60
STEP_MS = 5


def refine(args):
    raise NotImplementedError('timing refinement: delivered by the timing build agent')


def main(argv=None):
    import argparse
    from pathlib import Path
    from .store import DEFAULT_STORE
    p = argparse.ArgumentParser(prog='python -m haltere.obstacles.timing')
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('refine')
    r.add_argument('--store', type=Path, default=DEFAULT_STORE)
    r.add_argument('--search-ms', type=int, default=SEARCH_MS)
    r.add_argument('--step-ms', type=int, default=STEP_MS)
    r.add_argument('--flight-lock', default=None)
    args = p.parse_args(argv)
    return refine(args)


if __name__ == '__main__':
    main()
