"""Frame-store builder (offline). DELIVERED BY THE STORE BUILD AGENT; this is the contract.

``python -m haltere.obstacles.store build`` calls ``build(args)``; ``repose`` calls ``repose(args)``.

build(args) must:

1. Refuse to start without a flight lock path (thermal.require_flight_lock_path(args.flight_lock)),
   call thermal.limit_threads(2, cv2=True), and run ChunkGuard(lock).before_chunk() before every
   run and at least every 500 decoded frames (ffmpeg pipes simply block while paused).
2. Read configs/obstacles/folds.json (splits.load_folds_config) for every source's environment;
   never re-derive environments. Skip sealed sources unless --sealed-final, which writes them only
   to the 'sealed' store part (StoreWriter(part='sealed', sealed_final=True)).
3. Plan one source per physical flight, best first: geometry PNG (exact), capture dataset,
   good/fair aligned run video (unreliable only with --include-unreliable, grade UNRELIABLE).
   Flights in --secondary also get their next source, flagged SECONDARY_SOURCE. Derived
   data/vision subsets (derived_from) are skipped. Write the plan with StoreWriter.write_plan.
4. Per run: decode new (non-repeated) frames only; drop pause menu, countdown/LIFTOFF text,
   finish/results screens, frames outside the telemetry span, frames after the terminal impact
   and within POST_CONTACT_DROP_S after each contact. Keep every ``stride``-th remaining frame
   plus every frame with tti_s <= PRE_EVENT_DENSE_S and every clean-window frame
   (DENSE_EXTRA when off the stride grid). Resize to 448x252 with cv2.INTER_AREA from the
   1280x720 gameplay crop (x >= 648) or the 640x360 image.
5. Fill INDEX_DTYPE rows exactly as documented in store.py: t_wall (epoch s), t_phase (CSV
   phase clock), t_game, pose at t_wall (TELEMETRY_INTERP from the 100 Hz CSV where one exists;
   geometry PNG = WORKER_INTERP; capture datasets without CSV = SOURCE_COMPENSATED using
   telemetry_age_s, else SOURCE_RAW), vel, omega, launch-relative pos + origin_sim in runs.json,
   video offset = alignment.used_offset_s or offset_after_first_row_s, cue_uv (LOGGED from CSV
   cue columns when present, else NaN; DETECTED may be added later by overlays), luma, flags,
   tti_s/event_id from events.json.
6. Write events.json (store events: {store_event_id, run_id, kind: terminal_impact|contact,
   t_wall, t_phase, drone_pos_w, speed_mps, accel_mps2, source: sidecar|csv_contact}) and
   clean_windows.json ({run_id, source_id, start_phase_s, end_phase_s, criterion, origin}).
7. finalize(); print counts. Pass criteria (plan M1): >= 100k posed frames; Minus Two and Pine
   Valley >= 8k each; every clean-window and pre-impact frame present.

repose(args) must apply timing/refined.json (per-run timing_delta_s from timing.py) by
re-interpolating pos/quat/vel/omega/t_phase/t_game at the corrected times, set
Flag.TIMING_REFINED and timing_delta_s, keep the previous index as index.v<k>.npy, and rewrite
manifest.json (new index_sha256). Labels built on an older index become invalid (LabelSet checks).
"""
from __future__ import annotations


def build(args):
    raise NotImplementedError('store builder: delivered by the store build agent (see module docstring)')


def repose(args):
    raise NotImplementedError('store repose: delivered by the timing build agent (see module docstring)')
