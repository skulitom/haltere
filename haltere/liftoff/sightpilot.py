"""The by-sight pilot "Rabbit+": a virtual lead vehicle flies a smooth course through the gates the detector sees.

``TelemetryPilot.sight == 'rabbit'`` hands ``vision_goal`` to ``SightPilot.goal`` and the yaw stick to
``SightPilot.sight_yaw``; ``'legacy'`` keeps the older remembered-gate pilot.

Perception. Every detector frame is used once, placed with the pose at its screen grab (``TelemetryPilot.pose_at``).
A sighting is a world ray (the bearing is the accurate part) plus a range from the apparent width (the rough part,
bias-corrected by ``range_corr``). Apparent width only measures range per metre of gate width, and no course
announces how wide its arches are, so the pilot measures that too: every track keeps a four-unknown least squares
over all of its sightings - the arch's world point and, as a free unknown, its width - in which the bearings
constrain the point and the apparent widths constrain width x range. The drone's own movement across the line of
sight is what separates the two, and where there has been none the fit says so in its covariance instead of
inventing a number. Those per-arch fits go into one course-wide filter in log space (arches on a course are usually
one size), started at the width the detector's ranges assume with a prior wide enough to be told otherwise; every
newly seen arch starts at it, and when it moves, every estimate in hand slides along its ray with it and has to earn
its parallax again. A sighting's range then updates one of several per-arch Kalman filters with a covariance that
is tight across the ray and loose along it (twice as loose when the arch is cropped by the image edge). Arches are
associated
in metres along and across the ray and in log-range, so two arches lined up on one bearing stay two tracks. A track
confirms after three sightings over 0.2 s at a plausible height (hay bales and shadows on the ground do not); a
confirmed estimate that the camera should see but does not for 2 s is a ghost, and one nothing has seen for
``target_life`` while the detector is alive is dropped even when it is the target. Passed arches absorb their own
sightings from behind. One arch can leave two estimates, so a second pass within ``orphan_dedup_d`` of the one on
the books is that same gate: the pass declared from closer in is kept and the gate is counted once.

Guidance. The target is the nearest confirmed arch ahead. Its approach axis runs along the course (from the last gate,
turned toward the bisector with the next gate when that one is known) and pivots onto the exact bearing close up.
A rabbit, a point with bounded speed, acceleration, curvature and curvature rate, steers onto that axis with
line-of-sight guidance and flies through the gate, on for a few metres and, without a gate, around a search circle;
it never stops, and it waits for the drone (it keeps about 3 m ahead along its own trail). The brain's goal is the
rabbit (its horizontal part clipped to 5 m, its height to +-1.2 m), pulled onto the drone-to-gate bearing on the last
metres. The nose follows the rabbit's heading. Nothing that perception decides can make the goal jump: a new target
or a changed estimate only changes the rabbit's desired curvature and speed.

Speed. The brain holds a sensed cruise speed (about ``flow_ref`` m/s in the game), so the optic-flow/airflow gain is
scheduled as flow_ref / v_nom with a slow integral trim on the measured speed, between ``flow_min`` and ``flow_max``
(the fly command's --flow-gain).

Everything is in the pilot's world frame (x forward at the reset, y left, z up, metres from the reset point);
angles in the log are degrees. ``goal`` never raises: an internal error is reported once and the goal holds the last
carrot computed by a tick without an error while the yaw stick returns to centre; after 20 failed ticks in a row the
pilot starts over and the goal holds the drone's position until a tick succeeds.

Detector liveness. Every detector frame counts, whether it holds an arch or not; with no fresh frame for ``stall_s``
the detector is stalled (a dead capture thread, a hidden window, inference slower than ``stale_s``): an arch that is
not seen then is not a ghost, a remembered target is still flown at ``v_blind``, and without one the rabbit parks and
the drone holds its position (no launch leg, no search) until frames arrive again.
"""
from __future__ import annotations

import math
import sys
import traceback
from bisect import bisect_left
from dataclasses import dataclass, fields
from operator import itemgetter

import numpy as np

CENTRE_UP_M = 1.5                  # the arch's visual centre above the passage point (gates.CENTRE_UP_M)
GATE_WIDTH_NOM = 4.0               # the width a detection's range assumes when it does not say (gates.GATE_WIDTH_M);
                                   # the prior of the pilot's own estimate, never an answer it relies on
DEG = math.pi / 180.0
TWO_PI = 2.0 * math.pi
# true range / range from the apparent width, by the latter: measured on 2036 GateNet (gatenet8) detections of runs 17-21
# matched to the arch within 6 deg of their ray (median per bin; below 4.5 m extrapolated, beyond 30 m assumed)
RANGE_CORR = ((3.5, 0.75), (6.3, 0.81), (9.2, 0.88), (12.7, 0.99), (18.2, 1.08), (23.4, 1.08), (30.0, 1.10), (45.0, 1.0))
# the prototype's table (fitted to the rehearsal's synthetic detector; 5-10 % shorter than GateNet's at 6-18 m)
RANGE_CORR_SPEC = ((3.5, 0.58), (5.6, 0.72), (8.9, 0.84), (12.8, 0.90), (18.5, 1.0), (25.0, 1.15), (35.0, 1.0), (45.0, 0.90))

# numeric log columns (fly --log and the rehearsal), in this order; angles in degrees
LOG_COLUMNS = ['det_u', 'det_v', 'det_t',
               'rb_x', 'rb_y', 'rb_z', 'rb_psi', 'rb_kappa', 'rb_v', 'rb_vnom',
               'look', 'yaw_ref', 'sight_yaw',
               'tgt_id', 'tgt_x', 'tgt_y', 'tgt_z', 'tgt_sdlat', 'tgt_hits', 'tgt_age', 'axis_deg', 'next_id',
               'n_conf', 'n_tent', 'mode', 'n_passes', 'pass_kind',
               'flow_gain', 'rej_elev', 'rej_stale', 'rej_offaxis', 'absorbed', 'low', 'reseeds',
               'ghosts', 'orphans', 'unpasses', 'behind', 'goal_clips', 'sight_errors',
               'gate_w', 'gate_w_sd', 'w_updates', 'det_gap']
PASS_KIND = {'cross': 1, 'travel': 2, 'beside': 3, 'ghost': 4, 'unpass': 5}
MODE_NAMES = {0: 'ground', 1: 'cruise', 2: 'target', 3: 'search', 4: 'hold'}


def wrap(a: float) -> float:
    return (a + math.pi) % TWO_PI - math.pi


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class SightParams:
    """Every tunable of the rabbit pilot (stage A defaults of the merged spec). Speeds m/s, angles degrees."""
    # --- perception
    p_min: float = 0.5
    stale_s: float = 0.35                 # older detections (grab time) are not used
    stall_s: float = 1.0                  # no fresh detector frame (with or without an arch) for this long: stalled
    unseen_live_s: float = 0.2            # an arch in view counts as unseen only while a frame came this recently
    vision_lag: float = 0.0               # s subtracted from det.t before pairing with a pose
    elev_deg: tuple = (-40.0, 35.0)       # world elevation window of a sighting (attitude at the grab)
    range_corr: tuple | None = RANGE_CORR  # range from width -> multiplied by interp(range, table); None = off
    # The detector's range is f * stretch / width_px metres per metre of gate width, and an image cannot say how
    # wide an arch is: every range is proportional to an assumed width, so a course of 1.5 m arches reads 2.67x too
    # far and one of 8 m arches half as near, which desynchronises the whole tracker rather than merely biasing
    # the approach. The pilot therefore estimates the course's own gate width online. Each track keeps a four-
    # unknown least squares over all of its sightings - the arch's world point AND its width - in which the
    # bearings constrain the point and the apparent widths constrain the width times the range; parallax from the
    # drone's own movement is what separates them, and where there is none the fit says so in its own covariance
    # rather than inventing a number. Those per-arch fits go into one course-wide filter in log space.
    width_est: bool = True                # estimate the gate width online (False: keep the detector's nominal)
    width_prior_ln: float = 0.55          # prior std of ln(width) about the nominal: +-1 sigma is 0.58x-1.7x,
                                          # +-2 sigma 0.33x-3.0x, which spans the arches a race course can hold
    width_meas_ln: float = 0.25           # std of one arch's width fit beyond what that fit reports of itself: the
                                          # bearing scatter leaves a residual pull towards the drone that binning
                                          # reduces but does not remove, and an early fit understates it badly
    width_floor_ln: float = 0.18          # the posterior std never falls below this. Successive fits of one arch
                                          # share their error, so the filter must not treat forty of them as forty
                                          # independent looks: with a 0.10 floor one early, biased fit collapsed
                                          # the spread and an 8 m course was still being flown as a 7.1 m one at
                                          # the gate; at 0.18 the later, firmer fits can still move it (measured
                                          # on a straight approach, six seeds: a 4 m arch read 3.87-3.97, an 8 m
                                          # one 7.79-7.96, a 1.5 m one 1.43-1.50 on five seeds and 2.29 on the
                                          # sixth, which got one weak look and then accepted no others)
    width_reset_ln: float = 0.02          # a width revision this large makes every track earn its parallax again:
                                          # the parallax that had shrunk its along-ray variance was measured
                                          # against ranges on the old scale, so it does not license the new one
    width_span: tuple = (0.6, 20.0)       # m: the estimate is never taken outside this
    width_min_s: float = 0.25             # s between width updates from one track (15 Hz of them are one look)
    width_bin: int = 10                   # sightings averaged into one look before they enter a track's width fit:
                                          # per-frame centre scatter biases that fit towards the drone, and the
                                          # bias goes as the square of the scatter, so averaging is what kills it
    # two looks are the fewest at which the fit exists at all, and tri_sigma_frac is what actually decides whether
    # to believe it. A higher floor only makes the width unmeasurable on the courses that need it most: an arch
    # the tracker holds for 20 sightings before it is passed, ghosted or absorbed - which is most of them on a
    # course of 8 m arches, where the pilot thinks every gate is half as far as it is and cycles through them -
    # never produced a fit at 4. Measured on Straw Bale's layout with the rehearsal's detector, four seeds, 4 -> 2:
    # a 4 m course stays 7/7 and reads 3.5-4.3 m, an 8 m one goes 6,4,6,5 -> 7,4,7,6 gates and 4.2-6.6 -> 5.8-9.5 m,
    # a 1.5 m one is unchanged within its seed scatter.
    tri_min_rays: int = 2                 # such looks a track needs before its own width fit is used at all ...
    tri_sigma_frac: float = 0.5           # ... and the relative std it must have got down to (relative to the width
                                          # the pilot currently assumes, which is the scale its noise model was
                                          # built at - see _width_update). The fit's own std weights it in the
                                          # filter, so this only says when it is worth listening to: measured on a
                                          # straight approach with 5 px of centre scatter, 0.5 and 0.25 end within
                                          # 2 % of each other, and 0.5 gets there 2-5 m earlier on a narrow course
    range_corr_width: float = 4.0         # the arch width range_corr was fitted at: the table is really a function
                                          # of apparent pixel width, which is only a range once the width is known
    unit_range_clip: tuple = (0.25, 11.25)  # range per metre of gate width a sighting may claim: the old 1 m and
                                          # 45 m divided by the 4 m nominal. f / width_px, so a detector property
    range_common: float = 0.08            # the share of a width-range that successive sightings get wrong together
                                          # (the along-ray variance may only shrink below it with parallax)
    sig_along: tuple = (0.22, 0.3)        # along-ray std = a * range + b
    sig_cross: tuple = (0.05, 0.2)        # across-ray std = a * range + b
    crop_px: float = 3.0                  # the arch's box within this of the image edge = cropped
    crop_mult: float = 2.0                # along-ray std multiplier when cropped
    # off the optical axis the detector is a different instrument: within 20 px of the centre 89 % of its boxes are
    # the target arch, beyond 60 px (about 31 deg) only 28 % (task e). The angle is the box's image radius,
    # atan(hypot(u - cx, v - cy) / f), and the camera's 30 deg uptilt puts an arch level with the drone dead ahead at
    # 30 deg of it: over the six game flights the median sighting is 24-26 deg and the 95th percentile 32-37, so a
    # 35 deg cut slices the body of that distribution -- 23 % of every sighting made in search mode, and in replay
    # of w16 the pilot then sat on an unseen estimate for 45 s. At 40 deg that is 11 %, at 45 deg 6 %; both fix w16,
    # and 40 is the one the rehearsal likes (the synthetic detector has no off-axis degradation, so every sighting
    # let in there is a good one and the pilot commits to the turn out of gate 1 harder the wider this is opened).
    offaxis_max: float = 40.0             # a detection further than this off the optical axis is not used (0 = off);
                                          # 40 deg = 84 px of the 184 to the corner: the tail, not the bulk
    offaxis_sig_deg: float = 15.0         # its covariance is scaled by 1 + off_axis_deg / this (0 = off)
    q: float = 0.02                       # m^2/s process noise per track
    perp_gate: tuple = (1.5, 0.12)        # association: ray passes within max(a, b * along) of the estimate
    log_gate: float = 0.45                # |ln(range / along)| for confirmed tracks
    log_gate_tent: float = 0.60           # ... for tentative tracks
    confirm_hits: int = 3
    confirm_span: float = 0.2
    z_plaus_sigma: float = 0.7            # height plausibility applies once sigma_z < this ...
    low_sigma: float = 0.5                # ... and a confirmed track below the floor (see z_drop_max) is deleted
    no_update_d: float = 2.5              # the target is not updated within this horizontal distance ...
    no_update_crop_rng: float = 4.0       # ... nor when cropped and closer than this
    tent_life: float = 2.0
    conf_life: float = 30.0
    # the target is exempt from conf_life, and the ghost rule cannot retire it beyond ghost_range or while
    # ghost_keep_d holds it: nothing then retires an estimate that is simply never seen again (in replay the pilot
    # flew at one for 45 s). Time while the detector is stalled does not count: a remembered target is flown blind.
    target_life: float = 20.0             # a confirmed target unseen this long, the detector alive, is dropped
                                          # (0 = off; the longest honest gap over the six game flights is 15 s)
    passed_life: float = 90.0
    ghost_s: float = 2.0
    ghost_range: tuple = (6.0, 30.0)
    # the detector reports at most ONE arch per frame, so an arch in view is "unseen" on every frame that showed the
    # other one: the gate being flown at collected 2 s of it and was deleted 39 times in the game (task a)
    ghost_keep_d: float = 12.0            # never ghost the current target inside this horizontal range (0 = legacy)
    ghost_evidence: str = 'other'         # what a fresh frame must show to charge an in-view estimate: 'any' (legacy,
                                          # every frame), 'other' (its sighting went to another track off this one's
                                          # bearing), 'empty_or_other' (that, or the frame held no arch at all)
    ghost_other_deg: float = 4.0          # "off this one's bearing" = the rays differ by more than this
    ghost_s_hits: int = 20                # a confirmed track earns a longer ghost timer per this many sightings ...
    ghost_s_max: float = 5.0              # ... up to this (one with 88 hits died 2 s before its gate; 0 = legacy)
    orphan_d: float = 8.0                 # a dropped confirmed track that came this close was a gate, not a phantom:
    orphan_a: float = 1.5                 # ... once the drone is this far past it along its line ...
    orphan_lat: float = 5.0               # ... and within this of the line, register a 'travel' pass (0 = legacy)
    orphan_dedup_d: float = 15.0          # any pass within this of the one on the books is the same arch under a
                                          # second id, whichever path booked it: the nearer of the two is kept and
                                          # the gate is counted once (real gates on this course are 24-35 m apart,
                                          # and the measured near-duplicates sat at 9.6-14.6 m)
    orphan_life: float = 12.0             # s an orphan waits to be flown past before it is forgotten
    orphan_target_only: bool = True       # only the gate that was being flown at leaves an orphan (a track dropped
                                          # beside the course is not a gate the drone has just been through)
    merge_d: float = 1.5
    merge_lat: tuple = (1.5, 0.06)
    merge_log: float = 0.7
    gate_track_sigma: bool = True         # widen the association gate by the track's own across-ray uncertainty
    stale_penalty: float = 2.0            # selection score per second a track has gone unseen beyond stale_grace (cap 8)
    stale_grace: float = 1.0
    # --- target and approach axis
    eligible_d: float = 1.0
    eligible_bearing: float = 110.0
    switch_s: float = 0.3
    lock_s: tuple = (0.3, 0.6)            # on the latest ray up to a, blended to the filter until b
    bisector_cap: float = 45.0
    next_min_hits: int = 10               # sightings a confirmed track needs before it can set the approach bisector
                                          # (a briefly seen track swung gate 4's axis by 40 deg in the game)
    next_min_sep: float = 18.0            # ... and it must be this far from the target: real gates on this course are
                                          # 24-35 m apart, so anything nearer is a fragment of the target (task d)
    # a displaced sighting spawns a second confirmed track 5-10 m beyond the target along the same bearing; it is the
    # same arch badly ranged, so it must not be selected, act as the next gate, or survive (task d)
    frag_gap: float = 15.0                # a confirmed track this far ahead of the target along its ray is its
                                          # fragment (0 = off) ...
    frag_gap_frac: float = 0.6            # ... or this much of the course's own measured gate spacing, once known
    frag_lat_ahead: float = 4.0           # ... if it is also this close to the ray
    frag_absorb: bool = True              # fold such a fragment's sightings into the target instead of keeping it
    # these two retire the last pass as the course reference. Note that neither outlasts a leg of this course
    # (24-35 m, flown in 10-17 s): the reference expires BEFORE the gate on 40-60 % of the rows 3-12 m out, and the
    # axis there falls back to the drone's own chord, which also switches the turn rotation off (turn_gate rows drop
    # 30-73 %). Raising them (30 s / 50 m) does put the reference back, but what it buys is small and not
    # consistent -- replayed, the mean approach-axis error against the arch's own heading 3-12 m out goes 17.4 ->
    # 11.3 deg on w21's second lap and 12.8 -> 12.0 on w20's first, against 7.7 -> 9.4 on w20's second and no
    # change on w19 -- while it turns the bisector rotation back on over the 35 m leg into gate 1, and the drone
    # then flies an angled approach and a turning exit: in the rehearsal that peaks at 20.4 m/s^2 a metre past the
    # arch (the rest of the flight is 11.8 at the 99th percentile) and `liftoff score` calls gate 1 a hit on three
    # seeds of four. Not taken for that reason; to try it: --sight-set course_age_s=30 --sight-set course_back_m=50.
    course_age_s: float = 10.0            # a pass older than this no longer sets the approach course ...
    course_back_m: float = 25.0           # ... nor one the rabbit has run this far past (task b)
    course_win: tuple = (3.0, 5.0)        # then the course is the drone's own chord over this window, s ...
    course_min_m: float = 2.0             # ... if it moved at least this far in it (0 = never use it)
    turn_rot_min: float = 12.0            # no next gate: rotate the axis by min(0.5 |alpha|, turn_rot_max) beyond this
    turn_rot_max: float = 20.0
    turn_gate_alpha: float = 25.0
    turn_gate_bisector: float = 15.0
    pivot_max: float = 70.0               # axis may pivot this far from the ray at pivot_d[1], none at pivot_d[0]
    pivot_d: tuple = (3.0, 18.0)          # (the pivot onto the ray now starts further out: it undoes range error)
    axis_sigma_cap: float = 1.0           # m: cap |axis - bearing| at atan(this / along-range sigma), since an axis
                                          # tilted off the ray turns range error into lateral error (0 = off)
    pivot_rate: float = 40.0              # deg/s
    pivot_tau: float = 0.4
    pivot_freeze_d: float = 4.0
    turn_hints: tuple | None = None       # optional: heading change (deg, + left) at each gate in course order
    # --- pass
    airborne_guard: float = 2.0
    fresh_hits: int = 5
    fresh_age: float = 2.5
    fresh_rng: float = 10.0
    cross_before: float = -2.0
    cross_after: float = 0.3
    cross_lat: float = 4.0
    travel_age: float = 0.4
    travel_rng: float = 7.0
    travel_frac: float = 0.8
    beside_a: float = 3.0
    beside_d: float = 8.0
    beside_bearing: float = 100.0
    seen_ahead_age: float = 0.25
    seen_ahead_along: float = 3.0
    unpass_s: float = 1.5
    behind_s: float = 20.0
    pass_absorb_d: float = 4.0            # unpassed tracks this close to a gate being passed are its fragments
    frag_ahead: float = 10.0              # a pass is a short fragment when a fresh confirmed track lies this far ahead ...
    frag_lat: float = 3.0                 # ... within this of the approach line ...
    frag_fresh_s: float = 0.5             # ... seen this recently (gates on a course are further apart)
    # --- rabbit
    v_launch: float = 3.5                 # the launch leg was the slowest part of every game lap
    v_cruise: float = 2.5
    v_exit: float | None = None           # None = v_cruise
    v_gate: float = 3.0
    v_gate_turn: float = 2.5
    v_unsure: float = 2.5
    v_blind: float = 2.0
    v_search: float = 2.0
    # a saturated curvature collapsed the rabbit to sqrt(a_lat / kappa_max) = 2.19 m/s for whole legs: 2.58 now (task g)
    a_lat: float = 2.0
    a_acc: float = 0.8
    a_brk: float = 1.2
    t_lag: float = 1.2
    kappa_max: float = 0.30
    sharp: float = 0.06                   # 1/m^2
    k_head: float = 1.5                   # 1/s
    lead: float = 3.0                     # m at the gate; + lead_open in the open
    lead_open: float = 0.5
    lead_band: float = 1.5
    lead_fmax: float = 1.3
    rabbit_stop: float = 4.5              # the rabbit waits when this far from the drone
    reseed_d: float = 6.0
    reseed_s: float = 1.0
    bump_tau: float = 0.4
    bump_vmax: float = 2.5                # m/s: the reseed offset fades no faster than this
    d_on: float = 8.0                     # m flown straight on after a pass (4 m left the search circling too early)
    launch_t: float = 6.0
    search_radius: float = 8.0
    search_radius_wide: float = 12.0      # after a full circle
    search_leash: float = 20.0
    search_side: float = 1.0              # +1 left, -1 right (default side before the course has turned)
    tent_side_hits: int = 2               # a tentative track steers the search side after this many sightings
    snap_start: float = 12.0              # the terminal snap cancels range error: start it before the last 9 m ...
    snap_max: float = 2.0                 # ... and let it move the goal as far as the misses it undoes
    snap_rate: float = 0.8
    goal_max: float = 5.0
    goal_z: float = 1.2
    # --- altitude
    z_start: float = 1.5
    z_pass0: float = 1.2
    z_aim: float = 0.0                    # 0.3 in the spec: in the game the drone crossed gate 4 1.2 m high, into its top
    z_min: float = 1.2
    up_bias: float = 0.0                  # + min(up_bias, up_bias * sigma_z)
    climb_front: float = 0.5
    vz_frac: float = 0.35
    vz_min: float = 0.4
    vz_max: float = 1.0
    vz_max_down: float = 2.0              # a quad drops more easily than it climbs, and one limit for both
                                          # left a 10 m step down unflyable: the leg is over before the
                                          # reference arrives, so the drone reaches the gate still high
    az_max: float = 0.8
    grade_max: float = 0.35
    grade_len: float = 15.0
    # the estimate's height runs 1:1 into the altitude reference and from there into the drone: cap it at the grade
    # line z_aim_last + grade_last * min(s - last_pass.s, grade_len), so one high estimate cannot lift the approach.
    # Measured on the six game flights: a +2.5 m ceiling clips the real 5 m climbs into gates 5 and 6 by 1.2-1.7 m
    # (a crash), so the net is set where it never fights a climb and still catches a gross lift. Only the ceiling
    # rides the grade line: a floor that climbs with it would push the aim UP (+1.9 m on w21), which is the failure
    # this was written against, so the floor stays on the last passage height, as it was.
    z_window: tuple = (3.0, 6.0)          # target height within [min(z_ref, z_aim_last) - a, z_ref + b]
    z_window_ref: bool = True             # False: the old window around z_aim_last with no grade line
    # The Straw Bale course only ever climbs, and two rules quietly assumed every course does. A gate BELOW the
    # last passage height was refused confirmation and then deleted outright as ground clutter, and the aim could
    # not follow it down more than z_window[0] per gate. On the odd-course bench that cost every gate of a
    # descending course (low counter 21 against 0 at home) and overshot a 12 m -> 2 m step by 10.45 m, into the
    # top bar. A gate may now sit z_drop_max below the last passage point, and the aim may descend at the rate the
    # drone can actually fly - descend_slope metres down per metre along, which is what limits it in the air.
    z_drop_max: float = 8.0               # how far below the last passage height a gate may plausibly sit
    descend_slope: float = 0.35           # metres of descent per metre flown towards the gate
    # --- yaw
    yaw_rate: float = 2.3                 # rad/s per unit yaw stick (Liftoff 2.3; the simulator's rates 3.8)
    yaw_gain: float = 1.5                 # 1/s (2.5 limit-cycled against the simulator's lagging yaw-rate response)
    yaw_lead: float = 0.15                # s: phase lead from the measured heading rate, e + lead * (r_ref - psi_dot)
    yaw_rate_tau: float = 0.05            # s low-pass on the measured heading rate
    yaw_db: float = 2.0
    yaw_rate_cap: float = 1.0             # rad/s
    yaw_max: float = 0.35
    yaw_slew: float = 3.0                 # stick per s
    # detector quality is governed by how far off the optical axis the arch is, and extra yaw is free (the brain flies
    # a body-frame goal): keep the nose on the target instead of letting it sit 35 deg out of frame (task e)
    look_free: float = 10.0               # bearing beyond which the nose starts following the target (legacy 35)
    look_max: float = 45.0                # ... up to this much lead (legacy 25)
    look_tau: float = 0.3
    look_kappa: float = 0.20              # the look is given up as the rabbit's curvature approaches this (legacy .04)
    sweep: float = 35.0
    sweep_period: float = 5.0
    sweep_ramp: float = 1.5
    # --- speed sense
    flow_ref: float = 2.45                # m/s the brain holds as sensed cruise
    flow_min: float | None = None         # None: min(0.7, 0.9 flow_ref / max(v_cruise, v_gate))
    flow_max: float = 1.0                 # the fly command's --flow-gain
    flow_ki: float = 0.15
    flow_trim_range: tuple = (0.5, 1.3)
    flow_tau: float = 1.0
    flow_mode: str = 'global'             # 'along': scale only the speed along the rabbit's heading
    flow_alt: str = 'start'               # 'ground': altitude sense above the ground under the rabbit's reference

    def flow_bounds(self) -> tuple[float, float]:
        top = max(self.v_cruise, self.v_gate, self.flow_ref)
        lo = min(0.7, 0.9 * self.flow_ref / top) if self.flow_min is None else self.flow_min
        return min(lo, self.flow_max), self.flow_max

    def validate(self) -> None:
        """Raise ValueError on values the pilot cannot fly with: a wrong type, a zero it divides by, a bad table."""
        bad = []

        def num(x) -> bool:
            return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

        optional = {'range_corr', 'turn_hints', 'flow_min', 'v_exit'}
        choices = {'flow_mode': ('global', 'along'), 'flow_alt': ('start', 'ground'),
                   'ghost_evidence': ('any', 'other', 'empty_or_other')}
        for f in fields(self):
            v, d = getattr(self, f.name), f.default
            if f.name in optional:
                continue
            if isinstance(d, bool):
                ok = isinstance(v, bool)
            elif isinstance(d, (int, float)):
                ok = num(v)
            elif isinstance(d, tuple):
                ok = isinstance(v, (tuple, list)) and len(v) == len(d) and all(num(x) for x in v)
            elif isinstance(d, str):
                ok = v in choices.get(f.name, (v,))
            else:
                ok = True
            if not ok:
                bad.append(f'{f.name}={v!r}')
        if bad:
            raise ValueError('wrong type or value: ' + ', '.join(bad))
        for k in ('stale_s', 'stall_s', 'yaw_rate', 'yaw_db', 'yaw_rate_tau', 'yaw_slew', 'yaw_max', 'flow_ref',
                  'flow_max', 'flow_tau', 'look_tau', 'look_kappa', 'pivot_tau', 'bump_tau', 'bump_vmax', 'sweep_period',
                  'sweep_ramp', 'lead_band', 'log_gate', 'log_gate_tent', 'search_radius', 'search_radius_wide', 'a_lat',
                  'a_acc', 'a_brk', 'sharp', 'kappa_max', 'goal_max', 'goal_z', 'crop_mult', 'v_cruise', 'v_gate',
                  'v_gate_turn', 'ghost_s', 'orphan_life', 'course_age_s', 'grade_len', 'width_meas_ln',
                  'width_floor_ln', 'tri_sigma_frac', 'range_corr_width', 'range_common'):
            if not getattr(self, k) > 0:
                bad.append(f'{k}={getattr(self, k)!r} (must be > 0)')
        if self.tri_min_rays < 2:
            bad.append(f'tri_min_rays={self.tri_min_rays!r} (two rays are the fewest that triangulate)')
        if self.width_bin < 1:
            bad.append(f'width_bin={self.width_bin!r} (must be >= 1)')
        if not self.width_prior_ln >= self.width_floor_ln > 0:
            bad.append(f'width_prior_ln={self.width_prior_ln!r} (>= width_floor_ln={self.width_floor_ln!r} > 0)')
        for k in ('ghost_keep_d', 'ghost_other_deg', 'ghost_s_max', 'orphan_d', 'orphan_lat', 'orphan_dedup_d',
                  'offaxis_max', 'offaxis_sig_deg', 'frag_gap', 'frag_gap_frac', 'frag_lat_ahead', 'next_min_sep',
                  'course_back_m', 'course_min_m', 'axis_sigma_cap', 'look_free', 'look_max', 'target_life'):
            if getattr(self, k) < 0:
                bad.append(f'{k}={getattr(self, k)!r} (must be >= 0)')
        if not 0 < self.course_win[0] <= self.course_win[1]:
            bad.append(f'course_win={self.course_win!r} (0 < a <= b)')
        for k in ('sig_along', 'sig_cross'):
            a, b = getattr(self, k)
            if a < 0 or b <= 0:
                bad.append(f'{k}={getattr(self, k)!r} (a >= 0, b > 0)')
        if self.perp_gate[0] <= 0:
            bad.append(f'perp_gate={self.perp_gate!r} (a > 0)')
        for k in ('elev_deg', 'ghost_range', 'flow_trim_range', 'width_span', 'unit_range_clip'):
            lo, hi = getattr(self, k)
            if not lo < hi:
                bad.append(f'{k}={getattr(self, k)!r} (low < high)')
        if self.width_span[0] <= 0 or self.unit_range_clip[0] <= 0:
            bad.append(f'width_span={self.width_span!r} / unit_range_clip={self.unit_range_clip!r} (low > 0)')
        if self.flow_trim_range[0] <= 0:
            bad.append(f'flow_trim_range={self.flow_trim_range!r} (low > 0)')
        for k in ('flow_min', 'v_exit'):
            v = getattr(self, k)
            if v is not None and not (num(v) and v > 0):
                bad.append(f'{k}={v!r} (None or > 0)')
        rc = self.range_corr
        if rc is not None:
            try:
                d, kk = zip(*rc)
                ok = (len(d) >= 2 and all(num(x) for x in d + kk) and all(x > 0 for x in kk) and d[0] > 0
                      and all(b > a for a, b in zip(d, d[1:])))
            except (TypeError, ValueError):
                ok = False
            if not ok:
                bad.append(f'range_corr={rc!r} (pairs (distance, factor), distances increasing, factors > 0)')
        if self.turn_hints is not None and not (isinstance(self.turn_hints, (tuple, list))
                                                and all(num(x) for x in self.turn_hints)):
            bad.append(f'turn_hints={self.turn_hints!r}')
        if bad:
            raise ValueError(', '.join(bad))

    def describe(self) -> str:
        lo, hi = self.flow_bounds()
        return (f'cruise {self.v_cruise} m/s, gate {self.v_gate} (turn {self.v_gate_turn}), lead {self.lead} m, '
                f'a_lat {self.a_lat}, a_brk {self.a_brk}, yaw {self.yaw_rate} rad/s per stick gain {self.yaw_gain} '
                f'lead {self.yaw_lead} s max {self.yaw_max}, flow {lo:.2f}-{hi:.2f} (ref {self.flow_ref}, {self.flow_mode}, altitude '
                f'{self.flow_alt}), elevation {self.elev_deg}, vision lag {self.vision_lag} s, z aim {self.z_aim}, '
                + (f'arch width estimated from parallax (prior +-{100 * self.width_prior_ln:.0f}%), '
                   if self.width_est else 'arch width as the detector assumes it, ') +
                f'snap from {self.snap_start} m, search {"left" if self.search_side > 0 else "right"} '
                f'r {self.search_radius} m' + (f', turn hints {self.turn_hints}' if self.turn_hints else ''))


class Track:
    """One arch: a world Kalman estimate of its visual centre and what the pass logic needs.

    Alongside the filter it keeps the normal equations of a four-unknown least squares over ALL of its sightings:
    the arch's world point and, as a free unknown, how wide the arch is. Each sighting contributes its bearing
    (residual across the ray, the accurate part) and its apparent width (residual along the ray, which is the
    width times the sighting's unit range). Where the rays are parallel the point and the width trade off exactly
    and the fit says so in its own covariance; where the drone has moved across the line of sight they separate,
    and the width falls out of the geometry alone. Nothing here carries a prior, so what it returns is a
    measurement of the arch, not a restatement of what the pilot already assumed."""
    __slots__ = ('id', 'm', 'P', 'r0', 'dir_first', 'hits', 't_first', 't_last', 'p_last', 'r_last', 'rng_last_h',
                 'last_along', 'confirmed', 'passed', 't_passed', 'n_pass', 'min_a', 'unseen_in_view', 'min_d',
                 'N4', 'g4', 'rays', 't_width', 'bin')

    def __init__(self, tid: int, now: float, z: np.ndarray, Rm: np.ndarray, r: np.ndarray, pg: np.ndarray, rho: float):
        self.id = tid
        self.m = z.copy()
        self.P = Rm.copy()
        self.r0 = r.copy()
        self.N4 = np.zeros((4, 4))
        self.g4 = np.zeros(4)
        self.bin = [np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0, 0]     # sum of p, of r, of ur, of sa, of sc, count
        self.rays = 0                     # averaged sightings ("looks") that have gone into the fit
        self.t_width = -math.inf          # when this track last gave the width estimate a measurement
        dx, dy = float(z[0] - pg[0]), float(z[1] - pg[1])
        n = math.hypot(dx, dy)
        self.dir_first = (dx / n, dy / n) if n > 1e-6 else (1.0, 0.0)
        self.hits = 1
        self.t_first = self.t_last = now
        self.p_last = pg.copy()
        self.r_last = r.copy()
        self.rng_last_h = n
        self.last_along = rho
        self.confirmed = False
        self.passed = False
        self.t_passed = -math.inf
        self.n_pass = (1.0, 0.0)
        self.min_a = math.inf
        self.min_d = n                    # closest the drone has been, horizontally: a dropped track that came close
        self.unseen_in_view = 0.0         # was a gate the drone is about to fly past, not a phantom

    def dist_h(self, p) -> float:
        return math.hypot(float(self.m[0] - p[0]), float(self.m[1] - p[1]))

    def add_sight(self, pg: np.ndarray, r: np.ndarray, ur: float, sa: float, sc: float, group: int = 1) -> None:
        """One more sighting for the joint point-and-width fit: bearing ``r`` from ``pg`` with unit range ``ur``
        (range per metre of arch width), across-ray std ``sc`` and along-ray std ``sa``, both in metres.

        Sightings are averaged ``group`` at a time before they enter the fit. Fifteen frames a second from a drone
        that has moved a metre are one look at the arch measured fifteen times, not fifteen looks: they carry no
        extra parallax, and folding them in one at a time would let the detector's per-frame centre scatter bend
        the fit. Scatter in the bearings always bends it the same way - towards the drone, a gate read smaller than
        it is - because the residual it leaves is quadratic, so the cure is to average it down before it enters."""
        b = self.bin
        b[0] += pg
        b[1] += r
        b[2] += ur
        b[3] += sa
        b[4] += sc
        b[5] += 1
        if b[5] < max(1, int(group)):
            return
        n = float(b[5])
        pg = b[0] / n
        r = b[1] / n
        nr = float(np.linalg.norm(r))
        ur, sa, sc = b[2] / n, b[3] / n / math.sqrt(n), b[4] / n / math.sqrt(n)
        self.bin = [np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0, 0]
        if nr < 1e-6:
            return
        r = r / nr
        self._normal(pg, r, ur, sa, sc)

    def _normal(self, pg: np.ndarray, r: np.ndarray, ur: float, sa: float, sc: float) -> None:
        wc = 1.0 / max(sc, 1e-3) ** 2
        wa = 1.0 / max(sa, 1e-3) ** 2
        M = np.eye(3) - np.outer(r, r)                 # across the ray: M (x - pg) = 0, no width in it
        self.N4[:3, :3] += wc * M
        self.g4[:3] += wc * (M @ pg)
        j = np.array([r[0], r[1], r[2], -ur])          # along it: r.x - width * ur = r.pg
        self.N4 += wa * np.outer(j, j)
        self.g4 += wa * j * float(r @ pg)
        self.rays += 1

    def fit_width(self) -> tuple[float, float, np.ndarray] | None:
        """(arch width in metres, its std, the arch's world point) from every sighting so far, or None while the
        rays are still too nearly parallel for the two to be told apart."""
        if self.rays < 2:
            return None
        try:
            ev = np.linalg.eigvalsh(self.N4)
            if not ev[0] > max(1e-12, 1e-11 * float(ev[-1])):
                return None
            C = np.linalg.inv(self.N4)
            u = C @ self.g4
        except np.linalg.LinAlgError:
            return None
        w, var = float(u[3]), float(C[3, 3])
        if not (math.isfinite(w) and math.isfinite(var) and var >= 0.0):
            return None
        return w, math.sqrt(var), u[:3]


class SightPilot:
    """State and per-tick logic of the rabbit pilot. ``host`` is the TelemetryPilot (clock, last_R, last_vel, vision,
    pose_at, flow_gain; vision_gate_w / _passed / vision_passed_t / vision_status are kept up to date for logs)."""

    def __init__(self, host, params: SightParams | None = None):
        self.host = host
        self.params = params or SightParams()
        self.errors = 0
        self._err_logged = False
        self._consec_err = 0
        self.t_frame = -math.inf               # when the last fresh detector frame (arch or not) was consumed
        self.stalled = True
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        P = self.params
        self._init = False
        self.t_prev = None
        self.t_air = None
        self.grounded_for = 0.0
        self.det_seen = -1
        self.sight_yaw = 0.0
        self.flow_f = P.flow_max
        self.flow_trim = 1.0
        for k in ('rej_elev', 'rej_stale', 'rej_offaxis', 'absorbed', 'behind', 'ghosts', 'orphans_registered', 'low',
                  'unpasses', 'reseeds', 'goal_clips', 'dup_passes', 'stale_targets', 'w_updates', 'w_rejects'):
            setattr(self, k, 0)
        # how wide the arches of this course are, in log metres. It starts at whatever width the detector's ranges
        # assume (the first detection says so) and is a fact about the course, not about an attempt: a crash and a
        # fresh start throw the tracks away but not what the pilot has learned about the arches it is flying at.
        self.w_ln = math.log(GATE_WIDTH_NOM)
        self.w_var = P.width_prior_ln ** 2
        self.w_drift = 0.0                    # net revision since the tracks last had to earn their parallax
        self.w_seen = False                   # the nominal has been taken from a real detection
        self._next_id = 0
        self.pass_kind = 0
        self.carrot = None                    # the goal's world point of the last tick without an error
        self._carrot_new = None
        self._hold = None                     # world point held after repeated errors, until a tick succeeds
        self._last_rel = None
        self._full_init(np.zeros(3), 0.0, 0.0)
        self._init = False

    def _full_init(self, p: np.ndarray, psi_n: float, now: float) -> None:
        P = self.params
        self._init = True
        self.tracks: list[Track] = []
        self.target: Track | None = None
        self.pending: Track | None = None
        self.pending_since = 0.0
        self.next: Track | None = None
        self.n_ang: float | None = None
        self.turn_gate = False
        self.sd_lat = 0.0
        self.g_s = None
        self.d_gate = math.inf
        self.psi_launch = psi_n
        self.launch_p = (float(p[0]), float(p[1]))
        self._p = np.asarray(p, dtype=np.float64).copy()
        self.c = [float(p[0]) + 2.5 * math.cos(psi_n), float(p[1]) + 2.5 * math.sin(psi_n)]
        self.psi = psi_n
        self.kappa = 0.0
        self.v = 0.0
        self.v_nom = 0.0
        self.v_des = 0.0
        self.s = 0.0
        self.z_c = P.z_start
        self.vz_c = 0.0
        self.trail = [(0.0, self.c[0], self.c[1])]
        self.s_d_prev = -math.inf
        self.far_since = None
        self.o_snap = 0.0
        self.o_bump = np.zeros(3)
        self.last_pass = None                 # dict(t, m, n, s, kind)
        self.pass_backup = None
        self.n_passes = 0
        self.pass_xy: list = []               # where each pass was: the course's own gate spacing (fragment rule)
        self.spacing = None                   # median distance between successive passes, m
        self.orphans: list = []               # confirmed tracks dropped close in, still waiting to be flown past
        self.p_hist: list = []                # (t, x, y) of the drone, for its own smoothed course
        self.f_track = None                   # the track the last fresh detector frame's sighting went to ...
        self.f_dir = None                     # ... and that sighting's world ray
        self.z_pass_last = P.z_pass0
        self.z_aim_last = P.z_start
        self.grade_last = 0.0
        self.side = 1.0 if P.search_side >= 0 else -1.0
        self.mode = 0
        self.mode_since = now
        self.search_turned = 0.0
        self.last_tent_b = 0.0
        self.last_tent_t = -math.inf
        self.look = 0.0
        self.look_prev = 0.0
        self.psi_dot = 0.0
        self.yaw_ref = psi_n
        self.lead_now = 0.0
        self.t_launch = now                   # the launch leg runs launch_t from here (pushed on while holding)
        self.t_stall_end = now                # when the detector last came back: target staleness counts from here

    @property
    def ready(self) -> bool:
        """Initialised: the rabbit's heading and reference height mean something (for the optional flow modes)."""
        return self._init

    # ------------------------------------------------------------------ public tick
    def goal(self, pos_w) -> np.ndarray:
        """Body-frame goal vector for the brain. Never raises."""
        p = np.asarray(pos_w, dtype=np.float64)
        try:
            self._carrot_new = None
            rel_b = self._goal(p)
            c = self._carrot_new
            if not (np.all(np.isfinite(rel_b)) and c is not None and np.all(np.isfinite(c))):
                raise FloatingPointError(f'non-finite goal {rel_b} / carrot {self._carrot_new}')
            self.carrot = self._carrot_new                    # committed only by a tick that went all the way through
            self._consec_err = 0
            self._hold = None
            return rel_b
        except Exception:                                    # noqa: BLE001 - the 100 Hz loop must go on
            self.errors += 1
            self._consec_err += 1
            if not self._err_logged:
                self._err_logged = True
                print('SIGHT PILOT ERROR (reported once; holding the last carrot, then the drone position):\n'
                      + traceback.format_exc(), file=sys.stderr, flush=True)
            if self._consec_err >= 20:
                # start over on the next tick, and meanwhile hold where the drone is (never a new point ahead of it)
                if self._hold is None and np.all(np.isfinite(p)):
                    self._hold = np.array([p[0], p[1], clip(float(p[2]), self.params.z_min, 30.0)])
                self._init = False
            return self._safe_goal(p)

    def _safe_goal(self, p: np.ndarray) -> np.ndarray:
        try:
            R = self.host.last_R
            P = self.params
            point = self._hold if self._hold is not None else self.carrot
            if point is not None and np.all(np.isfinite(point)):
                rel = np.asarray(point, dtype=np.float64) - p
            else:
                rel = np.array([0.0, 0.0, P.z_start - float(p[2])])
            h = math.hypot(rel[0], rel[1])
            if h > P.goal_max:
                rel[:2] *= P.goal_max / h
            rel[2] = clip(float(rel[2]), -P.goal_z, P.goal_z)
            self.sight_yaw = clip(0.0, self.sight_yaw - P.yaw_slew * 0.01, self.sight_yaw + P.yaw_slew * 0.01)
            out = R.T @ rel
            return out if np.all(np.isfinite(out)) else np.zeros(3)
        except Exception:                                    # noqa: BLE001
            self.sight_yaw = 0.0
            return np.zeros(3)

    # ------------------------------------------------------------------ the tick
    def _goal(self, p: np.ndarray) -> np.ndarray:
        P = self.params
        h = self.host
        now = float(h.clock())
        self._p = p
        R = h.last_R
        vel = np.asarray(h.last_vel, dtype=np.float64)
        psi_n = math.atan2(R[1, 0], R[0, 0])
        if self.t_prev is None:
            self.t_prev = now
        dt = clip(now - self.t_prev, 0.0, 0.05)
        self.t_prev = now
        self.pass_kind = 0
        on_ground = p[2] < 0.3 and float(vel @ vel) < 0.25
        self.grounded_for = self.grounded_for + dt if on_ground else 0.0
        if not self._init or (on_ground and (self.t_air is None or self.grounded_for > 1.5)):
            self._full_init(p, psi_n, now)
            self.t_air = None
            if self._hold is not None:            # restarted after errors: the goal leaves the hold point smoothly
                self.o_bump = self._hold - self._carrot_world()
        elif self.t_air is None and p[2] > 0.8:
            self.t_air = self.t_launch = now

        # 1 intake: each detector frame once
        det = h.vision.get()
        if det.frames != self.det_seen:
            self.det_seen = det.frames
            tg = float(det.t) - P.vision_lag
            fresh = det.frames > 0 and now - tg <= P.stale_s
            if fresh:
                self.t_frame = now                # the detector is alive, whatever this frame holds
                self.f_track, self.f_dir = None, None   # ... and this frame has not shown an arch yet
            if det.frames > 0 and det.p_visible >= P.p_min and det.dist_m > 0.5:
                pose = h.pose_at(tg) if fresh else None
                if pose is None:
                    self.rej_stale += 1
                else:
                    self._intake(now, det, pose, p)
        self.stalled = now - self.t_frame > P.stall_s
        if self.stalled:
            self.t_stall_end = now            # a target unseen through a stall has not been missed: nothing was seen
        if not self.p_hist or now - self.p_hist[-1][0] >= 0.1:
            self.p_hist.append((now, float(p[0]), float(p[1])))
            while len(self.p_hist) > 2 and now - self.p_hist[0][0] > P.course_win[1] + 1.0:
                self.p_hist.pop(0)

        # 2 maintenance
        self._maintain(now, dt, p, R)

        # 3 target
        self._select(now, p)
        T = self.target
        self.next = None
        self.g_s = None
        self.d_gate = math.inf
        n_vec = None
        if T is not None:
            # 4 locked point, 5 approach axis
            age = now - T.t_last
            g_s = self._locked(T, age)
            self.g_s = g_s
            n_vec = self._axis(T, g_s, p, now, dt)
            # 6 pass
            T = self._pass_check(T, n_vec, p, now, age)
            if T is None:
                n_vec = None
                self.g_s = None
                self.d_gate = math.inf
        self._orphan_pass(now, p)
        h.vision_gate_w = None if self.target is None else self.target.m.copy()

        # guidance (mode and desired speed / curvature)
        self._guidance(now, dt, p, vel)
        # rabbit integration, reseed
        self._integrate(now, dt, p, vel, R)
        # altitude reference
        self._altitude(dt)
        # goal vector (its world point is committed by goal() once the whole tick has gone through)
        rel = self._goal_vector(dt, p, on_ground)
        # yaw and speed sense
        self._yaw(now, dt, psi_n, on_ground)
        self._flow(now, dt, vel)
        self._status(now)
        self._last_rel = rel
        return R.T @ rel

    # ------------------------------------------------------------------ 1 intake
    @property
    def gate_w(self) -> float:
        """The pilot's current estimate of how wide the arches of this course are, in metres."""
        return math.exp(self.w_ln)

    @property
    def gate_w_sd(self) -> float:
        """...and its 1-sigma spread, as a fraction (0.2 means +-20 %)."""
        return math.sqrt(max(self.w_var, 0.0))

    def _unit_range(self, det) -> float:
        """One detection's range per metre of gate width: f * stretch / width_px with the detector's range bias
        taken out. This is the whole of what apparent size measures; multiplying by a width makes it a range.

        The bias table is really a function of apparent pixel width, so it is read at the range the sighting would
        have if the arches were ``range_corr_width`` wide - which is how it was fitted - and not at whatever range
        the pilot currently believes. The clip is the detector's reach (f / width_px has a floor and a ceiling),
        not a fact about any course."""
        P = self.params
        w0 = float(getattr(det, 'width_m', 0.0) or 0.0) or GATE_WIDTH_NOM
        ur = float(det.dist_m) / w0
        if P.range_corr:
            d, k = zip(*P.range_corr)
            ur = ur * float(np.interp(ur * P.range_corr_width, d, k))
        return clip(ur, P.unit_range_clip[0], P.unit_range_clip[1])

    def _range(self, det) -> float:
        """The range of one sighting in metres: its unit range times the width the course seems to have."""
        return clip(self.gate_w * self._unit_range(det), 0.5, 400.0)

    @staticmethod
    def _off_axis(det, cam) -> float:
        """Degrees between the detection's ray and the camera's optical axis (0 when the camera is unknown)."""
        if cam is None:
            return 0.0
        fwd = cam.body_to_cam()[2]                     # camera forward, in body coordinates
        d = np.asarray(det.direction_body, dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return 0.0
        return math.degrees(math.acos(clip(float(d @ fwd) / n, -1.0, 1.0)))

    def _intake(self, now: float, det, pose, p: np.ndarray) -> None:
        P = self.params
        pg = np.asarray(pose[0], dtype=np.float64)
        Rg = pose[1]
        r = Rg @ np.asarray(det.direction_body, dtype=np.float64)
        r = r / max(float(np.linalg.norm(r)), 1e-9)
        el = math.asin(clip(float(r[2]), -1.0, 1.0))
        if not (P.elev_deg[0] * DEG < el < P.elev_deg[1] * DEG):
            self.rej_elev += 1
            return
        if not self.w_seen:               # the first sighting says which width the detector's ranges assume
            self.w_seen = True
            self.w_ln = math.log(clip(float(getattr(det, 'width_m', 0.0) or 0.0) or GATE_WIDTH_NOM,
                                      P.width_span[0], P.width_span[1]))
        ur = self._unit_range(det)
        rho = clip(self.gate_w * ur, 0.5, 400.0)
        cam = getattr(self.host.vision, 'cam', None)
        W, H = (float(cam.width), float(cam.height)) if cam is not None else (320.0, 180.0)
        off = self._off_axis(det, cam)
        if P.offaxis_max > 0 and off > P.offaxis_max:
            self.rej_offaxis += 1         # the image corners: 4 % of the sightings of the six flights, 11 % in search
            return
        hw = 0.5 * float(det.width_px)
        crop = (det.u - hw < P.crop_px or det.u + hw > W - P.crop_px or det.v - hw < P.crop_px
                or det.v + hw > H - P.crop_px)
        # an off-axis box is a worse measurement, not a wrong one
        k_off = math.sqrt(1.0 + off / P.offaxis_sig_deg) if P.offaxis_sig_deg > 0 and off > 0.0 else 1.0
        sa = (P.sig_along[0] * rho + P.sig_along[1]) * (P.crop_mult if crop else 1.0) * k_off
        sc = (P.sig_cross[0] * rho + P.sig_cross[1]) * k_off
        rr = np.outer(r, r)
        Rm = sa * sa * rr + sc * sc * (np.eye(3) - rr)
        z = pg + rho * r
        T = self._update(now, z, Rm, r, rho, pg, crop)
        if T is not None and not T.passed:
            # the same sighting also goes into the track's own point-and-width fit, which assumes no width at all;
            # once the drone has moved across the line of sight, that fit says how wide this arch is
            T.add_sight(pg, r, ur, sa, sc, P.width_bin)
            self._width_update(T, now)
        # the side to search: a tentative arch seen twice (a single sighting is as often a phantom or a flip)
        if T is not None and not T.confirmed and T.hits >= P.tent_side_hits:
            self.last_tent_b = wrap(math.atan2(float(T.m[1] - p[1]), float(T.m[0] - p[0])) - self.psi)
            self.last_tent_t = now

    def _width_update(self, T: Track, now: float) -> None:
        """What one arch's own sightings say its width is, folded into the course-wide estimate.

        Every arch on a course is normally the same size, so the per-track fits go into one filter in log space.
        A track is asked at most once every ``width_min_s`` (fifteen detections a second are one look at one arch,
        not fifteen independent ones), only when it is confirmed, and only when its fit has separated the width
        from the range to within ``tri_sigma_frac`` of itself - which is the drone's own movement across the line
        of sight doing the work, and is exactly what will not have happened on a first, distant sighting. What it
        then says is weighted by its own std, so an early rough fit moves the estimate and a later firm one fixes
        it; a fit that disagrees with the course by more than three sigma is a flip or a phantom and is dropped."""
        P = self.params
        if not P.width_est or not T.confirmed or T.rays < P.tri_min_rays or now - T.t_width < P.width_min_s:
            return
        fit = T.fit_width()
        if fit is None:
            return
        w_meas, sd, _ = fit
        if w_meas <= 0.0:
            return
        # The sightings went into the fit with the noise of a range the pilot believes, not the range the arch is
        # at, so the std comes back scaled by (what it believes) / (what the fit says): an arch read 2.7x too far
        # looks 2.7x noisier than it is, and the gate below would then be 2.7x too strict on exactly the course
        # that needs it. Dividing by the assumed width undoes that and makes the gate mean one thing everywhere.
        rel = sd / max(self.gate_w, 1e-6)
        if rel > P.tri_sigma_frac:
            return
        T.t_width = now
        if not (P.width_span[0] <= w_meas <= P.width_span[1]):
            self.w_rejects += 1
            return
        y = math.log(w_meas) - self.w_ln
        v = rel * rel + P.width_meas_ln ** 2
        s = self.w_var + v
        if abs(y) > 3.0 * math.sqrt(s):          # a flip, a phantom or a badly split track, not a gate size
            self.w_rejects += 1
            return
        k = self.w_var / s
        was = self.w_ln
        self.w_ln = clip(self.w_ln + k * y, math.log(P.width_span[0]), math.log(P.width_span[1]))
        self.w_var = max((1.0 - k) * self.w_var, P.width_floor_ln ** 2)
        self.w_updates += 1
        self._rescale(math.exp(self.w_ln - was))     # what the width actually moved, clip included

    def _rescale(self, g: float) -> None:
        """The course's arches just got g times wider, so every range the tracker holds was g times short: slide
        each unpassed estimate along the ray it was last seen on, from where it was seen, and scale its covariance
        with it (a point g times further away is g times more uncertain in metres, across the ray as well as along
        it). Passed arches are history and the drone's own path already vouches for them.

        An estimate that has just jumped is also worth less than it was. Every range that built it was measured at
        the old width, so the along-ray variance takes back the distance it moved, and a revision beyond
        ``width_reset_ln`` makes the track earn its parallax again - the parallax that had been allowed to shrink
        that variance was itself measured against ranges on the old scale. The filter then follows the new
        measurements instead of averaging them against a history taken on a different scale. Without this, an 8 m
        course was still being flown as a 7 m one at the gate, long after the width itself had been measured."""
        P = self.params
        if not math.isfinite(g) or abs(math.log(max(g, 1e-9))) < 1e-4:
            return
        # the parallax reference is retired on the NET drift since it was last set, not on one step: the estimate
        # jitters either way by a per cent or two at rest, which is no reason to throw a good estimate's parallax
        # away, while a course being walked from 4 m to 8 m drifts one way and so retires it
        self.w_drift += math.log(g)
        big = abs(self.w_drift) > P.width_reset_ln
        if big:
            self.w_drift = 0.0
        for T in self.tracks:
            if T.passed:
                continue
            rel = T.m - T.p_last
            n = float(np.linalg.norm(rel))
            T.m = T.p_last + g * rel
            T.P = T.P * (g * g)
            moved = abs(g - 1.0) * n
            if n > 1e-6 and moved > 0.0:
                u = rel / n
                T.P = T.P + (moved * moved) * np.outer(u, u)
            if big:
                T.r0 = T.r_last.copy()
            T.rng_last_h = T.dist_h(T.p_last)
            T.last_along *= g

    def _fit(self, T: Track, pg: np.ndarray, r: np.ndarray, rho: float, g: float) -> float | None:
        rel = T.m - pg
        al = float(rel @ r)
        if al <= 0.5:
            return None
        perp = float(np.linalg.norm(rel - al * r))
        P = self.params
        lim = max(P.perp_gate[0], P.perp_gate[1] * al)
        if P.gate_track_sigma:
            # a young estimate is itself uncertain across the ray: widen the gate by the total/measurement std ratio
            var_t = max(0.5 * (float(T.P[0, 0] + T.P[1, 1] + T.P[2, 2]) - float(r @ T.P @ r)), 0.0)
            sc = P.sig_cross[0] * al + P.sig_cross[1]
            lim *= math.sqrt(1.0 + var_t / (sc * sc))
        lr = abs(math.log(rho / al))
        if perp > lim or lr > g:
            return None
        return perp / lim + lr / g

    def _update(self, now, z, Rm, r, rho, pg, crop) -> Track | None:
        P = self.params
        for T in self.tracks:
            if not T.passed:
                continue
            nx, ny = T.n_pass
            ahead = float((z[0] - pg[0]) * nx + (z[1] - pg[1]) * ny) > 1.0
            # a passed arch absorbs its sightings from behind; after the un-pass window also from its entry side (the
            # drone came round in a search: the course does not go through the same arch again)
            if ((not ahead or now - T.t_passed > P.unpass_s)
                    and (math.hypot(float(z[0] - T.m[0]), float(z[1] - T.m[1])) < 4.0
                         or self._fit(T, pg, r, rho, 0.35) is not None)):
                self.absorbed += 1
                return None
            if (ahead and now - T.t_passed < P.unpass_s and self._fit(T, pg, r, rho, 0.45) is not None
                    and float((T.m[0] - pg[0]) * nx + (T.m[1] - pg[1]) * ny) > 2.0):
                self._unpass(T, now)
        lp = self.last_pass
        if lp is not None and now - lp['t'] < P.behind_s:
            if float((z[0] - lp['m'][0]) * lp['n'][0] + (z[1] - lp['m'][1]) * lp['n'][1]) < -1.0:
                self.behind += 1
                return None
        best, cost = None, math.inf
        for T in self.tracks:
            if T.passed:
                continue
            c = self._fit(T, pg, r, rho, P.log_gate if T.confirmed else P.log_gate_tent)
            if c is not None and c < cost:
                best, cost = T, c
        if best is None:
            self._next_id += 1
            T = Track(self._next_id, now, z, Rm, r, pg, rho)
            self.tracks.append(T)
            self.f_track, self.f_dir = T, r
            return T
        T = best
        self.f_track, self.f_dir = T, r    # this frame showed this arch: it says nothing about the others (ghost rule)
        T.hits += 1
        T.t_last = now
        T.p_last = pg.copy()
        T.r_last = r.copy()
        T.unseen_in_view = 0.0
        T.last_along = min(rho, float((T.m - pg) @ r))
        if not (T is self.target and (T.dist_h(pg) < P.no_update_d or (crop and rho < P.no_update_crop_rng))):
            S = T.P + Rm
            K = T.P @ np.linalg.inv(S)
            T.m = T.m + K @ (z - T.m)
            T.P = (np.eye(3) - K) @ T.P
            T.P = 0.5 * (T.P + T.P.T)
            # successive width ranges share their error: the along-ray variance may only shrink with parallax
            par = math.degrees(math.acos(clip(float(r @ T.r0), -1.0, 1.0)))
            sf = P.range_common * rho * (1.0 - min(par / 8.0, 1.0)) + 0.15
            vr = float(r @ T.P @ r)
            if vr < sf * sf:
                T.P = T.P + (sf * sf - vr) * np.outer(r, r)
        T.rng_last_h = T.dist_h(pg)
        if (not T.confirmed and T.hits >= P.confirm_hits and now - T.t_first >= P.confirm_span
                and (math.sqrt(max(T.P[2, 2], 0.0)) >= P.z_plaus_sigma or T.m[2] >= self._z_floor())):
            T.confirmed = True
        return T

    # ------------------------------------------------------------------ 2 maintenance
    def _in_view(self, T: Track, p: np.ndarray, R: np.ndarray, cam) -> bool:
        rel = T.m - p
        rng = float(np.linalg.norm(rel))
        lo, hi = self.params.ghost_range
        if cam is None or not (lo <= rng <= hi):
            return False
        px, ok = cam.project_body((R.T @ rel)[None])
        u, v = px[0]
        return bool(ok[0]) and 0.05 * cam.width < u < 0.95 * cam.width and 0.05 * cam.height < v < 0.95 * cam.height

    def _ghost_charge(self, T: Track, p: np.ndarray) -> bool:
        """Does the last fresh frame count against an estimate the camera should see? The detector reports at most one
        arch per frame, so a frame that showed a different arch elsewhere in the image says nothing about this one."""
        P = self.params
        if P.ghost_evidence == 'any':
            return True
        F = self.f_track
        if F is None:
            return P.ghost_evidence == 'empty_or_other'      # the frame held no arch at all
        if F is T or self.f_dir is None:
            return False
        b_t = math.atan2(float(T.m[1] - p[1]), float(T.m[0] - p[0]))
        b_f = math.atan2(float(self.f_dir[1]), float(self.f_dir[0]))
        return abs(wrap(b_f - b_t)) > P.ghost_other_deg * DEG

    def _ghost_s(self, T: Track) -> float:
        """A long-lived estimate has earned a longer timer: one with 88 sightings was ghosted 2 s before its gate."""
        P = self.params
        if P.ghost_s_hits <= 0 or P.ghost_s_max <= P.ghost_s:
            return P.ghost_s
        return min(P.ghost_s * max(1.0, T.hits / P.ghost_s_hits), P.ghost_s_max)

    def _orphan(self, T: Track, now: float, p: np.ndarray) -> None:
        """Remember a confirmed estimate the drone came close to but dropped: it was a gate, and flying past it still
        has to move the course reference and the height ladder on (a ghosted target never registered a pass)."""
        P = self.params
        if P.orphan_d <= 0 or not T.confirmed or T.passed or T.min_d > P.orphan_d:
            return
        if P.orphan_target_only and T is not self.target:
            return
        if self.n_ang is not None and T is self.target:
            nx, ny = math.cos(self.n_ang), math.sin(self.n_ang)
        else:
            dx, dy = float(T.m[0] - p[0]), float(T.m[1] - p[1])
            n = math.hypot(dx, dy)
            nx, ny = (dx / n, dy / n) if n > 1e-6 else (math.cos(self.psi), math.sin(self.psi))
        self.orphans.append({'T': T, 'm': (float(T.m[0]), float(T.m[1])), 'n': (nx, ny), 't': now})
        del self.orphans[:-4]

    def _orphan_pass(self, now: float, p: np.ndarray) -> None:
        """Register the pass of an orphan the drone has now flown past, without disturbing the current target."""
        P = self.params
        if P.orphan_d <= 0 or not self.orphans or self.t_air is None or now - self.t_air <= P.airborne_guard:
            return
        for o in list(self.orphans):
            if now - o['t'] > P.orphan_life:
                self.orphans.remove(o)
                continue
            nx, ny = o['n']
            dx, dy = float(p[0]) - o['m'][0], float(p[1]) - o['m'][1]
            if dx * nx + dy * ny < P.orphan_a or abs(-dx * ny + dy * nx) > P.orphan_lat:
                continue
            self.orphans.remove(o)
            self.orphans = [q for q in self.orphans     # the other fragments of this arch are the same gate
                            if math.hypot(q['m'][0] - o['m'][0], q['m'][1] - o['m'][1]) >= P.orphan_dedup_d]
            n = self.n_passes
            self._pass(o['T'], now, 'travel', o['n'], clear_target=False)
            if self.n_passes > n:              # _pass refuses a second pass of the arch already on the books
                self.orphans_registered += 1
            return

    def _maintain(self, now: float, dt: float, p: np.ndarray, R: np.ndarray) -> None:
        P = self.params
        cam = getattr(self.host.vision, 'cam', None)
        near, near_d = None, math.inf
        for T in self.tracks:
            T.P = T.P + (P.q * dt) * np.eye(3)
            if T.confirmed and not T.passed:
                T.min_d = min(T.min_d, T.dist_h(p))
                d = float(np.linalg.norm(T.m - p))
                if d < near_d and self._in_view(T, p, R, cam):
                    near, near_d = T, d
        if near is not None and now - self.t_frame < P.unseen_live_s and self._ghost_charge(near, p):
            near.unseen_in_view += dt                 # only while frames arrive: a stalled detector sees nothing
        keep = []
        for T in self.tracks:
            if T.passed:
                if now - T.t_passed <= P.passed_life:
                    keep.append(T)
                continue
            if not T.confirmed:
                if now - T.t_last <= P.tent_life:
                    keep.append(T)
                continue
            # an estimate nothing has seen for a long time is gone, the target included: the ghost rule cannot say so
            # beyond ghost_range or while ghost_keep_d holds the target, and a stalled detector saw nothing at all
            life = P.conf_life if T is not self.target else (P.target_life or math.inf)
            if now - max(T.t_last, self.t_stall_end) > life:
                self._orphan(T, now, p)
                if T is self.target:
                    self.stale_targets += 1
                    self.ghosts += 1
                    self.pass_kind = PASS_KIND['ghost']
                    self.target = None
                    self.n_ang = None
                continue
            if T.unseen_in_view > self._ghost_s(T):
                if T is self.target and P.ghost_keep_d > 0 and T.dist_h(p) < P.ghost_keep_d:
                    T.unseen_in_view = 0.0            # the gate being flown at is never deleted from close in: it is
                    keep.append(T)                    # about to be passed, or missed and dropped from further out
                    continue
                self.ghosts += 1
                self.pass_kind = PASS_KIND['ghost']
                self._orphan(T, now, p)
                if T is self.target:
                    self.target = None
                    self.n_ang = None
                continue
            if math.sqrt(max(T.P[2, 2], 0.0)) < P.low_sigma and T.m[2] < self._z_floor():
                self.low += 1                       # ground clutter: below anything a gate could plausibly be
                if T is self.target:
                    self.target = None
                    self.n_ang = None
                continue
            keep.append(T)
        # merge fragments of one arch (confirmed, unpassed), fusing the smaller into the larger
        out: list[Track] = []
        for T in sorted(keep, key=lambda t: -t.hits):
            twin = None
            if T.confirmed and not T.passed:
                for o in out:
                    if o.confirmed and not o.passed and self._same(o, T, p):
                        twin = o
                        break
            if twin is None:
                out.append(T)
                continue
            S = twin.P + T.P
            K = twin.P @ np.linalg.inv(S)
            twin.m = twin.m + K @ (T.m - twin.m)
            twin.P = (np.eye(3) - K) @ twin.P
            twin.P = 0.5 * (twin.P + twin.P.T)
            twin.hits += T.hits
            twin.N4 = twin.N4 + T.N4              # two estimates of one arch: one set of sightings of it
            twin.g4 = twin.g4 + T.g4
            twin.rays += T.rays
            if T.t_last > twin.t_last:
                twin.t_last, twin.p_last, twin.r_last = T.t_last, T.p_last, T.r_last
            twin.unseen_in_view = min(twin.unseen_in_view, T.unseen_in_view)
            if self.target is T:
                self.target = twin
            if self.pending is T:
                self.pending = twin
        self.tracks = out
        # a confirmed track strung out along the target's own bearing is that arch badly ranged, not the next gate:
        # fold its sightings in rather than let it steal the target or swing the bisector
        if P.frag_absorb and self.target is not None:
            for o in [o for o in self.tracks if o is not self.target and o.confirmed and not o.passed]:
                if self._ahead_fragment(o, self.target, p):
                    self.tracks.remove(o)
                    self.target.hits += o.hits
                    self.target.N4 = self.target.N4 + o.N4
                    self.target.g4 = self.target.g4 + o.g4
                    self.target.rays += o.rays
                    self.absorbed += 1
        if self.target is not None and self.target not in self.tracks:
            self.target = None
            self.n_ang = None

    def _frag_gap(self) -> float:
        """How far ahead of the target a confirmed track is still a fragment of it: the course's own gate spacing
        once two gates have been passed, otherwise the default."""
        P = self.params
        if P.frag_gap <= 0:
            return 0.0
        if self.spacing is not None and P.frag_gap_frac > 0:
            return max(P.frag_gap, P.frag_gap_frac * self.spacing)
        return P.frag_gap

    def _ahead_fragment(self, o: Track, T: Track | None, p: np.ndarray) -> bool:
        """Is ``o`` a fragment of the target sitting just beyond it on the same ray from the drone?"""
        if T is None or o is T:
            return False
        gap = self._frag_gap()
        if gap <= 0:
            return False
        dx, dy = float(T.m[0] - p[0]), float(T.m[1] - p[1])
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return False
        nx, ny = dx / n, dy / n
        ax, ay = float(o.m[0] - T.m[0]), float(o.m[1] - T.m[1])
        a = ax * nx + ay * ny
        return 0.5 < a <= gap and abs(-ax * ny + ay * nx) < self.params.frag_lat_ahead

    def _same(self, a: Track, b: Track, p: np.ndarray) -> bool:
        P = self.params
        if math.hypot(float(a.m[0] - b.m[0]), float(a.m[1] - b.m[1])) < P.merge_d:
            return True
        va, vb = a.m - p, b.m - p
        ra, rb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        if min(ra, rb) < 1.0 or float(va @ vb) <= 0.0:
            return False                      # the cross product below cannot tell a track behind from one ahead
        if ra <= rb:
            lat = float(np.linalg.norm(np.cross(va / ra, vb)))
        else:
            lat = float(np.linalg.norm(np.cross(vb / rb, va)))
        # never wider in range than the association gate: two arches kept apart there must not be fused here
        return (lat < max(P.merge_lat[0], P.merge_lat[1] * min(ra, rb))
                and abs(math.log(ra / rb)) < min(P.merge_log, P.log_gate))

    # ------------------------------------------------------------------ 3 target
    def _select(self, now: float, p: np.ndarray) -> None:
        P = self.params
        best, best_score = None, math.inf
        for T in self.tracks:
            if not T.confirmed or T.passed:
                continue
            dx, dy = float(T.m[0] - p[0]), float(T.m[1] - p[1])
            d = math.hypot(dx, dy)
            b = wrap(math.atan2(dy, dx) - self.psi)
            is_t = T is self.target
            if not is_t and (d < P.eligible_d or abs(b) > P.eligible_bearing * DEG):
                continue
            if not is_t and self._ahead_fragment(T, self.target, p):
                continue                  # the same arch 5-10 m further along the ray: never fly at the far copy
            score = d + 8.0 * (1.0 - math.cos(b)) - (4.0 if is_t else 0.0)
            # an estimate not seen for a while loses its standing against arches seen now
            score += min(P.stale_penalty * max(now - T.t_last - P.stale_grace, 0.0), 8.0)
            if score < best_score:
                best, best_score = T, score
        if self.target is None:
            if best is not None:
                self.target = best
                self.n_ang = None
            self.pending = None
        elif best is not self.target and best is not None:
            if self.pending is not best:
                self.pending, self.pending_since = best, now
            elif now - self.pending_since >= P.switch_s:
                self.target = best
                self.n_ang = None
                self.pending = None
        else:
            self.pending = None

    # ------------------------------------------------------------------ 4-5 locked point and approach axis
    def _course_stale(self, now: float) -> bool:
        """Is the last registered pass too old or too far behind to say where the course runs?"""
        P = self.params
        lp = self.last_pass
        return lp is None or now - lp['t'] > P.course_age_s or self.s - lp['s'] > P.course_back_m

    def _drone_course(self, now: float) -> float | None:
        """The drone's own heading over the last ``course_win`` seconds (None when it has not moved far enough)."""
        P = self.params
        if P.course_min_m <= 0:
            return None
        lo, hi = P.course_win
        old = None
        for t, x, y in self.p_hist:               # in time order: the oldest sample still inside the window
            if now - t <= hi:
                old = (t, x, y)
                break
        if old is None or now - old[0] < lo:
            return None
        dx, dy = float(self._p[0]) - old[1], float(self._p[1]) - old[2]
        return math.atan2(dy, dx) if math.hypot(dx, dy) >= P.course_min_m else None

    def _locked(self, T: Track, age: float) -> np.ndarray:
        a0, a1 = self.params.lock_s
        if age > a1:
            return T.m
        lock = T.p_last + T.r_last * max(float((T.m - T.p_last) @ T.r_last), 0.5)
        if age <= a0:
            return lock
        f = (age - a0) / max(a1 - a0, 1e-6)
        return lock + f * (T.m - lock)

    def _axis(self, T: Track, g_s: np.ndarray, p: np.ndarray, now: float, dt: float) -> tuple[float, float]:
        P = self.params
        gx, gy = float(g_s[0]), float(g_s[1])
        rx, ry = gx - float(p[0]), gy - float(p[1])
        d = math.hypot(rx, ry)
        self.d_gate = d
        a_ray = math.atan2(ry, rx)
        lp = self.last_pass
        stale = self._course_stale(now)
        if lp is not None and not stale and math.hypot(gx - lp['m'][0], gy - lp['m'][1]) > 4.0:
            a_cin = math.atan2(gy - lp['m'][1], gx - lp['m'][0])
        else:
            # no gate passed lately (a ghosted target never registered one): the drone's own course, not the bearing
            # the target happened to be first seen on, is what "along the course" means
            a_drone = self._drone_course(now)
            a_cin = a_drone if a_drone is not None else math.atan2(T.dir_first[1], T.dir_first[0])
        cx, cy = math.cos(a_cin), math.sin(a_cin)
        nxt, nxt_d = None, math.inf
        for o in self.tracks:
            if o is T or o.passed or not o.confirmed or o.hits < P.next_min_hits:
                continue                  # a briefly seen phantom beside the course would swing the approach axis
            vx, vy = float(o.m[0]) - gx, float(o.m[1]) - gy
            dv = math.hypot(vx, vy)
            # any arch not well behind the target along the course (turns up to about 107 deg; gate 2 turns 87), and
            # far enough to be a gate of its own: the gates of this course are 24-35 m apart
            if P.next_min_sep < dv < 45.0 and vx * cx + vy * cy > -0.3 * dv and dv < nxt_d:
                nxt, nxt_d = o, dv
        self.next = nxt
        hints = P.turn_hints
        if nxt is not None:
            ux, uy = (float(nxt.m[0]) - gx) / nxt_d, (float(nxt.m[1]) - gy) / nxt_d
            a_bis = math.atan2(cy + uy, cx + ux)
            rot = wrap(a_bis - a_cin)
            a_nom = a_cin + clip(rot, -P.bisector_cap * DEG, P.bisector_cap * DEG)
            self.turn_gate = abs(rot) > P.turn_gate_bisector * DEG
        elif hints and self.n_passes < len(hints):
            hint = float(hints[self.n_passes]) * DEG
            a_nom = a_cin + 0.5 * hint
            self.turn_gate = abs(hint) >= P.turn_gate_alpha * DEG
        elif lp is not None and not stale:
            alpha = wrap(a_cin - math.atan2(lp['n'][1], lp['n'][0]))
            if abs(alpha) >= P.turn_rot_min * DEG:
                a_nom = a_cin + math.copysign(min(0.5 * abs(alpha), P.turn_rot_max * DEG), alpha)
            else:
                a_nom = a_cin
            self.turn_gate = abs(alpha) >= P.turn_gate_alpha * DEG
        else:
            a_nom = a_cin
            self.turn_gate = False
        beta = wrap(a_nom - a_ray)
        lim = P.pivot_max * DEG * clip((d - P.pivot_d[0]) / max(P.pivot_d[1] - P.pivot_d[0], 1e-6), 0.0, 1.0)
        Pm = T.P
        if P.axis_sigma_cap > 0 and d > 1e-6:
            # an axis tilted off the ray turns range error into lateral error at the gate plane (w19 gate 3: +7.9 m
            # along the ray and 13.6 deg of tilt made the whole -1.9 m miss), so tilt only as far as the range is sure
            ux, uy = rx / d, ry / d
            sig_a = math.sqrt(max(ux * ux * Pm[0, 0] + 2 * ux * uy * Pm[0, 1] + uy * uy * Pm[1, 1], 0.0))
            lim = min(lim, math.atan(P.axis_sigma_cap / max(sig_a, 1.0)))
        a_app = a_ray + clip(beta, -lim, lim)
        if self.n_ang is None:
            self.n_ang = wrap(a_app)
        elif d >= P.pivot_freeze_d:
            step = wrap(a_app - self.n_ang) * min(1.0, dt / P.pivot_tau)
            self.n_ang = wrap(self.n_ang + clip(step, -P.pivot_rate * DEG * dt, P.pivot_rate * DEG * dt))
        nx, ny = math.cos(self.n_ang), math.sin(self.n_ang)
        lx, ly = -ny, nx
        self.sd_lat = math.sqrt(max(lx * lx * Pm[0, 0] + 2 * lx * ly * Pm[0, 1] + ly * ly * Pm[1, 1], 0.0))
        return nx, ny

    # ------------------------------------------------------------------ 6 pass
    def _pass_check(self, T: Track, n_vec, p: np.ndarray, now: float, age: float) -> Track | None:
        P = self.params
        nx, ny = n_vec
        dx, dy = float(p[0] - T.m[0]), float(p[1] - T.m[1])
        a_d = dx * nx + dy * ny
        lat_d = -dx * ny + dy * nx
        T.min_a = min(T.min_a, a_d)
        crossed = T.min_a <= P.cross_before and a_d >= P.cross_after and abs(lat_d) < P.cross_lat
        travelled = (age > P.travel_age and T.rng_last_h < P.travel_rng
                     and float((p[0] - T.p_last[0]) * nx + (p[1] - T.p_last[1]) * ny) >= P.travel_frac * T.rng_last_h)
        bear = abs(wrap(math.atan2(-dy, -dx) - self.psi))
        beside = a_d > P.beside_a or (self.d_gate < P.beside_d and bear > P.beside_bearing * DEG)
        seen_ahead = age < P.seen_ahead_age and T.last_along > P.seen_ahead_along
        self.dbg = (a_d, lat_d, T.min_a, crossed, travelled, beside, seen_ahead, self.d_gate, math.degrees(bear))
        if not (self.t_air is not None and now - self.t_air > P.airborne_guard and (crossed or travelled or beside)
                and not seen_ahead):
            return T
        frag = self._fragment_ahead(T, n_vec, now)
        if frag is not None:
            # T is a short estimate of an arch seen just ahead on the same line: not a gate of its own
            if T in self.tracks:
                self.tracks.remove(T)
            self.absorbed += 1
            frag.hits += T.hits
            frag.N4 = frag.N4 + T.N4
            frag.g4 = frag.g4 + T.g4
            frag.rays += T.rays
            self.target = frag
            self.pending = None
            return None
        if T.hits >= P.fresh_hits and (age < P.fresh_age or T.rng_last_h < P.fresh_rng):
            self._pass(T, now, 'cross' if crossed else 'travel' if travelled else 'beside', n_vec)
        else:
            if T in self.tracks:
                self.tracks.remove(T)
            self.ghosts += 1
            self.pass_kind = PASS_KIND['ghost']
            self._orphan(T, now, p)       # thin, but the drone is on top of it: flying past still moves the course on
            self.target = None
            self.n_ang = None
        return None

    def _fragment_ahead(self, T: Track, n_vec, now: float) -> Track | None:
        P = self.params
        if P.frag_ahead <= 0:
            return None
        nx, ny = n_vec
        best, best_a = None, math.inf
        for o in self.tracks:
            if o is T or o.passed or not o.confirmed or now - o.t_last > P.frag_fresh_s:
                continue
            dx, dy = float(o.m[0] - T.m[0]), float(o.m[1] - T.m[1])
            a = dx * nx + dy * ny
            if 0.5 < a <= P.frag_ahead and abs(-dx * ny + dy * nx) < P.frag_lat and a < best_a:
                best, best_a = o, a
        return best

    def _duplicate(self, T: Track, now: float, d_now: float) -> bool:
        """Is this pass the arch already on the books, under a second id? One arch leaves two estimates (an orphan
        and the track that survived it, or two fragments) 8-20 m apart, and the course between them is a few metres;
        the gates of a course are 24-35 m apart and a lap is a minute and more. The pass declared from closer in is
        the better one: it replaces the other, and the gate is counted once."""
        P = self.params
        lp = self.last_pass
        if lp is None or P.orphan_dedup_d <= 0 or now - lp['t'] >= P.behind_s:
            return False
        q = lp.get('p')
        if (math.hypot(float(T.m[0]) - lp['m'][0], float(T.m[1]) - lp['m'][1]) >= P.orphan_dedup_d
                and (q is None or math.hypot(float(self._p[0]) - q[0], float(self._p[1]) - q[1])
                     >= P.orphan_dedup_d)):
            return False                      # far from the last pass, and the drone has flown a leg since it
        if d_now >= lp.get('d', math.inf) or self.pass_backup is None:
            return True                       # the pass on the books was the nearer one: keep it, count no second
        (self.z_pass_last, self.z_aim_last, self.grade_last, self.last_pass, self.side,
         self.n_passes) = self.pass_backup    # ... otherwise undo it and let this one take its place
        if self.pass_xy:
            self.pass_xy.pop()
        self.dup_passes += 1
        return False

    def _pass(self, T: Track, now: float, kind: str, n_vec, clear_target: bool = True) -> None:
        """Book a pass. ``clear_target`` False registers one for an arch the drone has flown past while already
        flying at the next one (an orphan), so the course reference and the height ladder move on undisturbed."""
        P = self.params
        d_now = T.dist_h(self._p)
        dup = self._duplicate(T, now, d_now)
        lp = self.last_pass
        self.pass_backup = (self.z_pass_last, self.z_aim_last, self.grade_last, lp, self.side, self.n_passes)
        T.passed = True
        T.t_passed = now
        T.n_pass = (float(n_vec[0]), float(n_vec[1]))
        zp = float(T.m[2]) - CENTRE_UP_M
        if dup:
            self.dup_passes += 1              # the same arch again: pass it, but do not count or re-seed anything
            self.pass_backup = None           # ... and nothing is left to undo, so a third one cannot replace it
        else:
            if lp is not None:
                dist = math.hypot(float(T.m[0] - lp['m'][0]), float(T.m[1] - lp['m'][1]))
                self.grade_last = clip((zp - self.z_pass_last) / max(dist, 5.0), 0.0, P.grade_max)
                turn = wrap(self.n_ang - math.atan2(lp['n'][1], lp['n'][0])) if self.n_ang is not None else 0.0
                if abs(turn) > 10.0 * DEG:
                    self.side = math.copysign(1.0, turn)
            else:
                self.grade_last = 0.0
            self.z_pass_last = zp
            self.z_aim_last = clip(zp + P.z_aim, P.z_min, 30.0)
            self.last_pass = {'t': now, 'm': (float(T.m[0]), float(T.m[1]), float(T.m[2])), 'n': T.n_pass,
                              's': self.s, 'kind': kind, 'track': T, 'd': d_now,
                              'p': (float(self._p[0]), float(self._p[1]))}
            self.n_passes += 1
            # the course's own gate spacing, for the fragment rule (a real neighbour is 24-35 m away on this track)
            self.pass_xy.append((float(T.m[0]), float(T.m[1])))
            if len(self.pass_xy) >= 3:
                d = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.pass_xy, self.pass_xy[1:])]
                d = [x for x in d if x > 5.0]
                if d:
                    self.spacing = float(np.median(d))
        if clear_target:
            self.target = None
            self.n_ang = None
            self.pending = None
        self.pass_kind = PASS_KIND[kind]
        # fragments of the same arch (made while the target was frozen close up) would be passed again a moment later
        for o in [o for o in self.tracks if o is not T and not o.passed]:
            if o is self.target and not clear_target:
                continue                  # never absorb the gate now being flown at into one behind it
            if math.hypot(float(o.m[0] - T.m[0]), float(o.m[1] - T.m[1])) < P.pass_absorb_d:
                self.tracks.remove(o)
                self.absorbed += 1
        h = self.host
        if hasattr(h, '_passed'):
            h._passed.append((now, T.m.copy()))
        h.vision_passed_t = now

    def _unpass(self, T: Track, now: float) -> None:
        T.passed = False
        T.min_a = math.inf
        if self.pass_backup is not None:
            (self.z_pass_last, self.z_aim_last, self.grade_last, self.last_pass, self.side,
             self.n_passes) = self.pass_backup
            self.pass_backup = None
        self.target = T
        self.n_ang = None
        self.pending = None
        self.unpasses += 1
        self.pass_kind = PASS_KIND['unpass']
        h = self.host
        if getattr(h, '_passed', None):
            h._passed.pop()

    # ------------------------------------------------------------------ guidance
    def _guidance(self, now: float, dt: float, p: np.ndarray, vel: np.ndarray) -> None:
        P = self.params
        T = self.target
        mode_prev = self.mode
        lp = self.last_pass
        if self.t_air is None:
            self.mode = 0
            v_des, k_des = 0.0, 0.0
        elif T is not None and self.g_s is not None:
            self.mode = 2
            nx, ny = math.cos(self.n_ang), math.sin(self.n_ang)
            lx, ly = -ny, nx
            relx, rely = self.c[0] - float(self.g_s[0]), self.c[1] - float(self.g_s[1])
            a_r = relx * nx + rely * ny
            e = relx * lx + rely * ly
            if a_r < 0.0:
                psi_des = self.n_ang + math.atan2(-e, clip(0.6 * -a_r, 3.0, 8.0))
            else:
                psi_des = self.n_ang
            err = wrap(psi_des - self.psi)
            k_des = P.k_head * err / max(self.v_nom, 1.0)
            D = max(-a_r - 1.0, 1.0)
            k_req = min(max(abs(wrap(self.n_ang - self.psi)) / D, 4.0 * abs(e) / (D * D), abs(err) / 8.0), P.kappa_max)
            v_gate = (P.v_gate_turn if self.turn_gate else P.v_gate) * clip(1.15 - 0.5 * self.sd_lat, 0.75, 1.0)
            v_brake = math.sqrt(v_gate * v_gate + 2.0 * P.a_brk * max(0.0, -a_r - self.v_nom * P.t_lag))
            v_des = min(P.v_cruise, math.sqrt(P.a_lat / max(k_req, 1e-3)), v_brake)
            if self.d_gate < 10.0 and self.sd_lat > 0.7:
                v_des = min(v_des, P.v_unsure)
            if now - T.t_last > 2.5 or self.stalled:
                v_des = min(v_des, P.v_blind)
        elif self.stalled:
            # no detector frames: no launch leg, no search; the rabbit parks and the drone holds
            self.mode = 4
            v_des, k_des = 0.0, 0.0
            if lp is None:
                self.t_launch = now
        elif (lp is not None and self.s - lp['s'] < P.d_on) or (lp is None and now - self.t_launch < P.launch_t):
            self.mode = 1
            h_ref = math.atan2(lp['n'][1], lp['n'][0]) if lp is not None else self.psi_launch
            hints = P.turn_hints
            if lp is not None and hints and 0 < self.n_passes <= len(hints):
                h_ref += 0.5 * float(hints[self.n_passes - 1]) * DEG
            k_des = P.k_head * wrap(h_ref - self.psi) / max(self.v_nom, 1.0)
            v_des = (P.v_exit if P.v_exit is not None else P.v_cruise) if lp is not None else P.v_launch
        else:
            self.mode = 3
            if mode_prev != 3:
                self.mode_since = now
                self.search_turned = 0.0
            s_side = None
            hints = P.turn_hints
            if hints and 0 < self.n_passes <= len(hints) and hints[self.n_passes - 1]:
                s_side = math.copysign(1.0, float(hints[self.n_passes - 1]))
            if s_side is None and now - self.last_tent_t < 3.0 and 10.0 * DEG < abs(self.last_tent_b) < 120.0 * DEG:
                s_side = math.copysign(1.0, self.last_tent_b)
            if s_side is None:
                others = [o for o in self.tracks if o.confirmed and not o.passed]
                if others:
                    o = min(others, key=lambda o: o.dist_h(p))
                    b = wrap(math.atan2(float(o.m[1] - p[1]), float(o.m[0] - p[0])) - self.psi)
                    s_side = math.copysign(1.0, b)
            if s_side is None:
                s_side = self.side
            if lp is not None:
                ax, ay = lp['m'][0] + 6.0 * lp['n'][0], lp['m'][1] + 6.0 * lp['n'][1]
            else:
                ax = self.launch_p[0] + 10.0 * math.cos(self.psi_launch)
                ay = self.launch_p[1] + 10.0 * math.sin(self.psi_launch)
            if math.hypot(float(p[0]) - ax, float(p[1]) - ay) > P.search_leash:
                k_des = P.k_head * wrap(math.atan2(ay - self.c[1], ax - self.c[0]) - self.psi) / max(self.v_nom, 1.0)
            else:
                radius = P.search_radius if self.search_turned < TWO_PI else P.search_radius_wide
                k_des = s_side / radius
            v_des = P.v_search
            self.search_turned += abs(self.kappa * self.v * dt)
        self.v_des = v_des
        self.k_des = k_des

    # ------------------------------------------------------------------ rabbit integration
    def _drone_s(self, p: np.ndarray) -> float:
        tr = self.trail
        if len(tr) < 2:
            return self.s - math.hypot(self.c[0] - float(p[0]), self.c[1] - float(p[1]))
        lo = max(self.s_d_prev - 1.0, self.s - 15.0)
        i0 = max(bisect_left(tr, lo, key=itemgetter(0)) - 1, 0)
        seg = tr[i0:]
        if len(seg) < 2:
            seg = tr[-2:]
        A = np.array(seg + [(self.s, self.c[0], self.c[1])])
        S, X, Y = A[:, 0], A[:, 1], A[:, 2]
        ax, ay = X[:-1], Y[:-1]
        bx, by = X[1:] - ax, Y[1:] - ay
        L2 = np.maximum(bx * bx + by * by, 1e-9)
        f = np.clip(((float(p[0]) - ax) * bx + (float(p[1]) - ay) * by) / L2, 0.0, 1.0)
        qx, qy = ax + f * bx - float(p[0]), ay + f * by - float(p[1])
        i = int(np.argmin(qx * qx + qy * qy))
        self.s_d_prev = float(S[i] + f[i] * (S[i + 1] - S[i]))
        return self.s_d_prev

    def _carrot_world(self) -> np.ndarray:
        lx, ly = -math.sin(self.psi), math.cos(self.psi)
        return np.array([self.c[0] + self.o_snap * lx, self.c[1] + self.o_snap * ly, self.z_c]) + self.o_bump

    def _integrate(self, now: float, dt: float, p: np.ndarray, vel: np.ndarray, R: np.ndarray) -> None:
        P = self.params
        self.v_nom += clip(self.v_des - self.v_nom, -P.a_brk * dt, P.a_acc * dt)
        s_d = self._drone_s(p)
        lead0 = P.lead + P.lead_open * clip((self.d_gate - 4.0) / 6.0, 0.0, 1.0)
        self.lead_now = self.s - s_d
        f = clip(1.0 - (self.lead_now - lead0) / P.lead_band, 0.0, P.lead_fmax)
        dist_c = math.hypot(self.c[0] - float(p[0]), self.c[1] - float(p[1]))
        if dist_c > P.rabbit_stop:
            f = 0.0
        self.v += clip(self.v_nom * f - self.v, -2.0 * dt, 2.0 * dt)
        kmax = min(P.kappa_max, P.a_lat / max(self.v_nom, 0.5) ** 2)
        dk = P.sharp * max(self.v, 0.3) * dt
        self.kappa += clip(clip(self.k_des, -kmax, kmax) - self.kappa, -dk, dk)
        ds = self.v * dt
        self.psi = wrap(self.psi + self.kappa * ds)
        self.c[0] += ds * math.cos(self.psi)
        self.c[1] += ds * math.sin(self.psi)
        self.s += ds
        if self.s - self.trail[-1][0] >= 0.2:
            self.trail.append((self.s, self.c[0], self.c[1]))
            if len(self.trail) > 400:
                del self.trail[:100]
        # reseed when the drone has been far from the rabbit for a while (a bounce, a recovery)
        if dist_c > P.reseed_d:
            if self.far_since is None:
                self.far_since = now
            elif now - self.far_since > P.reseed_s:
                old = self._carrot_world()
                vh = math.hypot(float(vel[0]), float(vel[1]))
                if vh > 1.0:
                    ux, uy = float(vel[0]) / vh, float(vel[1]) / vh
                else:
                    a = math.atan2(R[1, 0], R[0, 0])
                    ux, uy = math.cos(a), math.sin(a)
                self.c = [float(p[0]) + 2.5 * ux, float(p[1]) + 2.5 * uy]
                self.psi = math.atan2(uy, ux)
                self.kappa = 0.0
                self.trail = [(self.s, self.c[0], self.c[1])]
                self.s_d_prev = -math.inf
                self.o_bump = np.zeros(3)
                self.o_bump = old - self._carrot_world()
                self.far_since = None
                self.reseeds += 1
        else:
            self.far_since = None
        nb = float(np.linalg.norm(self.o_bump))
        if nb > 0.0:
            shrink = min(nb * (1.0 - math.exp(-dt / P.bump_tau)), P.bump_vmax * dt)
            self.o_bump *= max(nb - shrink, 0.0) / nb

    # ------------------------------------------------------------------ altitude
    def _z_floor(self) -> float:
        """The lowest height a gate could plausibly sit at: a descending course is a course, ground clutter is not."""
        return max(self.params.z_min, self.z_pass_last - self.params.z_drop_max)

    def _z_ref(self) -> float:
        """The grade line: the last passage height carried on at the grade of the leg that led to it."""
        lp = self.last_pass
        if lp is None:
            return self.z_aim_last
        return self.z_aim_last + self.grade_last * min(max(self.s - lp['s'], 0.0), self.params.grade_len)

    def _altitude(self, dt: float) -> None:
        P = self.params
        T = self.target
        z_ref = self._z_ref()
        if T is not None and self.g_s is not None:
            sz = math.sqrt(max(T.P[2, 2], 0.0))
            z_tgt = float(T.m[2]) - CENTRE_UP_M + P.z_aim + min(P.up_bias, P.up_bias * sz)
            # the estimate's height runs 1:1 into the reference and from there into the drone, and it has been 1.3 m
            # out: cap it at the grade line, so one high estimate cannot lift the whole approach. The floor stays on
            # the last passage height: a floor that climbed with the grade line would itself lift the approach.
            base = z_ref if P.z_window_ref else self.z_aim_last
            # How far the aim may drop below the reference. The window is there to stop ONE bad estimate lifting or
            # dropping the whole approach, so it binds on uncertain estimates and gets out of the way of certain
            # ones: a confident height is the best information there is. While it is still uncertain, the limit is
            # what the drone could fly anyway - descend_slope metres down per metre along.
            drop = max(P.z_window[0], P.descend_slope * max(self.d_gate, 0.0))
            if sz < P.low_sigma:
                drop = max(drop, P.z_drop_max)
            z_tgt = clip(z_tgt, max(P.z_min, min(base, self.z_aim_last) - drop), base + P.z_window[1])
            T_z = max((self.d_gate - 1.0) / max(self.v_nom, 1.0), 1.0)
            if z_tgt > self.z_c:
                T_z = max(P.climb_front * T_z, 1.0)
        else:
            z_tgt = z_ref
            T_z = 1.5
        vlim = min(max(P.vz_frac * self.v_nom, P.vz_min), P.vz_max)
        vlim_dn = min(max(P.vz_frac * self.v_nom, P.vz_min), max(P.vz_max_down, P.vz_max))
        vz_des = clip((z_tgt - self.z_c) / T_z, -vlim_dn, vlim)
        self.vz_c += clip(vz_des - self.vz_c, -P.az_max * dt, P.az_max * dt)
        self.z_c += self.vz_c * dt

    # ------------------------------------------------------------------ goal
    def _goal_vector(self, dt: float, p: np.ndarray, on_ground: bool) -> np.ndarray:
        P = self.params
        snap_des = 0.0
        if self.mode == 2 and self.g_s is not None and self.d_gate < P.snap_start:
            rx, ry = float(self.g_s[0] - p[0]), float(self.g_s[1] - p[1])
            rn = max(math.hypot(rx, ry), 1e-6)
            rx, ry = rx / rn, ry / rn
            qx, qy = self.c[0] - float(p[0]), self.c[1] - float(p[1])
            al = qx * rx + qy * ry
            ox, oy = qx - al * rx, qy - al * ry
            lx, ly = -math.sin(self.psi), math.cos(self.psi)
            w = clip((P.snap_start - self.d_gate) / 4.0, 0.0, 1.0)
            snap_des = clip(-w * (ox * lx + oy * ly), -P.snap_max, P.snap_max)
        self.o_snap += clip(snap_des - self.o_snap, -P.snap_rate * dt, P.snap_rate * dt)
        carrot = self._carrot_world()
        self._carrot_new = carrot
        rel = carrot - p
        hh = math.hypot(float(rel[0]), float(rel[1]))
        if hh > P.goal_max:
            rel[0] *= P.goal_max / hh
            rel[1] *= P.goal_max / hh
            self.goal_clips += 1
        rel[2] = P.goal_z if on_ground else clip(float(rel[2]), -P.goal_z, P.goal_z)
        return rel

    # ------------------------------------------------------------------ yaw and speed sense
    def _yaw(self, now: float, dt: float, psi_n: float, on_ground: bool) -> None:
        P = self.params
        look_des = 0.0
        if self.mode == 2 and self.g_s is not None:
            p = self.host_pos
            b = wrap(math.atan2(float(self.g_s[1] - p[1]), float(self.g_s[0] - p[0])) - self.psi)
            look_des = (math.copysign(1.0, b) * clip(abs(b) - P.look_free * DEG, 0.0, P.look_max * DEG)
                        * clip((self.d_gate - 4.0) / 4.0, 0.0, 1.0) * clip(1.0 - abs(self.kappa) / P.look_kappa, 0.0, 1.0))
        elif self.mode == 3:
            ts = now - self.mode_since
            look_des = P.sweep * DEG * clip(ts / P.sweep_ramp, 0.0, 1.0) * math.sin(TWO_PI * ts / P.sweep_period)
        self.look += (look_des - self.look) * min(1.0, dt / P.look_tau)
        look_rate = clip((self.look - self.look_prev) / dt, -1.0, 1.0) if dt > 0 else 0.0
        self.look_prev = self.look
        self.yaw_ref = wrap(self.psi + self.look)
        om = getattr(self.host, 'omega', None)
        if om is not None and dt > 0:
            rate = float((self.host.last_R @ np.asarray(om, dtype=np.float64))[2])     # world heading rate
            self.psi_dot += (rate - self.psi_dot) * min(1.0, dt / P.yaw_rate_tau)
        r_ref = self.v * self.kappa + look_rate
        e = wrap(self.yaw_ref - psi_n) + P.yaw_lead * (r_ref - self.psi_dot)
        db = P.yaw_db * DEG
        e_s = e * e * e / (e * e + db * db)
        r_cmd = clip(r_ref + P.yaw_gain * e_s, -P.yaw_rate_cap, P.yaw_rate_cap)
        y_des = clip(-r_cmd / P.yaw_rate, -P.yaw_max, P.yaw_max)
        self.sight_yaw += clip(y_des - self.sight_yaw, -P.yaw_slew * dt, P.yaw_slew * dt)
        if on_ground:
            self.sight_yaw = 0.0

    def _flow(self, now: float, dt: float, vel: np.ndarray) -> None:
        P = self.params
        lo, hi = P.flow_bounds()
        f_ff = P.flow_ref / max(self.v_nom, P.flow_ref)
        if self.t_air is not None and now - self.t_air > 3.0 and self.v_nom > 1.0 and abs(self.v_des - self.v_nom) < 0.2:
            vh = math.hypot(float(vel[0]), float(vel[1]))
            trim = self.flow_trim + dt * P.flow_ki * (vh - self.v_nom) / self.v_nom
            # anti-windup: no trim beyond what the clipped output can use (a search below flow_ref pinned it at 1.3)
            trim = clip(trim, lo / f_ff, hi / f_ff)
            self.flow_trim = clip(trim, *P.flow_trim_range)
        self.flow_f += (clip(f_ff * self.flow_trim, lo, hi) - self.flow_f) * min(1.0, dt / P.flow_tau)
        self.host.flow_gain = self.flow_f

    def _status(self, now: float) -> None:
        T = self.target
        if self.mode == 2 and T is not None:
            seen = now - T.t_last < 0.3
            s = (f'gate {"seen" if seen else "remembered"} {self.d_gate:.1f} m; target {T.id} hits {T.hits} '
                 f'sd_lat {self.sd_lat:.2f}; axis {math.degrees(self.n_ang):.0f}{" turn" if self.turn_gate else ""}')
        elif self.mode == 1:
            s = 'flying on'
        elif self.mode == 3:
            s = 'no gate: searching'
        elif self.mode == 4:
            s = 'holding'
        else:
            s = 'on the ground'
        if self.stalled:
            gap = now - self.t_frame
            s = (f'DETECTOR STALLED ({gap:.1f} s without a fresh frame)' if math.isfinite(gap)
                 else 'DETECTOR STALLED (no frame yet)') + f'; {s}'
        self.host.vision_status = (f'{s}; rabbit v {self.v:.1f}/{self.v_nom:.1f} lead {self.lead_now:.1f}; '
                                   f'passes {self.n_passes}; flow {self.flow_f:.2f}; '
                                   f'arch {self.gate_w:.2f} m +-{100 * self.gate_w_sd:.0f}% ({self.w_updates})')

    @property
    def host_pos(self) -> np.ndarray:
        return self._p

    # ------------------------------------------------------------------ log
    def log_values(self, det=None, offset=(0.0, 0.0, 0.0)) -> list[float]:
        """Numeric values of LOG_COLUMNS (offset shifts world positions, e.g. to the rehearsal's gate frame)."""
        nan = math.nan
        ox, oy, oz = (float(x) for x in offset)
        du, dv, dtt = ((float(det.u), float(det.v), float(det.t)) if det is not None and det.frames else (nan, nan, nan))
        T = self.target
        now = self.t_prev if self.t_prev is not None else nan
        if T is not None:
            tgt = [T.id, float(T.m[0]) + ox, float(T.m[1]) + oy, float(T.m[2]) + oz, self.sd_lat, T.hits,
                   now - T.t_last, math.degrees(self.n_ang) if self.n_ang is not None else nan,
                   self.next.id if self.next is not None else -1]
        else:
            tgt = [-1, nan, nan, nan, nan, nan, nan, nan, -1]
        n_conf = sum(1 for t in self.tracks if t.confirmed and not t.passed)
        n_tent = sum(1 for t in self.tracks if not t.confirmed)
        return [du, dv, dtt,
                self.c[0] + ox, self.c[1] + oy, self.z_c + oz, math.degrees(self.psi), self.kappa, self.v, self.v_nom,
                math.degrees(self.look), math.degrees(self.yaw_ref), self.sight_yaw,
                *tgt,
                n_conf, n_tent, self.mode, self.n_passes, self.pass_kind,
                self.flow_f, self.rej_elev, self.rej_stale, self.rej_offaxis, self.absorbed, self.low, self.reseeds,
                self.ghosts, self.orphans_registered, self.unpasses, self.behind, self.goal_clips, self.errors,
                self.gate_w, self.gate_w_sd, self.w_updates,
                now - self.t_frame if math.isfinite(self.t_frame) else nan]

    @staticmethod
    def empty_log_values(det=None) -> list[float]:
        nan = math.nan
        du, dv, dtt = ((float(det.u), float(det.v), float(det.t)) if det is not None and det.frames else (nan, nan, nan))
        return [du, dv, dtt] + [nan] * (len(LOG_COLUMNS) - 3)


def smoke_test(params: SightParams, seconds: float = 14.0) -> int:
    """Fly the pilot on a kinematic stand-in for ``seconds`` (take-off, an arch ahead, its pass, the search, and a
    detector stall at the end) and let any exception through: bad parameters fail here instead of in the air. Returns
    the number of ticks flown."""
    from types import SimpleNamespace
    dt, arch = 0.01, np.array([14.0, 0.0, 2.7])

    class Host:
        def __init__(self):
            self.t, self.yaw = 1000.0, 0.0
            self.pos, self.last_vel, self.omega, self.last_R = np.zeros(3), np.zeros(3), np.zeros(3), np.eye(3)
            self.flow_gain, self.vision_gate_w, self.vision_passed_t, self.vision_status = 1.0, None, None, ''
            self._passed = []
            self.vision = SimpleNamespace(cam=None, get=lambda: self.det)
            self.det = SimpleNamespace(t=0.0, p_visible=0.0, u=160.0, v=90.0, width_px=30.0,
                                       direction_body=np.array([1.0, 0.0, 0.0]), dist_m=0.0, frames=0)

        def clock(self):
            return self.t

        def pose_at(self, t):
            return (self.pos.copy(), self.last_R.copy()) if abs(t - self.t) < 0.5 else None

    h = Host()
    sp = SightPilot(h, params)
    n = int(seconds / dt)
    for k in range(n):
        if k % 7 == 0 and k < n - 250:                   # 14 Hz frames, then the detector stalls
            rel = h.last_R.T @ (arch - h.pos)
            d = float(np.linalg.norm(rel))
            seen = rel[0] > 0.3 * d and d > 1.0
            h.det = SimpleNamespace(t=h.t - 0.05, p_visible=0.99 if seen else 0.05, u=160.0, v=90.0,
                                    width_px=max(4.0, 320.0 * 4.0 / max(d, 1.0) / 2.0),
                                    direction_body=rel / max(d, 1e-9), dist_m=d, frames=h.det.frames + 1)
        rel_b = sp._goal(h.pos.copy())
        if not (np.all(np.isfinite(rel_b)) and math.isfinite(sp.sight_yaw) and math.isfinite(float(h.flow_gain))):
            raise FloatingPointError(f'non-finite output at tick {k}: goal {rel_b}, yaw {sp.sight_yaw}, '
                                     f'flow {h.flow_gain}')
        gw = h.last_R @ rel_b
        g = float(np.linalg.norm(gw))
        v_des = gw / g * min(g, 2.5) if g > 1e-6 else np.zeros(3)
        h.last_vel = h.last_vel + (v_des - h.last_vel) * (dt / 0.5)
        h.pos = h.pos + h.last_vel * dt
        h.pos[2] = max(h.pos[2], 0.0)
        wz = -sp.sight_yaw * params.yaw_rate
        h.omega = np.array([0.0, 0.0, wz])
        h.yaw += wz * dt
        c, si = math.cos(h.yaw), math.sin(h.yaw)
        h.last_R = np.array([[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]])
        h.t += dt
    return n


# ---------------------------------------------------------------------------------------------------- CLI helpers

def add_cli_args(q, yaw_rate_default: float = 2.3) -> None:
    """The --sight* flags, shared by `liftoff fly` and `vision rehearse`."""
    g = q.add_argument_group('by-sight pilot (--sight rabbit)')
    g.add_argument('--sight', choices=['legacy', 'rabbit'], default='legacy',
                   help='by-sight pilot: legacy (remembered gate + carrot line) or rabbit (tracks + virtual lead vehicle)')
    g.add_argument('--sight-speed', type=float, default=2.5, help='rabbit cruise speed V_CRUISE (m/s); above flow_ref '
                   'the speed sense is scaled by flow_ref/speed (4 -> about 0.6)')
    g.add_argument('--sight-gate-speed', type=float, default=3.0, help='speed through a straight gate (m/s)')
    g.add_argument('--sight-turn-gate-speed', type=float, default=2.5, help='speed through a gate on a turn (m/s)')
    g.add_argument('--sight-lead', type=float, default=3.0, help='rabbit lead ahead of the drone at the gate (m)')
    g.add_argument('--sight-a-lat', type=float, default=2.0, help='rabbit lateral acceleration limit (m/s^2); with '
                   'kappa_max it sets the speed floor sqrt(a_lat / kappa_max) on a saturated turn')
    g.add_argument('--sight-a-brk', type=float, default=1.2, help='rabbit braking limit (m/s^2)')
    g.add_argument('--sight-yaw-rate', type=float, default=yaw_rate_default,
                   help='yaw rate per unit yaw stick (rad/s; Liftoff 2.3, the simulator 3.8)')
    g.add_argument('--sight-yaw-gain', type=float, default=1.5, help='heading error gain (1/s)')
    g.add_argument('--sight-yaw-lead', type=float, default=0.15,
                   help='phase lead on the heading error from the measured heading rate (s)')
    g.add_argument('--sight-yaw-max', type=float, default=0.35, help='yaw stick cap')
    g.add_argument('--sight-flow-ref', type=float, default=2.45, help='sensed cruise speed the brain holds (m/s)')
    g.add_argument('--sight-flow-min', type=float, default=None,
                   help='lowest speed-sense gain (default min(0.7, 0.9 flow_ref / max(speed, gate speed))); '
                        'the highest is --flow-gain')
    g.add_argument('--sight-flow', choices=['global', 'along'], default='global',
                   help='scale the whole horizontal speed sense, or only its part along the rabbit heading')
    g.add_argument('--sight-flow-alt', choices=['start', 'ground'], default='start',
                   help='optic-flow altitude: above the reset point (as trained), or above the ground under the '
                        'altitude reference (hills)')
    g.add_argument('--sight-search-side', choices=['left', 'right'], default='left',
                   help='search turn before the course has turned')
    g.add_argument('--sight-search-radius', type=float, default=8.0, help='search circle radius (m)')
    g.add_argument('--sight-elev', default='-40,35',
                   help='world elevation window of sightings, deg (write --sight-elev=-40,35)')
    g.add_argument('--vision-lag', type=float, default=0.0, help='s subtracted from the detection grab time')
    g.add_argument('--sight-range-corr', default='default',
                   help="range correction table 'd:k,d:k,...', 'default' (measured on GateNet), 'spec' (the prototype's) "
                        "or 'none'")
    g.add_argument('--sight-z-aim', type=float, default=0.0, help='fly this far above the passage point (m)')
    g.add_argument('--sight-climb-front', type=float, default=0.5, help='climbs finish by this fraction of the time to go')
    g.add_argument('--sight-snap-start', type=float, default=12.0,
                   help='pull the goal onto the gate bearing within this distance (m)')
    g.add_argument('--sight-turn-hints', default='', help='heading change at each gate, deg (+ left), e.g. 5,40,90')
    g.add_argument('--sight-set', action='append', default=[], metavar='NAME=VALUE',
                   help='override any SightParams field, e.g. --sight-set v_blind=2.5 --sight-set lead_open=1')


def params_from_args(a, flow_gain: float = 1.0) -> SightParams:
    import yaml
    P = SightParams(v_cruise=a.sight_speed, v_gate=a.sight_gate_speed, v_gate_turn=a.sight_turn_gate_speed,
                    lead=a.sight_lead, a_lat=a.sight_a_lat, a_brk=a.sight_a_brk, yaw_rate=a.sight_yaw_rate,
                    yaw_gain=a.sight_yaw_gain, yaw_lead=a.sight_yaw_lead, yaw_max=a.sight_yaw_max, flow_ref=a.sight_flow_ref,
                    flow_min=a.sight_flow_min, flow_max=flow_gain, flow_mode=a.sight_flow, flow_alt=a.sight_flow_alt,
                    search_side=1.0 if a.sight_search_side == 'left' else -1.0, search_radius=a.sight_search_radius,
                    vision_lag=a.vision_lag, z_aim=a.sight_z_aim, climb_front=a.sight_climb_front,
                    snap_start=a.sight_snap_start)
    P.search_radius_wide = 1.5 * P.search_radius
    lo, hi = (float(x) for x in str(a.sight_elev).split(','))
    P.elev_deg = (lo, hi)
    rc = str(a.sight_range_corr).strip().lower()
    if rc == 'none':
        P.range_corr = None
    elif rc == 'spec':
        P.range_corr = RANGE_CORR_SPEC
    elif rc != 'default':
        P.range_corr = tuple(tuple(float(y) for y in x.split(':')) for x in rc.split(','))
    if a.sight_turn_hints:
        P.turn_hints = tuple(float(x) for x in a.sight_turn_hints.split(','))
    names = {f.name: f for f in fields(SightParams)}
    for kv in a.sight_set:
        k, _, v = kv.partition('=')
        k = k.strip()
        if k not in names:
            raise SystemExit(f'--sight-set: SightParams has no field {k!r}')
        val = yaml.safe_load(v)
        if isinstance(val, list):
            val = tuple(tuple(x) if isinstance(x, list) else x for x in val)
        setattr(P, k, val)
    try:
        P.validate()
        smoke_test(P)                     # the pilot would swallow its errors in the air (and fly on a stale goal)
    except Exception as e:                # noqa: BLE001
        raise SystemExit(f'--sight rabbit: these parameters do not fly: {type(e).__name__}: {e}') from e
    return P
