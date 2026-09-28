"""Round 5 (arches): the looming governor's cap follows the ray of its evidence (configs/obstacles/stale_evidence.json,
`ClearanceRayConfig`) and the ring-marker reader rule (configs/pilot/ring_marker.json, `checkpoint_ring(annulus=...)`).

The governor tests replay the shape of minus-fast6-r4b-01 (a stand-off at a first wall, then short-TTC samples along a
ray 180 deg away, the wall behind the next arch); the reader tests paste a recorded patch of a checkered leg whose dark check passes
the earlier reader's hole tests (the right leg of the second Straw Bale start arch on straw-brain11cw13-r4b-noassist-02).
Development cases only; none of this is flight evidence.
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from haltere.liftoff import fast_race_cue as frc
from haltere.liftoff.camera_pose import CameraPoseHistory
from haltere.liftoff.fast_race_cue import (ClearanceRayConfig, FastRaceCue, TtcClearanceConfig, TtcClearanceGovernor,
                                           stale_evidence_configs)
from haltere.vision import race_cues
from haltere.vision.race_cues import checkpoint_ring, ring_marker_rule
from tests.test_visual_assistance import SENSOR, senses

ROOT = Path(__file__).resolve().parents[1]
RAY = ClearanceRayConfig()
A = np.array([np.cos(np.radians(28.)), np.sin(np.radians(28.)), 0.])      # the first wall's ray (minus r4b-01)
B = np.array([np.cos(np.radians(-155.)), np.sin(np.radians(-155.)), 0.])  # the travel ray toward the second wall


def canonical(obj):
    import hashlib
    body = {k: v for k, v in obj.items() if k not in ('frozen', 'frozen_at', 'sha256')}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
                          .encode('utf-8')).hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# Declarations
# ----------------------------------------------------------------------------------------------------------------
def test_declarations_are_frozen_hashed_and_loaded_by_the_runner():
    from haltere.liftoff.visual_brain import load_ring_marker, load_stale_evidence
    stale = json.loads((ROOT/'configs'/'obstacles'/'stale_evidence.json').read_text(encoding='utf-8'))
    ring = json.loads((ROOT/'configs'/'pilot'/'ring_marker.json').read_text(encoding='utf-8'))
    for obj in (stale, ring):
        assert obj['frozen'] is True and obj['sha256'] == canonical(obj) and obj['version'] == 1
    declaration, digest = load_stale_evidence()
    assert digest == stale['sha256']
    assert stale_evidence_configs(declaration) == dict(clearance_ray=ClearanceRayConfig(stale_deg=60.))
    declaration, digest = load_ring_marker()
    assert digest == ring['sha256']
    assert ring_marker_rule(declaration) == dict(min_white=.9, offsets_px=(1., 2.), samples=32)


def test_other_versions_and_edited_declarations_are_refused(tmp_path):
    from haltere.liftoff.visual_brain import load_ring_marker, load_stale_evidence
    for name, loader in (('obstacles/stale_evidence.json', load_stale_evidence),
                         ('pilot/ring_marker.json', load_ring_marker)):
        obj = json.loads((ROOT/'configs'/name).read_text(encoding='utf-8'))
        edited = dict(obj, note=obj['note']+' edited')
        path = tmp_path/'edited.json'
        path.write_text(json.dumps(edited), encoding='utf-8')
        with pytest.raises(ValueError, match='changed after the freeze'):
            loader(path)
        other = dict(obj, version=2)
        other['sha256'] = canonical(other)
        path.write_text(json.dumps(other), encoding='utf-8')
        with pytest.raises(ValueError, match='version 2'):
            loader(path)
    with pytest.raises(ValueError):
        stale_evidence_configs(dict(version=2, clearance_ray=dict(stale_deg=60.)))
    with pytest.raises(ValueError):
        stale_evidence_configs(dict(version=1))
    for bad in (dict(min_white=0., offsets_px=[1.], samples=32), dict(min_white=.9, offsets_px=[], samples=32),
                dict(min_white=.9, offsets_px=[1.], samples=4), dict(min_white=.9, offsets_px=[9.], samples=32),
                dict(min_white=.9, offsets_px=[1.])):
        with pytest.raises(ValueError):
            ring_marker_rule(dict(version=1, annulus=bad))


def test_config_validation():
    for bad in (0., 180., float('nan'), -5.):
        with pytest.raises(ValueError):
            ClearanceRayConfig(stale_deg=bad)
    with pytest.raises(ValueError, match='ClearanceRayConfig'):
        TtcClearanceGovernor(ray=dict(stale_deg=60.))
    with pytest.raises(ValueError, match='ClearanceRayConfig'):
        FastRaceCue(SENSOR, CameraPoseHistory(), clearance_ray=dict(stale_deg=60.))


# ----------------------------------------------------------------------------------------------------------------
# Governor
# ----------------------------------------------------------------------------------------------------------------
def stale_standoff_stream(gov, *, speed=6., standoff_at=0., switch_at=3., seconds=5.2, second_wall=True):
    """A stand-off at a wall along A (a slow closing speed, TTC well below 1 s), then flight along B at `speed` with a
    wall ahead along B from `switch_at` (TTC falling 1.1 -> 0.35 s): minus-fast6-r4b-01 in miniature. Returns per-tick
    (now, cap, ray, status, cap along B binds the request along B)."""
    rows, pending, frame = [], [], 1/10.
    next_frame = 0.
    for k in range(int(round(seconds/.01))):
        now = k*.01
        if now < 1.:
            pos, vel, ray, ttc, closing = np.zeros(3), A*.8, A, .25, .8           # creeping at the first wall
        elif now < switch_at:
            pos, vel, ray, ttc, closing = B*speed*(now-1.), B*speed, B, None, speed   # flying away: no evidence
        else:
            ttc = max(.35, 1.1-(now-switch_at)*.9) if second_wall else None
            pos, vel, ray, closing = B*speed*(now-1.), B*speed, B, speed
        if now >= next_frame-1e-9:
            pending.append((now+.08, now, pos.copy(), ray.copy(), ttc, closing))
            next_frame += frame
        while pending and pending[0][0] <= now+1e-9:
            _, t, p, r, s_ttc, c = pending.pop(0)
            gov.ingest(t, s_ttc, None if s_ttc is None else s_ttc*c, None, p, r, c, received=now)
        cap, cray, _ = gov.limits(pos, vel, now, .01, 3.5)
        binds = cap is not None and cray is not None and float(np.asarray(cray) @ B) > .5 and cap < speed-.05
        rows.append((now, cap, None if cray is None else np.array(cray), gov.status, binds))
    return rows


def test_a_stale_stand_off_ignores_the_wall_ahead_without_the_rule_and_brakes_for_it_with_it():
    old = stale_standoff_stream(TtcClearanceGovernor(TtcClearanceConfig()))
    new = stale_standoff_stream(TtcClearanceGovernor(TtcClearanceConfig(), ray=RAY))
    # both hold the stand-off at the first wall
    assert any(r[3] == 'standoff' for r in old if r[0] < 1.5) and any(r[3] == 'standoff' for r in new if r[0] < 1.5)
    # without the rule the cap stays on the first wall's ray and never bounds the flight toward the second wall
    late = [r for r in old if r[0] >= 3.]
    assert all(r[2] is not None and float(r[2] @ A) > .99 for r in late)
    assert not any(r[4] for r in late)
    # with the rule the cap re-seats on the travel ray once three samples below ttc_on (0.8 s; from 3.33 s on) have
    # arrived (at 10 Hz, 0.08 s late), and binds there
    reseat = next(r[0] for r in new if r[0] >= 3. and r[2] is not None and float(r[2] @ B) > .99)
    assert 3.33 < reseat <= 3.33+.4
    assert any(r[4] for r in new if r[0] >= reseat)
    assert min(r[1] for r in new if r[0] >= reseat) <= .75*6.


def test_the_old_stand_off_is_renewed_by_off_ray_samples_only_without_the_rule():
    old = TtcClearanceGovernor(TtcClearanceConfig())
    new = TtcClearanceGovernor(TtcClearanceConfig(), ray=RAY)
    stale_standoff_stream(old, seconds=4.6)
    stale_standoff_stream(new, seconds=4.6)
    assert old.standoff_until > 4.6                 # renewed by the samples along B (the live fault)
    assert new.standoff_until < 4.6                 # ended by the re-seat
    assert new.counts['reseats'] >= 1 and new.counts['stale_ray_samples'] >= 1
    assert 'reseats' not in old.counts and 'stale_ray_samples' not in old.counts


def random_stream(gov, seed, *, turn_deg=40.):
    """Random wall/terrain samples along a ray that turns by at most turn_deg (never stale at 60 deg)."""
    rng = np.random.default_rng(seed)
    out = []
    heading = 0.
    for k in range(600):
        now = k*.01
        heading = np.radians(turn_deg)*np.sin(now/2.)
        ray = np.array([np.cos(heading), np.sin(heading), 0.])
        speed = 5.+rng.normal()*.3
        pos = ray*speed*now
        if k % 5 == 0:
            ttc = None if rng.random() < .3 else float(rng.uniform(.2, 2.))
            below = None if rng.random() < .5 else float(rng.uniform(0., 1.))
            lower = None if rng.random() < .5 else float(rng.uniform(.2, 2.))
            gov.ingest(now-.05, ttc, None if ttc is None else ttc*speed, below, pos, ray, speed, received=now,
                       ttc_lower=lower)
        cap, cray, climb = gov.limits(pos, ray*speed, now, .01, 3.5)
        out.append((cap, None if cray is None else tuple(cray), climb, gov.status, gov.standoff_until))
    return out


@pytest.mark.parametrize('seed', range(6))
def test_the_rule_changes_nothing_while_every_sample_lies_within_stale_deg_of_the_cap_ray(seed):
    assert random_stream(TtcClearanceGovernor(TtcClearanceConfig()), seed) == \
        random_stream(TtcClearanceGovernor(TtcClearanceConfig(), ray=RAY), seed)


def pilot(**kw):
    history = CameraPoseHistory()
    return FastRaceCue(SENSOR, history, 6., reference_speed=6., **kw), history


def drive_pilot(p, history, seconds=5.2):
    """The stale stand-off stream through the whole pilot (a ring cue straight ahead of the travel ray)."""
    from tests.test_fast_race_cue import cue_toward
    out, pending = [], []
    next_frame = 0.
    for k in range(int(round(seconds/.01))):
        now = 10.+k*.01
        t = now-10.
        if t < 1.:
            pos, vel, ray, ttc = np.array([0., 0., 1.]), A*.8, A, .25
        else:
            pos, vel, ray = np.array([0., 0., 1.])+B*6.*(t-1.), B*6., B
            ttc = None if t < 3. else max(.35, 1.1-(t-3.)*.9)
        yaw = float(np.arctan2(ray[1], ray[0]))
        q = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
        history.append(now, pos, q)
        clearance = None
        if now >= next_frame-1e-9:
            pending.append((now+.08, now, ttc, float(np.linalg.norm(vel))))
            next_frame = now+.1
        while pending and pending[0][0] <= now+1e-9:
            _, tc, s_ttc, speed = pending.pop(0)
            clearance = dict(time=tc, ttc=s_ttc, distance=None if s_ttc is None else s_ttc*speed, below_fraction=None)
        s = senses(position=tuple(pos), velocity=tuple(vel), yaw=yaw)
        p.update(s, [0., 0., 0.], dict(race_cue=cue_toward([10., 0., 0.])), now, now,
                 **({} if clearance is None else dict(clearance=clearance)))
        out.append(p.velocity_command.copy())
    return np.array(out)


def test_shadow_flies_the_pilot_without_the_rule_bit_for_bit_and_logs_the_rule():
    plain, h0 = pilot()
    shadow, h1 = pilot(clearance_ray=RAY, stale_apply=False)
    applied, h2 = pilot(clearance_ray=RAY)
    a, b, c = drive_pilot(plain, h0), drive_pilot(shadow, h1), drive_pilot(applied, h2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert shadow.clearance.ray_rule is None and shadow.clearance_shadow.ray_rule == RAY
    assert shadow.clearance_shadow.counts['reseats'] >= 1 and applied.clearance.counts['reseats'] >= 1
    assert shadow.stale_log()['cap_reseat'] == shadow.clearance_shadow.counts['reseats']
    meta = shadow.metadata()['stale_evidence']
    assert meta['applied'] is False and meta['clearance_ray']['governor'] == 'shadow copy fed the same samples'
    assert 'stale_evidence' not in plain.metadata()
    assert np.isnan(plain.stale_log()['cap_reseat'])
    # with the rule applied the request along the travel ray is braked before the second wall
    speeds = c @ B
    assert speeds[-1] < a[-1] @ B-1.


def test_a_pilot_without_the_rule_builds_its_governor_exactly_as_before():
    p, h = pilot()
    drive_pilot(p, h, seconds=1.5)
    assert isinstance(p.clearance, TtcClearanceGovernor) and p.clearance.ray_rule is None
    assert p.clearance_shadow is None


def test_runner_log_columns_and_stack_wiring():
    from types import SimpleNamespace
    from haltere.liftoff.visual_brain import STALE_COLUMNS, stale_row
    assert STALE_COLUMNS == ('cap_ray_deg', 'cap_reseat')
    assert all(np.isnan(v) for v in stale_row(SimpleNamespace()))
    p, h = pilot(clearance_ray=RAY)
    drive_pilot(p, h)
    ray_deg, reseats = stale_row(p)
    assert ray_deg == pytest.approx(-155., abs=1.) and reseats >= 1


def test_the_obstacle_stack_passes_the_rule_to_the_pilot(monkeypatch):
    """The VisualController hands the declared rule and the stack's apply flag to FastRaceCue (no camera, no pad)."""
    from haltere.liftoff import visual_brain as vb
    seen = {}

    class Probe(FastRaceCue):
        def __init__(self, *a, **kw):
            seen.update(kw)
            super().__init__(*a, **kw)
    monkeypatch.setattr(frc, 'FastRaceCue', Probe)
    checkpoint = ROOT.parent.parent.parent/'runs'/'fast-brain-11-b-cw13'/'candidate.pt'
    if not checkpoint.exists():
        checkpoint = Path('C:/DEV/Haltere/runs/fast-brain-11-b-cw13/candidate.pt')
    mapping = Path('C:/DEV/Haltere/runs/pine-route-collection-01/liftoff-original-drone.yaml')
    if not checkpoint.exists() or not mapping.exists():
        pytest.skip('needs the local brain checkpoint and pad mapping')
    for apply in (True, False):
        seen.clear()
        vb.VisualController(str(checkpoint), str(mapping), 'cpu', pilot_assistance='race-cue', pilot_profile='fast',
                            assist_speed=6., motor_controller='brain', stale_evidence=str(vb.STALE_EVIDENCE_DECLARATION),
                            stale_apply=apply)
        assert seen['clearance_ray'] == RAY and seen['stale_apply'] is apply


# ----------------------------------------------------------------------------------------------------------------
# Ring-marker reader
# ----------------------------------------------------------------------------------------------------------------
RULE = dict(min_white=.9, offsets_px=(1., 2.), samples=32)


def marker(rgb, centre, outer=7.5, inner=3.6):
    """A HUD marker with the proportions measured on recorded frames (outer contour 15-16 px, hole contour ~60 px^2;
    the reader tests' 7/4 px drawing has a 1.5 px band inside the hole contour's radius, thinner than the game's)."""
    c = (int(centre[0]*16), int(centre[1]*16))
    cv2.circle(rgb, c, int(outer*16), (255, 255, 255), -1, shift=4)
    cv2.circle(rgb, c, int(inner*16), (20, 40, 25), -1, shift=4)
    return rgb


LEG_CHECK = ROOT/'tests'/'data'/'straw_start_arch_leg_check.png'


def leg_check(rgb, x0=700, y0=500):
    """The 48 x 48 px patch of the recorded frame of straw-brain11cw13-r4b-noassist-02 at 111.39 s (video time) centred
    on the dark check of the second start arch's right leg that the earlier reader takes for the marker (the live reader
    logged it at u 0.605, v 0.78 at 111.34-111.41 s), pasted into a frame."""
    patch = cv2.cvtColor(cv2.imread(str(LEG_CHECK)), cv2.COLOR_BGR2RGB)
    rgb[y0:y0+patch.shape[0], x0:x0+patch.shape[1]] = patch
    return rgb


def test_a_dark_check_of_a_checkered_leg_is_read_as_the_marker_only_without_the_rule():
    rgb = leg_check(np.zeros((720, 1280, 3), np.uint8))
    found = checkpoint_ring(rgb)
    assert found is not None and found['u'] == pytest.approx(725/1280, abs=.003)   # the earlier reader's false marker
    assert checkpoint_ring(rgb, annulus=RULE) is None
    # with the true marker in view as well: two hits (ambiguous, none) without the rule, the marker with it
    both = marker(rgb.copy(), (630, 600))
    assert checkpoint_ring(both) is None
    cue = checkpoint_ring(both, annulus=RULE)
    assert cue is not None and cue['u'] == pytest.approx(630/1280, abs=.002) and cue['v'] == pytest.approx(600/720)


@pytest.mark.parametrize('centre', [(640, 634), (19, 200), (640, 40), (1261, 360), (400, 700), (624, 83)])
def test_real_markers_keep_their_reading_with_the_rule(centre):
    rgb = marker(np.zeros((720, 1280, 3), np.uint8), centre)
    assert checkpoint_ring(rgb, annulus=RULE) == checkpoint_ring(rgb)
    assert checkpoint_ring(rgb) is not None


def test_a_marker_touching_a_white_arch_keeps_its_reading_with_the_rule():
    rgb = marker(np.zeros((720, 1280, 3), np.uint8), (640, 634))
    cv2.line(rgb, (633, 628), (610, 605), (255, 255, 255), 4)
    cv2.rectangle(rgb, (648, 560), (700, 700), (255, 255, 255), -1)   # a white leg touching the annulus
    marker(rgb, (640, 634))
    assert checkpoint_ring(rgb, annulus=RULE) is not None
    assert checkpoint_ring(rgb, annulus=RULE) == checkpoint_ring(rgb)


def test_without_the_rule_the_reader_is_the_m4b_reader_bit_for_bit(tmp_path):
    """The reader without a declared rule equals the m4b tree's reader on the reader tests' pictures and noise."""
    try:
        source = subprocess.run(['git', '-C', str(ROOT), 'show', 'm4b:haltere/vision/race_cues.py'],
                                capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip('needs the m4b branch in this repository')
    namespace = {}
    exec(compile(source, 'm4b_race_cues', 'exec'), namespace)
    old = namespace['checkpoint_ring']
    rng = np.random.default_rng(5)
    pictures = [marker(np.zeros((720, 1280, 3), np.uint8), (640, 634)),
                leg_check(np.zeros((720, 1280, 3), np.uint8)),
                leg_check(marker(np.zeros((720, 1280, 3), np.uint8), (300, 300)), 900, 100),
                marker(np.zeros((720, 1280, 3), np.uint8), (640, 634), 7., 4.)]
    for _ in range(4):
        noise = (rng.random((720, 1280, 3)) > .6).astype(np.uint8)*255
        pictures.append(marker(noise, (500, 500)))
    for rgb in pictures:
        assert checkpoint_ring(rgb) == old(rgb)


def test_the_camera_reader_argument():
    from haltere.liftoff.visual_brain import RING_MARKER_DECLARATION, race_cue_reader, resolve_ring_marker
    from types import SimpleNamespace
    assert race_cue_reader('none', None) is False
    assert race_cue_reader('race-cue', None) is True
    assert race_cue_reader('race-cue', str(RING_MARKER_DECLARATION)) == dict(annulus=RULE)
    assert resolve_ring_marker(SimpleNamespace(ring_marker=None, pilot_assistance='race-cue')) is None
    assert resolve_ring_marker(SimpleNamespace(ring_marker='off', pilot_assistance='race-cue')) is None
    assert resolve_ring_marker(SimpleNamespace(ring_marker='on', pilot_assistance='race-cue')) == \
        str(RING_MARKER_DECLARATION)
    with pytest.raises(ValueError, match='race-cue'):
        resolve_ring_marker(SimpleNamespace(ring_marker='on', pilot_assistance='none'))


def test_annulus_white_counts_only_in_image_points():
    mask = np.zeros((20, 20), np.uint8)
    mask[:, :10] = 255
    assert race_cues.annulus_white(mask, 0., 10., 3.) == pytest.approx(1.)
    assert np.isnan(race_cues.annulus_white(mask, -50., -50., 3.))
