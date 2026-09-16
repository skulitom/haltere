"""The odd-course bench: the synthetic courses are the shape the rehearsal reads, and each really does
break the Straw Bale assumption it is named for."""
import json
import math
from pathlib import Path

import numpy as np

from haltere.vision.gates import GATE_WIDTH_M
from haltere.vision.oddcourse import BUILDERS, build

STRAWBALE = Path(__file__).resolve().parents[1] / 'configs' / 'gates_strawbale.json'


def _xy(course):
    return np.array([g['pos'][:2] for g in course.gates], dtype=float)


def _leg_dirs(course):
    """Direction of every leg in degrees, the spawn counted as the start of the first one."""
    P = np.vstack([np.asarray(course.start[:2], dtype=float), _xy(course)])
    return [math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) for a, b in zip(P[:-1], P[1:])]


def _turns(course):
    d = _leg_dirs(course)
    return [(b - a + 180.0) % 360.0 - 180.0 for a, b in zip(d, d[1:])]


def test_every_course_has_the_shape_load_gate_file_returns():
    for name in BUILDERS:
        c = build(name)
        d = c.as_gate_file()
        assert set(d) == {'gate_width_m', 'centre_up_m', 'gates'}, name
        assert d['gates'] is c.gates and 6 <= len(c.gates) <= 8, name
        assert d['gate_width_m'] > 0 and d['centre_up_m'] == 1.5, name
        for i, g in enumerate(c.gates):
            assert set(g) >= {'pos', 'heading', 'gate'} and g['gate'] == i, (name, i)
            assert len(g['pos']) == 3 and all(isinstance(x, float) and math.isfinite(x) for x in g['pos']), (name, i)
            assert isinstance(g['heading'], float) and -math.pi <= g['heading'] <= math.pi, (name, i)
            assert g['pos'][2] >= 1.0, (name, i)                       # flyable: no gate in the ground
        assert len(c.start) == 4 and 150.0 <= c.budget_s <= 220.0, name
        assert c.breaks, name
        # the whole course is reachable: no leg longer than the leash of the search that has to find it blind
        assert all(leg > 3.0 for leg in c.legs_m), name
    assert len(BUILDERS) == 10


def test_home_is_the_straw_bale_course_and_only_climbs():
    c = build('home')
    z = [g['pos'][2] for g in c.gates]
    assert len(c.gates) == 7 and c.width_m == GATE_WIDTH_M
    assert all(b >= a for a, b in zip(z, z[1:])) and z[0] < 1.5 < 12.0 < z[-1]
    assert all(15.0 <= leg <= 40.0 for leg in c.legs_m)
    assert abs(_leg_dirs(c)[0]) < 5.0                                  # the first gate is straight ahead of the spawn
    if STRAWBALE.exists():
        f = json.loads(STRAWBALE.read_text(encoding='utf-8'))
        assert len(f['gates']) == len(c.gates) and f['gate_width_m'] == c.width_m
        for a, b in zip(c.gates, f['gates']):
            assert np.allclose(a['pos'], b['pos'], atol=1e-3)
            assert abs(a['heading'] - b['heading']) < 1e-6


def test_narrow_and_wide_change_only_the_width_and_are_warned_about():
    home, narrow, wide = build('home'), build('narrow'), build('wide')
    assert (narrow.width_m, wide.width_m) == (1.5, 8.0)
    assert narrow.gates == home.gates and wide.gates == home.gates
    assert home.width_warning() == ''
    for c in (narrow, wide):
        w = c.width_warning()
        assert 'GATE_WIDTH_M' in w and 'detection_geometry' in w and str(c.width_m) in w
        assert f'{GATE_WIDTH_M / c.width_m:.2f}x' in w                 # the raw range error, spelled out
        assert 'width_est' in w and 'through_w' in w                   # what takes it out, and what to read


def test_descending_falls_far_enough_to_matter():
    c = build('descending')
    z = [g['pos'][2] for g in c.gates]
    assert all(b < a for a, b in zip(z, z[1:-1])) and z[0] >= 9.0 and z[-1] <= 1.2
    # every drop but the last two is more than 1.2 m: only then does the next arch's visual centre
    # (passage + CENTRE_UP_M) sit below the last passage height, which is what the pilot's height filter reads
    assert sum(1 for a, b in zip(z, z[1:]) if a - b > 1.2) >= 3
    assert _xy(c).tolist() == _xy(build('home')).tolist()              # only the heights differ from home


def test_mixed_height_goes_up_and_down_repeatedly():
    c = build('mixed_height')
    z = [g['pos'][2] for g in c.gates]
    assert z == [1.5, 9.0, 3.0, 12.0, 2.0, 8.0, 4.0]
    steps = [b - a for a, b in zip(z, z[1:])]
    assert sum(1 for s in steps if s > 0) >= 3 and sum(1 for s in steps if s < 0) >= 3
    assert all(abs(a) > 1.2 and abs(b) > 1.2 and a * b < 0 for a, b in zip(steps, steps[1:]))


def test_gate_pair_has_two_pairs_6_m_apart_on_one_line():
    c = build('gate_pair')
    pairs = []
    for i, (a, b) in enumerate(zip(c.gates, c.gates[1:])):
        off = np.asarray(b['pos'][:2]) - np.asarray(a['pos'][:2])
        d = float(np.linalg.norm(off))
        if d > 10.0:
            continue
        n = np.array([math.cos(a['heading']), math.sin(a['heading'])])
        lat = abs(float(off[0] * -n[1] + off[1] * n[0]))
        pairs.append((i, d, lat, abs(b['heading'] - a['heading'])))
    assert len(pairs) == 2, pairs
    for i, d, lat, dh in pairs:
        assert abs(d - 6.0) < 0.05 and lat < 0.05 and dh < 1e-6        # 6 m apart, on the line, same facing
    assert [i for i, *_ in pairs] == [1, 4]
    assert all(leg > 20.0 for k, leg in enumerate(c.legs_m) if k not in (1, 4))


def test_long_legs_are_out_of_sight_of_each_other():
    c = build('long_legs')
    assert all(80.0 <= leg <= 100.0 for leg in c.legs_m), c.legs_m
    # an arch this far away is under the detector's 11 px floor (f = 100 px at the network's 320 px input)
    assert min(c.legs_m) > 100.0 * c.width_m / 11.0
    assert c.seconds == 150.0 and c.budget_s == 150.0          # a fixed budget: it is the first leg that decides


def test_hairpin_reverses_by_more_than_150_degrees():
    d = _leg_dirs(build('hairpin'))
    rev = max(abs((b - a + 180.0) % 360.0 - 180.0) for i, a in enumerate(d) for b in d[i + 1:])
    assert rev > 150.0, rev
    assert max(abs(t) for t in _turns(build('hairpin'))) > 85.0        # and a single gate turns more than any on home
    assert max(abs(t) for t in _turns(build('home'))) < 90.0


def test_clockwise_turns_the_other_way():
    home, cw = build('home'), build('clockwise')
    th, tc = _turns(home), _turns(cw)
    assert all(abs(a + b) < 1e-6 for a, b in zip(th, tc))
    assert sum(th) > 90.0 and sum(tc) < -90.0          # home turns anticlockwise on balance, the mirror clockwise
    assert np.allclose(_xy(cw)[:, 1], -_xy(home)[:, 1])
    assert np.allclose([g['pos'][2] for g in cw.gates], [g['pos'][2] for g in home.gates])


def test_offaxis_start_puts_the_first_gate_beside_the_spawn():
    c = build('offaxis_start')
    assert c.start[3] == 90.0 and c.gates == build('home').gates
    bearing = math.degrees(math.atan2(c.gates[0]['pos'][1] - c.start[1], c.gates[0]['pos'][0] - c.start[0]))
    off = abs((bearing - c.start[3] + 180.0) % 360.0 - 180.0)
    assert 80.0 < off < 100.0, off                                     # outside the 116 deg camera's half view of 58 deg
    assert off > 58.0


def test_unknown_course_is_refused():
    try:
        build('nope')
    except KeyError as e:
        assert 'home' in str(e)
    else:
        raise AssertionError('an unknown course name must raise')
