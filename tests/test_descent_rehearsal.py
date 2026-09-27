"""The scoring-only terrain model of the offline rehearsal (haltere.liftoff.fast_rehearsal: hill_course, CourseTerrain,
DescentScore) and the batched descent rehearsal's course sets. Development tools only; nothing here is flight evidence.
"""
import numpy as np
import pytest

from haltere.liftoff.fast_rehearsal import (HIGH_PASS_M, CourseTerrain, DescentScore, TerrainConfig, hill_course,
                                            path_below_view, synthetic_course)
from haltere.vision.camera import Camera

CAMERA = Camera(320, 180, 100., 30.)
LEVEL = np.array([1., 0., 0., 0.])


def test_terrain_config_validation():
    for bad in (dict(clearance_min_m=0.), dict(clearance_min_m=3., clearance_max_m=2.), dict(crest_max=.9),
                dict(after_pass_s=-1.), dict(min_drop_m=float('nan'))):
        with pytest.raises(ValueError):
            TerrainConfig(**bad)


def test_hill_course_climbs_then_descends_and_is_seeded():
    a, b = hill_course(6100), hill_course(6100)
    assert np.array_equal(a, b) and len(a) == 8
    assert not np.array_equal(a, hill_course(6101))
    heights = np.r_[0., a[:, 2]]
    assert (np.diff(heights) < -2).sum() >= 2          # several descending legs
    assert a[:, 2].min() >= 1.5


def test_hills_lie_under_descending_legs_with_clearance_below_each_checkpoint():
    course = np.array([[20., 0., 10.], [50., 0., 2.], [80., 0., 2.5], [110., 0., 12.]])
    terrain = CourseTerrain(course, 7)
    assert set(terrain.legs) == {1}                      # only the 8 m drop; the first leg never gets a hill
    leg = terrain.legs[1]
    c = TerrainConfig()
    top, _ = terrain.leg_ground(1, [20., 0.])
    bottom, _ = terrain.leg_ground(1, [50., 0.])
    assert c.clearance_min_m <= 10.-top <= c.clearance_max_m
    assert c.clearance_min_m <= 2.-bottom <= c.clearance_max_m
    assert terrain.leg_ground(1, [0., 0.])[0] == top     # crest height behind the upper checkpoint
    assert terrain.leg_ground(1, [70., 0.])[0] == bottom  # toe height beyond the lower one
    heights = [terrain.leg_ground(1, [x, 0.])[0] for x in np.linspace(20., 50., 61)]
    assert np.all(np.diff(heights) <= 1e-12)             # monotone down the hill
    assert terrain.leg_ground(1, [35., 30.])[0] == terrain.leg_ground(1, [35., 0.])[0]   # a broad hillside
    crest = terrain.leg_ground(1, [20.+leg['crest']*30.*.99, 0.])[0]
    assert crest == pytest.approx(top)                   # flat over the crest fraction
    assert terrain.leg_ground(0, [10., 0.]) is None and terrain.leg_ground(2, [60., 0.]) is None


def test_the_active_hill_follows_the_target_and_the_pass():
    course = np.array([[20., 0., 10.], [50., 0., 2.], [80., 0., 2.5]])
    terrain = CourseTerrain(course, 3)
    assert terrain.ground(0, None, 0., [30., 0.]) is None          # flying to checkpoint 0: no hill
    assert terrain.ground(1, 5., 6., [30., 0.])[2] == 1            # flying leg 1
    after = terrain.ground(2, 10., 10.5, [52., 0.])                # just passed checkpoint 1: its hill remains
    assert after is not None and after[2] == 1
    assert terrain.ground(2, 10., 11.5, [60., 0.]) is None         # after_pass_s later it is gone


def test_path_below_view_uses_the_camera_lower_edge():
    # level attitude: the lower image edge lies 12 degrees below the horizon (30 up, 42 half field of view)
    for angle, below in ((-10., False), (-14., True), (5., False), (-89., True)):
        v = np.array([np.cos(np.radians(angle)), 0., np.sin(np.radians(angle))])*5.
        assert path_below_view(CAMERA, v, LEVEL) is below, angle
    # nose 10 degrees down (pitch about body y): the edge drops to -22 degrees
    pitch = np.radians(10.)
    q = np.array([np.cos(pitch/2), 0., np.sin(pitch/2), 0.])
    v = np.array([np.cos(np.radians(-18.)), 0., np.sin(np.radians(-18.))])*5.
    assert path_below_view(CAMERA, v, q) is False and path_below_view(CAMERA, v, LEVEL) is True


def test_descent_score_counts_contacts_view_and_high_passes():
    course = np.array([[20., 0., 10.], [50., 0., 2.]])
    terrain = CourseTerrain(course, 3)
    score = DescentScore(course, CAMERA, terrain)
    score.passed(0, 1., np.array([20., 0., 10.]))
    ground = lambda x: terrain.leg_ground(1, [x, 0.])[0]
    dt = .01
    down = np.array([5., 0., -2.5])                  # a 27 degree descent: below the level camera's view
    t = 1.
    for x in np.arange(20., 30., .05):
        z = ground(x)+(.5 if x < 24 or x > 26 else -.2)   # one contact between x 24 and 26
        score.step(t, dt, 1, np.array([x, 0., z]), down, LEVEL)
        t += dt
    t += 1.
    for x in np.arange(30., 32., .05):                # a second, separate contact
        score.step(t, dt, 1, np.array([x, 0., ground(x)-.1]), down, LEVEL)
        t += dt
    score.passed(1, t, np.array([50., 0., 2.+HIGH_PASS_M+.2]))
    r = score.result()
    assert r['contacts'] == 2 and r['contact_legs'] == [1, 1]
    assert r['max_penetration_m'] == pytest.approx(.2, abs=1e-9) and r['min_clearance_m'] == pytest.approx(-.2)
    assert r['descent_s'] == pytest.approx(r['below_view_s']) and r['descent_s'] > 0
    assert r['high_passes'] == 1 and r['hill_legs'] == 1
    assert r['max_into_speed_mps'] > 0


def test_score_without_terrain_reports_view_and_passes_only():
    score = DescentScore(synthetic_course(1), CAMERA)
    score.step(0., .01, 0, np.array([0., 0., 5.]), np.array([5., 0., 0.]), LEVEL)
    r = score.result()
    assert 'contacts' not in r and r['descent_s'] == 0. and r['high_passes'] == 0


def test_course_sets_are_seeded():
    from haltere.liftoff.descent_rehearsal import course_set, parse_set
    assert parse_set('hill:6000-6002,6010') == ('hill', [6000, 6001, 6002, 6010])
    with pytest.raises(ValueError):
        parse_set('mountain:1-2')
    courses, terrains = course_set('flat', [3000, 3001])
    assert terrains == [None, None] and np.array_equal(courses[0], synthetic_course(3000))
    courses, terrains = course_set('steep', [3000])
    assert np.array_equal(courses[0], synthetic_course(3000, steep=.4)) and terrains[0] is not None
