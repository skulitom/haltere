import numpy as np
import pytest

from haltere.liftoff.section_scoring import score_sections


def fixture():
    geometry = dict(checkpoints=[dict(instance_id=i, center=[0, 3, i], normal=[0, 0, 1],
                                     half_width=2.5, half_height=2.5) for i in range(4)],
                    sections=[dict(id=str(i), kind='pillar', entry_checkpoint=2*i, exit_checkpoint=2*i+1)
                              for i in range(2)])
    times = np.arange(0, 4.51, .1)
    positions = np.column_stack([np.zeros_like(times), np.full_like(times, 3.), times-.5])
    return geometry, times, positions


def test_impact_counts_current_section_but_never_post_crash_sections():
    g, t, p = fixture()
    result = score_sections(g, t, p, impact_time=1.)
    assert [s['status'] for s in result['sections']] == ['impact_failure', 'unattempted']
    assert not result['game_finish_confirmed']


def test_success_then_impact_and_runtime_censor_are_distinct():
    g, t, p = fixture()
    impact = score_sections(g, t, p, impact_time=3.)
    assert [s['status'] for s in impact['sections']] == ['geometric_success', 'impact_failure']
    runtime = score_sections(g, t[t<3], p[t<3], runtime_stop=True)
    assert [s['status'] for s in runtime['sections']] == ['geometric_success', 'censored_runtime']


def test_wrong_direction_outside_gate_and_skipped_checkpoint_cannot_score():
    g, t, p = fixture()
    for positions in [p[::-1], p+np.array([3, 0, 0])]:
        assert score_sections(g, t, positions)['per_course']['geometric_sections_completed'] == 0
    # Starting beyond the first checkpoint cannot credit later gates out of order.
    assert score_sections(g, t, p+np.array([0, 0, 1]))['per_course']['geometric_sections_completed'] == 0


def test_gaps_censor_and_resets_need_separate_attempts():
    g, t, p = fixture()
    mask = (t<1) | (t>2)
    result = score_sections(g, t[mask], p[mask])
    assert [s['status'] for s in result['sections']] == ['censored_runtime', 'unattempted']
    with pytest.raises(ValueError, match='resets'):
        score_sections(g, t[::-1], p)
