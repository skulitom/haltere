import numpy as np
import pytest

from haltere.liftoff.race_eval import end_cause, opening, race_progress, race_sequence


def gate(gid, x, width=3., height=2.4, centred=False):
    return dict(id=gid, item='test', base=np.array([x, 0., 0.]), normal=np.array([1., 0., 0.]),
                side=np.array([0., 1., 0.]), up=np.array([0., 0., 1.]), sign=1, width=width, height=height,
                centred=centred, lap=0)


def fly(*points, dt=.02):
    """Straight segments between points, sampled every dt at 5 m/s."""
    path = [np.asarray(points[0], float)]
    for a, b in zip(points, points[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.linalg.norm(b-a)/(5*dt)))
        path += list(np.linspace(a, b, n)[1:])
    P = np.array(path)
    return np.arange(len(P))*dt, P


def test_openings_by_name_scale_and_family():
    one = dict(x=1., y=1., z=1.)
    assert opening('CheckpointBox10mX5m01', one) == (10., 5., True)
    assert opening('CheckpointBoxFlexible01', dict(x=.2, y=10., z=5.)) == (5., 10., True)
    assert opening('AirgateBig240H300B01FatShark01', one) == (3., 2.4, False)
    with pytest.raises(ValueError, match='No opening known'):
        opening('MysteryRing01', one)


def test_progress_needs_order_side_and_opening():
    sequence = [gate('s', 0.), gate('a', 10.), gate('b', 20.)]
    t, P = fly([-5, 0, 1.2], [25, 0, 1.2])
    result = race_progress(t, P, sequence)
    assert result['progress'] == 1. and result['crossed'] == 2 and result['next_expected'] is None
    # Beside gate a's opening: recorded, not credited, and b cannot be credited out of order.
    t, P = fly([-5, 0, 1.2], [5, 0, 1.2], [5, 4, 1.2], [25, 4, 1.2])
    result = race_progress(t, P, sequence)
    assert result['crossed'] == 0 and result['next_expected'] == 'a'
    assert [e['gate_id'] for e in result['outside_opening']] == ['a']
    # Over the top bar is outside too; through it backwards earns nothing.
    t, P = fly([-5, 0, 1.2], [2, 0, 1.2], [6, 0, 3.5], [15, 0, 3.5])
    assert race_progress(t, P, sequence)['crossed'] == 0
    t, P = fly([15, 0, 1.2], [-5, 0, 1.2])
    assert race_progress(t, P, sequence)['crossed'] == 0


def test_gap_and_reset_stop_credit():
    sequence = [gate('s', 0.), gate('a', 10.)]
    t, P = fly([-5, 0, 1.2], [12, 0, 1.2])
    hole = (P[:, 0] > 8) & (P[:, 0] < 11)
    t2, P2 = t[~hole], P[~hole]  # a telemetry gap across gate a
    result = race_progress(t2, P2, sequence)
    assert result['crossed'] == 0 and len(result['gaps']) == 1
    t, P = fly([-5, 0, 1.2], [5, 0, 1.2])
    P = np.vstack([P, [[-5, 0, 1.2], [20, 0, 1.2]]])  # respawn, then a jump through a
    t = np.append(t, [t[-1]+.02, t[-1]+.04])
    result = race_progress(t, P, sequence)
    assert result['reset_at'] is not None and result['crossed'] == 0


def test_race_sequence_repeats_the_lap_and_reads_flexible_axes(tmp_path):
    race = tmp_path/'race.xml'
    race.write_text('''<Race><requiredLaps>2</requiredLaps><checkPointPassages>
      <RaceCheckpointPassage><uniqueId>f</uniqueId><checkPointID>0</checkPointID><passageType>Finish</passageType>
        <nextPassageIDs/></RaceCheckpointPassage>
      <RaceCheckpointPassage><uniqueId>s</uniqueId><checkPointID>1</checkPointID><passageType>Start</passageType>
        <nextPassageIDs><string>p</string></nextPassageIDs></RaceCheckpointPassage>
      <RaceCheckpointPassage><uniqueId>p</uniqueId><checkPointID>2</checkPointID><passageType>Pass</passageType>
        <nextPassageIDs><string>f</string></nextPassageIDs></RaceCheckpointPassage>
      </checkPointPassages></Race>''')
    track = tmp_path/'track.xml'

    def bp(i, item, x, z, yaw=0., scale=''):
        return (f'<TrackBlueprint><itemID>{item}</itemID><instanceID>{i}</instanceID><position><x>{x}</x><y>0</y>'
                f'<z>{z}</z></position><rotation><x>0</x><y>{yaw}</y><z>0</z></rotation>{scale}</TrackBlueprint>')
    # Arches turned 90 deg face Unity x; the flexible box's thin axis is its own x, so it does too.
    track.write_text('<Track><blueprints>' + bp(1, 'AirgateBigLiftoffFinishWhite01', 0, 0, yaw=90)
                     + bp(2, 'CheckpointBoxFlexible01', 20, 0, scale='<scale><x>0.2</x><y>10</y><z>5</z></scale>')
                     + bp(0, 'AirgateBigLiftoffFinishWhite01', 40, 0, yaw=90) + '</blueprints></Track>')
    sequence, laps, lap_length = race_sequence(race, track, np.zeros(3))
    assert laps == 2 and lap_length == 2
    assert [g['id'] for g in sequence] == ['1', '2', '0', '2', '0'] and [g['lap'] for g in sequence] == [0, 1, 1, 2, 2]
    flexible = sequence[1]
    assert abs(flexible['normal'] @ np.array([0., -1., 0.])) > .99 and flexible['width'] == 5.
    assert [g['sign'] for g in sequence] == [1, 1, 1, -1, 1]  # lap 2 comes back to the box from the finish side


@pytest.mark.parametrize('meta, finish, expected', [
    (dict(ticks=10, stop_reason='Impact detected from flight motion', impact=dict(timestamp=1.)), None, 'impact'),
    (dict(ticks=10, stop_reason='Telemetry stale, paused or outside live flight'), None, 'needs_review'),
    (dict(ticks=10, stop_reason='Telemetry stale, paused or outside live flight'),
     dict(game_confirmed_finish=True), 'finish'),
    (dict(ticks=10, stop_reason='duration'), None, 'limit'),
    (dict(ticks=10, stop_reason='Flight distance limit exceeded'), None, 'limit'),
    (dict(ticks=10, stop_reason='Controller missed its real-time deadline'), None, 'runtime_error'),
    (dict(ticks=10, stop_reason='Fresh camera measurements unavailable'), None, 'runtime_error'),
    (dict(ticks=0, stop_reason='No fresh live image/telemetry: None'), None, 'runtime_error'),
])
def test_end_cause(meta, finish, expected):
    assert end_cause(meta, finish) == expected


def test_summary_headline_and_paired_difference(tmp_path):
    import json
    from haltere.liftoff.race_eval import summarise
    attempts = [dict(n=i+1, case=c, variant=v, rep=1, log=f'{c}-{v}-1.csv')
                for i, (c, v) in enumerate([(c, v) for c in ('a', 'b') for v in ('x', 'y')])]
    (tmp_path/'plan.json').write_text(json.dumps(dict(batch='t', attempts=attempts)))
    progress = {('a', 'x'): .2, ('a', 'y'): .4, ('b', 'x'): .0, ('b', 'y'): 1.}
    with open(tmp_path/'results.jsonl', 'w') as f:
        for a in attempts:
            done = progress[a['case'], a['variant']] == 1.
            f.write(json.dumps(dict(a, end_cause='finish' if done else 'impact', checkpoints='-', time_ratio=None,
                                    grade=dict(progress=progress[a['case'], a['variant']], finish=done,
                                               clean_finish=done, within_target=False)))+'\n')
    text = summarise(tmp_path)
    assert '| x | 0.100 [0.1, 0.1] |' in text and '| y | 0.700 [0.7, 0.7] |' in text
    assert 'y - x in mean progress: +0.600' in text and '| 1/2 | 1/2 | 0/2 |' in text
