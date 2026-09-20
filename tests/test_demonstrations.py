import csv
import json

import numpy as np
import pytest

from haltere.liftoff.dataset import DatasetWriter
from haltere.liftoff.telemetry import TelemetryFrame
from haltere.vision.demonstrations import DemonstrationSequences, load_capture, make_examples, prepare


def capture(tmp_path, name, colour=40):
    camera = tmp_path / 'camera.yaml'
    camera.write_text('width: 640\nheight: 360\nf: 200\ntilt_deg: 30\n')
    root = tmp_path / name
    writer = DatasetWriter(root, course=name, camera=camera)
    writer.meta.update(pilot='human', profile=name)
    with (root / 'telemetry.csv').open('w', newline='') as f:
        log = csv.writer(f)
        log.writerow(TelemetryFrame.columns())
        for i in range(81):
            t = i * .05
            # Unity +z is simulator +x, with constant 2 m/s velocity.
            frame = TelemetryFrame(timestamp=t, recv_time=100 + t,
                                   position=np.array([10., 1., 20. + 2*t]),
                                   velocity=np.array([0., 0., 2.]),
                                   input=np.array([.1, .2, .3, .4]))
            log.writerow(frame.as_row())
            image = np.full((24, 32, 3), colour, dtype=np.uint8)
            image[:4, :4] = i
            image[-4:, -4:] = 255
            writer.add(image, frame, frame.recv_time + .001, frame.recv_time + .002)
    writer.close('user_stop')
    return root


def test_future_path_rotates_to_body_and_never_crosses_segments(tmp_path):
    c = load_capture(capture(tmp_path, 'flight'))
    # 90 degrees yaw: world-forward is body-right (negative left).
    c['attitude'][:] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    x = make_examples(c, [{'start_s': .2, 'end_s': 1.4}, {'start_s': 2., 'end_s': 3.8}])
    np.testing.assert_allclose(x['future_body'][0], [[0, -.5, 0], [0, -1, 0], [0, -2, 0]], atol=1e-6)
    np.testing.assert_allclose(x['velocity_body'][0], [0, -2, 0], atol=1e-6)
    assert set(x['run_id']) == {0, 1}
    assert np.all(x['t'][x['segment_id'] == 0] + 1 <= 1.4)
    assert np.all(x['t'][x['segment_id'] == 1] + 1 <= 3.8)


def test_gap_blocks_lookahead_and_sequence_continuity(tmp_path):
    c = load_capture(capture(tmp_path, 'flight'))
    keep = (c['raw_ts'] < 1.5) | (c['raw_ts'] > 1.9)
    c['raw_ts'], c['raw_position'] = c['raw_ts'][keep], c['raw_position'][keep]
    # Retain real image/telemetry pairs only.
    image_keep = (c['t'] < 1.5) | (c['t'] > 1.9)
    for k in ('t', 'ts', 'wall_time', 'position', 'attitude', 'controls', 'files', 'velocity', 'valid_images'):
        c[k] = c[k][image_keep]
    x = make_examples(c, [{'start_s': 0, 'end_s': 4}], horizons=(.25, .5))
    assert not np.any((x['t'] > .95 + 1e-9) & (x['t'] < 1.9))
    assert len(np.unique(x['run_id'])) == 2


def plan_for(tmp_path, a, b):
    plan = {'schema': 1, 'horizons_s': [.25, .5, 1.], 'holdout_profiles': [b.name], 'takes': [
        {'id': p.name, 'source': str(p), 'profile': p.name, 'split': split,
         'segments': [{'start_s': .2, 'end_s': 3.8}]} for p, split in [(a, 'train'), (b, 'validation')]]}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    return path, plan


def test_prepare_loads_sequences_and_refuses_overwrite(tmp_path):
    a, b = capture(tmp_path, 'a'), capture(tmp_path, 'b', 90)
    plan, _ = plan_for(tmp_path, a, b)
    out = tmp_path / 'prepared'
    manifest = prepare(plan, out)
    assert manifest['gate_labels_available'] is False
    ds = DemonstrationSequences(out, length=8)
    item = ds[0]
    assert item['images'].shape == (8, 3, 90, 160)
    assert item['future_body'].shape == (8, 3, 3)
    assert item['take_id'] == 'a'
    assert DemonstrationSequences(out, split='validation')[0]['take_id'] == 'b'
    assert len(DemonstrationSequences(out, split='review')) == 0
    np.testing.assert_allclose(item['controls'][0], [.1, .2, .3, .4])
    with pytest.raises(FileExistsError):
        prepare(plan, out)
    entry, arrays = ds.takes[0]
    first = a / 'frames' / str(arrays['filename'][ds.windows[0][1]])
    first.write_bytes(b'changed')
    with pytest.raises(ValueError, match='source image changed'):
        ds[0]
    (out / 'a.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='prepared arrays changed'):
        DemonstrationSequences(out)


def test_rejects_copied_images_across_takes(tmp_path):
    a, b = capture(tmp_path, 'a'), capture(tmp_path, 'b')
    plan, _ = plan_for(tmp_path, a, b)
    with pytest.raises(ValueError, match='exact image'):
        prepare(plan, tmp_path / 'prepared')
    assert not (tmp_path / 'prepared').exists()


def test_rejects_whole_take_split_and_tampered_controls(tmp_path):
    a, b = capture(tmp_path, 'a'), capture(tmp_path, 'b', 90)
    path, plan = plan_for(tmp_path, a, b)
    plan['takes'][1]['source'] = str(a)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='whole take'):
        prepare(path, tmp_path / 'prepared')
    index = a / 'index.csv'
    text = index.read_text()
    index.write_text(text.replace('0.1,0.2,0.3,0.4', '0.9,0.2,0.3,0.4'))
    with pytest.raises(ValueError, match='mismatch'):
        load_capture(a)


def test_holdout_course_cannot_train(tmp_path):
    a, b = capture(tmp_path, 'a'), capture(tmp_path, 'b', 90)
    path, plan = plan_for(tmp_path, a, b)
    plan['holdout_profiles'] = ['a']
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='holdout course'):
        prepare(path, tmp_path / 'prepared')


def test_uniform_finish_images_are_excluded_not_mistaken_for_leakage(tmp_path):
    import cv2
    a, b = capture(tmp_path, 'a'), capture(tmp_path, 'b', 90)
    for source in (a, b):
        cv2.imwrite(str(source / 'frames' / '000030.jpg'), np.zeros((360, 640, 3), dtype=np.uint8))
    path, _ = plan_for(tmp_path, a, b)
    out = tmp_path / 'prepared'
    manifest = prepare(path, out)
    assert all('000030.jpg' in t['excluded_uniform_images'] for t in manifest['takes'])
    with np.load(out / 'a.npz') as x:
        assert '000030.jpg' not in x['filename']
        assert len(np.unique(x['run_id'])) == 2
