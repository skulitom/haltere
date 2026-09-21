"""Offline checks for collection alignment, data preservation and split leakage."""
import csv
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from haltere.liftoff.dataset import DatasetWriter
from haltere.liftoff.telemetry import TelemetryFrame
from haltere.vision.datasets import audit_split


def camera_file(tmp_path):
    camera = tmp_path / 'camera.yaml'
    camera.write_text('width: 640\nheight: 360\nf: 200\ntilt_deg: 30\n', encoding='utf-8')
    return camera


def telemetry(timestamp=1., recv_time=100., position=(1., 2., 3.)):
    return TelemetryFrame(timestamp=timestamp, recv_time=recv_time, position=np.array(position),
                          input=np.array([.1, .2, .3, .4]))


def test_capture_preserves_coordinate_and_input_contract(tmp_path):
    out = tmp_path / 'flight'
    writer = DatasetWriter(out, course='Pine Valley / development', camera=camera_file(tmp_path))
    image = np.zeros((90, 160, 3), dtype=np.uint8)
    assert writer.add(image, telemetry(), 100.01, 100.02)
    assert writer.add(image, telemetry(2., 101., (2., 4., 6.)), 101.01, 101.02)
    writer.close('duration')
    with (out / 'index.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert [float(rows[1][k]) for k in ('px', 'py', 'pz')] == [3., -1., 2.]
    assert [float(rows[0][k]) for k in ('qw', 'qx', 'qy', 'qz')] == [1., 0., 0., 0.]
    assert [float(rows[0][f'in_{k}']) for k in ('throttle', 'yaw', 'pitch', 'roll')] == [.1, .2, .3, .4]
    assert float(rows[1]['t']) == 1.
    assert cv2.imread(str(out / 'frames/000000.jpg')).shape == (360, 640, 3)
    meta = json.loads((out / 'capture.json').read_text())
    assert meta['frames'] == 2 and meta['origin_sim'] == [3., -1., 2.]
    assert meta['labels_reviewed'] is False and meta['course_complete'] is False
    assert meta['telemetry_age_p95_s'] == pytest.approx(.02)
    assert (out / 'camera.yaml').read_bytes() == (tmp_path / 'camera.yaml').read_bytes()
    with pytest.raises(FileExistsError):
        DatasetWriter(out, course='new', camera=tmp_path / 'camera.yaml')
    assert len(list((out / 'frames').iterdir())) == 2


def test_capture_rejects_stale_duplicate_and_invalid_pairs(tmp_path):
    writer = DatasetWriter(tmp_path / 'flight', course='test', camera=camera_file(tmp_path))
    image = np.zeros((8, 8, 3), np.uint8)
    assert not writer.add(image, telemetry(), 100.01, 100.2)
    assert not writer.add(image, telemetry(recv_time=101.), 100., 100.01)
    bad = telemetry()
    bad.attitude = np.zeros(4)
    assert not writer.add(image, bad, 100., 100.01)
    assert writer.add(image, telemetry(), 100.01, 100.02)
    assert not writer.add(image, telemetry(), 100.01, 100.02)
    assert not writer.add(image, telemetry(timestamp=float('nan')), 100.01, 100.02)
    writer.close('test')
    assert writer.frames == 1 and writer.rejected == 5


def test_passive_loop_stops_at_reset_and_closes_receiver(tmp_path, monkeypatch):
    from haltere.liftoff import commands, dataset, recorder

    class Receiver:
        closed = False
        def __init__(self, **kw):
            self.frames = iter([telemetry(1.), telemetry(2.), telemetry(0.)])
        def wait(self, timeout):
            return next(self.frames)
        def close(self):
            Receiver.closed = True

    class Screen:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def grab(self, region):
            return np.zeros((90, 160, 4), np.uint8)

    ticks = iter(np.arange(0., 100., .11))
    monkeypatch.setattr(dataset.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(dataset.time, 'time', lambda: 100.01)
    monkeypatch.setattr(dataset, 'TelemetryReceiver', Receiver)
    monkeypatch.setattr(dataset, 'read_config', lambda: None)
    monkeypatch.setattr(commands, 'find_game_window', lambda title: 1)
    monkeypatch.setattr(commands, 'game_window_active', lambda hwnd: True)
    monkeypatch.setattr(recorder, 'find_window_rect', lambda title: (0, 0, 160, 90))
    monkeypatch.setitem(sys.modules, 'mss', SimpleNamespace(mss=Screen))
    report = dataset.capture_dataset(tmp_path / 'capture', course='test', camera=camera_file(tmp_path), seconds=5)
    assert Receiver.closed and report['stop_reason'] == 'game_reset'
    assert report['frames'] == 2
    with (tmp_path / 'capture/telemetry.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 3  # original reset is preserved in raw telemetry


def labelled_dataset(path, colours=(50, 100)):
    (path / 'frames').mkdir(parents=True)
    labels = []
    for i, colour in enumerate(colours):
        name = f'{i}.jpg'
        cv2.imwrite(str(path / 'frames' / name), np.full((36, 64, 3), colour, np.uint8))
        labels.append(dict(file=name, visible=i % 2, u=32., v=18., width_px=12., gate=0))
    (path / 'labels.json').write_text(json.dumps(labels), encoding='utf-8')
    return path


def test_audit_rejects_aliases_and_copied_frames(tmp_path):
    first = labelled_dataset(tmp_path / 'train')
    with pytest.raises(ValueError, match='same dataset'):
        audit_split([first], [first / '..' / 'train'])
    second = labelled_dataset(tmp_path / 'validation')
    with pytest.raises(ValueError, match='exact images'):
        audit_split([first], [second])


def test_inventory_records_content_and_coverage_limits(tmp_path):
    first = labelled_dataset(tmp_path / 'train')
    second = labelled_dataset(tmp_path / 'validation', (25, 200))
    report = audit_split([first], [second])
    row = report['train'][0]
    assert report['split'] == 'whole_flights'
    assert row['positive'] == row['negative'] == 1 and row['labelled_gate_ids'] == [0]
    assert row['warnings'] and len(row['images_sha256']) == 64
    cv2.imwrite(str(first / 'frames/0.jpg'), np.zeros((36, 64, 3), np.uint8))
    changed = audit_split([first], [second])
    assert changed['train'][0]['images_sha256'] != row['images_sha256']
    assert changed['train'][0]['labels_sha256'] == row['labels_sha256']


def test_audit_rejects_resampled_frames_from_same_flight(tmp_path):
    first = labelled_dataset(tmp_path / 'train')
    second = labelled_dataset(tmp_path / 'validation', (25, 200))
    for path in (first,second):
        (path/'capture.json').write_text(json.dumps({'source_flight_sha256':'same-flight'}))
    with pytest.raises(ValueError,match='source flight'):
        audit_split([first],[second])


def test_projective_frames_require_calibrated_range_labels(tmp_path):
    from haltere.vision.train import GateFrames
    first=labelled_dataset(tmp_path/'train')
    metadata=first/'capture.json'
    metadata.write_text(json.dumps(dict(size_target='silhouette',camera=dict(width=640,f=200.))))
    with pytest.raises(ValueError,match='equivalent-range'):
        GateFrames([first],augment='projective')
    metadata.write_text(json.dumps(dict(size_target='equivalent 4m range',camera=dict(width=640,f=200.))))
    dataset=GateFrames([first],augment='projective')
    x,y=dataset[1]
    assert x.shape==(3,180,320) and y.shape==(4,)
    assert np.isfinite(y.numpy()).all()


def test_audit_rejects_missing_and_escaping_frame_paths(tmp_path):
    first = labelled_dataset(tmp_path / 'train')
    labels_path = first / 'labels.json'
    labels = json.loads(labels_path.read_text())
    labels[0]['file'] = '../../outside.jpg'
    labels_path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match='escaping'):
        audit_split([first])


def test_tiny_training_retains_split_provenance(tmp_path):
    import torch
    from haltere.vision.train import train

    first = labelled_dataset(tmp_path / 'train')
    second = labelled_dataset(tmp_path / 'validation', (25, 200))
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        out = train([str(first)], holdout=[str(second)], out_dir=str(tmp_path / 'run'),
                    epochs=1, batch=2, width=2, device='cpu', max_gpu_temp=0, batch_sleep=0)
        checkpoint = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
        mined = train([str(first)],holdout=[str(second)],out_dir=str(tmp_path/'mined'),
                      init=str(out/'best.pt'),hard_mining=True,epochs=1,batch=2,width=2,
                      device='cpu',max_gpu_temp=0,batch_sleep=0)
        mining_report=json.loads((mined/'datasets.json').read_text())
        assert len(mining_report['hard_example_sampling']['weights'])==2
        assert mining_report['init_sha256'] and mining_report['validation_kind']=='whole_flights'
    finally:
        torch.set_num_threads(previous)
    report = json.loads((out / 'datasets.json').read_text())
    assert checkpoint['data_provenance'] == report
    assert report['validation_kind'] == 'whole_flights' and report['seed'] == 0
    with pytest.raises(ValueError, match='same dataset'):
        train([str(first)], holdout=[str(first)], out_dir=str(tmp_path / 'bad'))
    assert not (tmp_path / 'bad').exists()


def test_hard_sampling_uses_only_clean_training_frames_and_is_bounded(tmp_path):
    import torch
    from haltere.vision.train import GateFrames,hard_example_weights

    path=labelled_dataset(tmp_path/'train',(20,40,60,80))
    labels=json.loads((path/'labels.json').read_text())
    # The fixed detector has one easy positive and one off-centre miss.
    labels[1].update(u=32.,v=18.)
    labels[3].update(u=60.,v=18.)
    (path/'labels.json').write_text(json.dumps(labels))
    dataset=GateFrames([path],augment='strong')
    class CentreDetector(torch.nn.Module):
        def forward(self,x):
            return x.new_tensor([5.,0.,0.,0.]).expand(len(x),-1)
    first=hard_example_weights(CentreDetector(),dataset,'cpu',2)
    second=hard_example_weights(CentreDetector(),dataset,'cpu',3)
    assert torch.equal(first,second)  # mining does not use the stochastic augmentation
    assert len(first)==len(labels) and torch.all((first>=1)&(first<=7))
    assert first[3]>4*first[1]  # spend more updates on the missed centre
    assert first[0]>2*first[1]  # and on false positives


def test_hard_sampling_requires_independent_validation_and_parent(tmp_path):
    from haltere.vision.train import train
    first=labelled_dataset(tmp_path/'train')
    second=labelled_dataset(tmp_path/'validation',(25,200))
    with pytest.raises(ValueError,match='initial detector'):
        train([str(first)],holdout=[str(second)],hard_mining=True,out_dir=str(tmp_path/'bad'))
    with pytest.raises(ValueError,match='whole-flight'):
        train([str(first)],init='unused.pt',hard_mining=True,out_dir=str(tmp_path/'bad'))
    assert not (tmp_path/'bad').exists()
