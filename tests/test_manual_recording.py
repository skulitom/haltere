import csv
import json

import cv2
import numpy as np

from haltere.liftoff.manual_recording import ManualTake, discontinuity, live_pose
from haltere.liftoff.telemetry import TelemetryFrame


def frame(t=1., p=(100., 5., 200.)):
    return TelemetryFrame(timestamp=t, position=np.array(p), recv_time=100.,
                          input=np.array([.1, .2, .3, .4]))


def test_end_of_race_zero_pose_is_not_a_training_frame():
    assert live_pose(frame())
    assert not live_pose(frame(2., (0., 0., 0.)))
    bad = frame()
    bad.attitude[:] = 0
    assert not live_pose(bad)
    assert discontinuity(frame(10.), frame(.1)) == 'game_reset'
    assert discontinuity(frame(), frame(1.01, (0., 0., 0.))) == 'position_jump'
    assert discontinuity(frame(), frame(1.01, (100.2, 5., 200.))) is None


def test_manual_take_saves_images_controls_video_and_human_provenance(tmp_path):
    camera = tmp_path / 'camera.yaml'
    camera.write_text('width: 640\nheight: 360\nf: 200\ntilt_deg: 30\n')
    take = ManualTake(tmp_path / 'takes', 3, camera, 20.)
    image = np.full((90, 160, 3), 90, np.uint8)
    fr = frame()
    take.telemetry(fr)
    take.image(image, fr, 100.01, 100.02)
    take.close('user_stop')
    metadata = json.loads((take.path / 'capture.json').read_text())
    assert metadata['pilot'] == 'human' and not metadata['oracle_route']
    assert metadata['profile'] == 'straw_bale_fence' and metadata['expected_laps'] is None
    assert metadata['frames'] == 1 and metadata['stop_reason'] == 'user_stop'
    assert not metadata['camera_alignment_verified'] and not metadata['map_label_verified']
    with (take.path / 'telemetry.csv').open() as f:
        row = next(csv.DictReader(f))
    assert float(row['in_throttle']) == .1 and float(row['px']) == 100.
    video = cv2.VideoCapture(str(take.path / 'preview.mp4'))
    ok, saved = video.read()
    video.release()
    assert ok and saved.shape == (360, 640, 3)
    second = ManualTake(tmp_path / 'takes', 3, camera, 20.)
    assert second.path != take.path
    second.close('recorder_closed')
