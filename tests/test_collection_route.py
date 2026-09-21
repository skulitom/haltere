import base64
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from haltere.liftoff.collection_route import CollectionRoute, prepare_route
from haltere.liftoff.frames import sim_vec_to_unity
from haltere.liftoff.pilot import LiftoffMapping, TelemetryPilot
from haltere.liftoff.telemetry import TelemetryFrame
from haltere.sim.tasks import HoverTask, HoverTaskConfig


class DummyBrain:
    channel_dims = dict(HoverTask.channels)

    def weight_matrix(self):
        return torch.zeros(1)

    def init_state(self, batch):
        return {}


def source_recording(tmp_path):
    # A right-angle line followed by a second lap that must never be included.
    points = np.array([[10, 20, 30], [10, 21, 30], [10, 21, 34], [13, 21, 34], [99, 99, 99]])
    states = np.column_stack([points, np.tile([0, 0, 0, 1], (5, 1)), np.zeros((5, 4)), np.arange(5)])
    root = ET.Element('LocalGhostStatesRecording')
    for k, v in {'gameVersion': '1.2.11', 'gamemode': 'Race', 'stateLayout': 'POSITION_STICKS',
                 'statesByte': base64.b64encode(states.astype('<f4').tobytes()).decode()}.items():
        ET.SubElement(root, k).text = v
    starts = ET.SubElement(root, 'lapStartIndices')
    for i in (0, 1, 3):
        ET.SubElement(starts, 'int').text = str(i)
    source = tmp_path / 'source.xml'
    source.write_bytes(ET.tostring(root))
    return source


def test_prefix_preserves_corners_altitude_and_exact_end(tmp_path):
    source = source_recording(tmp_path)
    out = tmp_path / 'route.json'
    route = prepare_route(source, out, length_m=6., speed_mps=2.)
    assert route['length_m'] == 6. and route['nominal_moving_seconds'] == 3.
    np.testing.assert_allclose(route['waypoints_unity'], [[10, 20, 30], [10, 21, 30], [10, 21, 34], [11, 21, 34]])
    with pytest.raises(FileExistsError):
        prepare_route(source, out)
    full = prepare_route(source, tmp_path / 'full.json', length_m=100)
    assert full['length_m'] == 8.
    assert full['waypoints_unity'][-1] == [13, 21, 34]


def test_live_origin_translation_preserves_world_route_and_does_not_rotate(tmp_path):
    out = tmp_path / 'route.json'
    prepare_route(source_recording(tmp_path), out, length_m=8.)
    route = CollectionRoute(out)
    frame = TelemetryFrame(position=np.array([10.3, 20., 30.2]), attitude=np.array([0., 2**-.5, 0., 2**-.5]))
    relative = route.relative_waypoints(frame)
    np.testing.assert_allclose(sim_vec_to_unity(relative) + frame.position, route.points)
    pilot = TelemetryPilot(DummyBrain(), HoverTaskConfig(), LiftoffMapping(), 'cpu')
    route.bind(pilot, frame)
    assert not pilot.loop and pilot.path_speed == 2.
    assert pilot.path_s[-1] == 8.
    np.testing.assert_allclose(pilot.path_point(1000), relative[-1])


@pytest.mark.parametrize('bad', ['spawn', 'moving', 'inverted', 'nan'])
def test_bad_live_start_is_rejected(tmp_path, bad):
    out = tmp_path / 'route.json'
    prepare_route(source_recording(tmp_path), out)
    route = CollectionRoute(out)
    frame = TelemetryFrame(position=np.array([10., 20., 30.]))
    if bad == 'spawn':
        frame.position[0] += 100
    elif bad == 'moving':
        frame.velocity[0] = 2
    elif bad == 'inverted':
        frame.attitude = np.array([1., 0., 0., 0.])
    else:
        frame.position[0] = np.nan
    with pytest.raises(ValueError):
        route.relative_waypoints(frame)


def test_open_path_never_adds_return_leg_or_wraps_progress():
    pilot = TelemetryPilot(DummyBrain(), HoverTaskConfig(), LiftoffMapping(), 'cpu',
                           waypoints=[[0, 0, 1], [4, 0, 1], [4, 3, 1]], loop=False)
    assert pilot.path_s[-1] == 7.
    np.testing.assert_allclose(pilot.path_point(-1), [0, 0, 1])
    np.testing.assert_allclose(pilot.path_point(100), [4, 3, 1])
    pilot.path_speed = 2.
    pilot.path_progress = 7.
    pilot.last_pos = np.array([4., 3., 1.])
    pilot.target_at(0.)
    np.testing.assert_allclose(pilot.target_at(1.), [4, 3, 1])
    assert pilot.path_progress == 7. and pilot.wp_index == 2
    loop = TelemetryPilot(DummyBrain(), HoverTaskConfig(), LiftoffMapping(), 'cpu',
                          waypoints=[[0, 0, 1], [4, 0, 1], [4, 3, 1]], loop=True)
    assert loop.path_s[-1] == 12.
    np.testing.assert_allclose(loop.path_point(13), loop.path_point(1))


def test_rejects_unsafe_speed_and_malformed_route(tmp_path):
    source = source_recording(tmp_path)
    with pytest.raises(ValueError):
        prepare_route(source, tmp_path / 'bad.json', speed_mps=20)
    out = tmp_path / 'route.json'
    data = prepare_route(source, out)
    data['waypoints_unity'][1] = data['waypoints_unity'][0]
    out.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        CollectionRoute(out)


@pytest.mark.parametrize('side', [-1., 0., 1.])
def test_cross_track_correction_opposes_side_error_without_changing_height_or_forward_goal(side):
    pilot = TelemetryPilot(DummyBrain(), HoverTaskConfig(), LiftoffMapping(), 'cpu',
                           waypoints=[[0, 0, 1], [10, 0, 1]], loop=False)
    pilot.path_speed = 2.
    pilot.path_progress = 3.
    pilot.last_pos = np.array([3., side * .2, 1.])
    original = pilot.target_at(0.).copy()
    pilot.path_cross_track_gain = 2.
    corrected = pilot.target_at(0.)
    np.testing.assert_allclose(corrected, original + [0., -side * .4, 0.])
    pilot.last_pos[1] = side * 20.
    assert abs(pilot.target_at(0.)[1]) <= .8


def test_cross_track_correction_handles_vertical_takeoff():
    pilot = TelemetryPilot(DummyBrain(), HoverTaskConfig(), LiftoffMapping(), 'cpu',
                           waypoints=[[0, 0, 0], [0, 0, 2], [5, 0, 2]], loop=False)
    pilot.path_speed = 2.
    pilot.path_cross_track_gain = 2.
    assert np.isfinite(pilot.target_at(0.)).all()


def test_cross_track_damping_brakes_sideways_motion_without_changing_forward_goal():
    pilot = TelemetryPilot(DummyBrain(), HoverTaskConfig(), LiftoffMapping(), 'cpu',
                           waypoints=[[0, 0, 1], [10, 0, 1]], loop=False)
    pilot.path_speed = 2.
    pilot.path_progress = 3.
    pilot.last_pos = np.array([3., 0., 1.])
    pilot.last_vel = np.array([2., .4, 0.])
    pilot.path_cross_track_damping = 1.
    np.testing.assert_allclose(pilot.target_at(0.), [4.5, -.4, 1.])


@pytest.mark.parametrize('field,value', [('cross_track_gain', float('nan')),
                                       ('cross_track_gain', -1),
                                       ('cross_track_damping', float('inf')),
                                       ('cross_track_damping', -1),
                                       ('cross_track_integral', float('nan')),
                                       ('cross_track_integral', -1)])
def test_invalid_cross_track_settings_rejected(tmp_path, field, value):
    out = tmp_path / 'route.json'
    data = prepare_route(source_recording(tmp_path), out)
    data[field] = value
    out.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Cross-track'):
        CollectionRoute(out)


def test_cross_track_integral_learns_bias_is_bounded_and_clears_on_reset():
    pilot = TelemetryPilot(DummyBrain(), HoverTaskConfig(), LiftoffMapping(), 'cpu',
                           waypoints=[[0, 0, 1], [100, 0, 1]], loop=False)
    pilot.path_speed = 2.
    pilot.path_cross_track_integral = .4
    pilot.last_pos = np.array([3., .2, 1.])
    for t in np.arange(0, 2.01, .01):
        pilot.target_at(float(t))
    assert .15 < pilot._path_lateral_bias < .17
    for t in np.arange(2.01, 30., .01):
        pilot.target_at(float(t))
    assert 0 < pilot._path_lateral_bias <= .8
    pilot.reset(None)
    assert pilot._path_lateral_bias == 0
    pilot.last_pos = np.array([0., .2, 0.])
    pilot.target_at(0.)
    pilot.target_at(.05)
    assert pilot._path_lateral_bias == 0


@pytest.mark.parametrize('valley_height', [.2, -.6])
def test_route_continues_below_launch_height_after_takeoff_and_rearms_on_reset(valley_height):
    pilot = TelemetryPilot(DummyBrain(), HoverTaskConfig(), LiftoffMapping(), 'cpu',
                           waypoints=[[0, 0, 1], [10, 0, 1], [20, 0, valley_height],
                                      [100, 0, valley_height]], loop=False)
    pilot.path_speed = 2.
    pilot.path_cross_track_integral = .4
    pilot.last_pos = np.array([0., .2, 0.])
    pilot.target_at(0.)
    pilot.target_at(.05)
    assert pilot.path_progress == 0 and pilot._path_lateral_bias == 0
    pilot.last_pos[2] = 1.
    pilot.target_at(.1)
    assert pilot.path_progress > 0
    pilot.path_progress = 30.
    pilot.last_pos = pilot.path_point(30.) + [0., .2, 0.]
    bias = pilot._path_lateral_bias
    pilot.target_at(.15)
    assert pilot.path_progress > 30.
    assert pilot._path_lateral_bias > bias
    pilot.reset(None)
    pilot.target_at(0.)
    pilot.target_at(.05)
    assert pilot.path_progress == 0 and not pilot._path_airborne
    assert pilot._path_lateral_bias == 0


def test_capture_records_teacher_provenance(tmp_path):
    from haltere.liftoff.dataset import DatasetWriter
    route = tmp_path / 'route.json'
    prepare_route(source_recording(tmp_path), route)
    camera = tmp_path / 'camera.yaml'
    camera.write_text('width: 640\nheight: 360\nf: 200\ntilt_deg: 30\n')
    writer = DatasetWriter(tmp_path / 'capture', course='Pine development', camera=camera, teacher_route=route)
    writer.close('test')
    report = json.loads((tmp_path / 'capture/capture.json').read_text())
    assert report['source'] == 'route_teacher_live' and report['oracle_route']
    assert report['teacher_route']['path'] == str(route.resolve())
    assert len(report['teacher_route']['sha256']) == 64


def test_telemetry_copy_reaches_independent_capture_without_stealing_pilot_packets():
    import socket
    import struct
    from haltere.liftoff.telemetry import TelemetryReceiver
    capture = TelemetryReceiver(port=0, stream=['Timestamp', 'Position'])
    pilot = TelemetryReceiver(port=0, stream=['Timestamp', 'Position'], forward_port=capture.sock.getsockname()[1])
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        payload = struct.pack('<4f', 2., 10., 20., 30.)
        tx.sendto(payload, ('127.0.0.1', pilot.sock.getsockname()[1]))
        first, second = pilot.wait(1), capture.wait(1)
        assert first.timestamp == second.timestamp == 2.
        np.testing.assert_array_equal(first.position, second.position)
        assert pilot.frames == capture.frames == 1
        with pytest.raises(ValueError):
            TelemetryReceiver(port=12345, forward_port=12345)
    finally:
        tx.close(); pilot.close(); capture.close()


@pytest.mark.parametrize('wrong_spawn', [False, True])
def test_world_route_command_drains_backlog_stops_on_reset_and_closes(tmp_path, monkeypatch, wrong_spawn):
    from types import SimpleNamespace
    from haltere.cli import main
    from haltere.liftoff import commands
    from haltere.train import bptt

    route_path = tmp_path / 'route.json'
    prepare_route(source_recording(tmp_path), route_path)

    class Brain(DummyBrain):
        device = torch.device('cpu')
        N = 1
        calls = 0
        def __call__(self, obs, state, weights):
            self.calls += 1
            return torch.zeros((1, 4)), state, {}

    brain = Brain()
    monkeypatch.setattr(bptt, 'load_checkpoint', lambda *args: (brain, SimpleNamespace(task=HoverTaskConfig()), None))
    monkeypatch.setattr(commands, 'load_mapping', lambda *args: LiftoffMapping())
    monkeypatch.setattr(commands, 'read_config', lambda: None)

    class Receiver:
        closed = False
        forwarding = []
        def __init__(self, **kw):
            self.forward_port = kw['forward_port']
            position = np.array([110., 20., 30.]) if wrong_spawn else np.array([10., 20., 30.])
            self.frames = iter([TelemetryFrame(timestamp=2., position=position, motor_rpm=np.full(4, 1000.)),
                                TelemetryFrame(timestamp=.5, position=position, motor_rpm=np.full(4, 1000.))])
        def wait(self, timeout):
            self.forwarding.append(self.forward_port)
            return next(self.frames)
        def close(self):
            Receiver.closed = True

    monkeypatch.setattr(commands, 'TelemetryReceiver', Receiver)
    argv = ['liftoff', 'fly', 'unused.pt', '--world-route', str(route_path), '--seconds', '5',
            '--log', str(tmp_path / 'flight.csv'), '--telemetry-copy-port', '9011', '--dry-run', '--device', 'cpu']
    if wrong_spawn:
        with pytest.raises(ValueError, match='correct course'):
            main(argv)
        assert brain.calls == 0 and Receiver.forwarding == [None]
    else:
        main(argv)
        assert brain.calls == 1 and Receiver.forwarding == [None, 9011]
        manifest = json.loads((tmp_path / 'flight.csv.route.json').read_text())
        assert manifest['origin_unity_xyz'] == [10., 20., 30.]
        assert manifest['oracle_route']
    assert Receiver.closed


def test_hover_copy_waits_until_reset_backlog_has_been_drained(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from haltere.cli import main
    from haltere.liftoff import commands
    from haltere.train import bptt

    class Brain(DummyBrain):
        device = torch.device('cpu')
        N = 1
        def __call__(self, obs, state, weights):
            return torch.zeros((1, 4)), state, {}

    monkeypatch.setattr(bptt, 'load_checkpoint', lambda *args: (
        Brain(), SimpleNamespace(task=HoverTaskConfig()), None))
    monkeypatch.setattr(commands, 'load_mapping', lambda *args: LiftoffMapping())
    monkeypatch.setattr(commands, 'read_config', lambda: None)

    class Receiver:
        closed = False
        forwarding = []
        def __init__(self, **kw):
            self.forward_port = kw['forward_port']
        def wait(self, timeout):
            self.forwarding.append(self.forward_port)
            if len(self.forwarding) > 1:
                raise KeyboardInterrupt
            return TelemetryFrame(timestamp=2., motor_rpm=np.full(4, 1000.))
        def close(self):
            Receiver.closed = True

    monkeypatch.setattr(commands, 'TelemetryReceiver', Receiver)
    main(['liftoff', 'fly', 'unused.pt', '--seconds', '5', '--dry-run', '--device', 'cpu',
          '--telemetry-copy-port', '9011', '--log', str(tmp_path / 'hover.csv')])
    assert Receiver.forwarding == [None, 9011]
    assert Receiver.closed


def test_legacy_pilot_rejects_visual_brains_before_opening_telemetry(monkeypatch):
    from types import SimpleNamespace
    from haltere.cli import main
    from haltere.liftoff import commands
    from haltere.train import bptt
    brain=SimpleNamespace(channel_dims={**HoverTask.channels,'retina':720})
    monkeypatch.setattr(bptt,'load_checkpoint',lambda *args:(brain,None,None))
    def unexpected_receiver(**kwargs):
        raise AssertionError('Wrong controller must fail before opening telemetry')
    monkeypatch.setattr(commands,'TelemetryReceiver',unexpected_receiver)
    with pytest.raises(SystemExit,match='visual fly-brain checkpoint'):
        main(['liftoff','fly','unused.pt','--dry-run','--device','cpu'])
