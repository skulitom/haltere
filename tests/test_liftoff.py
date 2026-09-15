import csv
import struct

import numpy as np
import torch

from haltere.liftoff import frames
from haltere.liftoff.sysid import fit_dynamics, infer_mapping, load_recording, to_sim_sticks
from haltere.liftoff.telemetry import DEFAULT_STREAM, FrameParser, TelemetryFrame, write_config, read_config
from haltere.sim.controller import RateControllerParams
from haltere.sim.quad import QuadParams, QuadState
from haltere.sim.vehicle import RatesConfig, Vehicle


def test_frame_parser_everything_layout():
    p = FrameParser(DEFAULT_STREAM)
    assert p.expected_size(4) == 97
    payload = struct.pack('<f3f4f3f3f4f2fB4f', 1.5, 1, 2, 3, 0, 0, 0, 1, 4, 5, 6, 7, 8, 9, -1, 0.1, 0.2, 0.3, 15.9, 0.8,
                          4, 100, 200, 300, 400)
    fr = p.parse(payload, recv_time=0.0)
    assert fr.timestamp == 1.5
    assert np.allclose(fr.position, [1, 2, 3]) and np.allclose(fr.attitude, [0, 0, 0, 1])
    assert np.allclose(fr.velocity, [4, 5, 6]) and np.allclose(fr.gyro, [7, 8, 9])
    assert np.allclose(fr.input, [-1, 0.1, 0.2, 0.3], atol=1e-6) and np.allclose(fr.battery, [15.9, 0.8], atol=1e-5)
    assert np.allclose(fr.motor_rpm, [100, 200, 300, 400])
    assert len(fr.as_row()) == len(TelemetryFrame.columns())


def test_frame_parser_custom_layout():
    p = FrameParser(['InputPitch', 'GyroPitch', 'InputRoll', 'GyroRoll', 'InputYaw', 'GyroYaw'])
    assert p.expected_size() == 24
    fr = p.parse(struct.pack('<6f', 0.1, 10, 0.2, 20, 0.3, 30))
    assert np.allclose(fr.input, [0, 0.3, 0.1, 0.2], atol=1e-6) and np.allclose(fr.gyro, [10, 20, 30])


def test_receiver_roundtrip():
    import socket
    from haltere.liftoff.telemetry import TelemetryReceiver
    rx = TelemetryReceiver(port=0)                       # OS-assigned port
    port = rx.sock.getsockname()[1]
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = struct.pack('<f3f4f3f3f4f2fB4f', 2.5, 1, 2, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 16, 0.9,
                          4, 1000, 1000, 1000, 1000)
    tx.sendto(payload, ('127.0.0.1', port))
    tx.sendto(payload[:10], ('127.0.0.1', port))         # a truncated datagram must be counted, not crash
    fr = rx.wait(timeout=2.0)
    assert fr is not None and fr.timestamp == 2.5 and rx.frames >= 1 and rx.bad + rx.frames >= 1
    rx.close()
    tx.close()


def test_fake_liftoff_frames_parse():
    from haltere.liftoff.fake_liftoff import FakeLiftoff
    from haltere.liftoff.telemetry import FrameParser
    fake = FakeLiftoff(telemetry_port=0, stick_port=0, verbose=False)
    data = fake.frame_bytes(np.array([-0.4, 0.1, -0.2, 0.3]))
    fr = FrameParser().parse(data)
    assert len(data) == 97 and np.allclose(fr.input, [-0.4, -0.3, 0.2, 0.1], atol=1e-6)
    assert np.allclose(fake.liftoff_to_sim_sticks(fr.input), [-0.4, 0.1, -0.2, 0.3], atol=1e-6)
    fake.tx.close()
    fake.rx.close()


def test_config_write_read(tmp_path):
    p = write_config(port=9123, root=tmp_path)
    cfg = read_config(tmp_path)
    assert p.exists() and cfg['EndPoint'] == '127.0.0.1:9123' and cfg['StreamFormat'] == DEFAULT_STREAM


def test_frame_conversions():
    fwd_u = np.array([0.0, 0.0, 1.0])  # Unity forward
    assert np.allclose(frames.unity_vec_to_sim(fwd_u), [1, 0, 0])
    assert np.allclose(frames.unity_vec_to_sim([0, 1, 0]), [0, 0, 1])   # Unity up -> sim up
    assert np.allclose(frames.unity_vec_to_sim([1, 0, 0]), [0, -1, 0])  # Unity right -> sim -y (right)
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)  # Unity quaternion x,y,z,w
        v_u = rng.normal(size=3)
        rotated_u = frames.quat_xyzw_to_mat(q) @ v_u
        q_s = frames.unity_quat_to_sim(q)
        rotated_s = frames.quat_wxyz_to_mat(q_s) @ frames.unity_vec_to_sim(v_u)
        assert np.allclose(rotated_s, frames.unity_vec_to_sim(rotated_u), atol=1e-9)
        back = frames.sim_quat_to_unity(q_s)
        assert np.allclose(np.abs(np.dot(back, q)), 1.0, atol=1e-9)


def test_omega_from_quats():
    q0 = np.array([1.0, 0, 0, 0])
    w = np.array([0.5, -0.2, 1.0])
    dt = 0.01
    dq = np.concatenate([[np.cos(np.linalg.norm(w) * dt / 2)], w / np.linalg.norm(w) * np.sin(np.linalg.norm(w) * dt / 2)])
    q1 = frames.quat_mul(q0, dq)
    assert np.allclose(frames.omega_from_quats(q0, q1, dt), w, atol=1e-6)


def synthetic_recording(path, seconds=12.0, dt=0.01, seed=0):
    """Fly the simulator with smooth random sticks and write a Liftoff-format CSV with deliberately
    different (but consistent) sign conventions, so the identification has something to discover."""
    torch.manual_seed(seed)
    quad = QuadParams(twr=5.5, thrust_exp=1.3, gyro_noise=0.0)
    veh = Vehicle(quad, RateControllerParams(), RatesConfig(), 'cpu', dt=dt)
    vs = veh.wrap(QuadState.hover(1, 'cpu', height=0.5))   # just above the ground (the sim has no ground contact)
    n = int(seconds / dt)
    hover = 2 * veh.sim.hover_command() - 1
    t_axis = np.arange(n) * dt
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.2, 0.8, size=(3, 2))
    phases = rng.uniform(0, 2 * np.pi, size=(3, 2))
    rows = []
    for k in range(n):
        st = vs.quad
        # a crude "angle mode" pilot: level the drone and hold altitude, plus sinusoidal excitation
        sens = veh.sim.sensors(st, noise=False)
        gb = sens['gravity_body'][0].numpy()
        exc = np.array([np.sum(0.12 * np.sin(2 * np.pi * freqs[i] * t_axis[k] + phases[i])) for i in range(3)])
        roll = 1.5 * gb[1] + exc[0]
        pitch = -1.5 * gb[0] + exc[1]
        yaw = exc[2]
        z, vz = float(st.pos[0, 2]), float(st.vel[0, 2])
        thr = hover + 0.25 * (6.0 - z) - 0.15 * vz
        sticks = np.clip(np.array([thr, roll, pitch, yaw]), -1.0, 1.0)
        q = st.quat[0].numpy()
        pos_u = frames.sim_vec_to_unity(st.pos[0].numpy())
        vel_u = frames.sim_vec_to_unity(st.vel[0].numpy())
        q_u = frames.sim_quat_to_unity(q)
        om = st.omega[0].numpy()
        gyro = np.rad2deg([om[1], -om[0], om[2]])                 # pitch, roll(flipped), yaw
        inp = [sticks[0], -sticks[3], -sticks[2], sticks[1]]      # throttle, yaw(flipped), pitch(flipped), roll
        rpm = st.motor[0].numpy() * 30000.0
        rows.append([k * dt, k * dt] + list(pos_u) + list(q_u) + list(vel_u) + list(gyro) + inp + [16.0, 0.9] + list(rpm))
        vs = veh.step(vs, torch.tensor(sticks, dtype=torch.float32)[None])
        assert not bool(vs.quad.crashed.any())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(TelemetryFrame.columns())
        w.writerows(rows)


def test_sysid_recovers_signs_and_parameters(tmp_path):
    csv_path = tmp_path / 'rec.csv'
    synthetic_recording(csv_path)
    rec = load_recording(str(csv_path))
    mapping, notes = infer_mapping(rec)
    assert mapping.stick_sign == (1.0, -1.0, -1.0)
    assert mapping.gyro_axis == (1, 0, 2) and mapping.gyro_sign == (-1.0, 1.0, 1.0)
    sticks = to_sim_sticks(rec['input'], mapping)
    assert sticks.shape == (len(rec['t']), 4)
    res = fit_dynamics(rec, mapping, QuadParams(twr=6.5, thrust_exp=1.6), RateControllerParams(), RatesConfig(),
                       iters=60, window=50, device='cpu', verbose=False)
    assert res['loss'][-1] < 0.5 * res['loss'][0]
    # thrust-to-weight and the thrust exponent are only jointly identifiable from a hover-dominated flight:
    # the identifiable quantity is the hover command (1/twr)^(1/exp)
    hover_true = (1 / 5.5) ** (1 / 1.3)
    hover_fit = (1 / res['quad']['twr']) ** (1 / res['quad']['thrust_exp'])
    hover_start = (1 / 6.5) ** (1 / 1.6)
    assert abs(hover_fit - hover_true) < 0.5 * abs(hover_start - hover_true)


def test_radial_sticks_invert_the_game_processing():
    from haltere.liftoff.pilot import LiftoffMapping
    from haltere.liftoff.stickcal import RadialSticks
    m = RadialSticks(0.25, {'roll': -1.0})
    rng = np.random.default_rng(0)
    for _ in range(500):
        p = rng.uniform(-1, 1, 2) * rng.uniform(0, 1)
        if np.hypot(*p) > 1:
            continue
        raw = m.raw_for('roll', 'pitch', *p)
        assert np.hypot(*raw) <= 1 + 1e-9
        assert np.allclose(m.processed('roll', 'pitch', *raw), p, atol=1e-9)
    # a small roll correction during a pitch cruise must arrive as commanded (a per-axis inverse made it 4x larger)
    mp = LiftoffMapping(stick_sign=(-1.0, 1.0, 1.0), hover_stick_sim=-0.43, hover_processed_game=0.111,
                        throttle_scale=0.8, stick_model=m)
    raw = mp.to_raw(np.array([-0.43, 0.05, 0.3, 0.2]))
    roll_p, pitch_p = m.processed('roll', 'pitch', raw[1], raw[2])
    thr_p, yaw_p = m.processed('throttle', 'yaw', raw[0], raw[3])
    assert np.allclose([roll_p, pitch_p, thr_p, yaw_p], [-0.05, 0.3, 0.111, 0.2], atol=1e-9)


def test_flight_log_scoring(tmp_path):
    from haltere.liftoff.flightlog import score_log
    cols = ['wall', 'ts', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz', 'in_thr',
            'in_yaw', 'in_pitch', 'in_roll', 'rpm', 'b_thr', 'b_roll', 'b_pitch', 'b_yaw', 'c_thr', 'c_roll',
            'c_pitch', 'c_yaw', 's_thr', 's_roll', 's_pitch', 's_yaw', 'gx', 'gy', 'gz', 'tx', 'ty', 'tz', 'phase',
            'crashed', 'det_p', 'det_w', 'det_age', 'status']
    path = tmp_path / 'log.csv'
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in range(3000):             # 30 s straight along x at 2 m/s, 2 m up, through a gate at x = 30
            t = k * 0.01
            row = dict.fromkeys(cols, 0.0)
            row.update(ts=t, px=2.0 * t, pz=2.0, vx=2.0, qw=1.0, phase=t + 5.0, status='')
            w.writerow([row[c] for c in cols])
    gates = tmp_path / 'gates.json'
    gates.write_text('{"gates": [{"pos": [30.0, 0.0, 1.2], "heading": 0.0}]}')
    (r,) = score_log(str(path), str(gates))
    assert r['gates_through'] == [0]
    assert abs(r['speed_median'] - 2.0) < 1e-6 and r['rate_shake_dps'] < 1e-6


def test_path_follower_profile_projection_and_frame():
    from haltere.liftoff.pathfollow import PathFollower, rotate_commands, rotate_senses, wrap
    # a 40 m straight, a 90 degree left turn of radius 5 m, another straight; the taught lap overlaps its start
    straight = [[x, 0.0, 1.2] for x in np.arange(0.0, 40.0, 1.0)]
    turn = [[40 + 5 * np.sin(a), 5 - 5 * np.cos(a), 1.2] for a in np.linspace(0, np.pi / 2, 9)[1:]]
    back = [[45.0, y, 1.2] for y in np.arange(6.0, 40.0, 1.0)]
    f = PathFollower(np.array(straight + turn + back), loop=False, v_max=5.0, a_lat=1.5)
    i_turn = f._i(42.0)
    assert f.v_ref[f._i(10.0)] == 5.0 and f.v_ref[i_turn] < 3.0         # slow in the bend, braking before it
    assert f.v_ref[f._i(36.0)] < 4.5                                     # braking starts about 9 m before it
    assert abs(f.project(np.array([12.0, 0.4, 1.2])) - 12.0) < 0.6
    assert abs(f.project(np.array([17.0, -0.3, 1.2])) - 17.0) < 0.6       # forward search from the last projection
    # overlap trimming: a closed lap that runs 10 m past its start closes at the start, not with a U-turn
    lap = [[20 * np.cos(a), 20 * np.sin(a), 1.2] for a in np.linspace(0, 2 * np.pi + 0.5, 140)]
    g = PathFollower(np.array(lap), loop=True)
    assert g.kappa.max() < 0.2 and abs(g.length - 2 * np.pi * 20) < 3.0
    # frame rotation: a tilt request made in the tangent frame arrives in the body frame rotated by delta
    delta = 0.7
    roll, pitch = rotate_commands(0.1, 0.3, delta)
    body = np.array([pitch, -roll])
    frame = np.array([0.3, -0.1])
    assert np.allclose(body, [np.cos(delta) * frame[0] - np.sin(delta) * frame[1],
                              np.sin(delta) * frame[0] + np.cos(delta) * frame[1]])
    s = {'gyro': torch.tensor([[0.2, -0.1, 0.3]]), 'gravity_body': torch.tensor([[0.1, 0.0, -0.99]]),
         'vel_body': torch.tensor([[2.0, 1.0, 0.0]]), 'yaw': torch.tensor([[0.0]])}
    s2, rel = rotate_senses(s, np.array([3.0, 0.0, 0.0]), delta, 1.0, torch)
    c, sn = np.cos(delta), np.sin(delta)
    assert np.allclose(s2['vel_body'][0, :2].numpy(), [c * 2 + sn * 1, -sn * 2 + c * 1], atol=1e-6)
    assert np.allclose(rel[:2], [3 * c, -3 * sn]) and float(s2['yaw']) == 1.0 and abs(abs(wrap(3 * np.pi)) - np.pi) < 1e-9
    # obstacle clearance: an obstacle 0.8 m right of the straight pushes the line 1.8 m away from it, smoothly
    h = PathFollower(np.array(straight + turn + back), loop=False, obstacles=[[20.0, -0.8, 1.4]], clearance=1.8)
    assert abs(h.point(20.0)[1] - 1.0) < 0.05 and abs(h.point(5.0)[1]) < 1e-9
