"""Implementations of the ``haltere liftoff ...`` sub-commands."""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from .telemetry import DEFAULT_STREAM, TelemetryFrame, TelemetryReceiver, config_path, read_config, write_config

SETUP_STEPS = """
Telemetry configuration written to:
  {path}
Liftoff re-reads it whenever the drone is reset, so start (or reset) a flight and run
  haltere liftoff listen
to confirm frames arrive on UDP port {port}.

Remaining one-time steps (these need you, not a script):
  1. Virtual gamepad driver (admin): install ViGEmBus 1.22.0 from
       https://github.com/nefarius/ViGEmBus/releases/tag/v1.22.0
     then:  set VGAMEPAD_SKIP_VIGEMBUS_INSTALL=true && uv pip install --python .venv/Scripts/python.exe vgamepad
  2. Steam -> Liftoff -> Properties -> Controller: set "Override for Liftoff" to "Disable Steam Input".
  3. In Liftoff: Settings -> Controls -> add controller. Run `haltere liftoff calibrate` and follow its
     prompts: it moves one virtual axis at a time (throttle, yaw, pitch, roll) when the wizard asks.
     Keep your real radio plugged in too: you fly manually for `haltere liftoff record`.
  4. Record two minutes of your own flying (`haltere liftoff record`), then `haltere liftoff fit` to
     match the simulator to Liftoff's drone and learn the stick/gyro sign conventions.
  5. Train (`haltere train`) and fly (`haltere liftoff fly runs/<run>/best.pt`).
"""


def cmd_setup(a):
    p = write_config(port=a.port, stream=DEFAULT_STREAM)
    print(SETUP_STEPS.format(path=p, port=a.port))


def cmd_doctor(a):
    """Check every prerequisite for flying in Liftoff and print the next step."""
    import importlib
    import subprocess
    ok = lambda b: 'OK  ' if b else 'MISSING'  # noqa: E731
    steps = []
    # 1. telemetry config
    cfg = read_config()
    print(f'[{ok(cfg is not None)}] Liftoff telemetry config {config_path()}')
    if cfg is None:
        steps.append('haltere liftoff setup')
    # 2. ViGEmBus driver
    try:
        out = subprocess.run(['sc', 'query', 'ViGEmBus'], capture_output=True, text=True).stdout
        vigem = 'RUNNING' in out or 'STATE' in out
    except Exception:
        vigem = False
    print(f'[{ok(vigem)}] ViGEmBus driver (virtual Xbox controller)')
    if not vigem:
        steps.append('install ViGEmBus 1.22.0 (admin): https://github.com/nefarius/ViGEmBus/releases/tag/v1.22.0')
    # 3. vgamepad
    try:
        importlib.import_module('vgamepad')
        vg = True
    except Exception:
        vg = False
    print(f'[{ok(vg)}] vgamepad Python package')
    if not vg:
        steps.append('set VGAMEPAD_SKIP_VIGEMBUS_INSTALL=true && uv pip install --python .venv/Scripts/python.exe vgamepad')
    # 4. controller mapping in Liftoff (heuristic: an InputSettings file mentioning an Xbox/XInput controller)
    from .telemetry import LIFTOFF_LOCALLOW
    mapped = False
    for f in (LIFTOFF_LOCALLOW / 'InputSettings').glob('*/*.inputsettings'):
        txt = f.read_text(encoding='utf-8', errors='ignore').lower()
        if 'xbox' in txt or 'xinput' in txt or 'gamepad' in txt:
            mapped = True
    print(f'[{ok(mapped)}] virtual pad mapped in Liftoff (Settings -> Controls; use `haltere liftoff calibrate`)')
    if not mapped:
        steps.append('in Liftoff add the Xbox controller and run: haltere liftoff calibrate')
    # 5. telemetry frames arriving (only meaningful while Liftoff is flying)
    frames = 0
    try:
        rx = TelemetryReceiver(port=a.port)
        t0 = time.time()
        while time.time() - t0 < 2.0:
            if rx.wait(0.5) is not None:
                frames = rx.frames
                break
        rx.close()
    except OSError as e:
        print(f'[MISSING] could not bind UDP port {a.port}: {e}')
    print(f'[{ok(frames > 0)}] telemetry frames on port {a.port} (needs Liftoff running with a drone in the air)')
    # 6. mapping / fit and checkpoint
    fit = os.path.exists(a.liftoff_config)
    print(f'[{ok(fit)}] stick/gyro conventions fitted ({a.liftoff_config})')
    if not fit:
        steps.append('fly manually while recording, then fit: haltere liftoff record --seconds 120 && haltere liftoff fit')
    ck = os.path.exists(a.ckpt)
    print(f'[{ok(ck)}] trained brain checkpoint ({a.ckpt})')
    print()
    if steps:
        print('next steps, in order:')
        for i, s in enumerate(steps, 1):
            print(f'  {i}. {s}')
    else:
        print(f'everything is in place: haltere liftoff fly {a.ckpt} --offset 0,0,2')


def cmd_listen(a):
    cfg = read_config()
    if cfg is None:
        print(f'no {config_path()} found; run `haltere liftoff setup` first', file=sys.stderr)
    stream = (cfg or {}).get('StreamFormat', DEFAULT_STREAM)
    rx = TelemetryReceiver(port=a.port, stream=stream)
    print(f'listening on udp://127.0.0.1:{a.port} for {a.seconds:.0f}s (stream: {", ".join(stream)})')
    t0 = time.time()
    last_print = 0.0
    n0 = 0
    while time.time() - t0 < a.seconds:
        fr = rx.wait(0.2)
        if fr is None:
            continue
        if time.time() - last_print > 0.5:
            rate = (rx.frames - n0) / (time.time() - last_print) if last_print else 0.0
            n0 = rx.frames
            last_print = time.time()
            print(f't={fr.timestamp:8.3f}s {rate:5.0f} Hz pos={np.round(fr.position, 2)} q={np.round(fr.attitude, 3)} '
                  f'gyro={np.round(fr.gyro, 1)} in={np.round(fr.input, 2)} rpm={np.round(fr.motor_rpm, 0)} '
                  f'bad={rx.bad}', flush=True)
    print(f'{rx.frames} frames received, {rx.bad} unparseable')
    rx.close()


def cmd_calibrate(a):
    from .gamepad import VirtualPad
    pad = VirtualPad()
    order = [s.strip().lower() for s in a.order.split(',')] if a.order else ['throttle', 'yaw', 'pitch', 'roll']
    print('Virtual Xbox 360 pad created (Windows sees an Xbox 360 controller). In Liftoff: Settings -> Controls ->')
    print('select it and start the axis wizard. Sticks rest at neutral (throttle low) between sweeps.')
    try:
        if a.auto > 0:
            print(f'auto mode: each axis is swept for {a.auto:.0f}s in the order {", ".join(order)}, with a '
                  f'{a.pause:.0f}s pause before each one; {a.rounds} round(s). Ctrl+C to stop.')
            for r in range(a.rounds):
                for axis in order:
                    for k in range(int(a.pause), 0, -1):
                        print(f'  {axis} in {k}s ...', flush=True)
                        time.sleep(1.0)
                    print(f'  >>> sweeping {axis} for {a.auto:.0f}s', flush=True)
                    pad.sweep(axis, seconds=a.auto)
            pad.neutral()
            print('done; sticks neutral. Run again with --order to repeat a single axis, e.g. --order roll --rounds 1')
            return
        print('When the wizard asks for an axis, press Enter here to sweep it (order: ' + ', '.join(order) + ').')
        for axis in order:
            input(f'  press Enter to sweep {axis} ... ')
            pad.sweep(axis, seconds=a.sweep)
            print(f'  {axis} swept (full range), sticks back to neutral')
        while True:
            s = input('type an axis name to sweep again, or Enter to finish: ').strip().lower()
            if not s:
                break
            if s in ('throttle', 'yaw', 'pitch', 'roll'):
                pad.sweep(s, seconds=a.sweep)
    except KeyboardInterrupt:
        pass
    finally:
        pad.close()


def cmd_pad(a):
    """Hold the virtual pad alive with neutral sticks (so Liftoff can see and select the controller).

    With --control-file, the pad also executes commands written to that file (one command, the file
    is emptied after it runs): ``sweep throttle|yaw|pitch|roll [seconds]``, ``hold AXIS VALUE``,
    ``neutral``, ``reconnect`` (re-plug the pad), ``quit``. This lets someone else (or a chat assistant) drive
    the wizard."""
    import socket
    import struct
    from .gamepad import VirtualPad
    pad = VirtualPad()
    ctl = Path(a.control_file) if a.control_file else None
    if ctl:
        ctl.parent.mkdir(parents=True, exist_ok=True)
        ctl.write_text('', encoding='utf-8')
    sock = None
    if a.udp_in:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', a.udp_in))
        sock.setblocking(False)
    log_f = open(a.log, 'w', newline='', encoding='utf-8') if a.log else None
    log = csv.writer(log_f) if log_f else None
    if log:
        log.writerow(['wall_time', 'throttle', 'roll', 'pitch', 'yaw', 'source'])
    print(f'virtual Xbox 360 pad is live for {a.seconds:.0f}s with neutral sticks (throttle low). Ctrl+C to stop.'
          + (f' Commands via {ctl}.' if ctl else '') + (f' Sticks via udp://127.0.0.1:{a.udp_in} (4 float32: '
          f'throttle, yaw, pitch, roll; falls back to neutral after 0.5 s of silence).' if sock else ''), flush=True)
    held = {'throttle': -1.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
    axes = ['throttle', 'roll', 'pitch', 'yaw']
    last_udp = 0.0
    script_until = 0.0
    ramp = None   # (axis, v0, v1, t0, t1)

    def emit(source):
        pad.send(held['throttle'], held['roll'], held['pitch'], held['yaw'])
        if log:
            log.writerow([f'{time.time():.4f}', *[f'{held[k]:.4f}' for k in axes], source])

    try:
        t_start = time.time()
        while time.time() - t_start < a.seconds:
            now = time.time()
            cmd = ''
            if ctl and ctl.exists():
                cmd = ctl.read_text(encoding='utf-8').strip()
                if cmd:
                    ctl.write_text('', encoding='utf-8')
            if cmd:
                parts = cmd.split()
                print(f'[{time.strftime("%H:%M:%S")}] command: {cmd}', flush=True)
                if parts[0] == 'sweep' and len(parts) >= 2 and parts[1] in held:
                    pad.sweep(parts[1], seconds=float(parts[2]) if len(parts) > 2 else 4.0)
                elif parts[0] == 'ramp' and len(parts) == 5 and parts[1] in held:   # ramp AXIS v0 v1 seconds
                    ramp = (parts[1], float(parts[2]), float(parts[3]), now, now + float(parts[4]))
                    script_until = ramp[4]
                elif parts[0] == 'hold' and len(parts) == 3 and parts[1] in held:
                    held[parts[1]] = max(-1.0, min(1.0, float(parts[2])))
                    script_until = now + 3600
                elif parts[0] == 'press' and len(parts) >= 2:       # press BUTTON [seconds]  (A, B, X, Y, START, BACK, ...)
                    pad.press(parts[1], seconds=float(parts[2]) if len(parts) > 2 else 0.15)
                elif parts[0] == 'neutral':
                    held.update({'throttle': -1.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0})
                    script_until = 0.0
                elif parts[0] == 'reconnect':
                    pad.reconnect()
                    print('virtual pad re-plugged', flush=True)
                elif parts[0] == 'quit':
                    break
            if ramp is not None:
                axis, v0, v1, t0, t1 = ramp
                f = min(1.0, (now - t0) / max(t1 - t0, 1e-6))
                held[axis] = v0 + (v1 - v0) * f
                if f >= 1.0:
                    ramp = None
                emit('script')
            elif sock is not None:
                latest = None
                while True:
                    try:
                        data, _ = sock.recvfrom(64)
                    except (BlockingIOError, OSError):
                        break
                    if data == b'RECONNECT':                           # the pilot noticed Liftoff dropped the pad
                        pad.reconnect()
                        print(f'[{time.strftime("%H:%M:%S")}] virtual pad re-plugged at the pilot request', flush=True)
                        last_udp = time.time()
                        continue
                    if data[:6] == b'PRESS ':                          # button press request from the pilot
                        try:
                            pad.press(data[6:].decode().strip() or 'A', seconds=0.15)
                        except Exception as e:  # unknown button name
                            print(f'press failed: {e}', flush=True)
                        continue
                    if len(data) >= 16:
                        latest = struct.unpack_from('<4f', data, 0)
                if latest is not None and now >= script_until:
                    thr, yaw, pitch, roll = latest
                    held.update({'throttle': thr, 'roll': roll, 'pitch': pitch, 'yaw': yaw})
                    last_udp = now
                    emit('udp')
                elif now - last_udp > 0.5 and now >= script_until:
                    held.update({'throttle': -1.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0})
                    emit('neutral')
                else:
                    emit('held')
            else:
                emit('held')
            time.sleep(0.005 if sock is not None else 0.05)
    except KeyboardInterrupt:
        pass
    finally:
        pad.close()
        if log_f:
            log_f.close()
        print('pad released', flush=True)


def cmd_record(a):
    cfg = read_config()
    stream = (cfg or {}).get('StreamFormat', DEFAULT_STREAM)
    rx = TelemetryReceiver(port=a.port, stream=stream)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'recording telemetry to {out} for {a.seconds:.0f}s. Fly the drone yourself: take off, hover,')
    print('then give clear roll, pitch, yaw and throttle inputs one at a time, then fly around normally.')
    n = 0
    t0 = time.time()
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(TelemetryFrame.columns())
        while time.time() - t0 < a.seconds:
            fr = rx.wait(0.2)
            if fr is None:
                continue
            w.writerow(fr.as_row())
            n += 1
            if n % 500 == 0:
                print(f'  {n} frames, t={fr.timestamp:.1f}s', flush=True)
    rx.close()
    print(f'{n} frames written to {out}')


def cmd_autotest(a):
    """Fly the automated axis-check / identification sequence, then infer the mapping (and optionally fit)."""
    from .autotest import AutoTest
    from .sysid import infer_mapping, load_recording
    res = AutoTest(a.port, a.out, max_throttle=a.max_throttle, climb=a.climb).run()
    print(json.dumps({k: v for k, v in res.items()}, indent=1))
    if res['frames'] < 200:
        return
    rec = load_recording(a.out)
    mapping, notes = infer_mapping(rec)
    print('inferred mapping from this flight:')
    print(f'  stick_sign (roll, pitch, yaw) = {mapping.stick_sign}; gyro_axis = {mapping.gyro_axis}; gyro_sign = {mapping.gyro_sign}')
    print('  correlations:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in notes.items() if k.startswith('corr')})
    print('  throttle range seen:', notes['throttle_range'], ' max rpm:', notes['max_rpm_seen'])
    if not res['error'] and a.write_mapping:
        out = {'mapping': {'stick_sign': list(mapping.stick_sign), 'gyro_axis': list(mapping.gyro_axis),
                           'gyro_sign': list(mapping.gyro_sign), 'use_quat_rates': True, 'max_rpm': mapping.max_rpm,
                           'throttle_scale': 1.0, 'notes': notes,
                           'hover_stick': res.get('hover'), 'lift_stick': res.get('lift_stick')}}
        with open(a.write_mapping, 'w', encoding='utf-8') as f:
            yaml.safe_dump(out, f, sort_keys=False)
        print(f'mapping written to {a.write_mapping}')


def cmd_fit(a):
    from ..config import load_yaml
    from ..train.bptt import ExperimentConfig
    from .sysid import fit_dynamics, infer_mapping, load_recording
    cfg = ExperimentConfig.from_dict(load_yaml(a.config))
    rec = load_recording(a.csv)
    print(f'{len(rec["t"])} frames, {rec["t"][-1] - rec["t"][0]:.1f}s of flight')
    mapping, notes = infer_mapping(rec)
    print('inferred mapping:', json.dumps({k: v for k, v in notes.items()}, indent=1))
    print(f'  stick_sign (roll, pitch, yaw) = {mapping.stick_sign}; gyro_axis = {mapping.gyro_axis}, '
          f'gyro_sign = {mapping.gyro_sign}')
    res = fit_dynamics(rec, mapping, cfg.quad, cfg.ctl, cfg.rates, iters=a.iters)
    out = {
        'mapping': {'stick_sign': list(mapping.stick_sign), 'gyro_axis': list(mapping.gyro_axis),
                    'gyro_sign': list(mapping.gyro_sign), 'use_quat_rates': True, 'max_rpm': mapping.max_rpm,
                    'throttle_scale': 1.0, 'notes': notes},
        'quad': res['quad'], 'ctl': res['ctl'],
        'fit': {'windows': res['windows'], 'dt': res['dt'], 'final_loss': res['loss'][-1]},
    }
    with open(a.out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f'wrote {a.out}. Copy its quad/ctl sections into configs/train.yaml (or pass --config) before training,')
    print('so the brain is trained on a simulator that matches Liftoff.')


def load_mapping(path: str):
    from .pilot import LiftoffMapping
    if not path or not os.path.exists(path):
        print(f'no {path}; using default stick/gyro mapping (run `haltere liftoff fit` first for best results)')
        return LiftoffMapping()
    d = yaml.safe_load(open(path, encoding='utf-8')) or {}
    m = d.get('mapping', {})
    hs = m.get('hover_stick')
    curves = None
    if d.get('stick_curves'):
        from .stickcal import StickCurves
        curves = StickCurves(d['stick_curves'])
        print('stick curves loaded:\n' + curves.describe())
    radial = None
    if d.get('stick_model', {}).get('type') == 'radial':
        from .stickcal import RadialSticks
        sm = d['stick_model']
        radial = RadialSticks(float(sm.get('deadzone', 0.25)), sm.get('sign'))
        print('stick model loaded: ' + radial.describe())
    hp = m.get('hover_processed')
    return LiftoffMapping(stick_sign=tuple(m.get('stick_sign', (1, 1, 1))), gyro_axis=tuple(m.get('gyro_axis', (1, 0, 2))),
                          gyro_sign=tuple(m.get('gyro_sign', (1, 1, 1))), use_quat_rates=bool(m.get('use_quat_rates', True)),
                          max_rpm=float(m.get('max_rpm', 30000.0)), throttle_scale=float(m.get('throttle_scale', 1.0)),
                          hover_stick_game=(float(hs) if hs is not None else None),
                          hover_processed_game=(float(hp) if hp is not None else None),
                          stick_curves=curves, stick_model=radial, notes=m.get('notes', {}))


def cmd_sticktest(a):
    """Measure how the game processes the pad's sticks in 2D: hold raw stick combinations through the pad bridge
    (``pad --udp-in``) with the drone on the ground and record the telemetry's processed Input. Throttle stays below
    the deadzone, so the drone does not lift. With --fit, the radial model is fitted and stored in the mapping file."""
    import socket
    import struct
    from .stickcal import AXES, fit_radial
    if not a.fit_only:
        rx = TelemetryReceiver(port=a.port, stream=(read_config() or {}).get('StreamFormat', DEFAULT_STREAM))
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host, port = a.udp_out.split(':')
        sweep = np.round(np.linspace(-1, 1, 41), 3)
        steps = [(-1.0, v, 0.0, 0.0) for v in sweep] + [(-1.0, 0.0, v, 0.0) for v in sweep] + \
                [(-1.0, 0.0, 0.0, v) for v in sweep]
        steps += [(-1.0, r, v, 0.0) for r in (0.15, 0.3, 0.5) for v in sweep]           # roll held, pitch swept
        steps += [(th, 0.0, 0.0, v) for th in (-0.6, -0.35, -0.2, 0.0) for v in sweep]  # throttle held, yaw swept
        steps += [(v, 0.0, 0.0, y) for y in (0.0, 0.4) for v in np.round(np.linspace(-1, 0.1, 23), 3)]
        print(f'{len(steps)} stick combinations, {a.hold:.2f} s each ({len(steps) * a.hold:.0f} s); keep the game '
              f'focused with the drone on the ground', flush=True)
        with open(a.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['t', *AXES, *[f'p_{x}' for x in AXES]])
            t0 = time.time()
            for k, (thr, roll, pitch, yaw) in enumerate(steps):
                t_end = time.time() + a.hold
                while time.time() < t_end:
                    sock.sendto(struct.pack('<4f', thr, yaw, pitch, roll), (host, int(port)))
                    fr = rx.wait(0.01)
                    if fr is not None and time.time() > t_end - 0.4 * a.hold:     # the settled part of the hold
                        p_thr, p_yaw, p_pitch, p_roll = (float(x) for x in fr.input)
                        w.writerow([f'{time.time() - t0:.3f}', thr, roll, pitch, yaw, f'{p_thr:.4f}', f'{p_roll:.4f}',
                                    f'{p_pitch:.4f}', f'{p_yaw:.4f}'])
                if k % 50 == 0:
                    print(f'  {k}/{len(steps)}', flush=True)
            for _ in range(20):
                sock.sendto(struct.pack('<4f', -1.0, 0.0, 0.0, 0.0), (host, int(port)))
                time.sleep(0.02)
        rx.close()
    model, info = fit_radial(a.out)
    print(f'{model.describe()}; rms error {info["rms"]:.4f} over {info["samples"]} samples')
    if a.fit:
        d = yaml.safe_load(open(a.liftoff_config, encoding='utf-8')) if os.path.exists(a.liftoff_config) else {}
        d['stick_model'] = {'type': 'radial', 'deadzone': info['deadzone'], 'sign': model.sign,
                            'fit': {'csv': a.out, 'rms': round(info['rms'], 5), 'samples': info['samples']}}
        with open(a.liftoff_config, 'w', encoding='utf-8') as f:
            yaml.safe_dump(d, f, sort_keys=False)
        print(f'stick model written to {a.liftoff_config}')


def cmd_stickcal(a):
    """Fit Liftoff's per-axis input curves from a pad log + telemetry recording and store them in the mapping file."""
    from .stickcal import StickCurves, fit_curves
    extra = [tuple(float(x) for x in p.split(':')) for p in a.throttle_points.split(',')] if a.throttle_points else None
    curves = fit_curves(a.pad_log, a.csv, latency=a.latency, extra_throttle_points=extra)
    for ax, c in curves.items():
        print(f'{ax:8s}: {c["samples"]} samples; raw -> processed: ' +
              ', '.join(f'{r:+.2f}->{p:+.2f}' for r, p in list(zip(c['raw'], c['processed']))[::max(1, len(c['raw']) // 8)]))
    d = yaml.safe_load(open(a.out, encoding='utf-8')) if os.path.exists(a.out) else {}
    d['stick_curves'] = curves
    with open(a.out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(d, f, sort_keys=False)
    print('inverse curves:\n' + StickCurves(curves).describe())
    print(f'stick curves written to {a.out}')


def load_waypoints(spec: str, path: str) -> list | None:
    if path:
        d = yaml.safe_load(open(path, encoding='utf-8')) or {}
        pts = d.get('waypoints', d) if isinstance(d, dict) else d
        return [[float(x) for x in p] for p in pts]
    if spec:
        return [[float(x) for x in w.split(',')] for w in spec.split(';')]
    return None


def cmd_waypoints(a):
    """Turn a manually flown recording into a waypoint list (a race track or a freestyle line)."""
    from .sysid import load_recording
    rec = load_recording(a.csv)
    pos = rec['pos'] - rec['pos'][0]            # relative to the reset point, like the pilot's frame
    alt = pos[:, 2]
    airborne = np.flatnonzero(alt > a.min_alt)
    if len(airborne) == 0:
        raise SystemExit('no airborne samples in the recording')
    pts = [pos[airborne[0]]]
    for i in airborne[1:]:
        if np.linalg.norm(pos[i] - pts[-1]) >= a.spacing:
            pts.append(pos[i])
    pts = [np.array([p[0], p[1], min(max(p[2], a.min_z), a.max_z)]).round(2).tolist() for p in pts]
    with open(a.out, 'w', encoding='utf-8') as f:
        yaml.safe_dump({'source': a.csv, 'spacing_m': a.spacing, 'frame': 'sim (x forward, y left, z up), relative to the reset point',
                        'waypoints': pts}, f, sort_keys=False)
    print(f'{len(pts)} waypoints ({a.spacing} m apart) written to {a.out}; fly them with:')
    print(f'  haltere liftoff fly runs/imJ_best.pt --waypoints-file {a.out} --advance-radius 1.0')


def find_game_window(title_substring: str = 'Liftoff') -> int:
    """Handle of the first visible window whose title contains the text (0 when none). Windows only."""
    import ctypes
    import ctypes.wintypes as wt
    user32 = ctypes.windll.user32
    found = []

    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if title_substring.lower() in buf.value.lower():
                    found.append(hwnd)
        return True

    user32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(cb), 0)
    return found[0] if found else 0


def game_window_active(hwnd: int) -> bool:
    """True when the game window is in the foreground and not minimized (the only state in which Liftoff
    reads the controller)."""
    import ctypes
    user32 = ctypes.windll.user32
    return bool(hwnd) and user32.GetForegroundWindow() == hwnd and not user32.IsIconic(hwnd)


def focus_game_window(title_substring: str = 'Liftoff') -> bool:
    """Restore and bring the game window to the foreground. Liftoff ignores the controller while its window
    is unfocused and minimizes itself whenever it loses focus, so the pilot does this at start, before
    every keystroke it sends, and whenever it notices the focus is gone. Windows only; returns False when
    no such window exists."""
    import ctypes
    user32 = ctypes.windll.user32
    hwnd = find_game_window(title_substring)
    if not hwnd:
        return False
    for _ in range(3):
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)          # SW_RESTORE
            time.sleep(0.8)
        if user32.GetForegroundWindow() != hwnd:
            user32.SwitchToThisWindow(hwnd, True)   # allowed to steal the foreground, unlike SetForegroundWindow
            time.sleep(0.5)
        if user32.GetForegroundWindow() == hwnd and not user32.IsIconic(hwnd):
            return True
    return user32.GetForegroundWindow() == hwnd


def press_key_in_window(key: str, title_substring: str = 'Liftoff') -> bool:
    """Send a keystroke to the game window (brings it to the foreground first). Windows only."""
    import ctypes
    user32 = ctypes.windll.user32
    found = focus_game_window(title_substring)
    time.sleep(0.15)
    vk = user32.VkKeyScanW(ord(key[0])) & 0xFF
    scan = user32.MapVirtualKeyW(vk, 0)
    user32.keybd_event(vk, scan, 0, 0)
    time.sleep(0.08)
    user32.keybd_event(vk, scan, 2, 0)   # KEYEVENTF_KEYUP
    return bool(found)


def cmd_fly(a):
    import torch
    from ..train.bptt import load_checkpoint
    from .frames import unity_vec_to_sim
    from .pilot import TelemetryPilot
    brain, cfg, graph = load_checkpoint(a.ckpt, a.device)
    mapping = load_mapping(a.liftoff_config)
    if a.stick_model == 'curves' and mapping.stick_model is not None:
        mapping.stick_model = None
        print('stick model: per-axis curves (the radial model is ignored)')
    if mapping.hover_stick_game is not None:
        hover_cmd = (1.0 / cfg.quad.twr) ** (1.0 / cfg.quad.thrust_exp)
        mapping.hover_stick_sim = 2.0 * hover_cmd - 1.0
        if a.throttle_scale:
            mapping.throttle_scale = a.throttle_scale
        print(f'throttle remap: brain hover stick {mapping.hover_stick_sim:+.2f} -> game hover stick '
              f'{mapping.hover_stick_game:+.2f}, gain {mapping.throttle_scale:.2f}')
    offset = [float(x) for x in a.offset.split(',')]
    waypoints = load_waypoints(a.waypoints, a.waypoints_file)
    if a.gyro == 'telemetry':
        mapping.use_quat_rates = False
    sg = [float(x) for x in str(a.stick_gain).split(',')]
    stick_gain = sg[0] if len(sg) == 1 else sg
    pilot = TelemetryPilot(brain, cfg.task, mapping, brain.device, offset=offset, waypoints=waypoints, dwell=a.dwell,
                           advance_radius=a.advance_radius, loop=not a.no_loop, pattern=a.pattern, radius=a.radius,
                           period=a.period, amplitude=a.amplitude, stick_gain=stick_gain, stick_lpf=a.stick_lpf,
                           face_gain=a.face_travel, face_max=a.face_max)
    pilot.face_ahead = a.face_ahead
    pilot.flow_gain = a.flow_gain
    if getattr(a, 'sight', 'legacy') == 'rabbit':
        from .sightpilot import params_from_args
        if not a.vision:
            raise SystemExit('--sight rabbit flies by sight: give --vision')
        if a.stick_lpf > 0:
            raise SystemExit('--sight rabbit does not run with --stick-lpf (its dt is wrong and any stick lag hurts)')
        pilot.sight = 'rabbit'
        pilot.sight_params = params_from_args(a, flow_gain=a.flow_gain)
        print(f'SIGHT: rabbit pilot; {pilot.sight_params.describe()}')
    pilot.face_wobble_deg, pilot.face_wobble_period = a.face_wobble, a.face_wobble_period
    if a.vision:
        import yaml
        from ..vision.camera import Camera
        from ..vision.runtime import GateVision
        c = yaml.safe_load(Path(a.camera).read_text(encoding='utf-8'))
        cam = Camera(int(c['width']), int(c['height']), float(c['f']), float(c['tilt_deg']))
        pilot.vision = GateVision(a.vision, cam, window_title=a.capture or 'Liftoff', fps=a.vision_fps, device=a.device).start()
        print(f'VISION: goal from the gate detector {a.vision} on the game view ({cam.hfov_deg:.0f} deg FOV, '
              f'tilt {cam.tilt_deg:.0f} deg) at {a.vision_fps:.0f} fps; telemetry positions are not used for the goal')
    if a.pattern:
        print(f'pattern {a.pattern}: radius {a.radius} m, period {a.period} s, amplitude {a.amplitude} m')
    if a.path_speed > 0 and waypoints:
        pilot.path_speed = a.path_speed
        pilot.path_lookahead = a.lookahead
        pilot.path_z_lead = a.z_lead if a.z_lead >= 0 else None
        print(f'path following: {len(waypoints)} waypoints, {pilot.path_s[-1]:.0f} m loop at {a.path_speed} m/s, '
              f'lookahead {a.lookahead} m')
    recorder = None
    shared = None
    if a.record or a.show or a.dataset:
        from .recorder import FlightRecorder, SharedFlightState
        shared = SharedFlightState(brain.N)
        rect = tuple(int(x) for x in a.capture_rect.split(',')) if a.capture_rect else None
        recorder = FlightRecorder(shared, cfg.train.graph, out=a.record or None,
                                  capture=(a.capture if (a.record or a.dataset) else None), rect=rect, fps=a.fps,
                                  show=bool(a.show), dataset=a.dataset or None, dataset_every=a.dataset_every)
        recorder.start()
    tcfg = read_config()
    stream = (tcfg or {}).get('StreamFormat', DEFAULT_STREAM)
    rx = TelemetryReceiver(port=a.port, stream=stream)
    pad = None
    if a.udp_out:
        from .gamepad import UdpSticks
        host, port = a.udp_out.split(':')
        pad = UdpSticks(host, int(port))
        mode = f'sticks -> udp://{a.udp_out}'
    elif not a.dry_run:
        from .gamepad import VirtualPad
        pad = VirtualPad()
        mode = 'virtual pad active'
    else:
        mode = 'DRY RUN (no gamepad)'
    print(f'brain: {brain.N} neurons on {brain.device}; target offset {offset} m; {mode}. Ctrl+C to stop.')
    if not a.dry_run:   # no-op when there is no game window (dry runs against the stand-in)
        if focus_game_window(a.capture or 'Liftoff'):
            print('game window restored and focused')
            if a.reset_key:
                # start from the reset point in a clean pose: a drone left lying on its side after the previous
                # run arms and tumbles instead of taking off
                press_key_in_window(a.reset_key, a.capture or 'Liftoff')
                print(f'sent the reset key {a.reset_key!r} to the game')
                time.sleep(1.5)
    last_print = 0.0
    last_frame_time = time.time()
    t_begin = time.time()
    game_hwnd = 0 if a.dry_run else find_game_window(a.capture or 'Liftoff')
    last_focus_check = 0.0
    game_active = True
    ignored_since = None    # the game reports mid throttle while we hold it low -> it dropped the pad
    last_reconnect = 0.0
    last_reset_attempt = 0.0
    dists = []
    stick_hist = []
    armed_since = None      # Liftoff arms only after the throttle has been low; hold it low briefly, then ramp in
    last_reset_ts = None
    grounded_since = None
    still_since = None
    crashed = False
    flog = None
    if a.log:
        # every telemetry frame: the pilot's pose, the processed input the game applied, what the brain asked for and
        # what was sent, and the goal (the 0.5 s console lines alias anything faster than 1 Hz)
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        flog_f = open(a.log, 'w', newline='', encoding='utf-8')
        flog = csv.writer(flog_f)
        flog.writerow(['wall', 'ts', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz',
                       'in_thr', 'in_yaw', 'in_pitch', 'in_roll', 'rpm', 'b_thr', 'b_roll', 'b_pitch', 'b_yaw',
                       'c_thr', 'c_roll', 'c_pitch', 'c_yaw', 's_thr', 's_roll', 's_pitch', 's_yaw',
                       'gx', 'gy', 'gz', 'tx', 'ty', 'tz', 'phase', 'crashed', 'det_p', 'det_w', 'det_age',
                       *(SIGHT_LOG_COLUMNS if pilot.sight == 'rabbit' else []), 'status'])
    try:
        while a.seconds <= 0 or time.time() - t_begin < a.seconds:
            fr = rx.wait(0.05)
            now = time.time()
            if fr is None:
                if pad is not None and now - last_frame_time > 0.5:
                    pad.neutral()
                continue
            last_frame_time = now
            if last_reset_ts is None or fr.timestamp < last_reset_ts - 0.5:
                armed_since = now
                grounded_since = None
                crashed = False
            last_reset_ts = fr.timestamp
            sticks = pilot.step(fr)
            phase = now - armed_since
            if game_hwnd and now - last_focus_check > 1.0:
                # Liftoff drops the controller whenever another window takes the focus (and minimizes itself):
                # take the focus back, and do not mistake the unresponsive drone for a crash meanwhile
                last_focus_check = now
                was_active, game_active = game_active, game_window_active(game_hwnd)
                if not game_active:
                    game_active = focus_game_window(a.capture or 'Liftoff')
                    print(f'[{time.strftime("%H:%M:%S")}] game window had lost the focus; '
                          + ('restored' if game_active else 'could not restore it'), flush=True)
                    grounded_since = None
            # crash detection: on the ground and not moving while the brain asks for thrust -> Liftoff will not
            # arm again until the drone is reset; hold the throttle low (and press the reset button if configured)
            alt = float(pilot.last_pos[2])
            still = float(np.linalg.norm(fr.velocity)) < 0.05
            grounded = alt < 0.15 and still and sticks[0] > -0.5
            # wedged in a gate frame or a hay bale: in the air, not moving, while the brain pushes hard (a fast
            # lap flew into gate 4 and hung there with the sticks saturated; only a fall was detected before)
            stuck = alt >= 0.15 and still and (abs(sticks[1]) > 0.4 or abs(sticks[2]) > 0.4 or sticks[0] > 0.6)
            still_since = still_since if (still and still_since is not None) else (now if still else None)
            # perfectly still for 5 s while the goal is elsewhere: sitting on something (a hovering brain that has
            # reached its goal can be this still, so the goal distance decides)
            stuck = stuck or (alt >= 0.15 and still and now - still_since > 5.0 and len(dists) > 0 and dists[-1] > 1.0)
            if game_active and phase > a.arm_hold + a.arm_ramp + 1.0 and (grounded or stuck):
                grounded_since = grounded_since or now
                if now - grounded_since > (1.5 if grounded else 3.0) and not crashed:
                    crashed = True
                    print(f'[{time.strftime("%H:%M:%S")}] drone appears {"crashed/grounded" if grounded else "stuck in the air"} '
                          f'at {np.round(pilot.last_pos, 2)}; '
                          f'holding throttle low' + (f', pressing {a.reset_button}' if a.reset_button else '')
                          + (f', sending key {a.reset_key}' if a.reset_key else ''), flush=True)
                    if a.reset_button and pad is not None and hasattr(pad, 'press'):
                        pad.press(a.reset_button)
                    if a.reset_key:
                        press_key_in_window(a.reset_key, a.capture or 'Liftoff')
                        last_reset_attempt = now
            else:
                grounded_since = None
            if crashed:
                sticks = np.array([-1.0, 0.0, 0.0, 0.0])
                # the game did not reset (no timestamp jump): the key did not reach it (window not in front) -> retry
                if a.reset_key and now - last_reset_attempt > 5.0:
                    last_reset_attempt = now
                    print(f'[{time.strftime("%H:%M:%S")}] still grounded after the reset key: sending {a.reset_key!r} again'
                          + ('' if game_window_active(game_hwnd) else ' (the game window is not in front)'), flush=True)
                    press_key_in_window(a.reset_key, a.capture or 'Liftoff')
            elif phase < a.arm_hold:
                sticks = np.array([-1.0, 0.0, 0.0, 0.0])
            elif phase < a.arm_hold + a.arm_ramp:
                f = (phase - a.arm_hold) / a.arm_ramp
                sticks[0] = -1.0 + f * (sticks[0] + 1.0)
                sticks[1:] *= f
            if pad is not None:
                pad.send(sticks[0], sticks[1], sticks[2], sticks[3])
                # Liftoff now and then stops reading the pad (its Input then sits at mid throttle whatever we
                # send); re-plugging the virtual pad brings it back
                if game_active and game_hwnd and sticks[0] <= -0.9 and float(fr.input[0]) > -0.5:
                    ignored_since = ignored_since or now
                    if now - ignored_since > 1.5 and now - last_reconnect > 8.0 and hasattr(pad, 'reconnect'):
                        last_reconnect = now
                        print(f'[{time.strftime("%H:%M:%S")}] Liftoff is not reading the pad (input {np.round(fr.input, 2)} '
                              f'while sending throttle low): re-plugging the virtual pad', flush=True)
                        pad.reconnect()
                        armed_since = now + 1.0     # arm again once the new pad is seen
                        ignored_since = None
                else:
                    ignored_since = None
            dists.append(float(np.linalg.norm(pilot.last_pos - pilot.last_target)))
            stick_hist.append(sticks.copy())
            if flog is not None:
                det = pilot.vision.get() if pilot.vision is not None else None
                v = unity_vec_to_sim(fr.velocity)
                flog.writerow([f'{now:.4f}', f'{fr.timestamp:.4f}', *np.round(pilot.last_pos, 4), *np.round(v, 4),
                               *np.round(pilot.last_quat, 5), *np.round(pilot.omega, 4), *np.round(fr.input, 4),
                               round(float(np.mean(fr.motor_rpm)), 1), *np.round(pilot.last_brain, 4),
                               *np.round(pilot.last_cmd, 4), *np.round(sticks, 4), *np.round(pilot.last_rel_b, 3),
                               *np.round(pilot.last_target, 3), round(phase, 3), int(crashed),
                               round(det.p_visible, 3) if det else '', round(det.width_px, 1) if det else '',
                               round(now - det.t, 3) if det else '',
                               *(_sight_log_cells(pilot, det) if pilot.sight == 'rabbit' else []),
                               pilot.vision_status.replace(',', ';') if pilot.vision is not None else ''])
            if recorder is not None:
                shared.publish(pilot.rates() if len(dists) % 2 == 0 else None, t=fr.timestamp - (pilot.t_start or 0.0),
                               dist=dists[-1], thr=sticks[0], roll=sticks[1], pitch=sticks[2], yaw=sticks[3],
                               px=pilot.last_pos[0], py=pilot.last_pos[1], pz=pilot.last_pos[2],
                               tx=pilot.last_target[0], ty=pilot.last_target[1], tz=pilot.last_target[2],
                               waypoint=pilot.wp_index, n_waypoints=len(pilot.waypoints),
                               qw=pilot.last_quat[0], qx=pilot.last_quat[1], qy=pilot.last_quat[2], qz=pilot.last_quat[3],
                               ts=fr.timestamp)
            if now - last_print > 0.5:
                last_print = now
                print(f't={fr.timestamp:7.2f}s pos={np.round(pilot.last_pos, 2)} target={np.round(pilot.last_target, 2)} '
                      f'dist={dists[-1]:.2f}m sticks thr={sticks[0]:+.2f} r={sticks[1]:+.2f} p={sticks[2]:+.2f} '
                      f'y={sticks[3]:+.2f}' + (f' | {pilot.vision_status}' if pilot.vision is not None else ''), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if pilot.vision is not None:
            pilot.vision.stop()
        if flog is not None:
            flog_f.close()
        if pad is not None:
            pad.close()
        rx.close()
        if recorder is not None:
            recorder.stop()
        if dists:
            d = np.asarray(dists)
            half = d[len(d) // 2:]
            print(f'{len(d)} frames; distance to target: mean {d.mean():.2f} m, second half mean {half.mean():.2f} m, '
                  f'within 0.5 m {100 * (half < 0.5).mean():.0f}% of the time', flush=True)
            if len(stick_hist) > 10:
                S = np.asarray(stick_hist)[len(stick_hist) // 2:]
                print(f'stick std [thr,roll,pitch,yaw] {np.round(S.std(0), 3)}; mean |change| per frame '
                      f'{np.round(np.abs(np.diff(S, axis=0)).mean(0), 4)}', flush=True)


def _sight_log_cells(pilot, det) -> list[str]:
    """The rabbit pilot's numeric log columns as CSV cells (blank for NaN, so liftoff score reads them)."""
    from .sightpilot import SightPilot
    vals = pilot.sightpilot.log_values(det) if pilot.sightpilot is not None else SightPilot.empty_log_values(det)
    out = []
    for v in vals:
        v = float(v)
        out.append('' if not np.isfinite(v) else (f'{v:.4f}' if abs(v) >= 1e5 else f'{v:.4g}' if v != int(v) else str(int(v))))
    return out


def cmd_score(a):
    from .flightlog import describe, score_log
    out = {}
    for path in a.logs:
        res = score_log(path, a.gates or None, a.track or None)
        out[path] = res
        print(describe(Path(path).stem, res))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1)


def cmd_fake(a):
    from .fake_liftoff import FakeLiftoff
    from ..sim.quad import QuadParams
    quad = QuadParams(twr=a.twr, thrust_exp=a.thrust_exp, motor_tau=0.04, drag_lin=(0.08, 0.08, 0.12))
    FakeLiftoff(telemetry_port=a.port, stick_port=a.stick_port, quad=quad, autopilot=not a.no_autopilot,
                seed=a.seed).run(a.seconds)
