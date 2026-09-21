"""Experimental camera-to-connectome flight/shadow runner; no navigation teacher.

Default is shadow mode (no controller is opened). --udp-out explicitly connects
the motor output to an already running, throttle-low virtual-pad bridge.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ..brain.retina import retina_input, visual_observation, brain_to_processed
from ..train.bptt import load_checkpoint
from ..vision.datasets import sha256
from .commands import load_mapping
from .pilot import TelemetryPilot
from .telemetry import TelemetryReceiver, read_config, DEFAULT_STREAM


class RetinaCamera:
    def __init__(self, title='Liftoff', fps=24, gate_sensor=None):
        self.title, self.fps = title, fps
        self.gate_sensor = gate_sensor
        self.detector = None
        if gate_sensor:
            from ..vision.train import load_gatenet
            from ..vision.camera import Camera
            if sha256(gate_sensor['checkpoint']) != gate_sensor['sha256']:
                raise ValueError('Camera detector differs from the trained sensory contract')
            self.detector = load_gatenet(gate_sensor['checkpoint'],'cpu')
            self.gate_camera = Camera(320,180,gate_sensor['focal_320'],gate_sensor['tilt_deg'])
            with torch.no_grad():
                self.detector(torch.zeros(1,3,180,320))
        self.latest = None
        self.error = None
        self.phase = 'starting'
        self.phase_started = time.monotonic()
        self.timings = deque(maxlen=4096)
        self.missing_frames = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run,daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.done.set()
        self.thread.join(timeout=2)

    def _phase(self, name):
        now = time.monotonic()
        self.phase, self.phase_started = name, now
        return now

    def diagnostics(self):
        samples = list(self.timings)
        result = dict(phase=self.phase,phase_age_ms=1000*(time.monotonic()-self.phase_started),
                      frames=len(samples),missing_frames=self.missing_frames,error=self.error)
        if samples:
            values = np.asarray(samples)*1000
            result['stages_ms'] = {name:dict(p50=float(np.percentile(values[:,i],50)),
                                            p95=float(np.percentile(values[:,i],95)),max=float(values[:,i].max()))
                                   for i,name in enumerate(('capture','preprocess','inference','publish','total'))}
        return result

    def run(self):
        import cv2
        import mss
        from .recorder import _capture_game_frame
        try:
            with mss.mss() as screen:
                while not self.done.is_set():
                    begin = self._phase('capture')
                    rgb = _capture_game_frame(screen,self.title)
                    captured = self._phase('preprocess')
                    if rgb is not None:
                        small = cv2.resize(rgb,(640,360),interpolation=cv2.INTER_LINEAR)
                        ok,enc = cv2.imencode('.jpg',cv2.cvtColor(small,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90])
                        if not ok:
                            raise RuntimeError('Image encoding failed')
                        small = cv2.cvtColor(cv2.imdecode(enc,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
                        prepared = self._phase('inference')
                        detection = None
                        if self.detector is not None:
                            from ..vision.model import decode
                            from ..vision.runtime import detection_geometry
                            gate_rgb = cv2.resize(small,(320,180),interpolation=cv2.INTER_AREA)
                            with torch.no_grad():
                                pixels = torch.tensor(gate_rgb.transpose(2,0,1)[None],dtype=torch.float32)/255
                                pred = decode(self.detector(pixels))[0].cpu().numpy()
                            p,u,v,width = pred
                            direction,distance = detection_geometry(self.gate_camera,(u+1)*160,(v+1)*90,width)
                            detection = dict(p=float(p),point=direction*distance,width=float(width))
                        if self.detector is None:
                            small = cv2.resize(small,(160,90),interpolation=cv2.INTER_AREA)
                            with torch.no_grad():
                                retina = retina_input(torch.tensor(small.transpose(2,0,1)[None],dtype=torch.float32)/255)
                        else:
                            retina = torch.zeros(1,720)
                        inferred = self._phase('publish')
                        self.latest = (begin,retina,detection)
                        published = self._phase('wait')
                        self.timings.append((captured-begin,prepared-captured,inferred-prepared,
                                             published-inferred,published-begin))
                    else:
                        self.missing_frames += 1
                    self.done.wait(max(0,1/self.fps-(time.monotonic()-begin)))
        except Exception as e:
            self.error = repr(e)


class VisualController:
    """All four axes come from the brain; optional camera gate measurement only."""
    def __init__(self, checkpoint, mapping_path, device='cuda'):
        self.brain,self.cfg,self.graph = load_checkpoint(checkpoint,device)
        ck = torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.meta = ck.get('visual_brain')
        if not self.meta or self.meta.get('runtime_requires_teacher') is not False:
            raise ValueError('Expected an exported, teacher-free visual brain')
        self.mapping = load_mapping(mapping_path)
        self.calibration = self.meta['calibration']
        c = self.calibration
        if self.mapping.hover_processed_game is None or not np.allclose(self.mapping.stick_sign,c['stick_sign']) or not np.isclose(
                self.mapping.hover_processed_game,c['hover_processed']):
            raise ValueError('Live controller mapping differs from training')
        self.mapping.hover_stick_sim = c['hover_stick_sim']
        self.mapping.throttle_scale = c['throttle_scale']
        self.mapping.max_rpm = c['max_rpm']
        if sha256(Path(self.cfg.train.graph).with_suffix('.npz')) != self.meta['graph_sha256']:
            raise ValueError('Connectome differs from the trained model')
        self.pose = TelemetryPilot(self.brain,self.cfg.task,self.mapping,self.brain.device)
        self.state = self.brain.init_state(1)
        self.W = self.brain.weight_matrix().detach()
        self.last_ts = None
        self.senses = self.motor = None
        self.gate_point = None
        self.gate_time = None
        self.last_detection_time = None
        self.relative_gate = np.zeros(3)
        self.gate_confidence = 0.
        from .camera_pose import CameraPoseHistory
        self.camera_poses = CameraPoseHistory()
        # CUDA's first sparse call can take hundreds of milliseconds. Initialize
        # kernels before the timed flight, then discard this synthetic state.
        with torch.no_grad():
            obs = {k:torch.zeros(1,d,device=self.brain.device) for k,d in self.brain.channel_dims.items()}
            warm = self.brain.init_state(1)
            for _ in range(8):
                _,warm,_ = self.brain(obs,warm,self.W)
            if self.brain.device.type=='cuda':
                torch.cuda.synchronize(self.brain.device)

    @torch.no_grad()
    def step(self,frame,retina,detection=None,capture_time=None,frame_time=None):
        if self.last_ts is not None and frame.timestamp < self.last_ts-.1:
            raise RuntimeError('Game reset; stop this attempt')
        if self.last_ts != frame.timestamp:
            previous = self.pose.prev_quat.copy() if self.pose.prev_quat is not None else None
            prev_omega = self.pose.omega.copy()
            delta = frame.timestamp-self.last_ts if self.last_ts is not None else self.cfg.brain.dt
            self.senses = self.pose.sensors(frame)
            if previous is not None:
                from .frames import omega_from_quats
                alpha = 1-.5**(max(.001,delta)/self.cfg.brain.dt)
                omega = (1-alpha)*prev_omega+alpha*omega_from_quats(previous,self.pose.prev_quat,max(.001,delta))
                self.pose.omega = omega
                self.senses['gyro'] = torch.tensor(omega,dtype=torch.float32,device=self.brain.device)[None]
            self.last_ts = frame.timestamp
            motor = float(np.clip(np.mean(frame.motor_rpm)/self.mapping.max_rpm,0,1))
            self.motor = torch.tensor([[motor]],device=self.brain.device)
            self.camera_poses.append(time.monotonic() if frame_time is None else frame_time,
                                     self.senses['pos'][0].cpu().numpy(),self.senses['quat'][0].cpu().numpy())
        obs = visual_observation(self.senses,self.motor,self.cfg.task,retina.to(self.brain.device))
        if self.meta.get('gate_sensor'):
            from ..brain.gate_senses import gate_observation
            from ..vision.camera import quat_wxyz_to_mat
            R = quat_wxyz_to_mat(self.senses['quat'][0].cpu().numpy())
            pos = self.senses['pos'][0].cpu().numpy()
            if capture_time != self.last_detection_time:
                self.last_detection_time = capture_time
                self.gate_confidence = detection['p'] if detection else 0.
                if detection and detection['p']>.8 and np.isfinite(detection['point']).all():
                    capture_pos,capture_quat = self.camera_poses.at(capture_time)
                    point = capture_pos+quat_wxyz_to_mat(capture_quat)@detection['point']
                    point[2] -= self.meta['gate_sensor']['centre_offset_m']
                    # Smooth only measured position, using odometry to remove
                    # camera rotation. No route or steering controller exists here.
                    alpha = 1. if self.gate_point is None else .2
                    self.gate_point = point if self.gate_point is None else (1-alpha)*self.gate_point+alpha*point
                    self.gate_time = capture_time
            if self.gate_time is None:
                raise RuntimeError('Camera gate measurement unavailable')
            self.relative_gate = R.T@(self.gate_point-pos)
            if not gate_memory_valid(capture_time-self.gate_time,self.relative_gate):
                raise RuntimeError('Camera gate measurement unavailable')
            obs = gate_observation(self.senses,self.motor,self.cfg.task,retina.to(self.brain.device),
                                   torch.tensor(self.relative_gate,dtype=torch.float32,device=self.brain.device)[None])
        action,self.state,_ = self.brain(obs,self.state,self.W)
        processed = brain_to_processed(action,self.calibration)[0].cpu().numpy()
        raw = np.clip(self.mapping.to_raw(action[0].cpu().numpy()),-1,1)
        return action[0].cpu().numpy(),processed,raw


def gate_memory_valid(age, relative_gate):
    """Bridge a close arch briefly leaving the camera; never extend frame freshness.

    Range and direction come solely from the last camera measurement, translated
    with odometry. Far/off-axis missing targets still expire after half a second.
    This retains a sensory point and never supplies a motor command.
    """
    nearby = np.linalg.norm(relative_gate) < 8. and relative_gate[0] > -1.
    return 0 <= age <= (2. if nearby else .5)


def flight_limit_reason(position, velocity, max_height, max_speed, max_distance):
    """Bound experimental control attempts; these limits do not steer the drone."""
    if position[2] > max_height:
        return 'Flight height limit exceeded'
    if np.linalg.norm(velocity) > max_speed:
        return 'Flight speed limit exceeded'
    if np.linalg.norm(position[:2]) > max_distance:
        return 'Flight distance limit exceeded'
    return None


def pause_active_game():
    """Pause only the still-foreground game, without changing desktop focus."""
    import ctypes
    from .commands import find_game_window, game_window_active
    if not game_window_active(find_game_window('Liftoff')):
        return False
    user32 = ctypes.windll.user32
    user32.keybd_event(0x1b,0x01,0,0)
    time.sleep(.04)
    user32.keybd_event(0x1b,0x01,2,0)
    return True


def run(args):
    from .gamepad import UdpSticks
    from .recorder import FlightRecorder, SharedFlightState
    from .manual_recording import live_pose
    if not 0 < args.seconds <= 1800:
        raise ValueError('Use a bounded run of 0 < seconds <= 1800')
    if not all(np.isfinite(v) and v>0 for v in (args.max_height,args.max_speed,args.max_distance)):
        raise ValueError('Use finite positive flight limits')
    log_path = Path(args.log)
    if log_path.exists() or log_path.with_suffix('.json').exists() or (args.record and Path(args.record).exists()):
        raise FileExistsError('Use new log and video paths')
    torch.set_num_threads(2)
    controller = VisualController(args.checkpoint,args.mapping,args.device)
    camera = RetinaCamera(gate_sensor=controller.meta.get('gate_sensor')).start()
    pad = None
    if args.udp_out:
        host,port = args.udp_out.rsplit(':',1)
        pad = UdpSticks(host,int(port))
    rx = TelemetryReceiver(port=args.port,stream=(read_config() or {}).get('StreamFormat',DEFAULT_STREAM))
    shared = SharedFlightState(controller.brain.N) if args.record else None
    recorder = FlightRecorder(shared,controller.cfg.train.graph,out=args.record,capture='Liftoff',fps=18) if shared else None
    if recorder:
        recorder.start()
        try:
            recorder.wait_ready()
        except Exception:
            recorder.stop()
            camera.stop()
            rx.close()
            if pad:
                pad.neutral()
                pad.close()
            raise
    log_path.parent.mkdir(parents=True,exist_ok=True)
    begin, next_tick, count, reason = time.monotonic(),time.monotonic(),0,'duration'
    frame, last_frame, first_ts, last_progress = None,begin,None,begin
    camera_failure = None
    last_timestamp = None
    try:
        with log_path.open('w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['wall','ts','image_age','shadow','thr','roll','pitch','yaw',
                             'processed_thr','processed_roll','processed_pitch','processed_yaw','x','y','z',
                             'in_thr','in_yaw','in_pitch','in_roll','raw_thr','raw_roll','raw_pitch','raw_yaw',
                             'vx','vy','vz','qw','qx','qy','qz','gate_p','gate_bx','gate_by','gate_bz','gate_age',
                             'capture_time','frame_time','det_bx','det_by','det_bz','det_width'])
            while time.monotonic()-begin < args.seconds:
                new = rx.wait(.001)
                now = time.monotonic()
                if new is not None:
                    frame,last_frame = new,now
                    if new.timestamp != last_timestamp:
                        last_progress = now
                        last_timestamp = new.timestamp
                if frame is None or camera.latest is None:
                    if pad:
                        pad.neutral()
                    if now-begin>5:
                        raise RuntimeError(f'No live image/telemetry: {camera.error}')
                    continue
                if now-last_frame>.12 or now-last_progress>.5 or not live_pose(frame):
                    raise RuntimeError('Telemetry stale, paused or outside live flight')
                capture_time,retina,detection = camera.latest
                if now-capture_time>.12:
                    camera_failure = camera.diagnostics()
                    raise RuntimeError(f'Image stale or game hidden: {camera.error}')
                if args.blank_retina:
                    retina = torch.zeros_like(retina)
                if now<next_tick:
                    continue
                if now-next_tick>.12 and count:
                    raise RuntimeError('Controller missed its real-time deadline')
                next_tick = max(next_tick+controller.cfg.brain.dt,now)
                action,processed,raw = controller.step(frame,retina,detection,capture_time,last_frame)
                if not np.isfinite(action).all():
                    raise RuntimeError('Nonfinite motor output')
                first_ts = frame.timestamp if first_ts is None else first_ts
                elapsed = frame.timestamp-first_ts
                pos = controller.senses['pos'][0].cpu().numpy()
                velocity = controller.senses['vel_world'][0].cpu().numpy()
                if pad:
                    limit = flight_limit_reason(pos,velocity,args.max_height,args.max_speed,args.max_distance)
                    if limit:
                        raise RuntimeError(limit)
                    if elapsed<1:
                        raw = np.array([-1.,0.,0.,0.])
                    else:
                        ramp = min(1.,(elapsed-1)/2)
                        raw[0] = -1+ramp*(raw[0]+1)
                        raw[1:] *= ramp
                    pad.send(*raw)
                q = controller.senses['quat'][0].cpu().numpy()
                writer.writerow([time.time(),frame.timestamp,now-capture_time,not bool(pad),*action,*processed,*pos,
                                 *frame.input,*raw,*velocity,*q,controller.gate_confidence,*controller.relative_gate,
                                 capture_time-controller.gate_time if controller.gate_time is not None else -1,
                                 capture_time,last_frame,*(detection['point'] if detection else [0.,0.,0.]),
                                 detection['width'] if detection else 0.])
                count += 1
                if shared:
                    rates = (controller.brain.cfg.rate_max*torch.sigmoid(controller.state['v'][:,0])).cpu().numpy()
                    target = controller.gate_point if controller.gate_point is not None else pos
                    shared.publish(rates,t=elapsed,dist=float(np.linalg.norm(target-pos)),thr=raw[0],roll=raw[1],pitch=raw[2],yaw=raw[3],
                                   px=pos[0],py=pos[1],pz=pos[2],tx=target[0],ty=target[1],tz=target[2],
                                   qw=q[0],qx=q[1],qy=q[2],qz=q[3],ts=frame.timestamp)
    except Exception as e:
        reason = str(e)
        raise
    finally:
        pause_key_sent = False
        if pad:
            pad.neutral()
            pad.close()
            # A terminal stop used to leave the drone falling while the video
            # encoder closed. Do not toggle an already-paused/stalled game.
            if args.pause_on_stop and time.monotonic()-last_progress<.15:
                try:
                    pause_key_sent = pause_active_game()
                except OSError:
                    pass
        camera_status = camera.diagnostics()
        camera.stop()
        rx.close()
        if recorder:
            recorder.stop()
        result = dict(checkpoint_sha256=sha256(args.checkpoint),runtime_requires_teacher=False,
                      control_mode='visual fly brain' if pad else 'shadow: no control output',
                      ticks=count,wall_s=time.monotonic()-begin,stop_reason=reason,
                      external_goal=bool(controller.meta.get('gate_sensor')),yaw_assistance=False,
                      goal_source='camera detector' if controller.meta.get('gate_sensor') else 'absent',
                      gate_sensor=controller.meta.get('gate_sensor'),runtime_route_oracle=False,
                      raw_retina_active=not (args.blank_retina or controller.meta.get('gate_sensor')),
                      camera_fps=camera.fps,
                      close_gate_memory_s=2.,
                      camera_pose_alignment='interpolated telemetry receipt times',
                      pause_key_sent=pause_key_sent,
                      camera_diagnostics=camera_status,camera_failure=camera_failure,
                      origin_sim=controller.pose.pos0.tolist() if controller.pose.pos0 is not None else None,
                      images_blanked=args.blank_retina,
                      limits=dict(height_m=args.max_height,speed_mps=args.max_speed,distance_m=args.max_distance),
                      process_session=windows_session_id())
        log_path.with_suffix('.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)


def windows_session_id():
    import ctypes
    import os
    session = ctypes.c_ulong()
    if not ctypes.windll.kernel32.ProcessIdToSessionId(os.getpid(),ctypes.byref(session)):
        raise OSError('Cannot determine process session')
    return session.value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint')
    p.add_argument('--mapping',required=True)
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--log',required=True)
    p.add_argument('--record',default='')
    p.add_argument('--udp-out',default='')
    p.add_argument('--port',type=int,default=9001)
    p.add_argument('--device',default='cuda')
    p.add_argument('--blank-retina',action='store_true',help='diagnostic ablation; zero image input, same brain weights')
    p.add_argument('--pause-on-stop',action='store_true',help='Pause the foreground game when a live control attempt ends')
    p.add_argument('--max-height',type=float,default=8.)
    p.add_argument('--max-speed',type=float,default=10.)
    p.add_argument('--max-distance',type=float,default=20.)
    run(p.parse_args())


if __name__=='__main__':
    main()
