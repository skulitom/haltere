"""Experimental camera-to-connectome flight/shadow runner; no navigation teacher.

Default is shadow mode (no controller is opened). --udp-out explicitly connects
the motor output to an already running, throttle-low virtual-pad bridge.
"""
from __future__ import annotations

import argparse
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
    def __init__(self, title='Liftoff', fps=18):
        self.title, self.fps = title, fps
        self.latest = None
        self.error = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run,daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.done.set()
        self.thread.join(timeout=2)

    def run(self):
        import cv2
        import mss
        from .recorder import _capture_game_frame
        try:
            with mss.mss() as screen:
                while not self.done.is_set():
                    begin = time.monotonic()
                    rgb = _capture_game_frame(screen,self.title)
                    if rgb is not None:
                        small = cv2.resize(rgb,(640,360),interpolation=cv2.INTER_LINEAR)
                        ok,enc = cv2.imencode('.jpg',cv2.cvtColor(small,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90])
                        if not ok:
                            raise RuntimeError('Image encoding failed')
                        small = cv2.cvtColor(cv2.imdecode(enc,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
                        small = cv2.resize(small,(160,90),interpolation=cv2.INTER_AREA)
                        with torch.no_grad():
                            retina = retina_input(torch.tensor(small.transpose(2,0,1)[None],dtype=torch.float32)/255)
                        self.latest = (time.monotonic(),retina)
                    self.done.wait(max(0,1/self.fps-(time.monotonic()-begin)))
        except Exception as e:
            self.error = repr(e)


class VisualController:
    """The exported fly brain controls all four axes; no external target/yaw logic."""
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
    def step(self,frame,retina):
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
        obs = visual_observation(self.senses,self.motor,self.cfg.task,retina.to(self.brain.device))
        action,self.state,_ = self.brain(obs,self.state,self.W)
        processed = brain_to_processed(action,self.calibration)[0].cpu().numpy()
        raw = np.clip(self.mapping.to_raw(action[0].cpu().numpy()),-1,1)
        return action[0].cpu().numpy(),processed,raw


def flight_limit_reason(position, velocity, max_height, max_speed, max_distance):
    """Bound experimental control attempts; these limits do not steer the drone."""
    if position[2] > max_height:
        return 'Flight height limit exceeded'
    if np.linalg.norm(velocity) > max_speed:
        return 'Flight speed limit exceeded'
    if np.linalg.norm(position[:2]) > max_distance:
        return 'Flight distance limit exceeded'
    return None


def run(args):
    from .gamepad import UdpSticks
    from .recorder import FlightRecorder, SharedFlightState
    from .manual_recording import live_pose
    if not 0 < args.seconds <= 120:
        raise ValueError('Use a bounded run of 0 < seconds <= 120')
    if not all(np.isfinite(v) and v>0 for v in (args.max_height,args.max_speed,args.max_distance)):
        raise ValueError('Use finite positive flight limits')
    log_path = Path(args.log)
    if log_path.exists() or log_path.with_suffix('.json').exists() or (args.record and Path(args.record).exists()):
        raise FileExistsError('Use new log and video paths')
    torch.set_num_threads(2)
    controller = VisualController(args.checkpoint,args.mapping,args.device)
    camera = RetinaCamera().start()
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
    last_timestamp = None
    try:
        with log_path.open('w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['wall','ts','image_age','shadow','thr','roll','pitch','yaw',
                             'processed_thr','processed_roll','processed_pitch','processed_yaw','x','y','z',
                             'in_thr','in_yaw','in_pitch','in_roll','raw_thr','raw_roll','raw_pitch','raw_yaw',
                             'vx','vy','vz','qw','qx','qy','qz'])
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
                capture_time,retina = camera.latest
                if now-capture_time>.12:
                    raise RuntimeError(f'Image stale or game hidden: {camera.error}')
                if now<next_tick:
                    continue
                if now-next_tick>.12 and count:
                    raise RuntimeError('Controller missed its real-time deadline')
                next_tick = max(next_tick+controller.cfg.brain.dt,now)
                action,processed,raw = controller.step(frame,retina)
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
                                 *frame.input,*raw,*velocity,*q])
                count += 1
                if shared:
                    rates = (controller.brain.cfg.rate_max*torch.sigmoid(controller.state['v'][:,0])).cpu().numpy()
                    shared.publish(rates,t=elapsed,dist=0,thr=raw[0],roll=raw[1],pitch=raw[2],yaw=raw[3],
                                   px=pos[0],py=pos[1],pz=pos[2],tx=pos[0],ty=pos[1],tz=pos[2],
                                   qw=q[0],qx=q[1],qy=q[2],qz=q[3],ts=frame.timestamp)
    except Exception as e:
        reason = str(e)
        raise
    finally:
        if pad:
            pad.neutral()
            pad.close()
        camera.stop()
        rx.close()
        if recorder:
            recorder.stop()
        result = dict(checkpoint_sha256=sha256(args.checkpoint),runtime_requires_teacher=False,
                      control_mode='visual fly brain' if pad else 'shadow: no control output',
                      ticks=count,wall_s=time.monotonic()-begin,stop_reason=reason,
                      external_goal=False,yaw_assistance=False,
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
    p.add_argument('--max-height',type=float,default=8.)
    p.add_argument('--max-speed',type=float,default=10.)
    p.add_argument('--max-distance',type=float,default=20.)
    run(p.parse_args())


if __name__=='__main__':
    main()
