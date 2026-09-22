"""Experimental camera-to-connectome flight/shadow runner, with optional pilot assistance.

Default is shadow mode (no controller is opened). --udp-out explicitly connects
the motor output to an already running, throttle-low virtual-pad bridge.
--pilot-assistance rabbit adds visual guidance, speed scheduling and yaw control.
The exported brain's training teachers are not loaded by either mode.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import gc
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
    def __init__(self, title='Liftoff', fps=24, gate_sensor=None, backend='mss',phase_status=None,on_frame=None,
                 race_cues=False,detector_device='cpu'):
        self.title, self.fps = title, fps
        self.gate_sensor = gate_sensor
        self.backend = backend
        self.phase_status = phase_status
        self.on_frame = on_frame
        self.race_cues = race_cues
        self.capture = None
        self.detector = None
        self.detector_device = torch.device(detector_device)
        if gate_sensor:
            from ..vision.train import load_gatenet
            from ..vision.camera import Camera
            if sha256(gate_sensor['checkpoint']) != gate_sensor['sha256']:
                raise ValueError('Camera detector differs from the trained sensory contract')
            if self.detector_device.type == 'cuda':
                # Keep the trained float32 sensory contract on either device.
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            self.detector = load_gatenet(gate_sensor['checkpoint'],self.detector_device)
            self.gate_camera = Camera(320,180,gate_sensor['focal_320'],gate_sensor['tilt_deg'])
            with torch.no_grad():
                self.detector(torch.zeros(1,3,180,320,device=self.detector_device))
        self.latest = None
        self.error = None
        self.phase = 'starting'
        self.phase_started = time.monotonic()
        self.timings = deque(maxlen=4096)
        self.frames = 0
        self.missing_frames = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run,daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.done.set()
        if self.capture is not None:
            self.capture.close()
        self.thread.join(timeout=2)

    def _phase(self, name):
        now = time.monotonic()
        self.phase, self.phase_started = name, now
        if self.phase_status is not None:
            from .camera_process import PHASES
            self.phase_status[0] = PHASES.index(name)
            self.phase_status[1] = now
        return now

    def diagnostics(self):
        samples = list(self.timings)
        result = dict(phase=self.phase,phase_age_ms=1000*(time.monotonic()-self.phase_started),
                      frames=self.frames,timing_samples=len(samples),missing_frames=self.missing_frames,error=self.error,
                      detector_device=str(self.detector_device))
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
            if self.backend=='dxgi':
                from .game_capture import DxGameCapture
                self.capture = DxGameCapture(self.title)
            with (self.capture if self.capture is not None else getattr(mss,'MSS',mss.mss)()) as screen:
                while not self.done.is_set():
                    begin = self._phase('capture')
                    capture_time = begin
                    if self.capture is not None:
                        frame = screen.read()
                        rgb = None if frame is None else frame[1]
                        if frame is not None:
                            capture_time = frame[0]
                    else:
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
                                pixels = torch.tensor(gate_rgb.transpose(2,0,1)[None],dtype=torch.float32,
                                                      device=self.detector_device)/255
                                pred = decode(self.detector(pixels))[0].cpu().numpy()
                            p,u,v,width = pred
                            direction,distance = detection_geometry(self.gate_camera,(u+1)*160,(v+1)*90,width)
                            detection = dict(p=float(p),point=direction*distance,width=float(width))
                        if (self.detector is not None and self.gate_sensor.get('raw_retina_active',False)
                                and self.gate_sensor.get('retina_mode')=='gatenet_scene_v1'):
                            from ..vision.scene_features import scene_map,project_scene
                            with torch.no_grad():
                                retina=project_scene(scene_map(self.detector,pixels),self.gate_sensor['scene_projection'])
                        elif self.detector is None or self.gate_sensor.get('raw_retina_active',False):
                            small = cv2.resize(small,(160,90),interpolation=cv2.INTER_AREA)
                            with torch.no_grad():
                                retina = retina_input(torch.tensor(small.transpose(2,0,1)[None],dtype=torch.float32)/255,
                                                      mode=(self.gate_sensor or {}).get('retina_mode','legacy'))
                        else:
                            retina = torch.zeros(1,720)
                        if self.race_cues:
                            from ..vision.race_cues import checkpoint_ring
                            cue = checkpoint_ring(rgb)
                            if detection is None:
                                detection = dict(p=0., point=np.zeros(3), width=0.)
                            detection['race_cue'] = cue
                        inferred = self._phase('publish')
                        # The controller and shared-memory camera transport are
                        # CPU consumers even when image inference uses CUDA.
                        retina = retina.detach().cpu()
                        self.latest = (capture_time,retina,detection)
                        if self.on_frame is not None:
                            self.on_frame(self.latest)
                        self.frames += 1
                        published = self._phase('wait')
                        self.timings.append((captured-begin,prepared-captured,inferred-prepared,
                                             published-inferred,published-begin))
                    else:
                        self.missing_frames += 1
                    self.done.wait(max(0,1/self.fps-(time.monotonic()-begin)))
        except Exception as e:
            self.error = repr(e)


class VisualController:
    """Visual brain with an explicit optional guidance/yaw assistant."""
    def __init__(self, checkpoint, mapping_path, device='cuda', *, stop_on_search_timeout=True,
                 pilot_assistance='none', assist_speed=2., motor_controller='brain'):
        if pilot_assistance not in ('none', 'rabbit', 'race-cue'):
            raise ValueError('Unknown pilot assistance mode')
        if motor_controller not in ('brain', 'pd'):
            raise ValueError('Unknown motor controller')
        if motor_controller == 'pd' and pilot_assistance != 'race-cue':
            raise ValueError('PD comparison requires the frozen race-cue guidance')
        # Offline demonstration replay must preserve missing-gate observations
        # even when the demonstrator safely continues beyond the live stop limit.
        # Every live caller retains the default stop condition.
        self.stop_on_search_timeout = stop_on_search_timeout
        self.brain,self.cfg,self.graph = load_checkpoint(checkpoint,device)
        ck = torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.meta = ck.get('visual_brain')
        if not self.meta or self.meta.get('runtime_requires_teacher') is not False:
            raise ValueError('Expected an exported, teacher-free visual brain')
        if self.meta.get('gate_sensor',{}).get('raw_retina_active',False):
            scene=self.meta.get('scene_training',{})
            if (not scene or scene.get('iteration',0)<scene.get('iterations',1)
                    or self.meta['gate_sensor'].get('retina_mode') not in ('scene_v2','gatenet_scene_v1')):
                raise ValueError('Expected a completed scene-adapter training checkpoint')
            if (self.meta['gate_sensor']['retina_mode']=='gatenet_scene_v1'
                    and self.meta['gate_sensor'].get('scene_projection',{}).get('detector_sha256')
                    !=self.meta['gate_sensor']['sha256']):
                raise ValueError('Scene features require their exact frozen detector')
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
        self.W = self.brain.inference_matrix() if self.brain.device.type=='cpu' else self.brain.weight_matrix().detach()
        self.last_ts = None
        self.senses = self.motor = None
        self.gate_point = None
        self.gate_time = None
        self.last_detection_time = None
        self.relative_gate = np.zeros(3)
        self.gate_confidence = 0.
        self.searching = False
        self.search_since = None
        self.search_height = None
        from .gate_memory import PassedGateMemory
        inhibition = self.meta.get('gate_sensor',{}).get('passed_gate_inhibition')
        if inhibition not in (None, PassedGateMemory.contract):
            raise ValueError('Unknown passed-gate sensory memory contract')
        self.passed_gate_memory = PassedGateMemory() if inhibition else None
        from .camera_pose import CameraPoseHistory
        self.camera_poses = CameraPoseHistory()
        self.assistance = None
        if pilot_assistance == 'rabbit':
            from .visual_assistance import VisualPilotAssistance
            self.assistance = VisualPilotAssistance(self.meta.get('gate_sensor'),self.camera_poses,assist_speed)
        elif pilot_assistance == 'race-cue':
            from .race_cue_assistance import RaceCueAssistance
            reference = self.meta.get('motor_tracking',{}).get('nominal_speed_mps',2.)
            self.assistance = RaceCueAssistance(self.meta.get('gate_sensor'),self.camera_poses,assist_speed,
                                                reference_speed=reference)
        self.assistance_mode = pilot_assistance
        self.motor_controller = motor_controller
        self.motor_baseline = None
        self.motor_metadata = dict(kind='brain', brain_controls_motors=True)
        if motor_controller == 'pd':
            from dataclasses import replace
            from ..brain.motor_baseline import MotorPD, MotorPDConfig
            from .fit_vertical import equivalent_power_curve
            profile = self.meta['gate_training']['dynamics']['profile']
            measured = profile['vertical_calibration']['mean']
            curve = equivalent_power_curve(measured, c, idle=self.cfg.ctl.idle)
            self.motor_baseline = MotorPD(replace(self.cfg.quad, **curve), self.cfg.rates, self.cfg.ctl.idle,
                                          MotorPDConfig(position_gain=max(.8, self.assistance.reference_speed/3.)))
            self.motor_metadata = dict(**self.motor_baseline.metadata(), vertical_fit=measured,
                                       effective_thrust_curve=curve, mixer_idle=self.cfg.ctl.idle)
        self.last_command = None
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
    def step(self,frame,retina,detection=None,capture_time=None,frame_time=None,observation_time=None):
        observation_time = time.monotonic() if observation_time is None else observation_time
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
        if self.assistance is not None:
            from ..brain.gate_senses import gate_observation
            self.relative_gate,assisted_senses = self.assistance.update(
                self.senses,self.pose.omega,detection,capture_time,observation_time)
            self.gate_confidence = detection['p'] if detection else 0.
            self.gate_point = self.assistance.pilot.carrot
            target = self.assistance.pilot.target
            self.gate_time = target.t_last if target is not None else None
            self.searching = False  # Rabbit owns search; neural-search timeouts do not apply.
            sensor = self.meta['gate_sensor']
            height_error = torch.zeros(1,1,device=self.brain.device) if sensor.get('search_height_anchor') else None
            obs = gate_observation(assisted_senses,self.motor,self.cfg.task,retina.to(self.brain.device),
                                   torch.tensor(self.relative_gate,dtype=torch.float32,device=self.brain.device)[None],
                                   height_invariant=sensor.get('height_invariant',False),
                                   gravity_aligned_height=sensor.get('gravity_aligned_height',False),
                                   search_height_error=height_error,
                                   raw_retina_active=sensor.get('raw_retina_active',False))
        elif self.meta.get('gate_sensor'):
            from ..brain.gate_senses import gate_observation
            from ..vision.camera import quat_wxyz_to_mat
            R = quat_wxyz_to_mat(self.senses['quat'][0].cpu().numpy())
            pos = self.senses['pos'][0].cpu().numpy()
            if self.passed_gate_memory is not None:
                passed = self.passed_gate_memory.advance(pos,observation_time)
                if passed is not None and self.gate_point is not None and np.linalg.norm(self.gate_point-passed)<2.:
                    self.gate_point = self.gate_time = None
            self.gate_confidence = detection['p'] if detection else 0.
            if capture_time != self.last_detection_time:
                self.last_detection_time = capture_time
                if detection and detection['p']>.8 and np.isfinite(detection['point']).all():
                    capture_pos,capture_quat = self.camera_poses.at(capture_time)
                    point = capture_pos+quat_wxyz_to_mat(capture_quat)@detection['point']
                    point[2] -= self.meta['gate_sensor']['centre_offset_m']
                    # Smooth only measured position, using odometry to remove
                    # camera rotation. No route or steering controller exists here.
                    if self.passed_gate_memory is None or not self.passed_gate_memory.rejects(point,observation_time,pos):
                        self.gate_point,self.gate_time = fuse_gate_detection(
                            self.gate_point,self.gate_time,point,capture_time,pos,R)
            if self.passed_gate_memory is not None:
                self.passed_gate_memory.observe(self.gate_point,self.gate_time,pos,observation_time)
            relative = R.T@(self.gate_point-pos) if self.gate_point is not None else np.zeros(3)
            # Memory must age on the control clock even if capture stops. Never
            # grant an old point a new lifetime by repeatedly reading its frame.
            age = observation_time-self.gate_time if self.gate_time is not None else float('inf')
            allow_search = self.meta['gate_sensor'].get('missing_gate')=='zero_goal_neural_search'
            self.relative_gate,self.searching = gate_measurement_or_search(age,relative,allow_search)
            if self.searching:
                if self.search_since is None:
                    self.search_height=float(pos[2])
                self.search_since = observation_time if self.search_since is None else self.search_since
                if self.stop_on_search_timeout and observation_time-self.search_since>15.:
                    raise RuntimeError('Neural gate search timed out')
                # A newly acquired gate must not be blended with an expired
                # target. This clears sensory memory; it supplies no steering.
                self.gate_point = self.gate_time = None
            else:
                self.search_since = None
                self.search_height = None
            height_error=None
            if self.meta['gate_sensor'].get('search_height_anchor',False):
                delta=float(pos[2])-self.search_height if self.search_height is not None else 0.
                height_error=torch.tensor([[delta]],dtype=torch.float32,device=self.brain.device)
            obs = gate_observation(self.senses,self.motor,self.cfg.task,retina.to(self.brain.device),
                                   torch.tensor(self.relative_gate,dtype=torch.float32,device=self.brain.device)[None],
                                   height_invariant=self.meta['gate_sensor'].get('height_invariant',False),
                                   gravity_aligned_height=self.meta['gate_sensor'].get('gravity_aligned_height',False),
                                   search_height_error=height_error,
                                   raw_retina_active=self.meta['gate_sensor'].get('raw_retina_active',False))
        self.last_observation = obs
        action,self.state,_ = self.brain(obs,self.state,self.W)
        brain_action = action[0].cpu().numpy()
        motor_action = brain_action
        if self.motor_baseline is not None:
            # Identical causal target and assisted velocity observations. The
            # cap matches the brain's trained target; slower settings use the
            # same sensory scaling in RaceCueAssistance for both motors.
            motor_action = self.motor_baseline.command(
                assisted_senses, action.new_tensor(self.relative_gate)[None],
                speed=self.assistance.reference_speed)[0].cpu().numpy()
        command = self.assistance.command(motor_action) if self.assistance is not None else motor_action.copy()
        self.last_command = command
        processed = brain_to_processed(action.new_tensor(command)[None],self.calibration)[0].cpu().numpy()
        raw = np.clip(self.mapping.to_raw(command),-1,1)
        return brain_action,processed,raw


def fuse_gate_detection(previous,previous_time,point,stamp,position,rotation):
    """Associate static visual landmarks without using course order or geometry.

    Close-up size regression can jump several metres to a distant arch. Retain
    a recent nearby point through that outlier, without refreshing its age. A
    passed/expired target can be replaced immediately rather than interpolating
    between two different gates. All positions originate in camera measurements.
    """
    if previous is None or previous_time is None:
        return point,stamp
    relative=rotation.T@(previous-position)
    age=stamp-previous_time
    innovation=np.linalg.norm(point-previous)
    if np.linalg.norm(relative)<6. and relative[0]>-.5 and 0<=age<=2. and innovation>2.:
        return previous,previous_time
    if age>2. or relative[0]<=-.5 or innovation>8.:
        return point,stamp
    return .8*previous+.2*point,stamp


def camera_measurement_fresh(age,allow_memory,foreground=True,allow_braking=False):
    """A short image gap can use odometry-backed memory, never stale pixels.

    This is restricted to gate/search-trained brains. Raw-retina controllers
    have no trained landmark memory and retain their original 120 ms deadline.
    """
    if not foreground:
        raise RuntimeError('Game hidden during visual control')
    limit = .5 if allow_memory and allow_braking else .25 if allow_memory else .12
    if not np.isfinite(age) or age<0 or age>limit:
        raise RuntimeError('Fresh camera measurements unavailable')
    return age<=.12


def gate_memory_valid(age, relative_gate):
    """Bridge a close arch briefly leaving the camera; never extend frame freshness.

    Range and direction come solely from the last camera measurement, translated
    with odometry. Far/off-axis missing targets still expire after half a second.
    This retains a sensory point and never supplies a motor command.
    """
    nearby = np.linalg.norm(relative_gate) < 8. and relative_gate[0] > -1.
    return 0 <= age <= (2. if nearby else .5)


def gate_measurement_or_search(age, relative_gate, allow_search=False):
    if gate_memory_valid(age,relative_gate):
        return relative_gate,False
    if not allow_search:
        raise RuntimeError('Camera gate measurement unavailable')
    return np.zeros(3),True


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


def telemetry_still_progressing(receiver):
    """Confirm live simulation before Escape; a recent cached frame isn't enough.

    An agent/user pause can race the 120 ms telemetry deadline. Sending Escape
    based on the cached frame would then resume the game during pad shutdown.
    Drain that frame and require fresh progress across a bounded quiet interval.
    A short export/game stall can outlast the first check; allow recovery for
    up to 1.4 s without ever resuming control or toggling a paused/results screen.
    """
    from .manual_recording import live_pose
    before = receiver.poll() or receiver.last
    if before is None or not live_pose(before):
        return False
    time.sleep(.2)
    for attempt in range(25):
        after = receiver.poll()
        if after is not None:
            if not live_pose(after) or after.timestamp < before.timestamp:
                return False
            if after.timestamp-before.timestamp > .03:
                return True
        if attempt < 24:
            time.sleep(.05)
    return False


def run(args):
    from .gamepad import UdpSticks
    from .recorder import FlightRecorder, SharedFlightState
    from .manual_recording import live_pose
    from .flight_guard import ImpactMonitor
    from .camera_process import ProcessRetinaCamera
    from .commands import find_game_window,game_window_active
    if not 0 < args.seconds <= 1800:
        raise ValueError('Use a bounded run of 0 < seconds <= 1800')
    if not all(np.isfinite(v) and v>0 for v in (args.max_height,args.max_speed,args.max_distance)):
        raise ValueError('Use finite positive flight limits')
    log_path = Path(args.log)
    if log_path.exists() or log_path.with_suffix('.json').exists() or (args.record and Path(args.record).exists()):
        raise FileExistsError('Use new log and video paths')
    torch.set_num_threads(2)
    controller = VisualController(args.checkpoint,args.mapping,args.device,
                                  pilot_assistance=getattr(args,'pilot_assistance','none'),
                                  assist_speed=getattr(args,'assist_speed',2.),
                                  motor_controller=getattr(args,'motor_controller','brain'))
    from .neural_replay import NeuralReplay,replay_camera_sensor
    replay_out = getattr(args,'replay_out','')
    replay = NeuralReplay(replay_out,controller.brain.channel_dims,
                          int(args.seconds/controller.cfg.brain.dt)+100,
                          passive_retina=args.blank_retina or not controller.meta.get('gate_sensor',{}).get('raw_retina_active',False)) if replay_out else None
    camera_sensor = replay_camera_sensor(controller.meta.get('gate_sensor'),bool(replay))
    gc.collect()
    camera = ProcessRetinaCamera(gate_sensor=camera_sensor,backend=args.capture_backend,fps=48,
                                  race_cues=controller.assistance_mode=='race-cue',
                                  detector_device=getattr(args,'vision_device','cpu')).start()
    pad = None
    if args.udp_out:
        host,port = args.udp_out.rsplit(':',1)
        pad = UdpSticks(host,int(port))
    rx = TelemetryReceiver(port=args.port,stream=(read_config() or {}).get('StreamFormat',DEFAULT_STREAM))
    shared = SharedFlightState(controller.brain.N) if args.record else None
    recorder = FlightRecorder(shared,controller.cfg.train.graph,out=args.record,capture='Liftoff',fps=18,
                              controller_label='PD MOTOR CONTROL | BRAIN IN SHADOW'
                              if controller.motor_baseline else '') if shared else None
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
    # Collection can pause every Python thread. Reclaim startup objects before
    # the bounded real-time interval, retaining normal reference-count cleanup.
    # Set this after spawning the recorder so video encoding keeps its existing
    # background priority. The independent camera raises its own priority.
    from .scheduling import flight_process_priority
    priority = flight_process_priority()
    game_hwnd = find_game_window('Liftoff')
    memory_gap_allowed = controller.meta.get('gate_sensor',{}).get('missing_gate')=='zero_goal_neural_search'
    camera_braking_allowed = memory_gap_allowed and controller.assistance_mode=='race-cue'
    gc_was_enabled = gc.isenabled()
    gc.disable()
    begin, next_tick, count, reason = time.monotonic(),time.monotonic(),0,'duration'
    frame, last_frame, first_ts, last_progress = None,begin,None,begin
    camera_failure = None
    impacts = ImpactMonitor()
    last_timestamp = None
    step_times = deque(maxlen=4096)
    deadline_failure = None
    telemetry_failure = None
    memory_only_ticks = 0
    max_image_age = 0.
    try:
        with log_path.open('w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['wall','ts','image_age','shadow','thr','roll','pitch','yaw',
                             'processed_thr','processed_roll','processed_pitch','processed_yaw','x','y','z',
                             'in_thr','in_yaw','in_pitch','in_roll','raw_thr','raw_roll','raw_pitch','raw_yaw',
                             'vx','vy','vz','qw','qx','qy','qz','gate_p','gate_bx','gate_by','gate_bz','gate_age',
                             'capture_time','frame_time','det_bx','det_by','det_bz','det_width','neural_search',
                             'motor_mean','omega_x','omega_y','omega_z','search_height_reference',
                             'command_thr','command_roll','command_pitch','command_yaw',
                             'pilot_assisted','pilot_mode','pilot_flow_gain','pilot_passes','phase',
                             'pilot_kind','cue_u','cue_v','cue_edge','cue_aim_u','motor_controller'])
            while time.monotonic()-begin < args.seconds:
                new = rx.wait(.001)
                camera_frame = camera.latest
                now = time.monotonic()
                if new is not None:
                    frame,last_frame = new,now
                    if new.timestamp != last_timestamp:
                        last_progress = now
                        last_timestamp = new.timestamp
                startup_image_stale = count==0 and camera_frame is not None and now-camera_frame[0]>.12
                if frame is None or camera_frame is None or startup_image_stale:
                    if pad:
                        pad.neutral()
                    if now-begin>5:
                        raise RuntimeError(f'No fresh live image/telemetry: {camera.error}')
                    continue
                if now-last_frame>.12 or now-last_progress>.5 or not live_pose(frame):
                    telemetry_failure = dict(receipt_age_ms=1000*(now-last_frame),
                                             progress_age_ms=1000*(now-last_progress),
                                             timestamp=float(frame.timestamp),
                                             live_pose=bool(live_pose(frame)),
                                             received_frames=rx.frames,invalid_packets=rx.bad)
                    raise RuntimeError('Telemetry stale, paused or outside live flight')
                capture_time,retina,detection = camera_frame
                image_age = now-capture_time
                max_image_age = max(max_image_age,image_age)
                try:
                    fresh = camera_measurement_fresh(image_age,memory_gap_allowed,game_window_active(game_hwnd),
                                                     allow_braking=camera_braking_allowed)
                    if camera.error:
                        raise RuntimeError(f'Camera failed: {camera.error}')
                except RuntimeError:
                    camera_failure = camera.diagnostics()
                    raise
                if not fresh:
                    retina,detection = torch.zeros_like(retina),None
                if args.blank_retina:
                    retina = torch.zeros_like(retina)
                if now<next_tick:
                    continue
                if now-next_tick>.12 and count:
                    deadline_failure = dict(stage='loop',late_ms=1000*(now-next_tick))
                    raise RuntimeError('Controller missed its real-time deadline')
                next_tick = max(next_tick+controller.cfg.brain.dt,now)
                step_begin = time.monotonic()
                action,processed,raw = controller.step(frame,retina,detection,capture_time,last_frame,now)
                memory_only_ticks += int(not fresh)
                step_times.append(time.monotonic()-step_begin)
                if not np.isfinite(np.concatenate((action,processed,raw))).all():
                    raise RuntimeError('Nonfinite motor output')
                first_ts = frame.timestamp if first_ts is None else first_ts
                elapsed = frame.timestamp-first_ts
                pos = controller.senses['pos'][0].cpu().numpy()
                velocity = controller.senses['vel_world'][0].cpu().numpy()
                q = controller.senses['quat'][0].cpu().numpy()
                if pad:
                    if time.monotonic()-now>.12:
                        deadline_failure = dict(stage='brain',late_ms=1000*(time.monotonic()-now))
                        raise RuntimeError('Brain step missed its real-time deadline; command discarded')
                    if impacts.update(frame.timestamp,pos,velocity,q):
                        raise RuntimeError('Impact detected from flight motion')
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
                writer.writerow([time.time(),frame.timestamp,now-capture_time,not bool(pad),*action,*processed,*pos,
                                 *frame.input,*raw,*velocity,*q,controller.gate_confidence,*controller.relative_gate,
                                 now-controller.gate_time if controller.gate_time is not None else -1,
                                 capture_time,last_frame,*(detection['point'] if detection else [0.,0.,0.]),
                                 detection['width'] if detection else 0.,controller.searching,
                                 float(controller.motor[0,0]),*controller.pose.omega,
                                 controller.search_height if controller.search_height is not None else float('nan'),
                                 *controller.last_command,controller.assistance is not None,
                                 controller.assistance.pilot.mode if controller.assistance else -1,
                                 controller.assistance.host.flow_gain if controller.assistance else 1.,
                                 controller.assistance.pilot.n_passes if controller.assistance else 0,elapsed,
                                 {'none':0,'rabbit':1,'race-cue':2}[controller.assistance_mode],
                                 *((detection.get('race_cue') or {}).get(k,-1) if detection else -1
                                   for k in ('u','v','edge','aim_u')),controller.motor_controller])
                if replay is not None:
                    replay.append(controller.last_observation,retina,action,pos,q,
                                  frame.timestamp,now,capture_time,fresh)
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
            if args.pause_on_stop and telemetry_still_progressing(rx):
                try:
                    pause_key_sent = pause_active_game()
                except OSError:
                    pass
        camera_status = camera.diagnostics()
        camera.stop()
        rx.close()
        if recorder:
            recorder.stop()
        if gc_was_enabled:
            gc.enable()
        assisted = controller.assistance is not None
        pilot_meta = controller.assistance.metadata() if assisted else dict(mode='none')
        if controller.motor_baseline:
            pilot_meta['motor_control'] = 'PD throttle/roll/pitch; assisted yaw; brain runs in shadow'
        result = dict(checkpoint_sha256=sha256(args.checkpoint),runtime_requires_teacher=False,
                      control_mode=('PD motor baseline; brain in shadow' if controller.motor_baseline else
                                    'pilot-assisted visual fly brain' if assisted else 'visual fly brain') if pad else 'shadow: no control output',
                      motor_controller=controller.motor_metadata,
                      pilot_assistance=pilot_meta,
                      navigation_predictor_loaded=False,
                      brain_action_columns=['thr','roll','pitch','yaw'],
                      command_columns=['command_thr','command_roll','command_pitch','command_yaw'],
                      raw_output_includes_arming_hold=True,
                      brain_device=str(controller.brain.device),
                      vision_device=getattr(args,'vision_device','cpu'),
                      telemetry_failure=telemetry_failure,
                      process_priority=priority,
                      image_freshness_s=.12,camera_outage_limit_s=.5 if camera_braking_allowed else .25 if memory_gap_allowed else .12,
                      camera_braking_after_s=.25 if camera_braking_allowed else None,
                      stale_image_memory_only_ticks=memory_only_ticks,max_image_age_ms=1000*max_image_age,
                      recurrent_matrix_layout=str(controller.W.layout),
                      ticks=count,wall_s=time.monotonic()-begin,stop_reason=reason,
                      external_goal=bool(controller.meta.get('gate_sensor')),yaw_assistance=assisted,
                      goal_source=controller.assistance.metadata()['goal_source'] if assisted else ('camera detector' if controller.meta.get('gate_sensor') else 'absent'),
                      visible_race_cues=controller.assistance_mode=='race-cue',
                      gate_sensor=controller.meta.get('gate_sensor'),runtime_route_oracle=False,
                      raw_retina_active=not args.blank_retina and (not controller.meta.get('gate_sensor')
                                         or controller.meta['gate_sensor'].get('raw_retina_active',False)),
                      camera_fps=camera.fps,
                      capture_backend=camera.backend,
                      close_gate_memory_s=2.,
                      close_gate_outlier_rejection=dict(radius_m=6.,innovation_m=2.,max_age_s=2.),
                      passed_gate_memory=dict(estimated_passages=controller.passed_gate_memory.passages,
                                              rejected_detections=controller.passed_gate_memory.rejected_detections,
                                              radius_m=2.,reverse_radius_m=5.,lifetime_s=30.) if controller.passed_gate_memory and not assisted else None,
                      neural_search_timeout_s=None if assisted else 15.,
                      camera_pose_alignment='interpolated telemetry receipt times',
                      pause_key_sent=pause_key_sent,
                      impact=impacts.impact,
                      camera_diagnostics=camera_status,camera_failure=camera_failure,
                      controller_step_ms=dict(p50=float(np.percentile(step_times,50)*1000),
                                              p95=float(np.percentile(step_times,95)*1000),
                                              max=float(max(step_times)*1000)) if step_times else None,
                      controller_deadline_failure=deadline_failure,
                      origin_sim=controller.pose.pos0.tolist() if controller.pose.pos0 is not None else None,
                      images_blanked=args.blank_retina,
                      limits=dict(height_m=args.max_height,speed_mps=args.max_speed,distance_m=args.max_distance),
                      process_session=windows_session_id())
        if replay is not None:
            result['neural_replay'] = replay.save()
            result['neural_replay']['sha256'] = sha256(replay.path)
            result['neural_replay']['pilot_assistance'] = controller.assistance_mode
            result['neural_replay']['action_source'] = ('shadow brain; not commanded' if controller.motor_baseline else
                                                      'brain before pilot yaw; senses include assistance' if assisted else 'brain')
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
    p.add_argument('--motor-controller',choices=['brain','pd'],default='brain',
                   help='PD is a matched diagnostic baseline; its brain panel is explicitly labelled as shadow')
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--log',required=True)
    p.add_argument('--record',default='')
    p.add_argument('--replay-out',default='',help='Save exact causal senses and passive frozen scene features for offline correction')
    p.add_argument('--udp-out',default='')
    p.add_argument('--port',type=int,default=9001)
    p.add_argument('--device',default='cuda')
    p.add_argument('--vision-device',choices=['cpu','cuda'],default='cpu',
                   help='Device for the frozen image model, independent of the brain device')
    p.add_argument('--blank-retina',action='store_true',help='diagnostic ablation; zero image input, same brain weights')
    p.add_argument('--pilot-assistance',choices=['none','rabbit','race-cue'],default='none',
                   help='Rabbit arch guidance or explicitly game-cue-assisted race guidance around the chosen motor controller')
    p.add_argument('--assist-speed',type=float,default=2.,help='Requested speed (0 < speed <= 5), capped by the trained motor reference; set per vehicle, not per course')
    p.add_argument('--pause-on-stop',action='store_true',help='Pause the foreground game when a live control attempt ends')
    p.add_argument('--capture-backend',choices=['mss','dxgi'],default='mss',help='DXGI uses original Windows frame timestamps')
    p.add_argument('--max-height',type=float,default=8.)
    p.add_argument('--max-speed',type=float,default=10.)
    p.add_argument('--max-distance',type=float,default=20.)
    run(p.parse_args())


if __name__=='__main__':
    main()
