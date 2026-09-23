"""Experimental camera-to-connectome flight/shadow runner, with optional pilot assistance.

Default is shadow mode (no controller is opened). --udp-out explicitly connects
the motor output to an already running, throttle-low virtual-pad bridge.
--pilot-assistance rabbit adds visual guidance, speed scheduling and yaw control.
The exported brain's training teachers are not loaded by either visual mode.
--collection-route is a separate privileged PD qualification/collection mode;
its footage and metadata explicitly exclude it from autonomous evaluation.
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
                 race_cues=False,detector_device='cpu',on_image=None):
        self.title, self.fps = title, fps
        self.on_image = on_image
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
                        image = small
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
                        if self.on_image is not None:
                            self.on_image(capture_time,image)
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
                 pilot_assistance='none', assist_speed=2., motor_controller='brain', collection_route=None,
                 dynamics_calibration=None, calibration_amplitudes=None, oracle_motor_diagnostic=False,
                 pilot_profile='standard', pd_profile='teacher', dynamics_profile=None):
        if pilot_profile not in ('standard', 'fast') or pd_profile not in ('teacher', 'fast'):
            raise ValueError('Unknown pilot or PD profile')
        if pilot_profile == 'fast' and pilot_assistance != 'race-cue':
            raise ValueError('The fast pilot profile is a race-cue guidance profile')
        if pd_profile == 'fast' and (motor_controller != 'pd' or pilot_profile != 'fast' or not dynamics_profile):
            raise ValueError('Fast PD requires PD motors, the fast race-cue profile and a measured dynamics profile')
        if dynamics_profile and pilot_profile != 'fast':
            raise ValueError('A dynamics profile is only used by the fast pilot and fast PD profiles')
        if pilot_assistance not in ('none', 'rabbit', 'race-cue'):
            raise ValueError('Unknown pilot assistance mode')
        if motor_controller not in ('brain', 'pd'):
            raise ValueError('Unknown motor controller')
        if oracle_motor_diagnostic and (not collection_route or motor_controller!='brain' or pilot_assistance!='none'):
            raise ValueError('Oracle motor diagnostic requires an explicit route, brain motors and no visual pilot')
        if collection_route and (motor_controller != 'pd' and not oracle_motor_diagnostic or pilot_assistance != 'none'):
            raise ValueError('Oracle collection requires explicit PD motors and no visual pilot mode')
        if dynamics_calibration and (motor_controller!='pd' or pilot_assistance!='none' or collection_route):
            raise ValueError('Dynamics calibration requires PD and no visual pilot or oracle route')
        if motor_controller == 'pd' and pilot_assistance != 'race-cue' and not collection_route and not dynamics_calibration:
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
        self.nominal_relative_gate = np.zeros(3)
        self.geometry_gate = None
        self.geometry_provider = None
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
        elif pilot_assistance == 'race-cue' and pilot_profile == 'fast':
            from .fast_race_cue import FastRaceCue, DEFAULT_YAW_CURVE
            reference = self.meta.get('motor_tracking',{}).get('nominal_speed_mps',2.)
            yaw_curve = DEFAULT_YAW_CURVE
            if dynamics_profile:
                import json
                axis = json.loads(Path(dynamics_profile).read_text())['axes']['yaw']
                if not axis.get('super_after_expo'):
                    raise ValueError('The fast pilot requires a measured post-expo yaw curve')
                yaw_curve = (axis['coefficient_deg_s'], axis['super_rate'], axis['expo'])
            fast_brain = self.meta.get('fast_motor_tracking') if motor_controller == 'brain' else None
            if fast_brain:
                reference = fast_brain['nominal_speed_mps']
                if assist_speed > reference+1e-9:
                    raise ValueError('The fast brain contract was trained up to its nominal speed')
            # A fast PD tracks the requested speed itself; other motor
            # controllers retain their trained reference as the ceiling.
            self.assistance = FastRaceCue(self.meta.get('gate_sensor'),self.camera_poses,assist_speed,
                                          reference_speed=assist_speed if pd_profile == 'fast' else reference,
                                          yaw_curve=yaw_curve,calibration=self.calibration,
                                          velocity_scale=fast_brain['velocity_scale'] if fast_brain else None)
        elif pilot_assistance == 'race-cue':
            from .race_cue_assistance import RaceCueAssistance
            reference = self.meta.get('motor_tracking',{}).get('nominal_speed_mps',2.)
            self.assistance = RaceCueAssistance(self.meta.get('gate_sensor'),self.camera_poses,assist_speed,
                                                reference_speed=reference)
        self.assistance_mode = pilot_assistance
        self.pilot_profile = pilot_profile
        self.fast_motor = None
        self.last_motor_time = None
        if dynamics_calibration:
            from .dynamics_calibration import DynamicsCalibration
            self.assistance = DynamicsCalibration(dynamics_calibration,c,self.cfg.rates,calibration_amplitudes)
            self.assistance_mode = 'dynamics-calibration'
        if collection_route:
            from .oracle_assistance import OracleCollectionAssistance
            self.assistance = OracleCollectionAssistance(collection_route,
                reference_speed=self.meta.get('motor_tracking',{}).get('nominal_speed_mps',2.),
                motor_controller=motor_controller)
            self.assistance_mode = 'oracle-route'
        self.motor_controller = motor_controller
        self.motor_baseline = None
        self.guidance_velocity = None
        self.motor_metadata = dict(kind='brain', brain_controls_motors=True)
        if self.assistance_mode in ('race-cue', 'oracle-route'):
            from .guidance_contract import GuidanceVelocityContract
            self.motor_speed = min(self.assistance.speed, self.assistance.reference_speed)
            self.guidance_velocity = GuidanceVelocityContract.brain(
                self.assistance.speed, self.assistance.reference_speed)
        if (motor_controller == 'brain' and pilot_profile == 'fast'
                and self.meta.get('fast_motor_tracking')):
            from .guidance_contract import GuidanceVelocityContract
            fast_brain = self.meta['fast_motor_tracking']
            # The brain's trained contract: goal = request * declared seconds.
            self.motor_speed = self.assistance.speed
            self.guidance_velocity = GuidanceVelocityContract(1./fast_brain['goal_seconds'],
                                                              1./fast_brain['vertical_goal_seconds'],
                                                              self.motor_speed, self.assistance.config.vertical_up)
            self.motor_metadata = dict(kind='brain', brain_controls_motors=True, contract='fast_velocity_brain_v1',
                                       fast_motor_tracking={k: fast_brain[k] for k in
                                                            ('nominal_speed_mps','goal_seconds','velocity_scale',
                                                             'vertical_goal_seconds','teacher')})
        if motor_controller == 'pd':
            from dataclasses import replace
            from ..brain.motor_baseline import MotorPD, MotorPDConfig
            from .fit_vertical import equivalent_power_curve
            profile = self.meta['gate_training']['dynamics']['profile']
            measured = profile['vertical_calibration']['mean']
            curve = equivalent_power_curve(measured, c, idle=self.cfg.ctl.idle)
            self.motor_speed = min(self.assistance.speed, self.assistance.reference_speed)
            self.motor_baseline = MotorPD(replace(self.cfg.quad, **curve), self.cfg.rates, self.cfg.ctl.idle,
                                          MotorPDConfig(position_gain=max(.8, self.motor_speed/3.)))
            from .guidance_contract import GuidanceVelocityContract
            self.guidance_velocity = GuidanceVelocityContract.pd(self.motor_baseline.config, self.motor_speed)
            self.motor_metadata = dict(**self.motor_baseline.metadata(), vertical_fit=measured,
                                       effective_thrust_curve=curve, mixer_idle=self.cfg.ctl.idle,
                                       speed_mps=self.motor_speed, velocity_input='unscaled measured motion',
                                       contract='motor_tracking_teacher_v1')
            if pd_profile == 'fast':
                from ..brain.motor_baseline import FastMotorPD
                profile_data = json.loads(Path(dynamics_profile).read_text())
                self.fast_motor = FastMotorPD(profile_data, c)
                self.motor_speed = self.assistance.speed
                self.guidance_velocity = GuidanceVelocityContract(1., 1., self.motor_speed,
                                                                  self.assistance.config.vertical_up)
                self.motor_metadata = dict(**self.fast_motor.metadata(), speed_mps=self.motor_speed,
                                           dynamics_profile=str(dynamics_profile),
                                           dynamics_profile_sha256=sha256(dynamics_profile),
                                           velocity_input='unscaled measured motion',
                                           contract='fast_velocity_pd_v1')
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
    def step(self,frame,retina,detection=None,capture_time=None,frame_time=None,observation_time=None,clearance=None):
        observation_time = time.monotonic() if observation_time is None else observation_time
        if self.last_ts is not None and frame.timestamp < self.last_ts-.1:
            raise RuntimeError('Game reset; stop this attempt')
        if self.last_ts != frame.timestamp:
            if self.assistance_mode in ('oracle-route','dynamics-calibration'):
                self.assistance.bind(frame)
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
            extra = dict(clearance=clearance) if clearance is not None and self.pilot_profile == 'fast' else {}
            self.relative_gate,assisted_senses = self.assistance.update(
                self.senses,self.pose.omega,detection,capture_time,observation_time,**extra)
            if self.pilot_profile == 'fast':
                from ..vision.camera import quat_wxyz_to_mat
                # Express the velocity request in each motor contract's own goal units.
                rotation=quat_wxyz_to_mat(self.senses['quat'][0].cpu().numpy())
                self.relative_gate=self.guidance_velocity.body_target(self.assistance.velocity_command,rotation)
            self.nominal_relative_gate = np.array(self.relative_gate,copy=True)
            if self.geometry_gate is not None:
                from ..vision.camera import quat_wxyz_to_mat
                rotation=quat_wxyz_to_mat(self.senses['quat'][0].cpu().numpy())
                desired=self.guidance_velocity.world_velocity(self.nominal_relative_gate, rotation)
                proposal=self.geometry_provider.latest()
                selected=self.geometry_gate.resolve(desired,self.senses['pos'][0].cpu().numpy(),
                                                      time.monotonic(),proposal)
                self.relative_gate=self.guidance_velocity.body_target(selected, rotation)
            self.gate_confidence = detection['p'] if detection else 0.
            self.gate_point = self.assistance.pilot.carrot
            if self.geometry_gate is not None:
                self.gate_point=self.senses['pos'][0].cpu().numpy()+rotation@self.relative_gate
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
        if self.fast_motor is not None:
            interval = float(np.clip(observation_time-self.last_motor_time,.005,.05)) if self.last_motor_time is not None else .01
            self.last_motor_time = observation_time
            motor_action = self.fast_motor.command(
                self.senses, action.new_tensor(self.assistance.velocity_command)[None],
                action.new_tensor(self.assistance.feedforward)[None], dt=interval)[0].cpu().numpy()
        elif self.motor_baseline is not None:
            # Match the training teacher: measured velocity and requested speed.
            # Sensory scaling belongs to the brain's fixed-speed readout only;
            # applying it to PD strengthens its velocity feedback incorrectly.
            # Both controllers retain the same causal target and speed cap.
            motor_action = self.motor_baseline.command(
                self.senses, action.new_tensor(self.relative_gate)[None],
                speed=self.motor_speed)[0].cpu().numpy()
        command = self.assistance.command(motor_action) if self.assistance is not None else motor_action.copy()
        self.last_command = command
        processed = brain_to_processed(action.new_tensor(command)[None],self.calibration)[0].cpu().numpy()
        raw = np.clip(self.mapping.to_raw(command),-1,1)
        return brain_action,processed,raw


def looming_row(sample, now):
    if not sample:
        return float('nan'),float('nan'),float('nan')
    return (sample['ttc'] if sample['ttc'] is not None else float('inf'),
            sample['distance'] if sample['distance'] is not None else float('inf'),now-sample['time'])


def clearance_row(assistance):
    governor = getattr(assistance,'clearance',None)
    if governor is None:
        return '',float('nan')
    return ('brake' if assistance.clearance_braking else governor.status,
            governor.cap if governor.cap is not None else float('nan'))


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
    camera_fps = getattr(args, 'camera_fps', 48.)
    geometry_control = getattr(args,'geometry_control',False)
    dense_checkpoint = getattr(args,'dense_obstacles',None)
    if dense_checkpoint and not (geometry_control or getattr(args,'geometry_shadow',False)):
        raise ValueError('Dense obstacle hypotheses require an explicit geometry mode')
    calibration_mode = getattr(args,'dynamics_calibration',None)
    if getattr(args,'calibration_amplitudes',None) is not None and calibration_mode in (None,'hover'):
        raise ValueError('Custom amplitudes require an explicit pulse calibration mode')
    if calibration_mode and (geometry_control or getattr(args,'geometry_shadow',False)
                             or not args.pause_on_stop or not args.udp_out):
        raise ValueError('Dynamics calibration requires explicit UDP control, pause-on-stop and no geometry mode')
    if geometry_control and (getattr(args,'geometry_shadow',False)
                             or getattr(args,'collection_route',None)
                             or getattr(args,'pilot_assistance','none')!='race-cue'):
        raise ValueError('Experimental geometry control requires race-cue guidance, no oracle route and no shadow flag')
    if getattr(args,'pilot_profile','standard')=='fast' and (geometry_control or getattr(args,'geometry_shadow',False)):
        raise ValueError('The fast pilot profile is not qualified with geometry guidance')
    if not np.isfinite(camera_fps) or not 12 <= camera_fps <= 60:
        raise ValueError('Camera rate must be between 12 and 60 fps for the freshness contract')
    copy_port = getattr(args, 'telemetry_copy_port', 0)
    if copy_port and (not 1 <= copy_port <= 65535 or copy_port == args.port):
        raise ValueError('Forward telemetry to a different valid local UDP port')
    log_path = Path(args.log)
    if log_path.exists() or log_path.with_suffix('.json').exists() or (args.record and Path(args.record).exists()):
        raise FileExistsError('Use new log and video paths')
    from .preflight import require_quiet
    preflight = require_quiet(log_path, getattr(args,'allow_workload_pid',()),getattr(args,'allow_workload_project',()))
    torch.set_num_threads(2)
    controller = VisualController(args.checkpoint,args.mapping,args.device,
                                  pilot_assistance=getattr(args,'pilot_assistance','none'),
                                  assist_speed=getattr(args,'assist_speed',2.),
                                  motor_controller=getattr(args,'motor_controller','brain'),
                                  collection_route=getattr(args,'collection_route',None),
                                  dynamics_calibration=calibration_mode,
                                  calibration_amplitudes=getattr(args,'calibration_amplitudes',None),
                                  oracle_motor_diagnostic=getattr(args,'oracle_motor_diagnostic',False),
                                  pilot_profile=getattr(args,'pilot_profile','standard'),
                                  pd_profile=getattr(args,'pd_profile','teacher'),
                                  dynamics_profile=getattr(args,'dynamics_profile',None))
    from .neural_replay import NeuralReplay,replay_camera_sensor
    replay_out = getattr(args,'replay_out','')
    replay = NeuralReplay(replay_out,controller.brain.channel_dims,
                          int(args.seconds/controller.cfg.brain.dt)+100,
                          passive_retina=args.blank_retina or not controller.meta.get('gate_sensor',{}).get('raw_retina_active',False)) if replay_out else None
    camera_sensor = replay_camera_sensor(controller.meta.get('gate_sensor'),bool(replay))
    geometry = None
    if getattr(args, 'geometry_shadow', False) or geometry_control:
        if controller.guidance_velocity is None:
            raise ValueError('Geometry requires a declared local target/velocity contract')
        if controller.motor_baseline is None and 'nominal_speed_mps' not in controller.meta.get('motor_tracking', {}):
            raise ValueError('Brain geometry requires a motor-tracking checkpoint with a declared reference speed')
        from .geometry_shadow import ProcessGeometryShadow,ProcessGeometryControl
        if geometry_control:
            from .geometry_control import GeometryControlGate
            geometry=ProcessGeometryControl(log_path.with_suffix('.geometry.jsonl'),camera_sensor,
                                            dense_checkpoint=dense_checkpoint,depth_device=getattr(args,'vision_device','cpu'))
            controller.geometry_gate=GeometryControlGate(speed=controller.motor_speed)
            controller.geometry_provider=geometry
        else:
            geometry = ProcessGeometryShadow(log_path.with_suffix('.geometry.jsonl'),camera_sensor,
                             source_route_oracle=controller.assistance_mode == 'oracle-route',
                             archive_images=getattr(args,'geometry_record_images',False),
                             dense_checkpoint=dense_checkpoint,depth_device=getattr(args,'vision_device','cpu'))
    gc.collect()
    looming = bool(getattr(args,'looming_brake',False))
    if looming and getattr(args,'pilot_profile','standard') != 'fast':
        raise ValueError('The looming brake is part of the fast pilot profile')
    camera = ProcessRetinaCamera(gate_sensor=camera_sensor,backend=args.capture_backend,fps=camera_fps,
                                  race_cues=controller.assistance_mode=='race-cue',
                                  detector_device=getattr(args,'vision_device','cpu'),looming=looming).start()
    pad = None
    if args.udp_out:
        host,port = args.udp_out.rsplit(':',1)
        pad = UdpSticks(host,int(port))
    rx = TelemetryReceiver(port=args.port,stream=(read_config() or {}).get('StreamFormat',DEFAULT_STREAM),
                           forward_port=copy_port or None)
    shared = SharedFlightState(controller.brain.N) if args.record else None
    recorder = FlightRecorder(shared,controller.cfg.train.graph,out=args.record,capture='Liftoff',fps=18,
                              encoder=getattr(args,'video_encoder','libx264'),
                              controller_label='SHADOW ONLY | NO CONTROL OUTPUT' if not pad else
                              ('VISUAL GEOMETRY | PD MOTORS | BRAIN IN SHADOW' if controller.motor_baseline else
                               'VISUAL GEOMETRY | BRAIN MOTORS | ASSISTED YAW') if geometry_control else
                              'DYNAMICS CALIBRATION | PD + PULSES | BRAIN IN SHADOW' if calibration_mode else
                              ('ORACLE ROUTE | PD MOTORS | BRAIN IN SHADOW' if controller.motor_baseline else
                               'ORACLE MOTOR DIAGNOSTIC | BRAIN MOTORS | ASSISTED YAW')
                              if controller.assistance_mode == 'oracle-route' else 'FAST PD MOTORS | FAST CUE PILOT | BRAIN IN SHADOW'
                              if controller.fast_motor is not None else 'PD MOTOR CONTROL | BRAIN IN SHADOW'
                              if controller.motor_baseline else 'BRAIN MOTORS | FAST CUE PILOT | ASSISTED YAW'
                              if controller.pilot_profile == 'fast' else '') if shared else None
    if recorder:
        try:
            recorder.start()
            recorder.wait_ready()
        except Exception:
            if recorder.proc.pid is not None:
                recorder.stop()
            camera.stop()
            rx.close()
            if pad:
                pad.neutral()
                pad.close()
            raise
    if geometry:
        try:
            geometry.start()
        except Exception:
            camera.stop()
            rx.close()
            if recorder:
                recorder.stop()
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
    previous_loop_phases = {}
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
                                 'pilot_kind','cue_u','cue_v','cue_edge','cue_aim_u','motor_controller',
                                 'nominal_gate_bx','nominal_gate_by','nominal_gate_bz','geometry_control_status',
                                 'rpm_lf','rpm_rf','rpm_lb','rpm_rb',
                                 'cmd_vx','cmd_vy','cmd_vz','pilot_state',
                                 'looming_ttc','looming_distance','looming_age','clearance_status','clearance_cap'])
            while time.monotonic()-begin < args.seconds:
                loop_mark = time.monotonic()
                loop_phases = {}
                if recorder and count % 100 == 0 and not recorder.proc.is_alive():
                    raise RuntimeError('Flight recorder stopped during the attempt')
                new = rx.wait(.001)
                loop_phases['telemetry_ms'] = 1000*(time.monotonic()-loop_mark)
                loop_mark = time.monotonic()
                camera_frame = camera.latest
                now = time.monotonic()
                loop_phases['camera_snapshot_ms'] = 1000*(now-loop_mark)
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
                    deadline_failure = dict(stage='loop',late_ms=1000*(now-next_tick),
                                            current_phases=loop_phases, previous_phases=previous_loop_phases)
                    raise RuntimeError('Controller missed its real-time deadline')
                next_tick = max(next_tick+controller.cfg.brain.dt,now)
                step_begin = time.monotonic()
                action,processed,raw = controller.step(frame,retina,detection,capture_time,last_frame,now,
                                                       clearance=camera.clearance if looming else None)
                memory_only_ticks += int(not fresh)
                step_times.append(time.monotonic()-step_begin)
                loop_phases['brain_ms'] = 1000*(time.monotonic()-step_begin)
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
                loop_mark = time.monotonic()
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
                                 {'none':0,'rabbit':1,'race-cue':2,'oracle-route':3,'dynamics-calibration':4}[controller.assistance_mode],
                                 *((detection.get('race_cue') or {}).get(k,-1) if detection else -1
                                   for k in ('u','v','edge','aim_u')),controller.motor_controller,
                                 *controller.nominal_relative_gate,
                                 controller.geometry_gate.status if controller.geometry_gate else 'disabled',
                                 *(list(frame.motor_rpm[:4])+[float('nan')]*max(0,4-len(frame.motor_rpm))),
                                 *(controller.assistance.velocity_command if getattr(controller.assistance,'velocity_command',None) is not None
                                   else [float('nan')]*3),
                                 getattr(controller.assistance,'state','') if controller.assistance else '',
                                 *looming_row(camera.clearance if looming else None,now),
                                 *clearance_row(controller.assistance)])
                loop_phases['csv_ms'] = 1000*(time.monotonic()-loop_mark)
                loop_mark = time.monotonic()
                if replay is not None:
                    replay.append(controller.last_observation,retina,action,pos,q,
                                  frame.timestamp,now,capture_time,fresh)
                loop_phases['replay_ms'] = 1000*(time.monotonic()-loop_mark)
                if looming:
                    camera.motion.publish(now,last_frame,frame.timestamp,pos,q,velocity,np.zeros(3))
                if geometry:
                    # Always publish the unmodified task goal, so a detour does
                    # not feed itself back as the next navigation objective.
                    loop_mark = time.monotonic()
                    from ..vision.camera import quat_wxyz_to_mat
                    desired = controller.guidance_velocity.world_velocity(
                        controller.nominal_relative_gate, quat_wxyz_to_mat(q))
                    geometry.buffer.publish(now,last_frame,frame.timestamp,pos,q,velocity,desired)
                    loop_phases['geometry_publish_ms'] = 1000*(time.monotonic()-loop_mark)
                count += 1
                if calibration_mode and controller.assistance.complete:
                    reason = 'Dynamics calibration sequence complete; stable hover restored'
                    break
                if controller.assistance_mode == 'oracle-route' and controller.assistance.complete:
                    reason = 'Oracle collection route endpoint reached; game finish requires separate confirmation'
                    break
                if shared:
                    loop_mark = time.monotonic()
                    rates = (controller.brain.cfg.rate_max*torch.sigmoid(controller.state['v'][:,0])).cpu().numpy()
                    target = controller.gate_point if controller.gate_point is not None else pos
                    shared.publish(rates,t=elapsed,dist=float(np.linalg.norm(target-pos)),thr=raw[0],roll=raw[1],pitch=raw[2],yaw=raw[3],
                                   px=pos[0],py=pos[1],pz=pos[2],tx=target[0],ty=target[1],tz=target[2],
                                   qw=q[0],qx=q[1],qy=q[2],qz=q[3],ts=frame.timestamp)
                    loop_phases['panel_publish_ms'] = 1000*(time.monotonic()-loop_mark)
                previous_loop_phases = loop_phases
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
        geometry_status = geometry.stop() if geometry else None
        rx.close()
        if recorder:
            recorder.stop()
        from .preflight import check_workloads
        try:
            postflight = check_workloads(getattr(args,'allow_workload_pid',()),getattr(args,'allow_workload_project',()))
        except Exception as exc:
            postflight = dict(passed=None, error=str(exc))
        if gc_was_enabled:
            gc.enable()
        assisted = controller.assistance is not None
        pilot_meta = controller.assistance.metadata() if assisted else dict(mode='none')
        if controller.motor_baseline and controller.assistance_mode not in ('oracle-route','dynamics-calibration'):
            pilot_meta['motor_control'] = 'PD throttle/roll/pitch; assisted yaw; brain runs in shadow'
        result = dict(checkpoint_sha256=sha256(args.checkpoint),checkpoint_requires_teacher=False,
                      runtime_requires_teacher=controller.assistance_mode == 'oracle-route',
                      control_mode=(('experimental visual geometry guidance; PD motors; brain in shadow' if controller.motor_baseline else
                                     'experimental visual geometry guidance; brain motors; assisted yaw') if geometry_control else
                                    'Dynamics calibration; PD and input pulses; brain in shadow' if calibration_mode else
                                    ('PRIVILEGED oracle collection; PD motors; brain in shadow' if controller.motor_baseline else
                                     'PRIVILEGED oracle motor diagnostic; brain motors; assisted yaw') if controller.assistance_mode == 'oracle-route' else
                                    'fast PD motor baseline with fast race-cue pilot; brain in shadow' if controller.fast_motor is not None else
                                    'PD motor baseline; brain in shadow' if controller.motor_baseline else
                                    'pilot-assisted visual fly brain' if assisted else 'visual fly brain') if pad else 'shadow: no control output',
                      motor_controller=controller.motor_metadata,
                      preflight=preflight,
                      postflight=postflight,
                      pilot_assistance=pilot_meta,
                      navigation_predictor_loaded=False,
                      geometry_shadow=geometry_status if not geometry_control else None,
                      geometry_perception=geometry_status if geometry_control else None,
                      geometry_control=controller.geometry_gate.metadata() if controller.geometry_gate else None,
                      geometry_goal_contract=controller.guidance_velocity.metadata() if geometry else None,
                      brain_action_columns=['thr','roll','pitch','yaw'],
                      command_columns=['command_thr','command_roll','command_pitch','command_yaw'],
                      raw_output_includes_arming_hold=True,
                      brain_device=str(controller.brain.device),
                      vision_device=getattr(args,'vision_device','cpu'),
                      telemetry_failure=telemetry_failure,
                      process_priority=priority,
                      recording=dict(enabled=bool(args.record), encoder=getattr(args,'video_encoder','libx264')
                                     if args.record else None, fps=18, cpu_encoder_threads=2),
                      image_freshness_s=.12,camera_outage_limit_s=.5 if camera_braking_allowed else .25 if memory_gap_allowed else .12,
                      camera_braking_after_s=.25 if camera_braking_allowed else None,
                      stale_image_memory_only_ticks=memory_only_ticks,max_image_age_ms=1000*max_image_age,
                      recurrent_matrix_layout=str(controller.W.layout),
                      ticks=count,wall_s=time.monotonic()-begin,stop_reason=reason,
                      external_goal=bool(controller.meta.get('gate_sensor')),yaw_assistance=assisted,
                      goal_source=controller.assistance.metadata()['goal_source'] if assisted else ('camera detector' if controller.meta.get('gate_sensor') else 'absent'),
                      visible_race_cues=controller.assistance_mode=='race-cue',
                      gate_sensor=controller.meta.get('gate_sensor'),
                      runtime_route_oracle=controller.assistance_mode == 'oracle-route',
                      autonomous_evaluation_eligible=controller.assistance_mode not in ('oracle-route','dynamics-calibration'),
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
                      telemetry_copy_port=copy_port or None,
                      images_blanked=args.blank_retina,
                      limits=dict(height_m=args.max_height,speed_mps=args.max_speed,distance_m=args.max_distance),
                      process_session=windows_session_id())
        if replay is not None:
            result['neural_replay'] = replay.save()
            result['neural_replay']['sha256'] = sha256(replay.path)
            result['neural_replay']['pilot_assistance'] = controller.assistance_mode
            result['neural_replay']['privileged_goal_observations'] = controller.assistance_mode == 'oracle-route'
            result['neural_replay']['autonomous_evaluation_eligible'] = controller.assistance_mode not in ('oracle-route','dynamics-calibration')
            if controller.assistance_mode == 'oracle-route':
                result['neural_replay']['route_labels_present'] = True
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
    p.add_argument('--collection-route', default=None,
                   help='PRIVILEGED route for course qualification/data collection only; requires PD and no visual pilot; never an autonomous evaluation')
    p.add_argument('--oracle-motor-diagnostic',action='store_true',
                   help='Explicitly allow brain motors with a stored collection route for motor tracking diagnostics; never autonomous navigation evidence')
    p.add_argument('--dynamics-calibration',choices=['hover','throttle','roll','pitch','yaw'],default=None,
                   help='PD hover and bounded identification pulses in an operator-verified empty arena; not a race or brain-control evaluation')
    geometry_options=p.add_mutually_exclusive_group()
    geometry_options.add_argument('--geometry-shadow', action='store_true',
                   help='Passive isolated image-geometry/trajectory audit alongside a declared motor guidance contract; never changes controls')
    geometry_options.add_argument('--geometry-control', action='store_true',
                   help='EXPERIMENTAL causal image-geometry guidance with race-cue PD or a motor-tracking brain; forbids oracle route, archives worker images')
    p.add_argument('--geometry-record-images', action='store_true',
                   help='Archive exact worker inputs in passive geometry mode too, for matched on/off runs')
    p.add_argument('--dense-obstacles',
                   help='EXPERIMENTAL verified TorchScript metric-depth export; adds transient obstacle hypotheses to explicit geometry mode')
    p.add_argument('--calibration-amplitudes',type=float,nargs='+',default=None,
                   help='Ascending processed-input magnitudes for a separately declared calibration validation batch')
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--log',required=True)
    p.add_argument('--record',default='')
    p.add_argument('--allow-workload-pid',type=int,action='append',default=[],
                   help='Operator-authorized workload exception; recorded and rejected if using >= 0.5 CPU core')
    p.add_argument('--allow-workload-project',action='append',default=[],
                   help='Explicitly authorized project with absolute Python script paths; resolve related idle compute PIDs on each preflight, retain busy-load refusal')
    p.add_argument('--video-encoder',choices=['libx264','h264_nvenc'],default='libx264',
                   help='Explicit encoder; tested before arming, never silently falls back')
    p.add_argument('--replay-out',default='',help='Save exact causal senses and passive frozen scene features for offline correction')
    p.add_argument('--udp-out',default='')
    p.add_argument('--port',type=int,default=9001)
    p.add_argument('--telemetry-copy-port',type=int,default=0,
                   help='Copy telemetry to a separate local passive image recorder; avoids competing UDP receivers')
    p.add_argument('--device',default='cuda')
    p.add_argument('--vision-device',choices=['cpu','cuda'],default='cpu',
                   help='Device for the frozen image model, independent of the brain device')
    p.add_argument('--camera-fps', type=float, default=48.,
                   help='Camera rate cap (12..60); freshness and stopping limits are unchanged')
    p.add_argument('--blank-retina',action='store_true',help='diagnostic ablation; zero image input, same brain weights')
    p.add_argument('--pilot-assistance',choices=['none','rabbit','race-cue'],default='none',
                   help='Rabbit arch guidance or explicitly game-cue-assisted race guidance around the chosen motor controller')
    p.add_argument('--assist-speed',type=float,default=2.,help='Requested speed (standard profile 0 < speed <= 5, fast profile <= 20), capped by the trained motor reference unless --pd-profile fast; set per vehicle, not per course')
    p.add_argument('--pilot-profile',choices=['standard','fast'],default='standard',
                   help='fast: velocity-level race-cue guidance (declared experimental profile); standard is unchanged')
    p.add_argument('--pd-profile',choices=['teacher','fast'],default='teacher',
                   help='fast: velocity-command PD on the measured full-envelope dynamics; requires --dynamics-profile')
    p.add_argument('--dynamics-profile',default=None,help='Measured original-drone dynamics profile JSON for --pd-profile fast')
    p.add_argument('--looming-brake',action='store_true',
                   help='EXPERIMENTAL fly-style looming (image expansion) time-to-contact caps speed toward surfaces ahead; fast pilot only')
    p.add_argument('--pause-on-stop',action='store_true',help='Pause the foreground game when a live control attempt ends')
    p.add_argument('--capture-backend',choices=['mss','dxgi'],default='mss',help='DXGI uses original Windows frame timestamps')
    p.add_argument('--max-height',type=float,default=8.)
    p.add_argument('--max-speed',type=float,default=10.)
    p.add_argument('--max-distance',type=float,default=20.)
    run(p.parse_args())


if __name__=='__main__':
    main()
