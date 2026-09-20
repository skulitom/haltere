"""Distil training-only path supervision into the actual visual fly brain.

Stage 1 prepares causal 100 Hz replay from immutable human takes. Stage 2 updates
the connectome with action imitation and a training-only neural path readout.
The exported checkpoint contains the fly brain and its sensory encoder only.
This is offline imitation, not evidence of successful closed-loop flight.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..brain.model import ConnectomeRNN
from ..brain.retina import RETINA_DIM, brain_to_processed, retina_input, visual_observation
from ..sim.quad import quat_to_mat, yaw_of
from ..vision.datasets import sha256
from .bptt import load_checkpoint
from .thermal import wait_if_hot


def causal_indices(available_time, query_time):
    """Latest fully available sample; -1 means no causal sample exists."""
    return np.searchsorted(available_time, query_time, side='right') - 1


def read_csv(path):
    return np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')


def columns(rows, names):
    return np.column_stack([rows[n] for n in names])


def prepare(dataset, teacher_path, brain_path, mapping_path, out, device='cuda', throttle_scale=.8):
    from ..liftoff.commands import load_mapping
    from ..liftoff.frames import unity_quat_to_sim, unity_vec_to_sim, omega_from_quats
    from ..sim.quad import QuadSim
    from ..vision.navigation import load_navigation
    from ..vision.train_navigation import CachedSequences

    out, dataset = Path(out), Path(dataset)
    if out.exists():
        raise FileExistsError(out)
    original, cfg, _ = load_checkpoint(brain_path, 'cpu')
    if not isinstance(original, ConnectomeRNN) or 'retina' in original.channel_dims:
        raise ValueError('Start from an ordinary connectome motor checkpoint')
    dt = cfg.brain.dt
    mapping = load_mapping(mapping_path)
    if mapping.hover_processed_game is None or throttle_scale <= 0:
        raise ValueError('A measured positive throttle calibration is required')
    calibration = dict(hover_processed=mapping.hover_processed_game, throttle_scale=throttle_scale,
                       hover_stick_sim=2*QuadSim(cfg.quad, 'cpu').hover_command()-1,
                       stick_sign=list(mapping.stick_sign), max_rpm=mapping.max_rpm)
    teacher, _ = load_navigation(teacher_path, device)
    teacher.requires_grad_(False)
    out.mkdir(parents=True)
    manifest = dict(schema=1, dt=dt, teacher_training_only=True,
                    teacher_sha256=sha256(teacher_path), initial_brain_sha256=sha256(brain_path),
                    graph_sha256=sha256(Path(cfg.train.graph).with_suffix('.npz')),
                    dataset_sha256=sha256(dataset/'manifest.json'), mapping_sha256=sha256(mapping_path),
                    calibration=calibration, teacher_horizons=teacher.horizons.tolist(),
                    timing='100 Hz sample-and-hold replay of lower-rate UDP; latest fully captured image; no future interpolation',
                    limitation='Physical camera/display delay uncalibrated; offline action imitation only',
                    takes=[])
    # Test/review takes stay untouched. Validation can select a checkpoint but
    # never supplies teacher targets or observations to an optimizer step.
    for split in ('train', 'validation'):
        data = CachedSequences(dataset, split, length=16, stride=1)
        for take_no, (entry, a) in enumerate(data.takes):
            source = (dataset/entry['source']).resolve()
            for name, digest in entry['source_hashes'].items():
                if sha256(source/name) != digest:
                    raise ValueError(f'Changed source: {source/name}')
            raw, index = read_csv(source/'telemetry.csv'), read_csv(source/'index.csv')
            teacher_paths = np.full((len(a['timestamp']), len(teacher.horizons), 3), np.nan, np.float32)
            windows = [start for n, start in data.windows if n == take_no]
            with torch.no_grad():
                for lo in range(0, len(windows), 64):
                    starts = windows[lo:lo+64]
                    ids = np.array(starts)[:, None] + np.arange(16)[None]
                    images = torch.tensor(data.pixels[take_no][ids], device=device, dtype=torch.float32)/255
                    vb = torch.tensor(a['velocity_body'][ids], device=device)
                    q = torch.tensor(a['attitude'][ids], device=device)
                    ts = a['timestamp'][ids]
                    ts = torch.tensor(ts-ts[:, :1], device=device, dtype=torch.float32)
                    teacher_paths[np.array(starts)+15] = teacher(images, vb, q, ts)[:, -1].cpu().numpy()
                retina = retina_input(torch.tensor(data.pixels[take_no], dtype=torch.float32)/255).numpy()

            raw_ts = raw['timestamp']
            q = np.array([unity_quat_to_sim(v) for v in columns(raw, ('qx','qy','qz','qw'))])
            omega = np.zeros((len(raw), 3), np.float32)
            for i in range(1, len(raw)):
                delta = max(.001, raw_ts[i]-raw_ts[i-1])
                alpha = 1 - .5**(delta/dt)
                omega[i] = (1-alpha)*omega[i-1] + alpha*omega_from_quats(q[i-1], q[i], delta)
            quaternion = torch.tensor(q, dtype=torch.float32)
            R = quat_to_mat(quaternion)
            velocity = torch.tensor(unity_vec_to_sim(columns(raw, ('vx','vy','vz'))), dtype=torch.float32)
            meta = json.loads((source/'capture.json').read_text())
            position = unity_vec_to_sim(columns(raw, ('px','py','pz'))) - np.array(meta['origin_sim'])
            sensors = dict(gyro=torch.tensor(omega), gravity_body=-R[:, 2, :],
                           vel_body=(R.transpose(-1,-2)@velocity[...,None]).squeeze(-1), vel_world=velocity,
                           pos=torch.tensor(position, dtype=torch.float32), quat=quaternion,
                           up=R[:,2,2], altitude=torch.tensor(position[:,2:3], dtype=torch.float32),
                           yaw=yaw_of(quaternion)[:,None])
            rpm = columns(raw, ('rpm_lf','rpm_rf','rpm_lb','rpm_rb')).mean(-1, keepdims=True)
            obs = visual_observation(sensors, torch.tensor(np.clip(rpm/mapping.max_rpm,0,1), dtype=torch.float32),
                                     cfg.task, torch.zeros(len(raw), RETINA_DIM))
            obs.pop('retina')
            actions = columns(raw, ('in_throttle','in_roll','in_pitch','in_yaw')).astype(np.float32)
            capture_end = index['capture_end'][a['source_row']]
            pieces = []
            for group in np.unique(a['run_id']):
                frame_ids = np.flatnonzero(a['run_id']==group)
                grid = np.arange(a['timestamp'][frame_ids[0]], a['timestamp'][frame_ids[-1]], dt)
                ri = causal_indices(raw_ts, grid)
                fi = causal_indices(capture_end, raw['recv_time'][ri])
                safe_fi = np.maximum(fi, 0)
                good = ((fi>=0) & (a['run_id'][safe_fi]==group)
                        & (grid-raw_ts[ri] <= .12)
                        & (raw['recv_time'][ri]-capture_end[safe_fi] <= .12)
                        & np.isfinite(teacher_paths[safe_fi]).all(axis=(1,2)))
                # Splitting invalid gaps keeps recurrent windows inside one continuous span.
                edges = np.flatnonzero(np.diff(np.r_[False,good,False]))
                for begin, end in zip(edges[::2], edges[1::2]):
                    if end-begin >= 64:
                        pieces.append((ri[begin:end], fi[begin:end]))
            if not pieces:
                raise ValueError(f'No continuous replay remains for {entry["id"]}')
            ri, fi = (np.concatenate([p[k] for p in pieces]) for k in (0,1))
            arrays = {key:value[ri].numpy() for key,value in obs.items()}
            arrays.update(retina=retina[fi], action=actions[ri], teacher=teacher_paths[fi],
                          run_id=np.concatenate([np.full(len(p[0]),i) for i,p in enumerate(pieces)]),
                          raw_row=ri, image_row=a['source_row'][fi])
            path = out/f'{entry["id"]}.npz'
            np.savez_compressed(path, **arrays)
            manifest['takes'].append(dict(id=entry['id'], split=split, arrays=path.name,
                                          sha256=sha256(path), ticks=len(ri)))
            print(f'Prepared {entry["id"]}: {len(ri)} causal neural ticks', flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


class Replay:
    def __init__(self, path, split, length=64, stride=32):
        path = Path(path)
        manifest = json.loads((path/'manifest.json').read_text())
        self.takes, self.windows = [], []
        for entry in manifest['takes']:
            if entry['split'] != split:
                continue
            if sha256(path/entry['arrays']) != entry['sha256']:
                raise ValueError('Changed prepared replay')
            with np.load(path/entry['arrays']) as z:
                arrays = {k:torch.from_numpy(z[k]) for k in z.files}
            n = len(self.takes)
            self.takes.append((entry,arrays))
            for group in arrays['run_id'].unique():
                ids = torch.where(arrays['run_id']==group)[0]
                self.windows.extend((n,i) for i in range(int(ids[0]),int(ids[-1])-length+2,stride))
        self.length = length
        if not self.windows:
            raise ValueError(f'No {split} sequences')

    def batch(self, ids, device):
        windows = [self.windows[i] for i in ids]
        keys = self.takes[0][1].keys()-{'run_id','raw_row','image_row'}
        return {k:torch.stack([self.takes[n][1][k][s:s+self.length] for n,s in windows]).to(device) for k in keys}


def student_from_motor(path, device):
    original, cfg, graph = load_checkpoint(path, device)
    if not isinstance(original, ConnectomeRNN):
        raise ValueError('A fly connectome checkpoint is required')
    cfg = copy.deepcopy(cfg)
    cfg.brain.sensory['retina'] = 'lptc'
    cfg.brain.encoder_gain = 1.0  # applies to the new retina; old parameters are restored below
    student = ConnectomeRNN(graph, {**original.channel_dims, 'retina':RETINA_DIM}, cfg.brain, device)
    missing, unexpected = student.load_state_dict(original.state_dict(), strict=False)
    if unexpected or any('retina' not in key for key in missing):
        raise ValueError('Unexpected checkpoint migration')
    # Keep normalization fixed: changing running statistics can masquerade as learning.
    student.eval()
    return student, cfg, graph


def rollout(brain, batch, detach_every=0, blank=False, probe=None):
    B, T = batch['action'].shape[:2]
    state, W, outputs, paths = brain.init_state(B), brain.weight_matrix(), [], []
    for t in range(T):
        if detach_every and t and t % detach_every == 0:
            state = brain.detach_state(state)
        obs = {k:batch[k][:,t] for k in brain.channel_dims}
        if blank:
            obs['retina'] = torch.zeros_like(obs['retina'])
        action, state, aux = brain(obs, state, W)
        outputs.append(action)
        if probe is not None:
            rates = brain.cfg.rate_max*torch.sigmoid(state['v'][probe.neuron_idx]).T
            paths.append(probe(rates))
    return torch.stack(outputs,1), (torch.stack(paths,1) if paths else None), aux


@torch.no_grad()
def evaluate(brain, replay, calibration, scale, ids, device, blank=False, mean_action=None, burn=20):
    errors, baseline, count = {}, {}, {}
    for lo in range(0,len(ids),8):
        ix = ids[lo:lo+8]
        batch = replay.batch(ix,device)
        action, _, _ = rollout(brain,batch,blank=blank)
        error = ((brain_to_processed(action,calibration)-batch['action'])[:,burn:]/scale).square()
        for j,i in enumerate(ix):
            name = replay.takes[replay.windows[i][0]][0]['id']
            errors[name] = errors.get(name,0)+error[j].sum(0).cpu().numpy()
            count[name] = count.get(name,0)+error.shape[1]
            if mean_action is not None:
                e = ((mean_action-batch['action'][j,burn:])/scale).square().sum(0).cpu().numpy()
                baseline[name] = baseline.get(name,0)+e
    return {k:dict(normalized_mse=float((v/count[k]).mean()),
                   processed_rmse=(np.sqrt(v/count[k])*scale.cpu().numpy()).tolist(),
                   constant_train_mean_mse=float((baseline[k]/count[k]).mean()) if k in baseline else None,
                   samples=count[k]) for k,v in errors.items()}


def export(path, brain, cfg, provenance, iteration):
    from .export import export_slim
    # No teacher or auxiliary path readout is ever saved into this inference checkpoint.
    ck = dict(model=brain.state_dict(), config=cfg.to_dict(), channels=brain.channel_dims,
              graph=cfg.train.graph, iter=iteration, visual_brain=dict(schema=1, **provenance))
    staging = path.with_suffix('.full.pt')
    torch.save(ck, staging)
    export_slim(staging,path)
    staging.unlink()


@torch.no_grad()
def evaluate_stream(checkpoint, prepared, out, device='cuda'):
    """Replay complete validation spans without resetting state every training window."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    brain,_,_ = load_checkpoint(checkpoint,device)
    manifest = json.loads((Path(prepared)/'manifest.json').read_text())
    train_data,val = Replay(prepared,'train'),Replay(prepared,'validation')
    all_actions = torch.cat([a['action'] for _,a in train_data.takes]).to(device)
    mean,scale = all_actions.mean(0),all_actions.std(0).clamp_min(.05)
    calibration = manifest['calibration']
    result = dict(checkpoint_sha256=sha256(checkpoint),state_reset='only at actual recording gaps',
                  teacher_loaded=False,closed_loop=False,takes={})
    W = brain.weight_matrix()
    for entry,arrays in val.takes:
        data = {k:v.to(device) for k,v in arrays.items() if k in brain.channel_dims or k=='action'}
        methods = {}
        for blank in (False,True):
            errors,baseline,count = torch.zeros(4,device=device),torch.zeros(4,device=device),0
            for group in arrays['run_id'].unique():
                ids = torch.where(arrays['run_id']==group)[0].tolist()
                state = brain.init_state(1)
                for j,i in enumerate(ids):
                    obs = {k:data[k][i:i+1] for k in brain.channel_dims}
                    if blank:
                        obs['retina'] = torch.zeros_like(obs['retina'])
                    action,state,_ = brain(obs,state,W)
                    if j>=20:
                        errors += (((brain_to_processed(action,calibration)[0]-data['action'][i])/scale)**2)
                        baseline += ((mean-data['action'][i])/scale)**2
                        count += 1
            methods['images_blanked' if blank else 'student'] = dict(
                normalized_mse=float((errors/count).mean()),
                processed_rmse=(torch.sqrt(errors/count)*scale).cpu().tolist(),
                constant_train_mean_mse=float((baseline/count).mean()),samples=count)
        result['takes'][entry['id']] = methods
        print(entry['id']+': '+json.dumps(methods),flush=True)
    out.write_text(json.dumps(result,indent=2))
    return result


def train(prepared, initial, out, config):
    out, prepared = Path(out), Path(prepared)
    if out.exists():
        raise FileExistsError(out)
    device = config.get('device','cuda')
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    manifest = json.loads((prepared/'manifest.json').read_text())
    if sha256(initial) != manifest['initial_brain_sha256']:
        raise ValueError('Initial brain differs from the prepared teacher run')
    brain, cfg, graph = student_from_motor(initial,device)
    cfg.brain.mask_motor_feedback = bool(config.get('mask_motor_feedback', False))
    if config.get('warm_start'):
        warm, warm_cfg, _ = load_checkpoint(config['warm_start'], device)
        warm_meta = torch.load(config['warm_start'], map_location='cpu', weights_only=True)['visual_brain']
        if (warm_meta['prepared_sha256'] != sha256(prepared/'manifest.json')
                or warm_cfg.brain.mask_motor_feedback != cfg.brain.mask_motor_feedback):
            raise ValueError('Warm start sensory/data contract differs')
        brain.load_state_dict(warm.state_dict())
        del warm
    if sha256(Path(cfg.train.graph).with_suffix('.npz')) != manifest['graph_sha256']:
        raise ValueError('Connectome changed after preparation')
    if cfg.brain.dt != manifest['dt']:
        raise ValueError('Neural replay clock differs from inference clock')
    train_data, val = Replay(prepared,'train'), Replay(prepared,'validation',stride=64)
    actions = torch.cat([a['action'] for _,a in train_data.takes]).to(device)
    mean_action, scale = actions.mean(0), actions.std(0).clamp_min(.05)
    # Auxiliary decoder from actual goal-population activity, used only in the loss.
    probe = nn.Linear(len(graph.population('goal')),3*len(manifest['teacher_horizons'])).to(device)
    probe.register_buffer('neuron_idx',torch.tensor(graph.population('goal'),device=device))
    groups = [dict(params=[p for n,p in brain.named_parameters() if p.requires_grad and n=='log_edge_gain'],lr=config['lr_edges']),
              dict(params=[p for n,p in brain.named_parameters() if p.requires_grad and n!='log_edge_gain'],lr=config['lr']),
              dict(params=probe.parameters(),lr=config['lr_probe'])]
    opt = torch.optim.Adam(groups)
    initial_params = {n:p.detach().cpu().clone() for n,p in brain.named_parameters()}
    # Fixed, spread-out validation windows from every take, selected before training.
    val_ids = []
    for n in range(len(val.takes)):
        ids = [i for i,(take,_) in enumerate(val.windows) if take==n]
        val_ids.extend(ids[i] for i in np.linspace(0,len(ids)-1,min(len(ids),config['validation_windows_per_take']),dtype=int))
    calibration = manifest['calibration']
    recovery = recovery_rollout = None
    if config.get('recovery_weight', 0) > 0:
        from .recovery import recovery_examples
        motor_teacher, motor_cfg, _ = load_checkpoint(initial, device)
        if config.get('recovery_rollout', False):
            from .recovery import RecoveryRollout
            motor_teacher.requires_grad_(False)
            recovery_rollout = RecoveryRollout(motor_teacher,motor_cfg,calibration,config['batch_size'])
        else:
            recovery = recovery_examples(motor_teacher, motor_cfg, calibration)
            del motor_teacher
    provenance = dict(calibration=calibration, teacher_training_only=True,
                      teacher_sha256=manifest['teacher_sha256'], prepared_sha256=sha256(prepared/'manifest.json'),
                      initial_brain_sha256=manifest['initial_brain_sha256'], graph_sha256=manifest['graph_sha256'],
                      external_goal=False, yaw_assistance=False, runtime_requires_teacher=False,
                      mask_motor_feedback=cfg.brain.mask_motor_feedback,
                      recovery_weight=config.get('recovery_weight', 0),
                      warm_start_sha256=sha256(config['warm_start']) if config.get('warm_start') else None,
                      recovery_rollout=bool(config.get('recovery_rollout', False)),
                      assessment='experimental offline imitation; not flight-qualified')
    out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps(dict(config=config,provenance=provenance),indent=2))
    baseline = evaluate(brain,val,calibration,scale,val_ids,device,mean_action=mean_action)
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2))
    print('Before training: '+json.dumps(baseline),flush=True)
    best = sum(v['normalized_mse'] for v in baseline.values())/len(baseline)
    best_reflex = float('inf')
    export(out/'initial.pt',brain,cfg,provenance,0)
    started = time.monotonic()
    by_take = [[i for i,(k,_) in enumerate(train_data.windows) if k==n] for n in range(len(train_data.takes))]
    for it in range(1,config['iterations']+1):
        if it % 10 == 1:
            wait_if_hot(config.get('max_gpu_temp',78))
        ids = [int(rng.choice(by_take[int(rng.integers(len(by_take)))])) for _ in range(config['batch_size'])]
        batch = train_data.batch(ids,device)
        action,path,aux = rollout(brain,batch,detach_every=8,probe=probe)
        burn = 20
        action_loss = (((brain_to_processed(action,calibration)-batch['action'])[:,burn:]/scale)**2).mean()
        # This target cannot enter the sensory dict; it only supplies a detached loss.
        teacher = batch['teacher'][:,burn:].flatten(-2).detach()
        path_loss = F.smooth_l1_loss(path[:,burn:]/5,teacher/5)
        loss = config.get('action_weight', 1.)*action_loss + config['teacher_weight']*path_loss + brain.regularization(aux)
        recovery_loss = torch.zeros((), device=device)
        if recovery is not None:
            ri = torch.randint(len(recovery['action']), (config['batch_size'],), device=device)
            rb = {k: v[ri] for k, v in recovery.items()}
            ra, _, _ = rollout(brain, rb, detach_every=8)
            recovery_loss = (((brain_to_processed(ra,calibration)-rb['action'])[:,burn:]/scale)**2).mean()
            loss = loss + config['recovery_weight'] * recovery_loss
        elif recovery_rollout is not None:
            recovery_loss = recovery_rollout.loss(brain, scale)
            loss = loss + config['recovery_weight'] * recovery_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if it == 1:
            gradients = {n:float(p.grad.abs().sum()) for n,p in brain.named_parameters() if p.grad is not None}
            if gradients.get('log_edge_gain',0)<=0 or gradients.get('log_gain',0)<=0:
                raise RuntimeError('Loss did not reach the recurrent brain')
            (out/'first_gradients.json').write_text(json.dumps(gradients,indent=2))
        norm = torch.nn.utils.clip_grad_norm_(brain.parameters(),1.)
        torch.nn.utils.clip_grad_norm_(probe.parameters(),1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError('Nonfinite training update')
        opt.step()
        row = dict(iteration=it,action_loss=float(action_loss.detach()),teacher_loss=float(path_loss.detach()),
                   recovery_loss=float(recovery_loss.detach()),
                   elapsed_s=time.monotonic()-started)
        with (out/'training.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if it==1 or it%10==0:
            print(json.dumps(row),flush=True)
        if it%config['eval_every']==0 or it==config['iterations']:
            result = evaluate(brain,val,calibration,scale,val_ids,device,mean_action=mean_action)
            score = sum(v['normalized_mse'] for v in result.values())/len(result)
            row = dict(iteration=it,score=score,takes=result)
            if config.get('qualify_motor', False):
                from .recovery import motor_check
                row['motor_check'] = motor_check(brain, cfg)
                if row['motor_check']['reflex_pass'] and score < best_reflex:
                    best_reflex = score
                    export(out/'motor-qualified.pt',brain,cfg,
                           {**provenance,'assessment':'simulator reflex check passed; not live-flight-qualified'},it)
            print('Validation: '+json.dumps(row),flush=True)
            with (out/'validation.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            if score < best:
                best = score
                export(out/'best.pt',brain,cfg,provenance,it)
            export(out/'last.pt',brain,cfg,provenance,it)
    differences = {n:float((p.detach().cpu()-initial_params[n]).abs().max()) for n,p in brain.named_parameters()}
    (out/'parameter_changes.json').write_text(json.dumps(differences,indent=2))
    selected = out/'best.pt' if (out/'best.pt').exists() else out/'initial.pt'
    brain, _, _ = load_checkpoint(selected,device)
    result = dict(checkpoint=str(selected),sha256=sha256(selected),predictor_loaded_in_training_process=False,
                  teacher_free_replay=evaluate(brain,val,calibration,scale,val_ids,device,mean_action=mean_action),
                  images_blanked=evaluate(brain,val,calibration,scale,val_ids,device,blank=True),
                  baseline=baseline,validation_windows=val_ids,flight_tested=False)
    if config.get('qualify_motor', False):
        result['motor_qualified_checkpoint'] = str(out/'motor-qualified.pt') if (out/'motor-qualified.pt').exists() else None
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print('Finished: '+json.dumps(result),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--dataset',required=True)
    a.add_argument('--teacher',required=True)
    a.add_argument('--brain',required=True)
    a.add_argument('--mapping',required=True)
    a.add_argument('--out',required=True)
    a.add_argument('--device',default='cuda')
    a = sub.add_parser('train')
    a.add_argument('--prepared',required=True)
    a.add_argument('--initial',required=True)
    a.add_argument('--out',required=True)
    a.add_argument('--config',required=True)
    a = sub.add_parser('evaluate-stream')
    a.add_argument('--checkpoint',required=True)
    a.add_argument('--prepared',required=True)
    a.add_argument('--out',required=True)
    a.add_argument('--device',default='cuda')
    args = p.parse_args()
    if args.command=='prepare':
        prepare(args.dataset,args.teacher,args.brain,args.mapping,args.out,args.device)
    elif args.command=='train':
        train(args.prepared,args.initial,args.out,json.loads(Path(args.config).read_text()))
    else:
        evaluate_stream(args.checkpoint,args.prepared,args.out,args.device)


if __name__ == '__main__':
    main()
