"""Train and evaluate a navigation predictor without controlling the game."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import sha256
from .demonstrations import DemonstrationSequences
from .navigation import NavigationNet, constant_velocity, constant_acceleration
from ..train.thermal import wait_if_hot


class CachedSequences(DemonstrationSequences):
    """Decode once; retain uint8 pixels in CPU RAM, never cache targets in inputs."""

    def __init__(self, *args, **kwargs):
        import cv2
        super().__init__(*args, **kwargs)
        self.pixels = []
        for entry, arrays in self.takes:
            images = []
            source = (self.root / entry['source']).resolve()
            for name in arrays['filename']:
                path = source / 'frames' / str(name)
                if sha256(path) != entry['frame_hashes'][str(name)]:
                    raise ValueError(f'Changed image: {path}')
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f'Cannot decode: {path}')
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
                images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
            self.pixels.append(np.stack(images))
            print(f'Cached {entry["id"]}: {len(images):,} frames', flush=True)

    def __getitem__(self, index):
        n, start = self.windows[index]
        stop = start + self.length
        entry, a = self.takes[n]
        return dict(images=self.pixels[n][start:stop].astype('float32')/255,
                    velocity_body=a['velocity_body'][start:stop], attitude=a['attitude'][start:stop],
                    time_s=(a['timestamp'][start:stop]-a['timestamp'][start]).astype('float32'),
                    future_body=a['future_body'][start:stop], take_id=entry['id'])


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def predict(model, batch):
    return model(batch['images'], batch['velocity_body'], batch['attitude'], batch['time_s'])


@torch.no_grad()
def evaluate(model, loader, device, burn_in=4, *, image_ablation=False):
    model.eval()
    errors = {}
    for batch in loader:
        b = to_device(batch, device)
        prediction = predict(model, b)
        methods = dict(model=prediction,
                       constant_velocity=constant_velocity(b['velocity_body'], model.horizons),
                       constant_acceleration=constant_acceleration(b['velocity_body'], b['attitude'], b['time_s'], model.horizons))
        if image_ablation and model.vision:
            original = b['images']
            b['images'] = torch.zeros_like(original)
            methods['images_blanked'] = predict(model, b)
            b['images'] = original
        for name, out in methods.items():
            error = (out[:, burn_in:] - b['future_body'][:, burn_in:]).norm(dim=-1).cpu().numpy()
            for i, take in enumerate(batch['take_id']):
                errors.setdefault(take, {}).setdefault(name, []).append(error[i])
    result = {}
    for take, methods in errors.items():
        result[take] = {}
        for method, pieces in methods.items():
            error = np.concatenate(pieces)
            result[take][method] = dict(frames=len(error), mean_error_m=error.mean(0).tolist(),
                                        p95_error_m=np.quantile(error, .95, axis=0).tolist())
    if not result:
        raise ValueError('No evaluation frames')
    return result


def selection_score(metrics):
    """Equal weight per take, relative to its constant-velocity one-second error."""
    return float(np.mean([m['model']['mean_error_m'][-1] / max(m['constant_velocity']['mean_error_m'][-1], .01)
                          for m in metrics.values()]))


def train_one(train_data, validation, out, config, *, vision):
    out.mkdir(parents=True, exist_ok=False)
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    horizons = train_data.takes[0][1]['horizons_s'].tolist()
    for _, a in train_data.takes + validation.dataset.takes:
        if not np.allclose(a['horizons_s'], horizons):
            raise ValueError('Takes have different prediction horizons')
    model = NavigationNet(vision=vision, hidden=config['hidden'], horizons=horizons).to(device)
    counts = Counter(n for n, _ in train_data.windows)
    weights = [1/counts[n] for n, _ in train_data.windows]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                     generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(train_data, batch_size=config['batch'], sampler=sampler, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['epochs'])
    best, best_epoch, started = float('inf'), 0, time.time()
    history = []
    parameters = sum(p.numel() for p in model.parameters())
    print(f'{out.name}: {parameters:,} parameters on {device}', flush=True)
    for epoch in range(1, config['epochs']+1):
        model.train()
        losses = []
        for step, batch in enumerate(loader):
            if device.type == 'cuda' and step % 10 == 0:
                wait_if_hot(config['max_gpu_temp'])
            b = to_device(batch, device)
            # Only photometric augmentation; one brightness/contrast value per clip.
            if vision:
                gain = torch.empty(len(b['images']), 1, 1, 1, 1, device=device).uniform_(.8, 1.2)
                bias = torch.empty_like(gain).uniform_(-.06, .06)
                b['images'] = (b['images']*gain + bias).clamp(0, 1)
            prediction = predict(model, b)[:, config['burn_in']:]
            target = b['future_body'][:, config['burn_in']:]
            # Normalize by horizon so short-range and long-range errors both matter.
            loss = F.smooth_l1_loss(prediction/model.horizons[:, None], target/model.horizons[:, None], beta=1.)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
            losses.append(float(loss.detach()))
            if config['batch_sleep']:
                time.sleep(config['batch_sleep'])
        scheduler.step()
        metrics = evaluate(model, validation, device, config['burn_in'])
        score = selection_score(metrics)
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_score=score,
                   elapsed_s=time.time()-started, metrics=metrics)
        history.append(row)
        with (out/'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        if score < best:
            best, best_epoch = score, epoch
            torch.save(dict(model=model.state_dict(), vision=vision, hidden=config['hidden'], horizons=horizons,
                            epoch=epoch, config=config, validation_score=score,
                            dataset_sha256=sha256(train_data.root/'manifest.json')), out/'best.pt')
        detail = ', '.join(f'{take} {m["model"]["mean_error_m"][-1]:.3f}m' for take, m in metrics.items())
        print(f'{out.name} {epoch:02d}/{config["epochs"]}: loss {row["train_loss"]:.4f}, score {score:.3f}, {detail}', flush=True)
    checkpoint = torch.load(out/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    metrics = evaluate(model, validation, device, config['burn_in'], image_ablation=True)
    summary = dict(best_epoch=best_epoch, selection_score=best, parameters=parameters,
                   elapsed_s=time.time()-started, metrics=metrics, checkpoint_sha256=sha256(out/'best.pt'))
    (out/'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def run(dataset, out, config):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if (config['epochs'] <= 0 or config['batch'] <= 0 or not 1 <= config['burn_in'] < config['length']
            or config['stride'] <= 0):
        raise ValueError('Invalid training configuration')
    torch.set_num_threads(4)
    train = CachedSequences(dataset, 'train', length=config['length'], stride=config['stride'])
    val_data = CachedSequences(dataset, 'validation', length=config['length'], stride=config['length']-config['burn_in'])
    if not len(train) or not len(val_data):
        raise ValueError('Training and validation sequences are required')
    # Review-only takes are never instantiated by this trainer.
    validation = DataLoader(val_data, batch_size=config['batch'], shuffle=False, num_workers=0)
    out.mkdir(parents=True)
    provenance = dict(config=config, dataset=str(Path(dataset).resolve()), dataset_sha256=sha256(Path(dataset)/'manifest.json'),
                      train_sequences=len(train), validation_sequences=len(val_data),
                      train_takes=[e['id'] for e, _ in train.takes], validation_takes=[e['id'] for e, _ in val_data.takes],
                      evaluation='Nonoverlapping scored frames after causal warmup; validation selects best epoch, not an untouched test set',
                      model_role='Offline local-path predictor; no controller or game integration',
                      overlays='Stick display, timer, compass and standings masked; scene race guidance markers remain visible')
    (out/'config.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    results = {}
    for vision, name in ((False, 'motion_only'), (True, 'vision_motion')):
        results[name] = train_one(train, validation, out/name, config, vision=vision)
    # Instantiate test data only after both checkpoints have been selected.
    test_data = CachedSequences(dataset, 'test', length=config['length'], stride=config['length']-config['burn_in'])
    if len(test_data):
        test = DataLoader(test_data, batch_size=config['batch'], shuffle=False, num_workers=0)
        device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        for name in results:
            ck = torch.load(out/name/'best.pt', map_location=device, weights_only=False)
            model = NavigationNet(vision=ck['vision'], hidden=ck['hidden'], horizons=ck['horizons']).to(device)
            model.load_state_dict(ck['model'])
            results[name]['test_metrics'] = evaluate(model, test, device, config['burn_in'], image_ablation=True)
        provenance['test_takes'] = [e['id'] for e, _ in test_data.takes]
        provenance['test_sequences'] = len(test_data)
        (out/'config.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    (out/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default='configs/train_navigation.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    run(args.dataset, args.out, config)


if __name__ == '__main__':
    main()
