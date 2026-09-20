"""Fit visual corrections while preserving a frozen, previously trained motion model."""
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
from .navigation import ResidualNavigationNet, load_navigation
from .train_navigation import CachedSequences, evaluate, predict, to_device
from ..train.thermal import wait_if_hot


def augment(images):
    """Clip-consistent colour/lighting changes; no geometric target changes."""
    shape = (len(images), 1, 1, 1, 1)
    gray = (images*images.new_tensor([.299, .587, .114])[None, None, :, None, None]).sum(2, keepdim=True)
    saturation = images.new_empty(shape).uniform_(0, 1.5)
    gamma = images.new_empty(shape).uniform_(.5, 2.)
    gain = images.new_empty(shape).uniform_(.55, 1.3)
    return ((gray + (images-gray)*saturation).clamp(0, 1).pow(gamma)*gain).clamp(0, 1)


def relative_errors(metrics, baseline):
    return {take: m['model']['mean_error_m'][-1]/baseline[take]['model']['mean_error_m'][-1]
            for take, m in metrics.items()}


def run(dataset, out, config):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if config['epochs'] <= 0 or not 1 <= config['burn_in'] < config['length']:
        raise ValueError('Invalid epoch or sequence configuration')
    torch.set_num_threads(4)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    base, ck = load_navigation(config['base_checkpoint'], device)
    if base.vision:
        raise ValueError('The residual base must be motion-only')
    model = ResidualNavigationNet(base, hidden=config['hidden'], max_correction_m=config['max_correction_m']).to(device)
    train = CachedSequences(dataset, 'train', length=config['length'], stride=config['stride'])
    val = CachedSequences(dataset, 'validation', length=config['length'], stride=config['length']-config['burn_in'])
    for _, a in train.takes + val.takes:
        if not np.allclose(a['horizons_s'], ck['horizons']):
            raise ValueError('Prediction horizons differ from the base checkpoint')
    validation = DataLoader(val, batch_size=config['batch'], num_workers=0)
    counts = Counter(n for n, _ in train.windows)
    sampler = WeightedRandomSampler([1/counts[n] for n, _ in train.windows], len(train),
                                     generator=torch.Generator().manual_seed(config['seed']))
    loader = DataLoader(train, batch_size=config['batch'], sampler=sampler, num_workers=0)
    frozen = {k: v.detach().clone() for k, v in base.state_dict().items()}
    baseline = evaluate(base, validation, device, config['burn_in'])
    out.mkdir(parents=True)
    provenance = dict(config=config, dataset_sha256=sha256(Path(dataset)/'manifest.json'),
                      base_sha256=sha256(config['base_checkpoint']),
                      test_status='Fence take 3 was evaluated in v1; now a retrospective regression check, not an untouched test',
                      selection='Mean relative 1-second error; each validation take must remain within 2% of frozen base')
    (out/'config.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    best, best_epoch = 1., 0

    def save(epoch):
        torch.save(dict(architecture='residual_navigation', model=model.state_dict(), vision=True,
                        hidden=config['hidden'], base_hidden=ck['hidden'], horizons=ck['horizons'],
                        max_correction_m=config['max_correction_m'], epoch=epoch, config=config,
                        dataset_sha256=provenance['dataset_sha256'], base_sha256=provenance['base_sha256']), out/'best.pt')

    save(0)  # Exact baseline remains eligible if training produces no acceptable model.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['lr'], weight_decay=.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config['epochs'])
    started = time.time()
    for epoch in range(1, config['epochs']+1):
        model.train(); losses = []
        for step, batch in enumerate(loader):
            if device.type == 'cuda' and step % 10 == 0:
                wait_if_hot(config['max_gpu_temp'])
            b = to_device(batch, device)
            b['images'] = augment(b['images'])
            prediction = predict(model, b)[:, config['burn_in']:]
            with torch.no_grad():
                anchor = predict(base, b)[:, config['burn_in']:]
            target = b['future_body'][:, config['burn_in']:]
            loss = F.smooth_l1_loss(prediction/model.horizons[:, None], target/model.horizons[:, None], beta=.5)
            loss = loss + .02*((prediction-anchor)/model.horizons[:, None]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.corrector.parameters(), 2.)
            opt.step(); losses.append(float(loss.detach()))
            time.sleep(config['batch_sleep'])
        sched.step()
        metrics = evaluate(model, validation, device, config['burn_in'])
        ratios = relative_errors(metrics, baseline)
        score = float(np.mean(list(ratios.values())))
        eligible = all(r <= 1.02 for r in ratios.values())
        if eligible and score < best:
            best, best_epoch = score, epoch
            save(epoch)
        row = dict(epoch=epoch, loss=float(np.mean(losses)), score=score, eligible=eligible, ratios=ratios, metrics=metrics)
        with (out/'history.jsonl').open('a', encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(f'epoch {epoch:02d}: loss {row["loss"]:.4f}, relative {score:.4f}, eligible={eligible}, ratios={ratios}', flush=True)
    assert all(torch.equal(frozen[k], v) for k, v in base.state_dict().items()), 'Frozen base changed'
    model, _ = load_navigation(out/'best.pt', device)
    result = dict(best_epoch=best_epoch, score=best, elapsed_s=time.time()-started, frozen_base_unchanged=True,
                  baseline=baseline, validation=evaluate(model, validation, device, config['burn_in'], image_ablation=True),
                  checkpoint_sha256=sha256(out/'best.pt'), provenance=provenance)
    test = CachedSequences(dataset, 'test', length=config['length'], stride=config['length']-config['burn_in'])
    if len(test):
        result['regression'] = evaluate(model, DataLoader(test, batch_size=config['batch']), device, config['burn_in'], image_ablation=True)
    (out/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset'); parser.add_argument('--out', required=True)
    parser.add_argument('--config', default='configs/train_residual_navigation.json')
    args = parser.parse_args()
    run(args.dataset, args.out, json.loads(Path(args.config).read_text(encoding='utf-8')))


if __name__ == '__main__':main()
