"""Label flight datasets with the recovered gates and train GateNet on them."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .calibrate import load_index
from .camera import Camera
from .gates import gate_label
from .model import IN_H, IN_W, GateNet, decode
from ..train.thermal import wait_if_hot


def label_dataset(dataset: str | Path, gates_path: str | Path, cam: Camera, verbose: bool = True) -> Path:
    """Write labels.json next to index.csv: one entry per frame with the next gate's image position and size."""
    from .gates import load_gate_file
    rows = load_index(dataset)
    gates, width_m, up_m = load_gate_file(gates_path)
    labels = []
    n_vis = 0
    for r in rows:
        pos = np.array([r['px'], r['py'], r['pz']])
        q = np.array([r['qw'], r['qx'], r['qy'], r['qz']])
        lab = gate_label(pos, q, gates, cam, width_m=width_m, up_m=up_m)
        lab['file'] = r['file']
        labels.append(lab)
        n_vis += lab['visible']
    out = Path(dataset) / 'labels.json'
    out.write_text(json.dumps(labels), encoding='utf-8')
    if verbose:
        print(f'{dataset}: {len(labels)} frames labelled, gate visible in {n_vis} ({100 * n_vis / max(len(labels), 1):.0f}%)')
    return out


class GateFrames(torch.utils.data.Dataset):
    """Frames + labels from one or more datasets; the images are resized to the network's input size."""

    def __init__(self, datasets: list[str | Path], augment: bool = True, cam: Camera | None = None):
        import cv2
        self.items = []
        for d in datasets:
            d = Path(d)
            labels = json.loads((d / 'labels.json').read_text(encoding='utf-8'))
            for lab in labels:
                self.items.append((d / 'frames' / lab['file'], lab))
        self.augment = augment
        self.cv2 = cv2
        self.src_w, self.src_h = (cam.width, cam.height) if cam else (640, 360)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        cv2 = self.cv2
        path, lab = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        vis = float(lab['visible'])
        lu, lv, lw = (lab['u'], lab['v'], lab['width_px']) if vis else (0.0, 0.0, 100.0)   # label in pixels
        if self.augment:
            # photometric jitter (Liftoff's lighting and clouds change), small crop + resize, horizontal flip
            if random.random() < 0.5:
                img = img[:, ::-1]
                lu = w - lu
            alpha = random.uniform(0.7, 1.3)
            beta = random.uniform(-25, 25)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if random.random() < 0.7:
                fx, fy = random.uniform(0.0, 0.08), random.uniform(0.0, 0.08)
                x0, y0 = int(fx * w * random.random()), int(fy * h * random.random())
                x1, y1 = w - int(fx * w * random.random()), h - int(fy * h * random.random())
                img = img[y0:y1, x0:x1]
                lu, lv = lu - x0, lv - y0
                w, h = x1 - x0, y1 - y0
        u = (lu / w * 2 - 1) if vis else 0.0        # normalised to [-1, 1] over the (cropped) image width
        v = (lv / h * 2 - 1) if vis else 0.0
        width_px = lw * IN_W / w if vis else 100.0
        if vis and (abs(u) > 1.1 or abs(v) > 1.1):
            vis, u, v, width_px = 0.0, 0.0, 0.0, 100.0
        img = cv2.resize(np.ascontiguousarray(img), (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.tensor([vis, u, v, float(np.log(max(width_px, 4.0) / 100.0))], dtype=torch.float32)
        return x, y


def train(datasets: list[str], out_dir: str = 'runs/gatenet', epochs: int = 25, batch: int = 64, lr: float = 1e-3,
          width: int = 32, device: str = 'cuda', val_frac: float = 0.1, seed: int = 0,
          max_gpu_temp: float = 70.0, batch_sleep: float = 0.15) -> Path:
    torch.manual_seed(seed)
    random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    full = GateFrames(datasets, augment=True)
    n_val = max(1, int(len(full) * val_frac))
    idx = list(range(len(full)))
    random.shuffle(idx)
    val_items = [full.items[i] for i in idx[:n_val]]
    train_items = [full.items[i] for i in idx[n_val:]]
    train_ds = GateFrames([], augment=True); train_ds.items = train_items; train_ds.cv2 = full.cv2
    val_ds = GateFrames([], augment=False); val_ds.items = val_items; val_ds.cv2 = full.cv2
    tl = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    vl = torch.utils.data.DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    net = GateNet(width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, epochs * len(tl)))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = float('inf')
    print(f'GateNet width {width}: {sum(p.numel() for p in net.parameters())} parameters; '
          f'{len(train_ds)} training frames, {len(val_ds)} validation frames on {dev}', flush=True)
    t0 = time.time()
    n_batches = 0
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            loss, parts = GateNet.loss(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            n_batches += 1
            if batch_sleep > 0:
                time.sleep(batch_sleep)        # duty cycle: this PC has shut down from heat during a long training
            if max_gpu_temp > 0 and n_batches % 10 == 0:
                wait_if_hot(max_gpu_temp)
        net.eval()
        vtot, n, err_px, n_vis, vis_ok = 0.0, 0, 0.0, 0, 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                o = net(x)
                loss, _ = GateNet.loss(o, y)
                vtot += float(loss) * len(x); n += len(x)
                d = decode(o)
                vis_ok += int(((d[:, 0] > 0.5).float() == y[:, 0]).sum())
                m = y[:, 0] > 0.5
                if m.any():
                    err_px += float(((d[m, 1:3] - y[m, 1:3]).abs() * torch.tensor([IN_W / 2, IN_H / 2], device=dev)).mean(1).sum())
                    n_vis += int(m.sum())
        vloss = vtot / max(n, 1)
        print(f'epoch {ep + 1:3d}: train {tot / max(len(tl), 1):.3f} val {vloss:.3f} '
              f'visible-acc {vis_ok / max(n, 1):.2f} centre-err {err_px / max(n_vis, 1):.1f} px ({n_vis} visible) {time.time() - t0:5.0f}s',
              flush=True)
        if vloss < best:
            best = vloss
            torch.save({'model': net.state_dict(), 'width': width, 'in_size': (IN_W, IN_H), 'epoch': ep + 1, 'val_loss': vloss},
                       out / 'best.pt')
    torch.save({'model': net.state_dict(), 'width': width, 'in_size': (IN_W, IN_H), 'epoch': epochs}, out / 'last.pt')
    return out


def load_gatenet(path: str | Path, device='cuda') -> GateNet:
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = GateNet(ck.get('width', 32)).to(dev)
    net.load_state_dict(ck['model'])
    net.eval()
    return net
