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
    """Write labels.json next to index.csv: one entry per frame with the nearest in-view arch's image position and size."""
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


def photometric(img: np.ndarray, cv2, strength: str = 'strong') -> np.ndarray:
    """Vary colour, lighting and sharpness. Pixels only: none of this moves the gate, so the label is untouched.

    'light' is brightness and contrast, which is all the first detectors saw. It leaves the palette of the track
    it was trained on intact, and a net that has only ever seen one environment learns that palette: the Straw
    Bale detector fired on almost nothing in Pine Valley. Hue, saturation, sharpness and colour altogether are
    what actually differ between environments, so 'strong' takes all of them away and leaves the shape.
    """
    if strength == 'light':
        return np.clip(img.astype(np.float32) * random.uniform(0.7, 1.3) + random.uniform(-25, 25), 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    if random.random() < 0.9:                                     # hue / saturation / value
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + random.randint(-90, 90)) % 180    # the whole circle: OpenCV hue is 0-179
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.0, 2.0), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.5, 1.5), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if random.random() < 0.15:                                    # drop colour entirely
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    elif random.random() < 0.1:                                   # or scramble it
        img = img[..., random.sample([0, 1, 2], 3)]
    img = np.clip(img.astype(np.float32) * random.uniform(0.75, 1.25) + random.uniform(-25, 25), 0, 255)
    if random.random() < 0.6:                                     # gamma: dusk and noon are not a linear scale apart
        img = 255.0 * np.power(img / 255.0, float(np.exp(random.uniform(np.log(0.45), np.log(2.4)))))
    img = img.astype(np.uint8)
    if random.random() < 0.25:                                    # motion blur along a random direction
        k = random.choice([3, 5, 7])
        ker = np.zeros((k, k), np.float32)
        if random.random() < 0.5:
            ker[k // 2, :] = 1.0 / k
        else:
            ker[:, k // 2] = 1.0 / k
        img = cv2.filter2D(img, -1, ker)
    elif random.random() < 0.25:
        img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.4, 1.4))
    if random.random() < 0.3:
        img = np.clip(img.astype(np.float32) + np.random.normal(0, random.uniform(2, 12), img.shape), 0, 255).astype(np.uint8)
    if random.random() < 0.25:                                    # the capture path is a JPEG in the game too
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 85)])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def occlude(img: np.ndarray, n_max: int = 2, frac: float = 0.12) -> np.ndarray:
    """Paste a few small patches of noise: foliage and course furniture cut across an arch on every track."""
    h, w = img.shape[:2]
    img = img.copy()
    for _ in range(random.randint(0, n_max)):
        bw, bh = int(w * random.uniform(0.03, frac)), int(h * random.uniform(0.03, frac))
        x0, y0 = random.randint(0, max(w - bw, 1)), random.randint(0, max(h - bh, 1))
        img[y0:y0 + bh, x0:x0 + bw] = np.random.randint(0, 256, (bh, bw, 3), dtype=np.uint8) if random.random() < 0.5 \
            else np.array(img[y0, x0], dtype=np.uint8)
    return img


def rotated_range_label(u, v, width, rotation, *, focal=100., image_width=IN_W, image_height=IN_H):
    """Rotate a calibrated view about its optical centre, preserving gate range.

    ``width`` encodes range, not a silhouette box. An affine width multiplier
    would change the range label incorrectly away from the optical axis.
    Return None if the labelled centre leaves the view; callers retain the
    original image rather than relabelling a different visible gate negative.
    """
    ray=np.array([(u-image_width/2)/focal,(v-image_height/2)/focal,1.])
    moved=np.asarray(rotation)@ray
    if moved[2]<=0:
        return None
    x,y=moved[:2]/moved[2]
    new_u,new_v=focal*x+image_width/2,focal*y+image_height/2
    if not (5<=new_u<image_width-5 and 5<=new_v<image_height-5):
        return None
    before=np.sqrt(1+ray[0]**2)*np.linalg.norm(ray)
    after=np.sqrt(1+x*x)*np.sqrt(1+x*x+y*y)
    return float(new_u),float(new_v),float(width*after/before)


class GateFrames(torch.utils.data.Dataset):
    """Frames + labels from one or more datasets; the images are resized to the network's input size."""

    def __init__(self, datasets: list[str | Path], augment: bool | str = True, cam: Camera | None = None):
        import cv2
        self.items = []
        for d in datasets:
            d = Path(d)
            if augment=='projective':
                meta=json.loads((d/'capture.json').read_text())
                camera=meta.get('camera',{})
                if ('equivalent 4m' not in meta.get('size_target','')
                        or not np.isclose(camera.get('f',0)*IN_W/camera.get('width',1),100.)):
                    raise ValueError('Projective augmentation requires calibrated equivalent-range labels')
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
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA)   # augment at the size the net sees:
        w, h = IN_W, IN_H                                                   # a quarter of the pixels, same picture
        vis = float(lab['visible'])
        sx, sy = IN_W / w0, IN_H / h0
        lu, lv, lw = (lab['u'] * sx, lab['v'] * sy, lab['width_px'] * sx) if vis else (0.0, 0.0, 100.0)  # label in pixels
        strength = 'strong' if self.augment is True else str(self.augment)
        if self.augment:
            # geometry first (it moves the label), then colour and sharpness (they do not)
            if random.random() < 0.5:
                img = img[:, ::-1]
                lu = w - lu
            if strength == 'projective':
                if vis and random.random()<.8:
                    # A camera rotation is an exact perspective warp for every
                    # depth. It changes framing without inventing gate distance.
                    R,_=cv2.Rodrigues(np.array([random.uniform(-.12,.12),
                                               random.uniform(-.2,.2),random.uniform(-.1,.1)]))
                    moved=rotated_range_label(lu,lv,lw,R)
                    if moved is not None:
                        K=np.array([[100.,0.,w/2],[0.,100.,h/2],[0.,0.,1.]])
                        img=cv2.warpPerspective(np.ascontiguousarray(img),K@R@np.linalg.inv(K),(w,h),
                                                flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                        lu,lv,lw=moved
            elif strength == 'appearance':
                # Range-encoded labels must retain the calibrated projection.
                # Mirroring above preserves range; arbitrary crops/warps do not.
                pass
            elif strength == 'light':
                if random.random() < 0.7:                      # the small crop the first detectors were trained with
                    fx, fy = random.uniform(0.0, 0.08), random.uniform(0.0, 0.08)
                    x0, y0 = int(fx * w * random.random()), int(fy * h * random.random())
                    x1, y1 = w - int(fx * w * random.random()), h - int(fy * h * random.random())
                    img = img[y0:y1, x0:x1]
                    lu, lv = lu - x0, lv - y0
                    w, h = x1 - x0, y1 - y0
            else:
                # One warp for bank angle, apparent size and framing. Scale matters more than it looks: apparent
                # size is (gate width / range), so a 1.5 m gate and an 8 m gate at the same distance differ by 5x.
                # The border is replicated, not reflected: a reflection mirrors the arch back into the frame as a
                # second, unlabelled one, which teaches exactly the wrong thing. Zooming out stays mild for the
                # same reason, and the frames already hold gates at every range.
                ang = random.uniform(-12, 12)
                s = float(np.exp(random.uniform(np.log(0.8), np.log(1.9))))
                M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
                M[0, 2] += random.uniform(-0.12, 0.12) * w
                M[1, 2] += random.uniform(-0.12, 0.12) * h
                img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                lu, lv = (M[0, 0] * lu + M[0, 1] * lv + M[0, 2]), (M[1, 0] * lu + M[1, 1] * lv + M[1, 2])
                lw = lw * s
            img = photometric(img, cv2, strength)
            if strength != 'light' and random.random() < 0.3:
                img = occlude(img)
        u = (lu / w * 2 - 1) if vis else 0.0        # normalised to [-1, 1] over the (cropped) image width
        v = (lv / h * 2 - 1) if vis else 0.0
        width_px = lw * IN_W / w if vis else 100.0
        if vis and (abs(u) > 1.1 or abs(v) > 1.1):
            vis, u, v, width_px = 0.0, 0.0, 0.0, 100.0
        img = cv2.resize(np.ascontiguousarray(img), (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.tensor([vis, u, v, float(np.log(max(width_px, 4.0) / 100.0))], dtype=torch.float32)
        return x, y


@torch.no_grad()
def hard_example_weights(net, dataset, device, batch=32):
    """Mine only the supplied training frames, without random augmentation.

    Bound each frame's sampling multiplier so easy frames and gate-less views
    remain represented. Validation frames must never enter this sampler.
    """
    clean=GateFrames([],augment=False);clean.items=dataset.items
    net.eval();weights=[]
    for x,y in torch.utils.data.DataLoader(clean,batch_size=batch,shuffle=False,num_workers=0):
        x,y=x.to(device),y.to(device);p=decode(net(x))
        centre=((p[:,1:3]-y[:,1:3])/torch.tensor([.1,.18],device=device)).square().mean(1).sqrt()
        error=(p[:,0]-y[:,0]).abs()
        weights.append((1+4*centre.clamp(0,1)*y[:,0]+2*error).cpu().double())
    return torch.cat(weights)


def train(datasets: list[str], out_dir: str = 'runs/gatenet', epochs: int = 25, batch: int = 64, lr: float = 1e-3,
          width: int = 32, device: str = 'cuda', val_frac: float = 0.1, seed: int = 0,
          max_gpu_temp: float = 70.0, batch_sleep: float = 0.15, init: str = '', augment: str = 'strong',
          holdout: list[str] | None = None, hard_mining: bool = False) -> Path:
    from .datasets import audit_split
    # Run before loading/training: aliases and copied frames otherwise leak into a
    # nominal whole-flight holdout. Retain exactly which labels and images were used.
    provenance = audit_split(datasets, holdout or [])
    if epochs <= 0 or batch <= 0 or not 0 < val_frac < 1:
        raise ValueError('epochs and batch must be positive, and val_frac must be in (0, 1)')
    if hard_mining and (not init or not holdout):
        raise ValueError('Hard-example sampling requires an initial detector and whole-flight validation')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    full = GateFrames(datasets, augment=augment)
    if holdout:
        # A random split over frames of the same flights is not a test: neighbouring frames are the same picture,
        # so the held-out frames are in the training set in all but name. Whole datasets held out - better still,
        # a whole environment - is the only split that answers "does this work somewhere it has never been".
        val_ds = GateFrames(holdout, augment=False)
        train_ds = GateFrames([], augment=augment); train_ds.items = full.items; train_ds.cv2 = full.cv2
    else:
        n_val = max(1, int(len(full) * val_frac))
        idx = list(range(len(full)))
        random.shuffle(idx)
        val_items = [full.items[i] for i in idx[:n_val]]
        train_items = [full.items[i] for i in idx[n_val:]]
        train_ds = GateFrames([], augment=augment); train_ds.items = train_items; train_ds.cv2 = full.cv2
        val_ds = GateFrames([], augment=False); val_ds.items = val_items; val_ds.cv2 = full.cv2
    if len(train_ds) < batch or not len(val_ds):
        raise ValueError('Need at least one full training batch and a nonempty validation split')
    vl = torch.utils.data.DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    net = GateNet(width).to(dev)
    if init:
        ck = torch.load(init, map_location=dev, weights_only=False)
        net.load_state_dict(ck['model'])
        print(f'initialised from {init} (epoch {ck.get("epoch")})', flush=True)
    sampler=None
    if hard_mining:
        weights=hard_example_weights(net,train_ds,dev,batch)
        sampler=torch.utils.data.WeightedRandomSampler(weights,len(train_ds),replacement=True,
                                                       generator=torch.Generator().manual_seed(seed))
        provenance['hard_example_sampling']=dict(source='unaugmented training frames only',
            visibility_weight=2.,centre_weight=4.,centre_scale_normalized=[.1,.18],
            weights=weights.tolist(),replacement=True)
    tl = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=sampler is None,
                                    sampler=sampler,num_workers=0,drop_last=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, epochs * len(tl)))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    provenance.update(seed=seed, init=str(init), init_sha256=None,
                      augment=augment, epochs=epochs, batch=batch, lr=lr,
                      validation_kind='whole_flights' if holdout else 'random_frames_legacy')
    if init:
        from .datasets import sha256
        provenance['init_sha256'] = sha256(init)
    (out / 'datasets.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
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
            torch.save({'model': net.state_dict(), 'width': width, 'in_size': (IN_W, IN_H), 'epoch': ep + 1, 'val_loss': vloss,
                        'datasets': [str(d) for d in datasets], 'augment': augment, 'holdout': [str(d) for d in (holdout or [])],
                        'data_provenance': provenance},
                       out / 'best.pt')
    torch.save({'model': net.state_dict(), 'width': width, 'in_size': (IN_W, IN_H), 'epoch': epochs,
                'datasets': [str(d) for d in datasets], 'augment': augment, 'holdout': [str(d) for d in (holdout or [])],
                'data_provenance': provenance}, out / 'last.pt')
    return out


def load_gatenet(path: str | Path, device='cuda') -> GateNet:
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = GateNet(ck.get('width', 32)).to(dev)
    net.load_state_dict(ck['model'])
    net.eval()
    return net
