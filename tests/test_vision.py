"""Gate vision: camera conventions, gate labels, the network's shapes and the pilot's goal logic."""
import json
import random
import numpy as np
import torch

from haltere.vision.camera import Camera, body_to_cam_matrix
from haltere.vision.gates import gate_label, next_gate_index
from haltere.vision.model import IN_H, IN_W, GateNet, decode


def test_camera_conventions():
    R = body_to_cam_matrix(30.0)
    assert abs(np.linalg.det(R) - 1) < 1e-9
    cam = Camera(640, 360, 300.0, 30.0)
    px, ok = cam.project_body(np.array([[10.0, 0.0, 0.0],                          # ahead, level
                                        [10.0, 0.0, 10 * np.tan(np.deg2rad(30))],   # on the optical axis
                                        [10.0, 3.0, 0.0],                           # ahead-left
                                        [-5.0, 0.0, 0.0]]))                         # behind
    assert ok[:3].all() and not ok[3]
    assert abs(px[0, 0] - 320) < 1e-6 and px[0, 1] > 180          # a level point sits below the centre (camera looks up)
    assert np.allclose(px[1], [320, 180], atol=1e-6)
    assert px[2, 0] < 320                                          # left of the drone -> left in the image
    d = cam.unproject_body(px[:3])
    for v, dd in zip([[10.0, 0, 0], [10.0, 0, 10 * np.tan(np.deg2rad(30))], [10.0, 3.0, 0]], d):
        assert np.allclose(dd, np.asarray(v) / np.linalg.norm(v), atol=1e-6)


def test_gate_labels_and_next_gate():
    cam = Camera(640, 360, 300.0, 30.0)
    gates = [{'pos': [10.0, 0.0, 1.5], 'heading': 0.0}, {'pos': [40.0, 5.0, 1.5], 'heading': 0.0}]
    lab = gate_label(np.array([0.0, 0.0, 1.5]), np.array([1.0, 0, 0, 0]), gates, cam, up_m=0.0)
    assert lab['visible'] == 1 and abs(lab['u'] - 320) < 1e-6 and lab['v'] > 180 and abs(lab['dist_m'] - 10) < 1e-6
    yaw = np.deg2rad(20)
    lab = gate_label(np.array([0.0, 0.0, 1.5]), np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), gates, cam, up_m=0.0)
    assert lab['u'] > 320                                          # yawed left -> the gate moves right in the image
    assert next_gate_index(np.array([12.0, 0.0, 1.5]), gates) == 1
    assert next_gate_index(np.array([0.0, 0.0, 1.5]), gates) == 0
    assert gate_label(np.array([50.0, 0.0, 1.5]), np.array([1.0, 0, 0, 0]), gates, cam, up_m=0.0)['visible'] == 0


def test_gatenet_shapes_and_loss():
    net = GateNet(8).eval()
    x = torch.rand(2, 3, IN_H, IN_W)
    with torch.no_grad():
        o = net(x)
    assert o.shape == (2, 4)
    y = torch.tensor([[1.0, 0.2, -0.1, 0.0], [0.0, 0.0, 0.0, 0.0]])
    loss, parts = GateNet.loss(o, y)
    assert torch.isfinite(loss) and set(parts) == {'vis', 'pos', 'size'}
    d = decode(o)
    assert d.shape == (2, 4) and (0 <= d[:, 0]).all() and (d[:, 0] <= 1).all() and (d[:, 3] > 0).all()


def test_triangulation_recovers_a_gate():
    from haltere.vision.camera import world_to_body
    from haltere.vision.triangulate import intersect_rays, ray_world
    cam = Camera(640, 360, 300.0, 30.0)
    gate = np.array([20.0, 3.0, 2.0])
    origins, dirs = [], []
    for i, x in enumerate([0.0, 4.0, 8.0, 12.0]):
        row = {'px': x, 'py': 0.0, 'pz': 1.5, 'qw': 1.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0}
        px, ok = cam.project_body(world_to_body(gate[None], np.array([x, 0, 1.5]), np.array([1.0, 0, 0, 0])))
        assert ok[0]
        o, d = ray_world(row, px[0, 0] + 2.0, px[0, 1] - 2.0, cam)
        origins.append(o)
        dirs.append(d)
    p, rms = intersect_rays(np.array(origins), np.array(dirs))
    assert np.linalg.norm(p - gate) < 0.3 and rms < 0.2


def test_strong_augmentation_carries_the_label_with_the_picture(tmp_path, monkeypatch):
    """The warp moves the gate; if the label does not move with it the whole training run is wasted.

    A bright square stands in for the arch: wherever the augmentation puts it, the returned centre must
    still land on it, and the returned width must still be its width. Colour is held still for this one
    (the next test covers it) so that the only thing under test is the geometry.
    """
    import cv2

    from haltere.vision import train as train_mod
    from haltere.vision.train import GateFrames

    monkeypatch.setattr(train_mod, 'photometric', lambda img, cv2, strength='strong': img)
    monkeypatch.setattr(train_mod, 'occlude', lambda img, **kw: img)

    (tmp_path / 'frames').mkdir()
    img = np.full((360, 640, 3), 20, np.uint8)
    img[160:200, 180:220] = 245                                   # a 40 px square centred at (200, 180)
    cv2.imwrite(str(tmp_path / 'frames' / 'a.jpg'), img)
    lab = {'file': 'a.jpg', 'visible': 1, 'u': 200.0, 'v': 180.0, 'width_px': 40.0}
    (tmp_path / 'labels.json').write_text(json.dumps([lab]), encoding='utf-8')

    ds = GateFrames([tmp_path], augment='strong')
    random.seed(0)
    checked = 0
    for _ in range(40):
        x, y = ds[0]
        if y[0] < 0.5 or abs(float(y[1])) > 0.85 or abs(float(y[2])) > 0.85:
            continue                                              # the warp pushed it to or past the edge
        a = x.permute(1, 2, 0).numpy().mean(2)
        ys, xs = np.nonzero(a >= 0.5 * (float(a.max()) + float(a.min())))
        if not len(xs) or len(xs) > 0.2 * a.size:                 # the square left the frame; nothing to check
            continue
        u = (float(y[1]) + 1) / 2 * IN_W
        v = (float(y[2]) + 1) / 2 * IN_H
        assert abs(u - xs.mean()) < 16 and abs(v - ys.mean()) < 16, 'label left the gate behind'
        w_pred = float(np.exp(float(y[3]))) * 100.0 * IN_W / 640
        assert 0.4 < w_pred / max(np.ptp(xs), 1) < 2.5, 'width does not track the apparent size'
        checked += 1
    assert checked >= 10, 'too many draws lost the gate out of frame'


def test_light_augmentation_leaves_the_geometry_alone(tmp_path):
    """The 'light' setting must stay what the earlier detectors were trained with, so an A/B means something."""
    import cv2

    from haltere.vision.train import GateFrames, photometric

    (tmp_path / 'frames').mkdir()
    img = (np.random.default_rng(0).random((360, 640, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / 'frames' / 'a.jpg'), img)
    (tmp_path / 'labels.json').write_text(
        json.dumps([{'file': 'a.jpg', 'visible': 1, 'u': 320.0, 'v': 180.0, 'width_px': 80.0}]), encoding='utf-8')
    ds = GateFrames([tmp_path], augment='light')
    random.seed(1)
    us = [float(ds[0][1][1]) for _ in range(25)]
    assert max(abs(u) for u in us) < 0.35, 'light augmentation should not move a centred gate far'

    base = np.full((40, 60, 3), 128, np.uint8)
    random.seed(2)
    outs = [photometric(base.copy(), cv2, 'strong') for _ in range(12)]
    assert all(o.shape == base.shape and o.dtype == np.uint8 for o in outs)
    hues = [cv2.cvtColor(o, cv2.COLOR_RGB2HSV)[..., 0].mean() for o in outs]
    assert max(hues) - min(hues) > 5, 'strong augmentation is not varying colour'


def test_perturbations_change_appearance_and_nothing_else():
    """The colour battery must leave the geometry alone, or its recall drop would mean nothing."""
    import cv2

    from haltere.vision.measure import PERTURBATIONS, perturb
    rng = np.random.default_rng(0)
    img = (rng.random((IN_H, IN_W, 3)) * 70).astype(np.uint8)  # a dark, textured background
    img[40:80, 100:160] = 250                                  # a landmark whose position must not move
    for kind in PERTURBATIONS:
        out = perturb(img, kind)
        assert out.shape == img.shape and out.dtype == np.uint8, kind
        a = out.mean(2)
        ys, xs = np.nonzero(a >= 0.5 * (float(a.max()) + float(a.min())))
        if kind in ('identity', 'grayscale', 'dark', 'gamma2.2'):
            assert abs(xs.mean() - 130) < 12 and abs(ys.mean() - 60) < 12, kind
        if kind != 'identity':
            assert not np.array_equal(out, img), f'{kind} changed nothing'
    hsv_in = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 0].astype(float)
    hsv_out = cv2.cvtColor(perturb(img, 'hue180'), cv2.COLOR_RGB2HSV)[..., 0].astype(float)
    assert abs(float(np.median((hsv_out - hsv_in) % 180)) - 90) < 2      # 180 degrees is 90 in OpenCV units
