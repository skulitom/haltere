"""Gate vision: camera conventions, gate labels, the network's shapes and the pilot's goal logic."""
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
    lab = gate_label(np.array([0.0, 0.0, 1.5]), np.array([1.0, 0, 0, 0]), gates, cam)
    assert lab['visible'] == 1 and abs(lab['u'] - 320) < 1e-6 and lab['v'] > 180 and abs(lab['dist_m'] - 10) < 1e-6
    yaw = np.deg2rad(20)
    lab = gate_label(np.array([0.0, 0.0, 1.5]), np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), gates, cam)
    assert lab['u'] > 320                                          # yawed left -> the gate moves right in the image
    assert next_gate_index(np.array([12.0, 0.0, 1.5]), gates) == 1
    assert next_gate_index(np.array([0.0, 0.0, 1.5]), gates) == 0
    assert gate_label(np.array([50.0, 0.0, 1.5]), np.array([1.0, 0, 0, 0]), gates, cam)['visible'] == 0


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
