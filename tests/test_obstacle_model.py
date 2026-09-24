"""ClearanceNet v0: output contract, gravity-only fan geometry and loss parity with the numpy references."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from haltere.obstacles import contract
from haltere.obstacles import model as M
from haltere.obstacles import train as T
from haltere.obstacles.labels import LabelKind


def _quats(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_fan_projection_matches_contract_from_gravity_alone():
    q = _quats(200)
    g = torch.tensor(contract.gravity_camera(q), dtype=torch.float32)
    rays, bx, fd = M._geometry_constants()
    uvn, inview, d_c = M.fan_projection(g, torch.tensor(bx), torch.tensor(fd))
    ref = contract.fan_directions_camera(q).reshape(200, 36, 3)
    ok = np.isfinite(ref).all(-1)
    assert np.abs(d_c.numpy()[ok] - ref[ok]).max() < 1e-5
    uv, front = contract.project_camera(ref)
    want = contract.in_image(uv, front) & ok
    assert ((inview.numpy() > 0.5) == want).mean() > 0.995     # float32 vs float64 at the image border
    assert np.abs(uvn.numpy()).max() <= 1.0


def test_cell_heading_angle_matches_contract():
    q = _quats(50, 1)
    ang = M.cell_heading_angle_deg(contract.gravity_camera(q))
    rays, _, _ = M._geometry_constants()
    xh = contract.heading_frame(q)[..., 0]
    rw = np.einsum('nij,jhw->nihw', contract.camera_to_world(q), rays.astype(np.float64))
    ref = np.degrees(np.arccos(np.clip(np.einsum('ni,nihw->nhw', xh, rw), -1, 1)))
    assert np.nanmax(np.abs(ang - ref)) < 0.01


@pytest.mark.parametrize('arch', ['resnet18fpn', 'dav2s'])
def test_build_model_output_contract(arch):
    if arch == 'dav2s':
        try:
            M._import_transformers()
        except ImportError:
            pytest.skip('transformers not available')
    torch.manual_seed(0)
    net = M.build_model(arch, pretrained=False).eval()
    g = torch.tensor(contract.gravity_camera(_quats(2, 2)), dtype=torch.float32)
    with torch.no_grad():
        out = net(torch.randn(2, *M.INPUT_SHAPE), g)
    assert set(out) == set(M.OUTPUT_SPEC)
    for k, shape in M.OUTPUT_SPEC.items():
        assert tuple(out[k].shape) == (2,) + shape
        assert torch.isfinite(out[k]).all()
    assert (out['grid_log_range'][:, 0] <= out['grid_log_range'][:, 1]).all()
    assert (out['fan_log_free'][:, 0] <= out['fan_log_free'][:, 1]).all()
    arr = M.outputs_to_numpy(out)
    assert arr['grid_q50'].shape == (2, 18, 32) and arr['fan_p8'].shape == (2, 4, 9)
    # the architecture rebuilds without weights and reloads its own state dict (evaluation path)
    net2 = M.build_model(arch, pretrained=False)
    net2.load_state_dict(net.state_dict())


def test_validity_channel_starts_at_zero_weight():
    try:
        M._import_transformers()
    except ImportError:
        pytest.skip('transformers not available')
    net = M.build_model('dav2s', pretrained=False)
    w = net.da.backbone.embeddings.patch_embeddings.projection.weight
    assert w.shape[1] == 4 and float(w[:, 3].abs().max()) == 0.0


def test_torch_losses_match_numpy_references():
    rng = np.random.default_rng(3)
    n = 4000
    q50 = rng.normal(1.5, 1.2, n)
    q20 = q50 - np.abs(rng.normal(0.3, 0.4, n))
    v = np.exp(rng.normal(1.5, 1.0, n))
    v[rng.random(n) < 0.1] = np.nan
    k = rng.integers(0, 4, n)
    ref = T.laplace_censored_nll(q20, q50, v, k)
    got = T.laplace_censored_nll_torch(torch.tensor(q20), torch.tensor(q50), torch.tensor(v), torch.tensor(k))
    assert np.abs(ref - got.numpy()).max() < 1e-5
    logit = rng.normal(0, 3, n)
    for d in (4.0, 8.0):
        t, w = T.fan_blocked_targets(v, k, d)
        ref = T.fan_bce(1 / (1 + np.exp(-logit)), v, k, d)
        got = T.fan_bce_torch(torch.tensor(logit), torch.tensor(t, dtype=torch.float64),
                              torch.tensor(w, dtype=torch.float64))
        assert np.abs(ref - got.numpy()).max() < 1e-5


def test_nll_gradients_are_finite_on_all_kinds():
    q50 = torch.linspace(-2, 5, 64, requires_grad=True)
    q20 = (q50 - 0.2).detach().requires_grad_(True)
    v = torch.exp(torch.linspace(-1, 4, 64))
    v[::7] = float('nan')
    k = torch.tensor([int(x) for x in np.resize([0, 1, 2, 3], 64)])
    T.laplace_censored_nll_torch(q20, q50, v, k).sum().backward()
    assert torch.isfinite(q50.grad).all() and torch.isfinite(q20.grad).all()


def test_teacher_shape_loss_is_affine_invariant():
    rng = np.random.default_rng(4)
    t = np.exp(rng.normal(0, 1, (3, 36, 64))).astype(np.float32)
    lt = np.log(np.maximum(t.reshape(3, 18, 2, 32, 2).max(axis=(2, 4)), 1e-3))
    q50 = -(2.5 * lt + 0.7)
    loss = T.teacher_shape_loss(torch.tensor(q50), torch.tensor(t))
    assert float(loss) < 1e-4
    loss2 = T.teacher_shape_loss(torch.tensor(q50 + rng.normal(0, 0.5, q50.shape).astype(np.float32)), torch.tensor(t))
    assert float(loss2) > 0.1


def test_grid_mirror_matches_fan_mirror_of_geometry():
    """A horizontal flip maps fan column j to 8 - j when gravity x is negated (the flip augmentation)."""
    q = _quats(20, 5)
    g = contract.gravity_camera(q)
    gm = g.copy()
    gm[:, 0] = -gm[:, 0]
    rays, bx, fd = M._geometry_constants()
    uvn, iv, _ = M.fan_projection(torch.tensor(g, dtype=torch.float32), torch.tensor(bx), torch.tensor(fd))
    uvm, ivm, _ = M.fan_projection(torch.tensor(gm, dtype=torch.float32), torch.tensor(bx), torch.tensor(fd))
    u = uvn.numpy().reshape(20, 4, 9, 2)
    um = uvm.numpy().reshape(20, 4, 9, 2)
    both = (iv.numpy().reshape(20, 4, 9) > 0.5) & (contract.mirror_fan(ivm.numpy().reshape(20, 4, 9)) > 0.5)
    mirrored = contract.mirror_fan(um.transpose(0, 3, 1, 2)).transpose(0, 2, 3, 1)
    assert np.abs(u[..., 0][both] + mirrored[..., 0][both]).max() < 1e-4
    assert np.abs(u[..., 1][both] - mirrored[..., 1][both]).max() < 1e-4


def test_model_module_is_runtime_safe():
    import ast
    import inspect
    src = inspect.getsource(M)
    tree = ast.parse(src)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mods.add(node.module or '')
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
    assert not any('labels' in m or 'store' in m or 'splits' in m for m in mods)
