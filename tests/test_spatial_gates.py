import pytest
import torch

from haltere.vision.model import GateNet, SpatialGateNet, decode, make_gatenet
from haltere.vision.train import load_gatenet


def test_location_moves_with_the_peak_and_background_suppresses_visibility():
    maps = torch.full((2, 2, 23, 40), -30.)
    maps[:, 1] = -.7
    maps[0, 0, 15, 7] = 12.
    maps[1, 0, 15, 32] = 12.
    raw = SpatialGateNet.decode_maps(maps)
    prediction = decode(raw)
    assert torch.allclose(raw[:, 1], torch.tensor([-.625, .625]), atol=1e-5)
    assert torch.allclose(raw[:, 2], torch.full((2,), (15.5 / 23) * 2 - 1), atol=1e-5)
    assert (prediction[:, 0] > .99).all()
    assert torch.allclose(raw[:, 3], torch.full((2,), -.7))
    maps[:, 0] = -12
    assert (decode(SpatialGateNet.decode_maps(maps))[:, 0] < .001).all()


def test_spatial_supervision_reaches_image_filters_and_ignores_negative_coordinates():
    torch.manual_seed(32)
    net = SpatialGateNet(4).eval()
    pixels = torch.rand(2, 3, 180, 320)
    target = torch.tensor([[1., -.6, .4, -.8], [0., 0., 0., 0.]])
    output, loss, parts = net.predict_and_loss(pixels, target)
    assert output.shape == (2, 4) and torch.isfinite(loss) and parts['heatmap'] > 0
    loss.backward()
    assert net.features[0][0].weight.grad.abs().sum() > 0
    assert net.spatial_head[-1].weight.grad.abs().sum() > 0
    other = target.clone(); other[1, 1:] = torch.tensor([100., -100., 50.])
    _, same, _ = net.predict_and_loss(pixels, other)
    assert torch.equal(loss.detach(), same.detach())


@pytest.mark.parametrize('architecture', ['regression', 'spatial_v1'])
def test_checkpoint_loader_preserves_detector_architecture(tmp_path, architecture):
    net = make_gatenet(architecture, 4).eval()
    path = tmp_path / 'detector.pt'
    checkpoint = dict(model=net.state_dict(), width=4)
    if architecture != 'regression':
        checkpoint['architecture'] = architecture
    torch.save(checkpoint, path)
    loaded = load_gatenet(path, 'cpu')
    pixels = torch.rand(1, 3, 180, 320)
    with torch.no_grad():
        assert torch.equal(net(pixels), loaded(pixels))
    assert loaded.architecture == architecture


def test_unknown_architecture_is_not_silently_replaced():
    with pytest.raises(ValueError, match='Unknown'):
        make_gatenet('unknown')


def test_detector_refit_preserves_brain_scene_features_including_batch_norm():
    from haltere.vision.scene_features import scene_map, scene_feature_fingerprint, freeze_scene_backbone
    net = SpatialGateNet(4).eval()
    pixels = torch.rand(2,3,180,320)
    target = torch.tensor([[1.,-.6,.4,-.8],[0.,0.,0.,0.]])
    before = scene_map(net,pixels).detach().clone()
    fingerprint = scene_feature_fingerprint(net)
    original_head = net.spatial_head[-1].weight.detach().clone()
    freeze_scene_backbone(net)
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.001)
    for _ in range(2):
        net.train();freeze_scene_backbone(net)
        _,loss,_ = net.predict_and_loss(pixels,target)
        optimizer.zero_grad();loss.backward();optimizer.step()
    net.eval()
    assert torch.equal(before,scene_map(net,pixels))
    assert fingerprint == scene_feature_fingerprint(net)
    assert not torch.equal(original_head,net.spatial_head[-1].weight)
