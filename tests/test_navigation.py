import numpy as np
import torch

from haltere.vision.navigation import NavigationNet, constant_acceleration, mask_hud, motion_features


def batch(t=12):
    return (torch.rand(2,t,3,90,160), torch.randn(2,t,3),
            torch.tensor([1.,0,0,0]).expand(2,t,4).clone(), torch.arange(t).expand(2,t)/18)


def test_predictions_are_causal_and_start_at_constant_velocity():
    torch.manual_seed(3)
    m=NavigationNet().eval()
    images,v,q,t=batch()
    with torch.no_grad():
        out=m(images,v,q,t)
    torch.testing.assert_close(out, v[...,None,:]*m.horizons[:,None])
    # Nonzero readout is necessary for a meaningful causality test.
    torch.nn.init.normal_(m.head[-1].weight, std=.1)
    with torch.no_grad():
        before=m(images,v,q,t)
        images[:,7:]=torch.rand_like(images[:,7:]);v[:,7:]+=10
        after=m(images,v,q,t)
    torch.testing.assert_close(before[:,:7],after[:,:7])


def test_inertial_features_ignore_global_yaw_and_quaternion_sign():
    _,v,q,t=batch()
    q_yaw=q.clone();q_yaw[...,0]=np.cos(.7);q_yaw[...,3]=np.sin(.7)
    torch.testing.assert_close(motion_features(v,q,t),motion_features(v,q_yaw,t))
    q_yaw[:,::2]*=-1
    torch.testing.assert_close(motion_features(v,q,t),motion_features(v,q_yaw,t))


def test_constant_acceleration_uses_actual_timestamps():
    _,v,q,t=batch()
    t=t.square();v.zero_();v[...,0]=3*t
    horizons=torch.tensor([.25,.5,1.])
    pred=constant_acceleration(v,q,t,horizons)
    expected=v[:,1:,None,:]*horizons[:,None]
    expected[...,0]+=1.5*horizons.square()
    torch.testing.assert_close(pred[:,1:],expected,atol=1e-5,rtol=1e-5)


def test_hud_stick_values_cannot_change_predictions():
    images,v,q,t=batch()
    masked=mask_hud(images)
    changed=images.clone();changed[...,78:,:]=torch.rand_like(changed[...,78:,:])
    torch.testing.assert_close(masked,mask_hud(changed))
    assert torch.all(masked[...,78:,:]==0)


def test_training_step_reaches_visual_encoder():
    model=NavigationNet()
    torch.nn.init.normal_(model.head[-1].weight,std=.1)
    images,v,q,t=batch(8)
    loss=(model(images,v,q,t)-torch.randn(2,8,3,3)).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.features[0].weight.grad.abs().sum()>0


def test_tiny_training_writes_selected_checkpoint_and_metrics(tmp_path):
    from torch.utils.data import DataLoader, Dataset
    from haltere.vision.train_navigation import train_one

    class Flights(Dataset):
        def __init__(self):
            self.root=tmp_path
            self.takes=[({'id':'synthetic'}, {'horizons_s':np.array([.25,.5,1.])})]
            self.windows=[(0,i) for i in range(4)]
        def __len__(self):return 4
        def __getitem__(self,i):
            images,v,q,t=batch(6)
            v[:]=1
            h=torch.tensor([.25,.5,1.])
            return dict(images=images[0],velocity_body=v[0],attitude=q[0],time_s=t[0],
                        future_body=v[0,:,None,:]*h[:,None],take_id='synthetic')
    (tmp_path/'manifest.json').write_text('{}')
    ds=Flights()
    config=dict(seed=1,device='cpu',hidden=16,batch=2,lr=.001,weight_decay=.001,
                epochs=1,burn_in=2,max_gpu_temp=70,batch_sleep=0)
    result=train_one(ds,DataLoader(ds,batch_size=2),tmp_path/'run',config,vision=False)
    assert result['best_epoch']==1
    assert result['metrics']['synthetic']['model']['frames']==16
    assert (tmp_path/'run'/'best.pt').is_file()


def test_residual_freezes_base_bounds_corrections_and_falls_back_on_blank():
    from haltere.vision.navigation import ResidualNavigationNet
    base=NavigationNet(vision=False)
    model=ResidualNavigationNet(base,max_correction_m=.6).train()
    images,v,q,t=batch(8)
    anchor=base(images,v,q,t).detach()
    torch.testing.assert_close(model(images,v,q,t),anchor)
    frozen={k:a.clone() for k,a in base.state_dict().items()}
    torch.nn.init.normal_(model.corrector.head[-1].weight,std=4.)
    output=model(images,v,q,t)
    bound=.6*model.horizons.square()
    assert torch.all((output-anchor).norm(dim=-1) <= bound+1e-5)
    output.square().mean().backward()
    assert all(p.grad is None for p in base.parameters())
    assert all(torch.equal(frozen[k],a) for k,a in base.state_dict().items())
    assert base.training is False
    torch.testing.assert_close(model(torch.zeros_like(images),v,q,t),anchor)


def test_residual_checkpoint_roundtrip(tmp_path):
    from haltere.vision.navigation import ResidualNavigationNet,load_navigation
    m=ResidualNavigationNet(NavigationNet(vision=False)).eval()
    path=tmp_path/'model.pt'
    torch.save(dict(architecture='residual_navigation',model=m.state_dict(),base_hidden=64,
                    hidden=64,horizons=[.25,.5,1.],max_correction_m=.6),path)
    loaded,_=load_navigation(path)
    args=batch(8)
    with torch.no_grad():torch.testing.assert_close(m(*args),loaded(*args))
