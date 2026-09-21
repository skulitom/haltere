"""Frozen image features for the brain's visual sensory encoder, never a path.

Reuse GateNet's early convolutional filters with the observed-stick HUD masked.
Compression statistics are fitted on training images only. Neither future
positions nor controls nor track geometry enter this visual representation.
"""
import torch
from torch.nn import functional as F
from ..brain.retina import mask_retina_pixels


def scene_map(net,pixels):
    x=mask_retina_pixels(pixels,'scene_v2')
    return F.adaptive_avg_pool2d(net.features[:6]((x-net.mean)/net.std),(6,10))


def fit_projection(training_features):
    if training_features.ndim!=4 or training_features.shape[1]<12:
        raise ValueError('Expected training feature maps with at least 12 channels')
    x=training_features.detach().cpu().double().permute(0,2,3,1).reshape(-1,training_features.shape[1])
    if not torch.isfinite(x).all() or len(x)<13:raise ValueError('Invalid training feature maps')
    mean=x.mean(0);x=x-mean
    values,vectors=torch.linalg.eigh(x.T@x/(len(x)-1))
    # Whiten retained components so broad scene colour cannot dominate the
    # smaller obstacle features merely through its activation magnitude.
    matrix=vectors[:,-12:]/values[-12:].clamp_min(1e-4).sqrt()[None]
    return dict(schema=1,layer_count=6,grid=[6,10],mean=mean.tolist(),matrix=matrix.tolist(),scale=3.)


def project_scene(features,projection):
    mean=features.new_tensor(projection['mean']);matrix=features.new_tensor(projection['matrix'])
    if (projection.get('schema')!=1 or projection.get('layer_count')!=6 or projection.get('grid')!=[6,10]
            or features.shape[1:]!=(len(mean),6,10) or matrix.shape!=(len(mean),12)
            or not torch.isfinite(matrix).all() or not torch.isfinite(mean).all()
            or projection.get('scale')!=3.):
        raise ValueError('Invalid frozen scene feature contract')
    x=(features.permute(0,2,3,1)-mean)@matrix
    return torch.tanh(x/3.).permute(0,3,1,2).reshape(len(features),720)
