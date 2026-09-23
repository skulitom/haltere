"""Optional pretrained metric-depth obstacle hypotheses, explicitly unqualified.

No course data or fitted race-specific scale is used. Source-domain depth errors
can create false or misplaced obstacles. This never certifies free space and
does not replace direct triangulation. The caller must opt into live authority.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

from .camera import quat_wxyz_to_mat
from .geometry_mask import liftoff_geometry_mask


def metric_obstacle_points(depth, rgb, camera, position, quaternion, *, stride=12, max_range=12.):
    depth = np.asarray(depth)
    if (depth.shape != (camera.height, camera.width) or not np.isfinite(depth).all()
            or np.asarray(rgb).shape != (*depth.shape, 3)):
        raise ValueError('Use finite calibrated depth and matching RGB arrays')
    if not isinstance(stride, int) or stride <= 0 or not np.isfinite(max_range) or max_range <= .5:
        raise ValueError('Use positive sampling stride and metric range')
    mask = liftoff_geometry_mask(rgb, exclude_white=False)
    yy, xx = np.mgrid[stride//2:camera.height:stride, stride//2:camera.width:stride]
    pixels = np.column_stack([xx.ravel(), yy.ravel()])
    rays = camera.unproject_body(pixels)
    optical = (rays@camera.body_to_cam().T)[:, 2]
    ranges = depth[yy, xx].ravel()/optical
    good = (mask[yy, xx].ravel() > 0) & (ranges >= .5) & (ranges <= max_range) & (optical > 0)
    rays, ranges, pixels = rays[good], ranges[good], pixels[good]
    return dict(points=np.asarray(position)+ranges[:, None]*(rays@quat_wxyz_to_mat(quaternion).T),
                sigma=np.maximum(.25, .25*ranges), pixels=pixels, establishes_free_space=False)


class MetricObstacleDepth:
    def __init__(self, checkpoint, device='cuda'):
        import torch
        path = Path(checkpoint)
        self.source = json.loads(path.with_suffix('.json').read_text())
        if (self.source.get('schema') != 'haltere.pretrained_metric_depth.v1'
                or self.source.get('image_size') != [640, 360]
                or self.source.get('input_shape') != [1, 3, 336, 602]
                or hashlib.sha256(path.read_bytes()).hexdigest() != self.source.get('export_sha256')):
            raise ValueError('Use the verified fixed-shape pretrained depth export and its metadata')
        self.device = torch.device(device)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = torch.jit.load(str(path), map_location=self.device).eval()
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, 336, 602, device=self.device))

    def predict(self, rgb):
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tvf
        rgb = np.asarray(rgb)
        if rgb.shape != (360, 640, 3) or rgb.dtype != np.uint8:
            raise ValueError('Dense depth export requires 640x360 uint8 RGB')
        with torch.inference_mode():
            pixels = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None]
            resized = tvf.resize(pixels, [336, 602], interpolation=InterpolationMode.BICUBIC, antialias=True).float()
            pixels = tvf.normalize(resized, torch.tensor([.485, .456, .406])*255.,
                                   torch.tensor([.229, .224, .225])*255.)
            depth = self.model(pixels.to(self.device))[0].cpu().numpy()
        if depth.shape != (360, 640) or not np.isfinite(depth).all():
            raise RuntimeError('Invalid pretrained depth output')
        return depth

    def metadata(self):
        return dict(model=self.source['model'], revision=self.source['revision'],
                    export_sha256=self.source['export_sha256'], device=str(self.device),
                    source='pretrained indoor metric depth; no Liftoff fitting',
                    uncertainty='heuristic max(0.25 m, 25% range); not a calibrated error bound',
                    establishes_free_space=False, brain_weights_changed=False)
