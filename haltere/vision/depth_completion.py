"""Experimental metric completion of relative depth from causal sparse anchors.

This is a perception diagnostic, not a free-space or collision certificate.
Anchors must already pass the caller's pose, visibility and uncertainty checks.
The consistency split checks one image; it is not independent flight evidence.
"""
from __future__ import annotations

import numpy as np

from .camera import quat_wxyz_to_mat


def calibrate_relative_depth(relative, anchors):
    """Fit positive inverse-depth scale/shift and retain a consistency check.

    Anchors are rows of pixel u/v, optical depth, optical-depth uncertainty.
    The optional fifth column declares the fit/check split (zero = check).
    Unsupported calibration returns None, never an invented metric scale.
    """
    import cv2
    relative, anchors = np.asarray(relative, np.float32), np.asarray(anchors, float)
    if (relative.ndim != 2 or not np.isfinite(relative).all() or anchors.ndim != 2
            or anchors.shape[1] not in (4, 5) or not np.isfinite(anchors).all()):
        raise ValueError('Use finite depth and sparse metric anchor arrays')
    if len(anchors) < 8:
        return None
    u, v, z, sigma = anchors[:, :4].T
    if (np.any((u < 0) | (u >= relative.shape[1]) | (v < 0) | (v >= relative.shape[0]))
            or np.any(z <= 0) or np.any(sigma < 0)):
        raise ValueError('Use in-image anchors with positive depths and nonnegative uncertainty')
    # Retain the strict geometric uncertainty filter. A good fit alone did not
    # reject erroneous metric extrapolation from less certain depth anchors.
    valid = sigma < .1*z
    d = cv2.remap(relative, u.astype(np.float32)[:, None], v.astype(np.float32)[:, None],
                  cv2.INTER_LINEAR).ravel()
    split = anchors[:, 4] if anchors.shape[1] == 5 else np.arange(len(anchors)) % 4
    train, check = valid & (split != 0), valid & (split == 0)
    if train.sum() < 6 or check.sum() < 2 or np.ptp(1/z[train]) < .02 or np.ptp(d[train]) < 1e-6:
        return None
    x, y = np.column_stack([d, np.ones(len(d))]), 1/z
    indices = np.flatnonzero(train)
    rng, best = np.random.default_rng(0), None
    for _ in range(128):
        pair = rng.choice(indices, 2, replace=False)
        if abs(d[pair[0]]-d[pair[1]]) < 1e-6:
            continue
        coef = np.linalg.solve(x[pair], y[pair])
        if coef[0] <= 0:
            continue
        error = abs(x@coef-y)/y
        keep = train & (error < .15)
        rank = (int(keep.sum()), -float(np.median(error[train])))
        if best is None or rank > best[0]:
            best = rank, keep
    if best is None or best[1].sum() < 6 or best[1].sum()/train.sum() < .7:
        return None
    keep = best[1]
    weights = np.clip(1/np.maximum(sigma/z**2, .002), 1, 500)
    coef = np.linalg.lstsq(x[keep]*weights[keep, None], y[keep]*weights[keep], rcond=None)[0]
    inv = x[check]@coef
    if coef[0] <= 0 or np.any(inv <= 0):
        return None
    errors = abs(1/inv-z[check])/z[check]
    if np.mean(errors) >= .2 or np.mean(1/inv > 1.25*z[check]) >= .25:
        return None
    support = keep | check
    return dict(coefficients=coef, anchors=anchors[support, :4].copy(),
                relative_support=d[support], fit_anchors=int(keep.sum()), check_anchors=int(check.sum()),
                check_relative_error=float(np.mean(errors)),
                relative_uncertainty=max(.1, float(np.max(errors)), float(np.max(sigma[support]/z[support]))),
                establishes_free_space=False)


def supported_metric_points(relative, calibration, mask, camera, position, quaternion,
                            *, stride=8, max_support_pixels=64., max_range=12.):
    """Sample only near anchor pixels and inside their relative-depth support.

    Keep dense samples separate from direct triangulations: their uncertainty
    is a declared heuristic floor, not a calibrated statistical confidence bound.
    A failed calibration supplies no points. It does not mean the scene is empty.
    """
    if calibration is None:
        return dict(points=np.empty((0, 3)), sigma=np.empty(0), pixels=np.empty((0, 2)),
                    establishes_free_space=False)
    relative, mask = np.asarray(relative), np.asarray(mask)
    if (relative.shape != (camera.height, camera.width) or mask.shape != relative.shape
            or not np.isfinite(relative).all() or not isinstance(stride, int) or stride < 1
            or not np.isfinite([max_support_pixels, max_range]).all()
            or min(max_support_pixels, max_range) <= 0):
        raise ValueError('Use matching image arrays and finite positive sampling limits')
    yy, xx = np.mgrid[stride//2:camera.height:stride, stride//2:camera.width:stride]
    pixels = np.column_stack([xx.ravel(), yy.ravel()])
    d = relative[yy, xx].ravel()
    inverse = calibration['coefficients'][0]*d+calibration['coefficients'][1]
    support = calibration['relative_support']
    # No inverse-affine extrapolation beyond the observed value range. Such
    # extrapolation can look accurate at anchors yet place a nearby wall far away.
    nearby = np.min(np.sum((pixels[:, None, :]-calibration['anchors'][None, :, :2])**2, axis=2), axis=1)
    valid = ((mask[yy, xx].ravel() > 0) & (inverse > 0) & (d >= np.min(support))
             & (d <= np.max(support)) & (nearby <= max_support_pixels**2))
    pixels, inverse = pixels[valid], inverse[valid]
    rays = camera.unproject_body(pixels)
    optical = (rays@camera.body_to_cam().T)[:, 2]
    ranges = 1/inverse/optical
    valid = (optical > 0) & (ranges >= .25) & (ranges <= max_range)
    ranges, rays, pixels = ranges[valid], rays[valid], pixels[valid]
    return dict(points=np.asarray(position)+ranges[:, None]*(rays@quat_wxyz_to_mat(quaternion).T),
                sigma=np.maximum(.15, calibration['relative_uncertainty']*ranges), pixels=pixels,
                establishes_free_space=False)
