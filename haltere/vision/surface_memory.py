"""Short-lived surface hypotheses from causal, uncertain image-depth points.

Planar patches interpolate only nearby observed points. Large gaps remain holes;
no observations never means free space. This module is an offline/shadow geometry
prototype, not an enabled flight planner or a guarantee of collision avoidance.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, QhullError


def triangle_distance(points, triangles):
    """Euclidean distances from N points to M finite triangles, including edges."""
    points,triangles=np.asarray(points,float),np.asarray(triangles,float)
    if points.ndim!=2 or points.shape[1]!=3 or triangles.shape[1:]!=(3,3):
        raise ValueError('Expected Nx3 points and Mx3x3 triangles')
    if not np.isfinite(points).all() or not np.isfinite(triangles).all():
        raise ValueError('Use finite geometry')
    if not len(triangles):return np.full((len(points),0),np.inf)
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    u,v=b-a,c-a
    normal=np.cross(u,v);norm=np.linalg.norm(normal,axis=1)
    normal=normal/np.maximum(norm[:,None],1e-12)
    delta=points[:,None,:]-a
    signed=np.einsum('nmi,mi->nm',delta,normal)
    projected=delta-signed[:,:,None]*normal
    uu,vv,uv=np.sum(u*u,axis=1),np.sum(v*v,axis=1),np.sum(u*v,axis=1)
    pu,pv=np.einsum('nmi,mi->nm',projected,u),np.einsum('nmi,mi->nm',projected,v)
    denominator=np.maximum(uu*vv-uv*uv,1e-12)
    s,t=(vv*pu-uv*pv)/denominator,(uu*pv-uv*pu)/denominator
    inside=(s>=0)&(t>=0)&(s+t<=1)&(norm>1e-8)
    result=np.where(inside,abs(signed),np.inf)
    for start,end in [(a,b),(b,c),(c,a)]:
        edge=end-start
        fraction=np.clip(np.einsum('nmi,mi->nm',points[:,None,:]-start,edge)
                         /np.maximum(np.sum(edge*edge,axis=1),1e-12),0,1)
        residual=points[:,None,:]-(start+fraction[:,:,None]*edge)
        result=np.minimum(result,np.linalg.norm(residual,axis=2))
    return result


def surface_patches(points,sigma,*,max_edge=3.5,plane_tolerance=.2,max_planes=4):
    """Deterministic robust planes, triangulated without bridging large gaps."""
    points,sigma=np.asarray(points,float),np.asarray(sigma,float)
    if (points.ndim!=2 or points.shape[1]!=3 or sigma.shape!=(len(points),)
            or not np.isfinite(points).all() or not np.isfinite(sigma).all() or (sigma<0).any()
            or not np.isfinite([max_edge,plane_tolerance]).all() or min(max_edge,plane_tolerance)<=0
            or max_planes<1):
        raise ValueError('Use finite depth points, uncertainties and positive patch limits')
    triangles,uncertainty=[],[]
    remaining=np.arange(len(points));rng=np.random.default_rng(0)
    for _ in range(max_planes):
        if len(remaining)<6:break
        cloud=points[remaining];best=np.zeros(len(cloud),bool)
        for _ in range(64):
            a,b,c=cloud[rng.choice(len(cloud),3,replace=False)]
            normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
            if length<.02:continue
            inliers=abs((cloud-a)@(normal/length))<=plane_tolerance
            if inliers.sum()>best.sum():best=inliers
        if best.sum()<6:break
        group=remaining[best];remaining=remaining[~best]
        center=np.median(points[group],axis=0)
        _,singular,basis=np.linalg.svd(points[group]-center,full_matrices=False)
        if len(singular)<2 or singular[1]<.1:continue
        xy=(points[group]-center)@basis[:2].T
        try:indices=Delaunay(xy).simplices
        except QhullError:continue
        # Keep the measured vertices instead of forcing them onto an exact plane.
        patches=points[group][indices]
        edges=np.linalg.norm(patches-np.roll(patches,1,axis=1),axis=2)
        area=np.linalg.norm(np.cross(patches[:,1]-patches[:,0],patches[:,2]-patches[:,0]),axis=1)/2
        keep=(edges.max(axis=1)<=max_edge)&(area>.01)
        triangles.extend(patches[keep])
        uncertainty.extend(np.median(sigma[group][indices[keep]],axis=1))
    return np.asarray(triangles).reshape(-1,3,3),np.asarray(uncertainty)


class SurfaceMemory:
    def __init__(self,*,lifetime=3.,range_m=12.,voxel=.25,max_sigma=1.5):
        if not np.isfinite([lifetime,range_m,voxel,max_sigma]).all() or min(lifetime,range_m,voxel,max_sigma)<=0:
            raise ValueError('Use finite positive memory limits')
        self.lifetime,self.range,self.voxel,self.max_sigma=lifetime,range_m,voxel,max_sigma
        self.cells={};self.last_time=None

    def update(self,position,timestamp,points=(),sigma=()):
        position=np.asarray(position,float);points=np.asarray(points,float).reshape(-1,3);sigma=np.asarray(sigma,float)
        if (position.shape!=(3,) or sigma.shape!=(len(points),)
                or not np.isfinite(position).all() or not np.isfinite(points).all()
                or not np.isfinite(sigma).all() or (sigma<0).any() or not np.isfinite(timestamp)
                or self.last_time is not None and timestamp<self.last_time):
            raise ValueError('Use finite causal points, position and uncertainty')
        self.last_time=timestamp
        for point,uncertainty in zip(points,sigma):
            if uncertainty>self.max_sigma or np.linalg.norm(point-position)>self.range:continue
            key=tuple(np.floor(point/self.voxel).astype(int))
            # Prefer the lower-uncertainty observed point; refresh its receipt only
            # when the same voxel is actually observed again.
            old=self.cells.get(key)
            value=(point,uncertainty) if old is None or uncertainty<old[2] else (old[1],old[2])
            self.cells[key]=(timestamp,*value)
        self.cells={key:value for key,value in self.cells.items()
                    if timestamp-value[0]<=self.lifetime and np.linalg.norm(value[1]-position)<=self.range}
        values=list(self.cells.values())
        cloud=np.array([v[1] for v in values]).reshape(-1,3)
        errors=np.array([v[2] for v in values])
        patches,patch_sigma=surface_patches(cloud,errors)
        return dict(points=cloud,sigma=errors,triangles=patches,triangle_sigma=patch_sigma,
                    timestamp=timestamp,coverage_certified=False)


def observed_path_margin(path,surfaces,*,vehicle_radius=.35):
    """Minimum distance to observed surfaces, reduced by their range uncertainty.

    An infinite/positive margin describes only the observed surfaces. It cannot
    certify safety through unknown space. Sample trajectories densely enough to
    include motion between controller ticks.
    """
    path=np.asarray(path,float)
    if not np.isfinite(vehicle_radius) or vehicle_radius<=0:
        raise ValueError('Use a positive vehicle radius')
    distances=triangle_distance(path,surfaces['triangles'])
    margin=float(np.min(distances-surfaces['triangle_sigma'][None,:]-vehicle_radius)) if distances.size else float('inf')
    if len(surfaces['points']):
        distances=np.linalg.norm(path[:,None,:]-surfaces['points'],axis=2)
        margin=min(margin,float(np.min(distances-surfaces['sigma'][None,:]-vehicle_radius)))
    return dict(margin_m=margin,observed_collision=margin<0,coverage_certified=False)
