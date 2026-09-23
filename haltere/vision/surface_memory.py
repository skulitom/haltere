"""Local surface hypotheses from causal, uncertain image-depth points.

Planar patches interpolate only nearby observed points. Large gaps remain holes;
no observations never means free space. This module supplies experimental geometry
hypotheses, not a flight planner or a guarantee of collision avoidance.
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
    def __init__(self,*,lifetime=3.,range_m=12.,voxel=.25,max_sigma=1.5,max_points=None,patch_lifetime=None,
                 build_patches=True):
        if (not np.isfinite([range_m,voxel,max_sigma]).all() or min(range_m,voxel,max_sigma)<=0
                or lifetime is not None and (not np.isfinite(lifetime) or lifetime<=0)
                or patch_lifetime is not None and (not np.isfinite(patch_lifetime) or patch_lifetime<=0)
                or max_points is not None and (not isinstance(max_points,int) or max_points<1)
                or lifetime is None and max_points is None):
            raise ValueError('Use finite positive memory limits')
        self.lifetime,self.range,self.voxel,self.max_sigma=lifetime,range_m,voxel,max_sigma
        self.max_points=max_points
        self.patch_lifetime=patch_lifetime
        self.build_patches=bool(build_patches)
        self.capacity_evictions=0
        self._cached_cloud=self._cached_errors=self._cached_patches=self._cached_patch_sigma=None
        self.cells={};self.last_time=None

    def metadata(self):
        return dict(lifetime_s=self.lifetime,range_m=self.range,voxel_m=self.voxel,
                    max_sigma_m=self.max_sigma,max_points=self.max_points,patch_lifetime_s=self.patch_lifetime,
                    interpolated_patches=self.build_patches,
                    capacity_evictions=self.capacity_evictions,
                    policy='retain nearby static obstacles until outside radius or farthest-first capacity eviction'
                           if self.lifetime is None else 'time-limited observations',
                    limits='Assumes stationary surfaces in the telemetry frame; does not clear false points or certify free space')

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
                    if (self.lifetime is None or timestamp-value[0]<=self.lifetime)
                    and np.linalg.norm(value[1]-position)<=self.range}
        # A lack of parallax while braking is not evidence that an observed wall
        # disappeared. Static-scene mode keeps nearby observations, with their
        # original receipt times, and a fixed spatial/capacity bound. Far points
        # leave first when full; eviction is logged, never called free space.
        if self.max_points is not None and len(self.cells)>self.max_points:
            self.capacity_evictions+=len(self.cells)-self.max_points
            self.cells=dict(sorted(self.cells.items(),key=lambda item:
                np.linalg.norm(item[1][1]-position))[:self.max_points])
        values=list(self.cells.values())
        cloud=np.array([v[1] for v in values]).reshape(-1,3)
        errors=np.array([v[2] for v in values])
        # Interpolated interiors are weaker evidence than measured points: their
        # apparent plane can bridge a real opening. Require recent support for
        # these patches without erasing the underlying obstacle measurements.
        fresh=np.array([self.patch_lifetime is None or timestamp-v[0]<=self.patch_lifetime
                        for v in values],dtype=bool)
        patch_cloud,patch_errors=cloud[fresh],errors[fresh]
        if (self._cached_cloud is None or not np.array_equal(patch_cloud,self._cached_cloud)
                or not np.array_equal(patch_errors,self._cached_errors)):
            self._cached_patches,self._cached_patch_sigma=(surface_patches(patch_cloud,patch_errors)
                if self.build_patches else (np.empty((0,3,3)),np.empty(0)))
            self._cached_cloud,self._cached_errors=patch_cloud.copy(),patch_errors.copy()
        # Keep cached geometry independent of arrays returned to callers. Receipt
        # ages and the current query timestamp still update on every observation.
        patches,patch_sigma=self._cached_patches.copy(),self._cached_patch_sigma.copy()
        return dict(points=cloud,sigma=errors,triangles=patches,triangle_sigma=patch_sigma,
                    timestamp=timestamp,coverage_certified=False,
                    oldest_observation_age_s=max(timestamp-v[0] for v in values) if values else None,
                    capacity_evictions=self.capacity_evictions,patch_support_points=int(fresh.sum()))


def observed_path_margin(path,surfaces,*,vehicle_radius=.35):
    """Minimum distance to observed surfaces, reduced by their range uncertainty.

    An infinite/positive margin describes only the observed surfaces. It cannot
    certify safety through unknown space. Sample trajectories densely enough to
    include motion between controller ticks.
    """
    path=np.asarray(path,float)
    if not np.isfinite(vehicle_radius) or vehicle_radius<=0:
        raise ValueError('Use a positive vehicle radius')
    triangles=np.asarray(surfaces['triangles'],float)
    if (path.ndim!=2 or path.shape[1]!=3 or not len(path) or not np.isfinite(path).all()
            or triangles.ndim!=3 or triangles.shape[1:]!=(3,3) or not np.isfinite(triangles).all()):
        raise ValueError('Use finite nonempty path and finite triangles')
    margin=float('inf')
    if len(surfaces['points']):
        distances=np.linalg.norm(path[:,None,:]-surfaces['points'],axis=2)
        margin=min(margin,float(np.min(distances-surfaces['sigma'][None,:]-vehicle_radius)))
    if len(triangles):
        # Point distances supply an exact upper bound on the final minimum.
        # Disjoint axis-aligned boxes supply conservative lower bounds for each
        # triangle. A triangle beyond that upper bound cannot change the answer.
        separation=np.maximum(0.,np.maximum(triangles.min(axis=1)-path.max(axis=0),
                                            path.min(axis=0)-triangles.max(axis=1)))
        lower=np.linalg.norm(separation,axis=1)-surfaces['triangle_sigma']-vehicle_radius
        keep=lower<=margin+1e-10
        distances=triangle_distance(path,triangles[keep])
        if distances.size:
            margin=min(margin,float(np.min(distances-surfaces['triangle_sigma'][keep][None,:]-vehicle_radius)))
    return dict(margin_m=margin,observed_collision=margin<0,coverage_certified=False)


def observed_escape(path,surfaces,*,vehicle_radius=.35,required_margin=.15,slack=.01,require_exit=True):
    """Test recovery from an already violated uncertain clearance constraint.

    Each initially overlapping primitive must recede monotonically (within a
    declared 1 cm motion/numerical slack) until clearance is restored. Other
    primitives retain the full required margin. The complete braking endpoint
    must clear every observation. ``require_exit=False`` checks only a shared
    reaction prefix to reject impossible recoveries before enumerating paths.
    No obstacle is deleted or declared free.
    """
    path=np.asarray(path,float)
    if (path.ndim!=2 or path.shape[1]!=3 or len(path)<2 or not np.isfinite(path).all()
            or not np.isfinite([vehicle_radius,required_margin,slack]).all()
            or min(vehicle_radius,required_margin)<=0 or slack<0):
        raise ValueError('Use a finite recovery path and valid clearance limits')
    terminal=observed_path_margin(path[-1:],surfaces,vehicle_radius=vehicle_radius)['margin_m']
    if require_exit and terminal<required_margin:
        return dict(allowed=False,terminal_margin_m=terminal)
    initially_violated=False
    for kind in ('points','triangles'):
        geometry=surfaces[kind]
        if not len(geometry):
            continue
        sigma=surfaces['sigma' if kind=='points' else 'triangle_sigma']
        lower=geometry if kind=='points' else geometry.min(axis=1)
        upper=geometry if kind=='points' else geometry.max(axis=1)
        separation=np.maximum(0.,np.maximum(lower-path.max(axis=0),path.min(axis=0)-upper))
        # A lower distance bound already above the required margin cannot
        # violate recovery or ordinary clearance anywhere on this path.
        keep=np.linalg.norm(separation,axis=1)-sigma-vehicle_radius<required_margin+1e-10
        if not keep.any():
            continue
        distances=(np.linalg.norm(path[:,None,:]-geometry[keep],axis=2) if kind=='points'
                   else triangle_distance(path,geometry[keep]))
        margins=distances-sigma[keep][None,:]-vehicle_radius
        initial=margins[0]<required_margin
        initially_violated |= bool(initial.any())
        clipped=np.minimum(margins[:,initial],required_margin)
        if (not np.all(margins[:,~initial]>=required_margin)
                or not np.all(clipped>=np.maximum.accumulate(clipped,axis=0)-slack)):
            return dict(allowed=False,terminal_margin_m=terminal)
    return dict(allowed=initially_violated,terminal_margin_m=terminal)
