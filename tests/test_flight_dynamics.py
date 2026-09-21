import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from haltere.liftoff.fit_drag import fit_drag


def test_horizontal_drag_recovers_known_coasting_motion():
    # With level attitude and vertical thrust balancing gravity, horizontal
    # coasting acceleration is exactly the body drag. Excite both axes.
    dt=.01
    times=np.arange(0,20,dt)
    velocity=np.zeros((len(times),3))
    velocity[0]=[3.,2.,0.]
    linear=np.array([.03,.04,0.])
    quadratic=np.array([.02,.03,0.])
    for i in range(1,len(times)):
        velocity[i]=velocity[i-1]-dt*(linear*velocity[i-1]+quadratic*np.abs(velocity[i-1])*velocity[i-1])
    # Rotate the entire scene: coefficients must be fitted in body coordinates.
    rotation=Rotation.from_euler('z',1.1)
    world=rotation.apply(velocity)
    q=rotation.as_quat()
    frame=pd.DataFrame(dict(ts=times,x=0.,y=0.,z=1.,vx=world[:,0],vy=world[:,1],vz=world[:,2],
                            qx=q[0],qy=q[1],qz=q[2],qw=q[3]))
    result=fit_drag(frame,start_s=2,end_s=18)
    for name,a,b in [('forward',.03,.02),('lateral',.04,.03)]:
        assert abs(result[name]['linear_per_s']-a)<.002
        assert abs(result[name]['quadratic_per_m']-b)<.002
        assert result[name]['acceleration_rms']<.003


def test_impact_monitor_distinguishes_thrust_from_a_side_collision():
    from haltere.liftoff.flight_guard import ImpactMonitor
    q=np.array([1.,0.,0.,0.])
    pos=np.array([0.,0.,1.])
    thrust=ImpactMonitor()
    for i in range(8):
        assert not thrust.update(i*.01,pos,np.array([0.,0.,i*.6]),q)
    hit=ImpactMonitor()
    for i in range(4):
        assert not hit.update(i*.01,pos,np.array([2.,0.,0.]),q)
    assert hit.update(.04,pos,np.zeros(3),q)
    assert hit.impact['unexplained_mps2']>60
    ground=ImpactMonitor()
    for i in range(4):
        assert not ground.update(i*.01,pos*0,np.array([0.,0.,-i]),q)
