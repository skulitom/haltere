import numpy as np
import pytest

from haltere.liftoff.geometry_control import GeometryControlGate,ProposalBuffer


def proposal(**changes):
    row=dict(available_at=1.1,capture_time=1.,requested_goal_time=1.05,
             status='observed_obstacle_detour',position=[0,0,0],requested_velocity=[2,0,0],
             proposal_velocity=[1,1,0],changed=True)
    return {**row,**changes}


def test_proposal_transport_is_latest_only_and_nonblocking():
    buffer=ProposalBuffer()
    assert buffer.read() is None
    row=proposal()
    assert buffer.publish(row,row['available_at'])
    got=buffer.read()
    np.testing.assert_array_equal(got['proposal_velocity'],[1,1,0])
    with buffer.lock:
        assert buffer.read() is None
        assert not buffer.publish(row,1.2)


def test_nominal_leaves_current_goal_unchanged_and_detours_are_bounded():
    gate=GeometryControlGate()
    np.testing.assert_allclose(gate.resolve([2.1,0,0],[0,0,0],1.2,
        proposal(changed=False,status='nominal_unverified')),[2.1,0,0])
    np.testing.assert_allclose(gate.resolve([2,0,0],[0,0,0],1.21,proposal()),[1,1,0])
    bounded=gate.resolve([2,0,0],[0,0,0],1.22,proposal(proposal_velocity=[10,10,10]))
    assert np.linalg.norm(bounded[:2])==pytest.approx(2.5)
    assert bounded[2]==1.2
    assert not gate.metadata()['coverage_certified']


@pytest.mark.parametrize('row,position,requested,status',[
    (proposal(capture_time=.5),[0,0,0],[2,0,0],'stale_proposal'),
    (proposal(available_at=2.),[0,0,0],[2,0,0],'future_proposal'),
    (proposal(),[3,0,0],[2,0,0],'position_mismatch'),
    (proposal(),[0,0,0],[-2,0,0],'task_changed'),
    (proposal(status='stale_geometry_brake'),[0,0,0],[2,0,0],'geometry_unavailable'),
])
def test_mismatched_proposals_brake_instead_of_reusing_a_detour(row,position,requested,status):
    gate=GeometryControlGate()
    np.testing.assert_array_equal(gate.resolve(requested,position,1.2,row),[0,0,0])
    assert gate.status==status


def test_missing_worker_eventually_stops_and_valid_snapshot_does_not_refresh_its_age():
    gate=GeometryControlGate()
    gate.resolve([2,0,0],[0,0,0],1.2,proposal())
    np.testing.assert_array_equal(gate.resolve([2,0,0],[0,0,0],1.5,proposal()),[0,0,0])
    with pytest.raises(RuntimeError,match='unavailable'):
        gate.resolve([2,0,0],[0,0,0],2.21,proposal())
    empty=GeometryControlGate()
    empty.resolve([2,0,0],[0,0,0],10.,None)
    with pytest.raises(RuntimeError,match='unavailable'):
        empty.resolve([2,0,0],[0,0,0],15.01,None)


def test_explicit_escape_proposal_survives_transport_and_freshness_gate():
    buffer=ProposalBuffer();row=proposal(status='observed_obstacle_escape')
    assert buffer.publish(row,row['available_at'])
    assert buffer.read()['status']=='observed_obstacle_escape'
    gate=GeometryControlGate()
    np.testing.assert_array_equal(gate.resolve([2,0,0],[0,0,0],1.2,buffer.read()),[1,1,0])
    assert gate.status=='escape'


@pytest.mark.parametrize('change',[{'collection_route':'known-route.json'},
                                 {'motor_controller':'brain'},{'pilot_assistance':'rabbit'},
                                 {'geometry_shadow':True}])
def test_runtime_refuses_oracle_or_unsupported_geometry_control_before_loading(change):
    from types import SimpleNamespace
    from haltere.liftoff.visual_brain import run
    options=dict(seconds=10,max_height=10,max_speed=5,max_distance=20,geometry_control=True,
                 geometry_shadow=False,motor_controller='pd',pilot_assistance='race-cue',collection_route=None)
    with pytest.raises(ValueError,match='requires race-cue PD'):
        run(SimpleNamespace(**{**options,**change}))
