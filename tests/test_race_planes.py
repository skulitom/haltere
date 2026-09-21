import numpy as np
import pytest
from haltere.liftoff.gate_evaluation import ordered_race_planes, ordered_plane_progress


def test_wrong_side_and_skipped_gate_do_not_advance_progress():
    chain = [dict(gate_id=1, normal_direction=1), dict(gate_id=2, normal_direction=-1)]
    events = [dict(gate_id=g, normal_direction=d, seconds=float(t))
              for t, (g,d) in enumerate([(2,-1),(1,-1),(1,1),(1,1),(2,1),(2,-1),(1,1)])]
    result = ordered_plane_progress(events, chain)
    assert result['completed_plane_cycles'] == 1 and result['current_cycle_plane_count'] == 1
    assert result['next_expected']['gate_id'] == 2
    assert [e['seconds'] for e in result['accepted']] == [2,5,6]
    assert [e['reason'] for e in result['rejected']] == [
        'not_expected_checkpoint','wrong_approach_side','not_expected_checkpoint','wrong_approach_side']
    assert result['race_completion_verified'] is False and result['aperture_verified'] is False


def test_passage_links_determine_order_and_geometry_determines_mesh_side(tmp_path):
    path = tmp_path/'race.xml'
    # Deliberately stored in reverse order. Prefab directionality isn't a mesh-normal sign.
    path.write_text('''<Race><checkPointPassages>
      <RaceCheckpointPassage><uniqueId>b</uniqueId><checkPointID>2</checkPointID>
      <passageType>Finish</passageType><directionality>LeftToRight</directionality><nextPassageIDs/></RaceCheckpointPassage>
      <RaceCheckpointPassage><uniqueId>a</uniqueId><checkPointID>1</checkPointID>
      <passageType>Start</passageType><directionality>LeftToRight</directionality>
      <nextPassageIDs><string>b</string></nextPassageIDs></RaceCheckpointPassage>
      </checkPointPassages></Race>''')
    planes = [dict(id=i,base=np.array([x,0.,0.]),normal=np.array([1.,0.,0.])) for i,x in [(1,2.),(2,1.)]]
    assert ordered_race_planes(path, planes) == [dict(gate_id=1,normal_direction=1),dict(gate_id=2,normal_direction=-1)]
    with pytest.raises(ValueError, match='without evaluated planes'):
        ordered_race_planes(path, planes[:1])
