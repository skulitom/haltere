import numpy as np
import pytest

from haltere.vision.camera import Camera
from haltere.vision.flight_gate_labels import PanelClock, opening_label
from haltere.vision.runtime import detection_geometry


@pytest.mark.parametrize('centre', ([8.,0.,1.2], [12.,8.,4.], [3.,-2.,1.5]))
def test_opening_range_round_trips_off_axis(centre):
    camera = Camera(640,360,200.,30.)
    centre = np.asarray(centre)
    position = np.array([.5,0.,1.])
    up = np.array([0.,.1,np.sqrt(.99)])
    label = opening_label(position,np.eye(3),[dict(base=centre-1.2*up,up=up,id=3)],camera)
    direction,distance = detection_geometry(camera,label['u'],label['v'],label['width_px'])
    np.testing.assert_allclose(direction*distance,centre-position,atol=1e-12)
    scaled = camera.scaled(320,180)
    direction,distance = detection_geometry(scaled,label['u']/2,label['v']/2,label['width_px']/2)
    np.testing.assert_allclose(direction*distance,centre-position,atol=1e-12)


def test_nearest_visible_opening_and_empty_view():
    camera = Camera(640,360,200.,30.)
    def gate(x,y,i):
        return dict(base=np.array([x,y,0.]),up=np.array([0.,0.,1.]),id=i)
    gates = [gate(-2,0,0),gate(12,0,1),gate(8,0,2),gate(3,100,3)]
    label = opening_label(np.zeros(3),np.eye(3),gates,camera)
    assert label['gate'] == 2
    assert opening_label(np.zeros(3),np.eye(3),gates[:1],camera) == dict(visible=0)


def test_panel_clock_matches_rendered_time():
    from PIL import Image,ImageDraw
    from haltere.viz.fastpanel import _font
    image = Image.new('RGB',(1928,720))
    ImageDraw.Draw(image).text((14,26),'t =   12.3 s   distance to target 4.00 m',
                               fill=(255,255,255),font=_font(12))
    value,error = PanelClock(20).read(np.asarray(image)[:,:,::-1].copy(),12.7)
    assert value == 12.3
    assert error == 0.


def test_indexed_labels_preserve_original_recording(tmp_path,monkeypatch):
    import json
    import pandas as pd
    from haltere.vision import flight_gate_labels as module
    source,out=tmp_path/'source',tmp_path/'labelled'
    (source/'frames').mkdir(parents=True)
    (source/'frames/a.jpg').write_bytes(b'original image')
    (source/'capture.json').write_text(json.dumps(dict(origin_sim=[0,0,0],course='test')))
    (source/'camera.yaml').write_text('width: 640\nheight: 360\nf: 200\ntilt_deg: 30\n')
    pd.DataFrame([dict(file='a.jpg',px=0,py=0,pz=1,qw=1,qx=0,qy=0,qz=0)]).to_csv(source/'index.csv',index=False)
    track=tmp_path/'track.xml'; track.write_text('fixture')
    monkeypatch.setattr(module,'track_planes',lambda *a:[dict(base=np.array([10,0,0]),up=np.array([0,0,1]),id=0)])
    original={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    module.prepare_indexed(source,track,out)
    assert (out/'frames/a.jpg').read_bytes()==b'original image'
    assert json.loads((out/'labels.json').read_text())[0]['visible']==1
    assert json.loads((out/'capture.json').read_text())['runtime_uses_track'] is False
    assert original=={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        module.prepare_indexed(source,track,out)
