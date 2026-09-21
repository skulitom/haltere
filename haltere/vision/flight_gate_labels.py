"""Offline opening-centre/range labels from a posed, recorded flight.

Track geometry is used only to supervise the visual frontend. No track, race
order or coordinates are part of its deployed inputs. Review the projected
labels: a visible projection alone does not prove absence of an occluder.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .camera import Camera,quat_wxyz_to_mat
from .datasets import sha256
from ..liftoff.gate_evaluation import track_planes


def opening_label(position,rotation,gates,camera,height=1.2,max_distance=45.):
    candidates=[]
    for gate in gates:
        centre=gate['base']+height*gate['up']
        body=rotation.T@(centre-position)
        pixel,front=camera.project_body(body[None])
        distance=float(np.linalg.norm(body))
        u,v=pixel[0]
        if (front[0] and .8<distance<max_distance
                and 0<=u<camera.width and 0<=v<camera.height):
            du,dv=u-camera.width/2,v-camera.height/2
            stretch=np.sqrt(camera.f**2+du**2)*np.sqrt(camera.f**2+du**2+dv**2)/camera.f**2
            # This is a range encoding, NOT a bounding-box width. Its inverse
            # is detection_geometry, so an oblique gate does not appear farther
            # away simply because its silhouette is foreshortened.
            width=camera.f*4.*stretch/distance
            candidates.append(dict(visible=1,u=float(u),v=float(v),width_px=float(width),
                                   dist_m=distance,gate=gate['id']))
    return min(candidates,key=lambda x:x['dist_m']) if candidates else dict(visible=0)


class PanelClock:
    """Read our own fixed-font clock to align video with telemetry (0.1 s)."""
    def __init__(self,seconds):
        from PIL import Image,ImageDraw
        from ..viz.fastpanel import _font
        font=_font(12)
        self.values=np.arange(int(seconds*10)+31)/10
        images=[]
        for value in self.values:
            im=Image.new('L',(98,16))
            ImageDraw.Draw(im).text((0,0),f't = {value:6.1f} s',fill=255,font=font)
            images.append(np.asarray(im,dtype=np.float32))
        self.templates=np.stack(images)

    def read(self,bgr,video_seconds):
        import cv2
        observed=cv2.cvtColor(bgr[26:42,14:112],cv2.COLOR_BGR2GRAY).astype(np.float32)
        indices=np.flatnonzero(abs(self.values-video_seconds)<3.)
        errors=np.mean((self.templates[indices]-observed)**2,axis=(1,2))
        k=int(errors.argmin())
        return float(self.values[indices[k]]),float(np.sqrt(errors[k]))


def prepare(run,track,out,*,fps=3.,height=1.2):
    import cv2
    run,out=Path(run),Path(out)
    if out.exists():
        raise FileExistsError(out)
    d=pd.read_csv(run.with_suffix('.csv'))
    metadata=json.loads(run.with_suffix('.json').read_text())
    gates=track_planes(track,metadata['origin_sim'])
    camera=Camera(640,360,200.,30.)
    cap=cv2.VideoCapture(str(run.with_suffix('.mp4')))
    video_fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or video_fps<=0:
        raise ValueError('Could not open the flight recording')
    duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/video_fps
    clock=PanelClock(duration)
    elapsed=d.ts.to_numpy()-d.ts.iloc[0]
    out.mkdir(parents=True)
    (out/'frames').mkdir()
    labels,rows=[],[]
    skipped=0
    for seconds in np.arange(1.,min(duration-1.,elapsed[-1]-.5),1/fps):
        cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
        ok,image=cap.read()
        if not ok or image.shape[:2]!=(720,1928):
            raise ValueError('Expected the original 1928x720 brain/flight recording')
        game_time,error=clock.read(image,seconds)
        if error>18 or game_time<.1 or abs(elapsed-game_time).min()>.1:
            skipped+=1
            continue
        row=d.iloc[abs(elapsed-game_time).argmin()]
        position=row[['x','y','z']].to_numpy(dtype=float)
        rotation=quat_wxyz_to_mat(row[['qw','qx','qy','qz']].to_numpy(dtype=float))
        label=opening_label(position,rotation,gates,camera,height)
        name=f'{len(labels):06d}.jpg'
        frame=cv2.resize(image[:,648:],(640,360),interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out/'frames'/name),frame,[cv2.IMWRITE_JPEG_QUALITY,90])
        label['file']=name
        labels.append(label)
        rows.append(dict(file=name,video_seconds=float(seconds),game_elapsed=game_time,clock_rms=error,
                         ts=float(row.ts),px=position[0],py=position[1],pz=position[2],
                         qw=float(row.qw),qx=float(row.qx),qy=float(row.qy),qz=float(row.qz)))
    cap.release()
    if not labels:
        raise ValueError('No reliably synchronized frames')
    (out/'labels.json').write_text(json.dumps(labels,indent=2))
    pd.DataFrame(rows).to_csv(out/'index.csv',index=False)
    provenance=dict(source='offline_track_projection',course='Straw Bale / Field Day',
                    source_flight_sha256=sha256(run.with_suffix('.csv')),
                    labels_reviewed=False,course_complete=False,frames=len(labels),skipped_clock=skipped,
                    opening_height_m=height,centre_offset_m=0.,camera=dict(width=640,height=360,f=200.,tilt_deg=30.),
                    size_target='equivalent 4m frontal width encoding true range; not silhouette width',
                    source_hashes={str(p):sha256(p) for p in [run.with_suffix('.csv'),run.with_suffix('.mp4'),Path(track)]},
                    runtime_uses_track=False)
    (out/'capture.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(dict(out=str(out),frames=len(labels),positive=sum(l['visible'] for l in labels),
                          skipped_clock=skipped,max_clock_rms=max(r['clock_rms'] for r in rows))),flush=True)


def prepare_indexed(source,track,out,*,height=1.2):
    """Relabel original DatasetWriter frames without modifying a human recording."""
    import yaml
    source,out=Path(source),Path(out)
    if out.exists():
        raise FileExistsError(out)
    metadata=json.loads((source/'capture.json').read_text())
    calibration=yaml.safe_load((source/'camera.yaml').read_text())
    camera=Camera(**{k:calibration[k] for k in ('width','height','f','tilt_deg')})
    gates=track_planes(track,metadata['origin_sim'])
    rows=pd.read_csv(source/'index.csv')
    out.mkdir(parents=True)
    (out/'frames').mkdir()
    labels=[]
    for row in rows.itertuples():
        image=(source/'frames'/row.file).resolve()
        if not image.is_relative_to((source/'frames').resolve()):
            raise ValueError('Frame path leaves source recording')
        rotation=quat_wxyz_to_mat(np.array([row.qw,row.qx,row.qy,row.qz]))
        label=opening_label(np.array([row.px,row.py,row.pz]),rotation,gates,camera,height)
        label['file']=row.file
        shutil.copy2(image,out/'frames'/row.file)
        labels.append(label)
    shutil.copy2(source/'index.csv',out/'index.csv')
    (out/'labels.json').write_text(json.dumps(labels,indent=2))
    provenance=dict(source='offline_track_projection',course=metadata.get('course'),
                    source_flight_sha256=sha256(source/'index.csv'),labels_reviewed=False,
                    course_complete=False,frames=len(labels),opening_height_m=height,
                    centre_offset_m=0.,camera=vars(camera),runtime_uses_track=False,
                    size_target='equivalent 4m frontal width encoding true range; not silhouette width',
                    source_hashes={str(p):sha256(p) for p in [source/'index.csv',source/'capture.json',Path(track)]})
    (out/'capture.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(dict(out=str(out),frames=len(labels),positive=sum(l['visible'] for l in labels))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run');p.add_argument('--track',required=True);p.add_argument('--out',required=True)
    p.add_argument('--fps',type=float,default=3.);p.add_argument('--height',type=float,default=1.2)
    p.add_argument('--indexed',action='store_true',help='Use a DatasetWriter directory with original frames/index.csv')
    a=p.parse_args()
    if not np.isfinite(a.fps) or not 0<a.fps<=18 or not np.isfinite(a.height) or a.height<=0:
        raise ValueError('Use a positive height and 0 < fps <= 18')
    if a.indexed:
        prepare_indexed(a.run,a.track,a.out,height=a.height)
    else:
        prepare(a.run,a.track,a.out,fps=a.fps,height=a.height)


if __name__=='__main__':
    main()
