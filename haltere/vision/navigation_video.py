"""Render an explicitly labelled offline forecast comparison on recorded human FPV."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .datasets import sha256
from .navigation import constant_velocity, load_navigation
from .train_navigation import CachedSequences


def render(dataset, checkpoint, take_id, out, *, fps=18., speed=2.):
    import cv2
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg or fps <= 0 or speed <= 0:
        raise ValueError('ffmpeg and positive playback rates are required')
    manifest = json.loads((Path(dataset)/'manifest.json').read_text(encoding='utf-8'))
    entry = next(t for t in manifest['takes'] if t['id'] == take_id)
    ds = CachedSequences(dataset, entry['split'], length=16, stride=1)
    n = next(i for i,(e,_) in enumerate(ds.takes) if e['id'] == take_id)
    entry, a = ds.takes[n]
    source = (ds.root/entry['source']).resolve()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    model, _ = load_navigation(checkpoint, device)
    starts = {start for take, start in ds.windows if take == n}
    # Replay on actual timestamps; every model context ends at its displayed frame.
    times = np.arange(a['t'][15], a['t'][-1], speed/fps)
    ids = np.searchsorted(a['t'], times, side='right')-1
    ids = [int(i) for i in ids if i-15 in starts]
    predictions, bases = [], []
    with torch.no_grad():
        for offset in range(0, len(ids), 24):
            current = ids[offset:offset+24]
            rows = np.array([np.arange(i-15,i+1) for i in current])
            pixels = torch.from_numpy(ds.pixels[n][rows].astype('float32')/255).to(device)
            velocity = torch.from_numpy(a['velocity_body'][rows]).to(device)
            attitude = torch.from_numpy(a['attitude'][rows]).to(device)
            ts = a['timestamp'][rows]; ts = torch.from_numpy((ts-ts[:, :1]).astype('float32')).to(device)
            predictions.extend(model(pixels,velocity,attitude,ts)[:, -1].cpu().numpy())
            bases.extend(constant_velocity(velocity,model.horizons)[:, -1].cpu().numpy())
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix('.ffmpeg.log')
    width, height = 1024, 576
    colours = {'human':(240,240,240),'model':(194,219,100),'baseline':(123,155,231)}
    extent = max(5., float(np.quantile(np.linalg.norm(a['velocity_body'],axis=1),.95))*1.2)

    def text(canvas, label, xy, scale=.55, colour=(221,226,234), thickness=1):
        cv2.putText(canvas,label,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,colour,thickness,cv2.LINE_AA)

    def chart_point(p):
        # Horizontal right = negative body-left. Vertical up = body-forward.
        x = 834 - float(p[1])/extent*148
        y = 421 - float(p[0])/extent*300
        return int(x),int(y)

    with log_path.open('w',encoding='utf-8') as log:
        proc = subprocess.Popen([ffmpeg,'-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}',
                                 '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20',
                                 '-pix_fmt','yuv420p','-movflags','+faststart',str(out)],
                                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            for i, prediction, baseline in zip(ids,predictions,bases):
                canvas = np.full((height,width,3),(35,25,17),dtype=np.uint8)
                image = cv2.imread(str(source/'frames'/str(a['filename'][i])))
                canvas[98:458,16:656] = image
                text(canvas,'HALTERE  |  learned path forecasts',(18,31),.8,thickness=2)
                text(canvas,'OFFLINE COMPARISON - RECORDED HUMAN FLIGHT',(18,62),.58,(110,211,241))
                text(canvas,take_id.replace('_',' ') + f'  |  {speed:g}x playback',(18,87),.5)
                cv2.rectangle(canvas,(670,98),(1004,464),(68,57,45),1)
                for frac in (-.5,0,.5):
                    x=int(834+frac*296);cv2.line(canvas,(x,99),(x,463),(58,47,36),1)
                for frac in (0,.25,.5,.75,1):
                    y=int(421-frac*300);cv2.line(canvas,(671,y),(1003,y),(58,47,36),1)
                text(canvas,'TOP-DOWN: forward is up',(688,118),.45)
                text(canvas,f'Grid height: {extent:.1f} metres',(688,141),.42)
                human = a['future_body'][i]
                for path,colour in ((baseline,colours['baseline']),(human,colours['human']),(prediction,colours['model'])):
                    points=np.array([chart_point(p) for p in np.vstack((np.zeros(3),path))])-np.array([671,99])
                    chart=canvas[99:463,671:1003]
                    cv2.polylines(chart,[points],False,colour,2,cv2.LINE_AA)
                    for point in points[1:]:cv2.circle(chart,tuple(point),3,colour,-1,cv2.LINE_AA)
                text(canvas,'Human future',(685,484),.45,colours['human'])
                text(canvas,'Learned model',(835,484),.45,colours['model'])
                text(canvas,'Constant velocity',(685,506),.45,colours['baseline'])
                learned=float(np.linalg.norm(prediction[-1]-human[-1])); simple=float(np.linalg.norm(baseline[-1]-human[-1]))
                text(canvas,f'Time {a["t"][i]:5.1f}s   Speed {np.linalg.norm(a["velocity_body"][i]):.1f} m/s',(18,484),.58)
                text(canvas,f'1-second 3D error: learned {learned:.2f}m / baseline {simple:.2f}m',(18,513),.55)
                text(canvas,'Model masks HUD/sticks. Human future is shown only for comparison.',(18,547),.49)
                proc.stdin.write(canvas.tobytes())
        finally:
            proc.stdin.close()
            returncode=proc.wait()
    if returncode:
        raise RuntimeError(log_path.read_text(encoding='utf-8'))
    metadata=dict(take=take_id,source_split=entry['split'],checkpoint_sha256=sha256(checkpoint),
                  dataset_sha256=sha256(Path(dataset)/'manifest.json'),video_sha256=sha256(out),
                  type='offline forecasts on human gameplay, not autonomous flight',frames=len(ids),fps=fps,speed=speed,
                  source_time_s=[float(a['t'][ids[0]]),float(a['t'][ids[-1]])],context_frames=16,
                  metrics_note='Displayed context always has 16 frames; benchmark scores include shorter post-warmup contexts')
    out.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'{out}: {len(ids)} frames, {len(ids)/fps:.1f} seconds',flush=True)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset');p.add_argument('checkpoint');p.add_argument('--take',required=True);p.add_argument('--out',required=True)
    p.add_argument('--fps',type=float,default=18.);p.add_argument('--speed',type=float,default=2.)
    a=p.parse_args();render(a.dataset,a.checkpoint,a.take,a.out,fps=a.fps,speed=a.speed)


if __name__=='__main__':main()
