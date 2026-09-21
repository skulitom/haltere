"""Replace raw scene samples with frozen, training-normalized visual features."""
import argparse,copy,json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from ..vision.datasets import sha256
from ..vision.train import load_gatenet
from ..vision.scene_features import scene_map,fit_projection,project_scene


@torch.no_grad()
def prepare(prepared,dataset,out):
    torch.set_num_threads(2);prepared,dataset,out=map(Path,(prepared,dataset,out))
    manifest=json.loads((prepared/'manifest.json').read_text())
    if manifest['dataset_sha256']!=sha256(dataset/'manifest.json'):raise ValueError('Source manifest changed')
    sensor=manifest['gate_sensor']
    if sha256(sensor['checkpoint'])!=sensor['sha256']:raise ValueError('Detector changed')
    net=load_gatenet(sensor['checkpoint'],'cuda');net.requires_grad_(False)
    sources={t['id']:t for t in json.loads((dataset/'manifest.json').read_text())['takes']}
    out.mkdir(parents=True,exist_ok=False);training=[];cached=[]
    for entry in manifest['takes']:
        source=sources[entry['id']];root=(dataset/source['source']).resolve()
        for name,digest in source['source_hashes'].items():
            if sha256(root/name)!=digest:raise ValueError('Raw recording changed')
        if sha256(prepared/entry['arrays'])!=entry['sha256']:raise ValueError('Scene replay changed')
        with np.load(prepared/entry['arrays']) as z:image_ids=np.unique(z['image_row'])
        index=pd.read_csv(root/'index.csv');features=[]
        for start in range(0,len(image_ids),32):
            pixels=[]
            for i in image_ids[start:start+32]:
                rgb=cv2.cvtColor(cv2.imread(str(root/'frames'/index.iloc[i]['file'])),cv2.COLOR_BGR2RGB)
                pixels.append(cv2.resize(rgb,(320,180),interpolation=cv2.INTER_AREA))
            x=torch.tensor(np.stack(pixels).transpose(0,3,1,2),device='cuda',dtype=torch.float32)/255
            features.append(scene_map(net,x).cpu())
        features=torch.cat(features);cached.append((entry,image_ids,features))
        if entry['split']=='train':training.append(features)
        print('Features',entry['id'],len(features),flush=True)
    projection=fit_projection(torch.cat(training));projection['detector_sha256']=sensor['sha256']
    result=copy.deepcopy(manifest);result.update(parent_replay_sha256=sha256(prepared/'manifest.json'),
        feature_preparation_code_sha256=sha256(__file__),retina_mode='gatenet_scene_v1',scene_projection=projection,
        projection_fit_split='train',takes=[])
    for entry,ids,features in cached:
        with np.load(prepared/entry['arrays']) as z:a={k:z[k] for k in z.files}
        a['retina']=project_scene(features,projection).numpy()[np.searchsorted(ids,a['image_row'])]
        np.savez_compressed(out/entry['arrays'],**a)
        result['takes'].append({**entry,'sha256':sha256(out/entry['arrays'])})
    (out/'manifest.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    prepare(a.prepared,a.dataset,a.out)
