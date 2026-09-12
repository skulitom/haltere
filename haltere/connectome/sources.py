"""Data sources for the male-CNS connectome.

Primary source: the public "flat connectome" release in the ``gs://flyem-male-cns`` bucket
(Apache Arrow feather files, no login required). Optional: neuPrint (see ``neuprint_source``),
which requires an auth token from https://neuprint.janelia.org.
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

BUCKET = "https://storage.googleapis.com/flyem-male-cns"
FLAT = BUCKET + "/{version}/connectome-data/flat-connectome/"
FILES = {
    'annotations': 'body-annotations-male-cns-{version}-minconf-0.5.feather',
    'neurotransmitters': 'body-neurotransmitters-male-cns-{version}.feather',
    'weights': 'connectome-weights-male-cns-{version}-minconf-0.5-traced-only.feather',
}
DEFAULT_VERSION = 'v1.0'


def project_root() -> Path:
    return Path(os.environ.get('HALTERE_ROOT', Path(__file__).resolve().parents[2]))


def raw_dir(root: Path | None = None, version: str = DEFAULT_VERSION) -> Path:
    root = project_root() if root is None else Path(root)
    return root / 'data' / 'raw' / f'male-cns-{version}'


def file_path(kind: str, root: Path | None = None, version: str = DEFAULT_VERSION) -> Path:
    return raw_dir(root, version) / FILES[kind].format(version=version)


def _download(url: str, dest: Path, resume: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    start = tmp.stat().st_size if (resume and tmp.exists()) else 0
    req = urllib.request.Request(url)
    if start:
        req.add_header('Range', f'bytes={start}-')
    with urllib.request.urlopen(req) as resp:
        total = resp.headers.get('Content-Length')
        total = (int(total) + start) if total else None
        if resp.status == 200 and start:  # server ignored the range
            start = 0
        mode = 'ab' if start else 'wb'
        done = start
        try:
            from tqdm import tqdm
            bar = tqdm(total=total, initial=start, unit='B', unit_scale=True, desc=dest.name, file=sys.stderr)
        except Exception:  # pragma: no cover
            bar = None
        with open(tmp, mode) as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if bar:
                    bar.update(len(chunk))
        if bar:
            bar.close()
    shutil.move(str(tmp), str(dest))


def fetch(root: Path | None = None, version: str = DEFAULT_VERSION, which: list[str] | None = None,
          force: bool = False) -> dict[str, Path]:
    """Download the flat-connectome files (about 570 MB in total). Returns kind -> local path."""
    out = {}
    for kind in (which or list(FILES)):
        dest = file_path(kind, root, version)
        if dest.exists() and not force:
            out[kind] = dest
            continue
        url = FLAT.format(version=version) + FILES[kind].format(version=version)
        print(f'downloading {url} -> {dest}', file=sys.stderr)
        _download(url, dest)
        out[kind] = dest
    return out


ANNOTATION_COLUMNS = ['bodyId', 'type', 'instance', 'class', 'superclass', 'subclass', 'somaSide', 'rootSide',
                      'somaNeuromere', 'entryNerve', 'exitNerve', 'status', 'statusLabel', 'flywireType',
                      'hemibrainType', 'mancType', 'group', 'dimorphism']


def load_annotations(root: Path | None = None, version: str = DEFAULT_VERSION) -> pd.DataFrame:
    path = file_path('annotations', root, version)
    if not path.exists():
        raise FileNotFoundError(f'{path} missing; run `haltere fetch` first')
    df = feather.read_feather(path)
    cols = [c for c in ANNOTATION_COLUMNS if c in df.columns]
    df = df[cols].copy()
    df['bodyId'] = df['bodyId'].astype(np.int64)
    return df


def load_neurotransmitters(root: Path | None = None, version: str = DEFAULT_VERSION) -> pd.DataFrame:
    path = file_path('neurotransmitters', root, version)
    if not path.exists():
        raise FileNotFoundError(f'{path} missing; run `haltere fetch` first')
    df = feather.read_feather(path)
    keep = ['body', 'consensus_nt', 'predicted_nt', 'predicted_nt_confidence', 'celltype_predicted_nt',
            'celltype_predicted_nt_confidence', 'ground_truth']
    df = df[[c for c in keep if c in df.columns]].rename(columns={'body': 'bodyId'})
    df['bodyId'] = df['bodyId'].astype(np.int64)
    return df


def load_weights(root: Path | None = None, version: str = DEFAULT_VERSION) -> pd.DataFrame:
    """Segment-to-segment synapse counts: columns body_pre, body_post, weight."""
    path = file_path('weights', root, version)
    if not path.exists():
        raise FileNotFoundError(f'{path} missing; run `haltere fetch` first')
    tbl = feather.read_table(path, columns=['body_pre', 'body_post', 'weight'])
    return tbl.to_pandas()
