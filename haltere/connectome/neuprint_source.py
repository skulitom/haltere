"""Optional neuPrint path (https://neuprint.janelia.org) using neuprint-python.

Needs an auth token: log in at neuprint.janelia.org with a Google account, open Account (top right)
and copy the Auth Token into the ``NEUPRINT_APPLICATION_CREDENTIALS`` environment variable.

The bulk flat-connectome files are the fast path for building the whole graph; this module offers
(1) a connectivity check, (2) fetching the same three tables (annotations, neurotransmitters,
weights) for a set of neurons straight from neuPrint, e.g. to rebuild the graph from a newer
snapshot or to cross-check the flat files, and (3) ad-hoc Cypher.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

SERVER = 'neuprint.janelia.org'
DATASET = 'male-cns:v1.0'
ENV_TOKEN = 'NEUPRINT_APPLICATION_CREDENTIALS'

ANNOTATION_PROPS = ['bodyId', 'type', 'instance', 'class', 'superclass', 'subclass', 'somaSide', 'rootSide',
                    'somaNeuromere', 'entryNerve', 'exitNerve', 'status', 'statusLabel', 'flywireType',
                    'hemibrainType', 'mancType', 'group', 'dimorphism']
NT_PROPS = ['bodyId', 'consensusNt', 'predictedNt', 'predictedNtConfidence', 'celltypePredictedNt',
            'celltypePredictedNtConfidence']


def client(dataset: str = DATASET, server: str = SERVER, token: str | None = None):
    from neuprint import Client
    token = token or os.environ.get(ENV_TOKEN)
    if not token:
        raise RuntimeError(f'set {ENV_TOKEN} to your neuPrint auth token (neuprint.janelia.org -> Account)')
    return Client(server, dataset=dataset, token=token)


def check(dataset: str = DATASET) -> dict:
    c = client(dataset)
    meta = c.fetch_custom('MATCH (m:Meta) RETURN m.dataset AS dataset, m.lastDatabaseEdit AS edited, '
                          'm.totalPreCount AS pre, m.totalPostCount AS post')
    n = c.fetch_custom("MATCH (n:Neuron) WHERE n.status = 'Traced' RETURN count(n) AS traced")
    return {'server': SERVER, 'dataset': dataset, 'version': c.fetch_version(), 'meta': meta.iloc[0].to_dict(),
            'traced_neurons': int(n['traced'].iloc[0]), 'neuron_keys': c.fetch_neuron_keys()}


def fetch_annotations(dataset: str = DATASET, status: tuple[str, ...] = ('Traced',)) -> pd.DataFrame:
    """Neuron property table with the same columns as the flat-connectome annotation file."""
    from neuprint import NeuronCriteria as NC, fetch_neurons
    c = client(dataset)
    df = fetch_neurons(NC(status=list(status), client=c), omit_rois=True, returned_columns=ANNOTATION_PROPS + NT_PROPS,
                       client=c)
    df['bodyId'] = df['bodyId'].astype(np.int64)
    return df


def neurotransmitters_from_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape neuPrint's per-neuron NT properties into the flat-file layout used by graph building."""
    out = pd.DataFrame({
        'bodyId': df['bodyId'].astype(np.int64),
        'consensus_nt': df.get('consensusNt', pd.Series(['unclear'] * len(df))).fillna('unclear'),
        'predicted_nt': df.get('predictedNt', pd.Series(['unclear'] * len(df))).fillna('unclear'),
        'predicted_nt_confidence': df.get('predictedNtConfidence', pd.Series([0.0] * len(df))).fillna(0.0),
        'celltype_predicted_nt': df.get('celltypePredictedNt', pd.Series(['unclear'] * len(df))).fillna('unclear'),
        'celltype_predicted_nt_confidence': df.get('celltypePredictedNtConfidence', pd.Series([0.0] * len(df))).fillna(0.0),
    })
    return out


def fetch_weights(body_ids: np.ndarray, dataset: str = DATASET, min_weight: int = 1, batch_size: int = 200,
                  threads: int = 4) -> pd.DataFrame:
    """Total connection weights among ``body_ids`` (columns body_pre, body_post, weight). Slow for >10k bodies."""
    from neuprint import fetch_adjacencies
    c = client(dataset)
    ids = [int(b) for b in body_ids]
    _, conn = fetch_adjacencies(ids, ids, min_total_weight=min_weight, omit_rois=True, batch_size=batch_size,
                                threads=threads, client=c)
    return conn.rename(columns={'bodyId_pre': 'body_pre', 'bodyId_post': 'body_post'})[['body_pre', 'body_post', 'weight']]


def tables_from_neuprint(dataset: str = DATASET, status: tuple[str, ...] = ('Traced',), body_ids=None,
                         min_weight: int = 1) -> dict[str, pd.DataFrame]:
    """The three tables ``build_graph`` needs, fetched from neuPrint instead of the flat files."""
    ann = fetch_annotations(dataset, status)
    nt = neurotransmitters_from_annotations(ann)
    ids = ann['bodyId'].to_numpy() if body_ids is None else np.asarray(body_ids)
    w = fetch_weights(ids, dataset, min_weight)
    return {'annotations': ann[ANNOTATION_PROPS], 'neurotransmitters': nt, 'weights': w}
