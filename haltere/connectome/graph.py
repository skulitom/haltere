"""Build a connectome-constrained graph (neurons, signed synapse counts, named populations)
from the male-CNS flat connectome.

The "flight graph" keeps the sensory and motor populations, whole circuits of interest
(central complex, descending neurons), and every neuron that lies on a short path from a sensory
population to a motor population.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import sources
from .populations import DEFAULT_FLIGHT_POPULATIONS, select_populations, describe_populations

DEFAULT_SIGNS = {  # neurotransmitter -> synaptic sign (0 = unknown / learnable)
    'acetylcholine': +1,
    'gaba': -1,
    'glutamate': -1,   # GluCl-mediated inhibition dominates in the fly CNS
    'histamine': -1,
    'dopamine': 0,
    'octopamine': 0,
    'serotonin': 0,
    'unclear': 0,
}


@dataclass
class GraphConfig:
    version: str = sources.DEFAULT_VERSION
    min_synapses: int = 3
    status: tuple[str, ...] = ('Traced',)
    exclude_superclass: tuple[str, ...] = ('ol_intrinsic', 'ol_sensory')
    populations: dict = field(default_factory=lambda: {k: [dict(s) for s in v] for k, v in DEFAULT_FLIGHT_POPULATIONS.items()})
    include: tuple[str, ...] = ('descending', 'cx')
    sources: tuple[str, ...] = ('haltere', 'wing_cs', 'lptc', 'ocelli', 'jo', 'compass', 'goal')
    sinks: tuple[str, ...] = ('wing_mn',)
    hops_forward: int = 2
    hops_backward: int = 2
    max_neurons: int = 30000
    signs: dict = field(default_factory=lambda: dict(DEFAULT_SIGNS))
    sign_min_confidence: float = 0.0
    whole_cns: bool = False   # ignore the path-based selection and keep every neuron (large!)
    # populations defined by connectivity after node selection: name -> {presynaptic_of: <population>,
    # min_synapses: int, exclude_sensory: bool}
    derived: dict = field(default_factory=lambda: {'premotor': {'presynaptic_of': 'wing_mn', 'min_synapses': 5,
                                                                'exclude_sensory': True}})

    @staticmethod
    def from_dict(d: dict) -> "GraphConfig":
        cfg = GraphConfig()
        for k, v in d.items():
            if not hasattr(cfg, k):
                raise KeyError(f'unknown graph config key: {k}')
            if isinstance(getattr(cfg, k), tuple):
                v = tuple(v)
            setattr(cfg, k, v)
        return cfg


class BrainGraph:
    """Nodes are neurons (index 0..N-1), edges carry synapse counts from pre to post."""

    def __init__(self, nodes: pd.DataFrame, pre: np.ndarray, post: np.ndarray, n_syn: np.ndarray,
                 populations: dict[str, np.ndarray], meta: dict | None = None):
        self.nodes = nodes.reset_index(drop=True)
        self.pre = pre.astype(np.int64)
        self.post = post.astype(np.int64)
        self.n_syn = n_syn.astype(np.float32)
        self.populations = {k: np.asarray(v, dtype=np.int64) for k, v in populations.items()}
        self.meta = meta or {}

    @property
    def N(self) -> int:
        return len(self.nodes)

    @property
    def E(self) -> int:
        return len(self.pre)

    @property
    def sign(self) -> np.ndarray:
        return self.nodes['sign'].to_numpy().astype(np.int8)

    def population(self, name: str) -> np.ndarray:
        return self.populations[name]

    def summary(self) -> str:
        lines = [f'BrainGraph: {self.N} neurons, {self.E} edges, {int(self.n_syn.sum())} synapses']
        s = self.sign
        lines.append(f'  signs: +{int((s > 0).sum())} / -{int((s < 0).sum())} / unknown {int((s == 0).sum())}')
        for k, v in self.populations.items():
            lines.append(f'  {k:12s} {len(v):6d}')
        return '\n'.join(lines)

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix('.npz'), pre=self.pre, post=self.post, n_syn=self.n_syn,
                            **{f'pop__{k}': v for k, v in self.populations.items()})
        self.nodes.to_parquet(path.with_suffix('.nodes.parquet'), index=False)
        with open(path.with_suffix('.meta.json'), 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, indent=2, default=str)

    @staticmethod
    def load(path: Path) -> "BrainGraph":
        path = Path(path)
        z = np.load(path.with_suffix('.npz'))
        pops = {k[len('pop__'):]: z[k] for k in z.files if k.startswith('pop__')}
        nodes = pd.read_parquet(path.with_suffix('.nodes.parquet'))
        meta_path = path.with_suffix('.meta.json')
        meta = json.load(open(meta_path, encoding='utf-8')) if meta_path.exists() else {}
        return BrainGraph(nodes, z['pre'], z['post'], z['n_syn'], pops, meta)


# ---------------------------------------------------------------------- building

def _reach(A: sp.csr_matrix, seed: np.ndarray, hops: int, reverse: bool = False) -> np.ndarray:
    """Boolean mask of nodes reachable from ``seed`` within ``hops`` steps (following edges, or against them)."""
    M = A.T.tocsr() if reverse else A
    reached = seed.copy()
    frontier = seed.copy()
    for _ in range(hops):
        nxt = (M.T @ frontier.astype(np.float32)) > 0  # nodes that receive an edge from the frontier
        nxt &= ~reached
        if not nxt.any():
            break
        reached |= nxt
        frontier = nxt
    return reached


def assign_signs(nodes: pd.DataFrame, nt: pd.DataFrame, signs: dict, min_conf: float) -> pd.DataFrame:
    nt = nt.set_index('bodyId')
    nt = nt.reindex(nodes['bodyId'].to_numpy())
    body_nt = nt['consensus_nt'].fillna('unclear').to_numpy().astype(object)
    body_conf = nt['predicted_nt_confidence'].fillna(0.0).to_numpy()
    ct_nt = nt['celltype_predicted_nt'].fillna('unclear').to_numpy().astype(object)
    ct_conf = nt['celltype_predicted_nt_confidence'].fillna(0.0).to_numpy()
    chosen = np.where((body_nt != 'unclear') & (body_conf >= min_conf), body_nt,
                      np.where((ct_nt != 'unclear') & (ct_conf >= min_conf), ct_nt, 'unclear'))
    source = np.where((body_nt != 'unclear') & (body_conf >= min_conf), 'body',
                      np.where((ct_nt != 'unclear') & (ct_conf >= min_conf), 'celltype', 'none'))
    sign = np.array([signs.get(x, 0) for x in chosen], dtype=np.int8)
    nodes = nodes.copy()
    nodes['nt'] = chosen
    nodes['nt_source'] = source
    nodes['sign'] = sign
    return nodes


def build_graph(cfg: GraphConfig, root: Path | None = None, verbose: bool = True,
                tables: dict[str, pd.DataFrame] | None = None) -> BrainGraph:
    """Build the graph from the flat-connectome files under ``root`` (default) or from ``tables``
    (dict with 'annotations', 'neurotransmitters', 'weights' DataFrames, e.g. from neuPrint)."""
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    tables = tables or {}

    ann = tables['annotations'] if 'annotations' in tables else sources.load_annotations(root, cfg.version)
    keep = ann['status'].isin(cfg.status).to_numpy().copy()
    if cfg.exclude_superclass:
        keep &= ~ann['superclass'].isin(cfg.exclude_superclass).fillna(False).to_numpy()
    ann = ann[keep].reset_index(drop=True)
    log(f'[graph] {len(ann)} candidate neurons after status/superclass filter ({time.time()-t0:.1f}s)')

    masks = select_populations(ann, cfg.populations)
    log('[graph] populations:\n' + describe_populations(ann, masks))
    for name in list(cfg.include) + list(cfg.sources) + list(cfg.sinks):
        if name not in masks:
            raise KeyError(f'population {name!r} is referenced but not defined')

    # --- adjacency among candidates
    W = tables['weights'] if 'weights' in tables else sources.load_weights(root, cfg.version)
    W = W[W['weight'] >= cfg.min_synapses]
    body_index = pd.Series(np.arange(len(ann)), index=ann['bodyId'].to_numpy())
    pre_idx = body_index.reindex(W['body_pre'].to_numpy()).to_numpy()
    post_idx = body_index.reindex(W['body_post'].to_numpy()).to_numpy()
    ok = ~np.isnan(pre_idx) & ~np.isnan(post_idx)
    pre_idx = pre_idx[ok].astype(np.int64)
    post_idx = post_idx[ok].astype(np.int64)
    w = W['weight'].to_numpy()[ok].astype(np.float32)
    del W
    n = len(ann)
    A = sp.csr_matrix((w, (pre_idx, post_idx)), shape=(n, n))
    log(f'[graph] {len(w)} edges >= {cfg.min_synapses} synapses among candidates ({time.time()-t0:.1f}s)')

    # --- node selection
    core = np.zeros(n, dtype=bool)
    for name in list(cfg.include) + list(cfg.sources) + list(cfg.sinks):
        core |= masks[name]
    if cfg.whole_cns:
        selected = np.ones(n, dtype=bool)
    else:
        src = np.zeros(n, dtype=bool)
        for name in cfg.sources:
            src |= masks[name]
        snk = np.zeros(n, dtype=bool)
        for name in cfg.sinks:
            snk |= masks[name]
        fwd = _reach(A, src, cfg.hops_forward)
        bwd = _reach(A, snk, cfg.hops_backward, reverse=True)
        path_nodes = fwd & bwd
        selected = core | path_nodes
        log(f'[graph] core {int(core.sum())}, forward-reach {int(fwd.sum())}, backward-reach {int(bwd.sum())}, '
            f'on-path {int(path_nodes.sum())}, selected {int(selected.sum())}')
        if selected.sum() > cfg.max_neurons:
            # rank non-core nodes by synaptic coupling with the selected set, keep the strongest
            sel_f = selected.astype(np.float32)
            coupling = (A @ sel_f) + (A.T @ sel_f)   # out-weight to selected + in-weight from selected
            cand = np.where(selected & ~core)[0]
            budget = max(cfg.max_neurons - int(core.sum()), 0)
            order = cand[np.argsort(-coupling[cand], kind='stable')]
            selected = core.copy()
            selected[order[:budget]] = True
            log(f'[graph] capped to {int(selected.sum())} neurons (budget {budget} beyond core)')

    idx = np.where(selected)[0]
    remap = -np.ones(n, dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    sub = A[idx][:, idx].tocoo()
    nodes = ann.iloc[idx].reset_index(drop=True)

    # --- signs from neurotransmitter predictions
    nt = tables['neurotransmitters'] if 'neurotransmitters' in tables else sources.load_neurotransmitters(root, cfg.version)
    nodes = assign_signs(nodes, nt, cfg.signs, cfg.sign_min_confidence)

    pops = {name: remap[np.where(m)[0]] for name, m in masks.items()}
    pops = {k: v[v >= 0] for k, v in pops.items()}
    # derived populations defined by connectivity, e.g. the premotor partners of the wing motor neurons
    for name, spec in (cfg.derived or {}).items():
        target = pops[spec['presynaptic_of']]
        strong = sub.data >= float(spec.get('min_synapses', 5))
        onto = np.isin(sub.col, target) & strong
        cand = np.unique(sub.row[onto])
        cand = cand[~np.isin(cand, target)]
        sc = nodes['superclass'].fillna('').astype(str).to_numpy()
        if spec.get('exclude_sensory', True):
            keep = ~np.array([('sensory' in s) for s in sc[cand]], dtype=bool)
            cand = cand[keep]
        if spec.get('superclass'):
            cand = cand[np.isin(sc[cand], list(spec['superclass']))]
        pops[name] = cand.astype(np.int64)
    for name in pops:
        nodes[f'pop_{name}'] = False
        nodes.loc[pops[name], f'pop_{name}'] = True

    meta = {'config': asdict(cfg), 'built': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_candidates': int(n), 'n_nodes': int(len(idx)), 'n_edges': int(sub.nnz)}
    g = BrainGraph(nodes, sub.row, sub.col, sub.data, pops, meta)
    log(f'[graph] done in {time.time()-t0:.1f}s\n' + g.summary())
    return g
