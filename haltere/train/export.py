"""Slim inference checkpoints: parameters and running statistics only (no optimiser, no buffers that
are rebuilt from the graph), about 12 MB instead of 200 MB, for publishing."""
from __future__ import annotations

from pathlib import Path

import torch


def export_slim(src: str | Path, dst: str | Path) -> dict:
    ck = torch.load(src, map_location='cpu')
    sd = ck['model']
    keep = {}
    for k, v in sd.items():
        if k.startswith(('pre', 'post', 'n_syn', 'nnorm', 'edge_index', 'edge_index_t', 'perm_t', 'sign_fixed',
                         'sign_known', 'unknown_pos', 'motor_idx')) or k.startswith('idx_') or k.endswith('.group') \
                or k.endswith('.U') and False:
            continue   # rebuilt from the graph at load time
        keep[k] = v
    slim = {'model': keep, 'config': ck['config'], 'iter': ck.get('iter', 0), 'graph': ck['graph'],
            'channels': ck['channels'], 'slim': True}
    torch.save(slim, dst)
    size = Path(dst).stat().st_size
    return {'src': str(src), 'dst': str(dst), 'params_kept': len(keep), 'bytes': size}
