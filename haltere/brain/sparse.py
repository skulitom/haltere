"""Sparse recurrent matmul with a memory- and compute-efficient backward.

``torch.sparse.mm`` differentiates w.r.t. the sparse values by forming a dense (N x N) product and
masking it, which for a 30k-neuron graph is a 3.6 GB, 230 GFLOP detour per time step. Here the
gradient for the values is computed edge-wise (sum over the batch of grad_out[post] * r[pre]) in
chunks, and the gradient for the dense input uses the pre-built transposed sparse matrix.
"""
from __future__ import annotations

import torch


class SparseRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, r, indices, indices_t, perm_t, N, chunk):
        W = torch.sparse_coo_tensor(indices, values, (N, N), is_coalesced=True)
        out = torch.sparse.mm(W, r)
        ctx.save_for_backward(values, r, indices, indices_t, perm_t)
        ctx.N = N
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, grad_out):
        values, r, indices, indices_t, perm_t = ctx.saved_tensors
        N = ctx.N
        grad_values = grad_r = None
        if ctx.needs_input_grad[1]:
            WT = torch.sparse_coo_tensor(indices_t, values[perm_t], (N, N), is_coalesced=True)
            grad_r = torch.sparse.mm(WT, grad_out)
        if ctx.needs_input_grad[0]:
            post, pre = indices
            E = values.shape[0]
            grad_values = torch.empty_like(values)
            for s in range(0, E, ctx.chunk):
                e = min(s + ctx.chunk, E)
                grad_values[s:e] = (grad_out.index_select(0, post[s:e]) * r.index_select(0, pre[s:e])).sum(dim=1)
        return grad_values, grad_r, None, None, None, None, None


def transpose_order(post: torch.Tensor, pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Given edges sorted by (post, pre), return (indices of the transposed matrix sorted by (pre, post),
    permutation that reorders the values accordingly)."""
    N = int(max(post.max(), pre.max())) + 1
    key = pre * N + post
    perm = torch.argsort(key)
    indices_t = torch.stack([pre[perm], post[perm]])
    return indices_t, perm


def sparse_recurrent(values, r, indices, indices_t, perm_t, N, chunk=400_000):
    return SparseRecurrent.apply(values, r, indices, indices_t, perm_t, N, chunk)
