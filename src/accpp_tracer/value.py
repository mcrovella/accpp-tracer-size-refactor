"""Value inputs of attention-head firings (a third input type next to query and key).

ACC++ traces what drives a head firing's attention score: the query side at the
destination ``d`` and the key side at the source ``s``. What the firing *copies*
comes from its value input, the residual stream at ``s``. With the attention
pattern and the ln1 / post-attention norm scales frozen (as ACC++ freezes them),
the firing's output at ``d`` is linear in the normalized source residual ``z_s``:

    out = A[h, d, s] (z_s W_V[h] + b_V[h]) W_O[h]   (times diag(w_post) / s_post on Gemma-2)

Its *readers* are the graph's out-edges of the firing: heads (ACC++'s signal
coordinates on the query or key side), the root objective, MLPs (the MLP trace's
attribution operator) and the value side of other firings. Their operators
(``D x r_i``, raw residual at ``d``) are stacked into ``W^T`` (``D x R``). Pulling
``W^T`` back through the value gives ``M = A W_V W_O W^T``, which maps a vector in
``z_s`` coordinates to the readers' coordinates. ``z_s`` decomposes exactly into
upstream components ``o_c`` (embedding, head-source pairs and attention biases of
layers ``< l``, MLPs ``< l``), so ``t_c = o_c M`` and ``T = sum_c t_c`` (``b_V`` is a
constant leaf). Greedy forward selection picks components until
``||T - sum_sel t_c|| <= eps ||T||`` or ``k_max`` components; a selected
component's edge weight is its share ``t_c . T / ||T||^2``.

The firing then becomes a reader of its value inputs through
``W_v^T = C M / s_ln1`` (raw residual at ``s``), so value inputs compose along
chains of value edges, as MLP operators do (``MLPExpansionOptions``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer

from .mlp import ComponentSet
from .models import ModelConfig


@dataclass
class ValueInputOptions:
    """Options for tracing the value inputs of head firings.

    Attributes:
        eps: Greedy selection stops when the unexplained part of the readers'
            value signal is at most ``eps`` of it (relative norm).
        k_max: At most this many value inputs per firing.
        tau: A selected value input whose share ``|t_c . T| / ||T||^2`` is
            below ``tau`` keeps its edge (``child_decision="pruned"``) but is
            not traced further. 0 traces every selected input.
        flow_share: Root flow (``node["flow"]``): the fraction of a firing's
            flow passed to its value inputs; the rest goes to its query/key
            inputs (each group split in proportion to ``|weight|``). Used for
            ``MLPExpansionOptions.min_flow`` pruning. A firing whose output is
            the product of its attention and its value needs both, so the
            default splits evenly.
        recurse: Trace selected value inputs further (heads: their query, key
            and value inputs; MLPs: path-summed expansion, if MLP expansion is
            on). False adds one level of value inputs only.
        bos_leaf: Do not value-trace firings that read position 0 (the BOS /
            attention-sink position), and do not trace value inputs that are
            heads reading position 0.
        max_traces: Cap on the number of firings value-traced (a warning is
            issued when reached).
    """

    eps: float = 0.1
    k_max: int = 10
    tau: float = 0.0
    flow_share: float = 0.5
    recurse: bool = True
    bos_leaf: bool = False
    max_traces: int = 20000

    def __post_init__(self):
        if not 0 < self.eps < 1:
            raise ValueError("eps must be in (0, 1)")
        if self.k_max < 1:
            raise ValueError("k_max must be >= 1")
        if self.tau < 0:
            raise ValueError("tau must be >= 0")
        if not 0 <= self.flow_share <= 1:
            raise ValueError("flow_share must be in [0, 1]")
        if self.max_traces < 1:
            raise ValueError("max_traces must be >= 1")


def resolve_value_inputs(opt) -> ValueInputOptions | None:
    """Accept None / False (off), True (defaults), or a ValueInputOptions."""
    if opt is None or opt is False:
        return None
    if opt is True:
        return ValueInputOptions()
    if isinstance(opt, ValueInputOptions):
        return opt
    raise TypeError("value_inputs must be None, a bool, or a ValueInputOptions instance "
                    f"(got {type(opt).__name__}).")


def head_input_components(
    model: HookedTransformer,
    config: ModelConfig,
    center: bool,
    cache: ActivationCache,
    prompt_idx: int,
    layer: int,
    pos: int,
) -> ComponentSet:
    """Decompose layer ``layer``'s normalized attention input (``ln1.hook_normalized``) at ``pos``.

    Component ids follow ACC++'s ``(layer, comp_id, src_token)``: ``comp_id < n_heads``
    a head reading ``src_token``, ``n_heads`` an MLP, ``n_heads + 1`` an attention
    bias, ``n_heads + 2`` the embedding. Components are mean-centered for LayerNorm
    models and divided by the frozen ln1 scale, so they sum to ``ln1.hook_normalized``.
    """
    H, D, n_tok = model.cfg.n_heads, model.cfg.d_model, pos + 1
    z = cache[f"blocks.{layer}.ln1.hook_normalized"][prompt_idx, pos].float()
    O = ComponentSet(D, center, cache[f"blocks.{layer}.ln1.hook_scale"][prompt_idx, pos], z.device)
    O.add_dense(cache["blocks.0.hook_resid_pre"][prompt_idx, pos].unsqueeze(0), [(0, H + 2, pos)])
    for l in range(layer):
        attn = model.blocks[l].attn
        A = cache[f"blocks.{l}.attn.hook_pattern"][prompt_idx, :, pos, :n_tok]
        V = cache[f"blocks.{l}.attn.hook_v"][prompt_idx, :n_tok]
        if config.has_gqa:
            V = torch.repeat_interleave(V, config.gqa_repeats, dim=1)
        post = sc = None
        b_O = attn.b_O.float()
        if config.has_post_attn_ln:
            post = model.blocks[l].ln1_post.w.float()
            sc = cache[f"blocks.{l}.ln1_post.hook_scale"][prompt_idx, pos].float()
            b_O = b_O * post / sc
        O.add_heads(A, V, attn.W_O, post, sc, l)
        O.add_dense(b_O.unsqueeze(0), [(l, H + 1, pos)])
        O.add_dense(cache[f"blocks.{l}.hook_mlp_out"][prompt_idx, pos].unsqueeze(0), [(l, H, pos)])
    err = (O.sum() - z).norm() / z.norm().clamp_min(1e-12)
    if err > 1e-3:
        raise RuntimeError(f"Head input decomposition does not reconstruct ln1.hook_normalized "
                           f"(layer {layer}, pos {pos}, relative error {err:.2e}).")
    return O


def value_operator(
    model: HookedTransformer,
    config: ModelConfig,
    cache: ActivationCache,
    prompt_idx: int,
    layer: int,
    head: int,
    dest: int,
    src: int,
    WT: Tensor,
) -> Tensor:
    """``M = A[h, d, s] W_V[h] W_O[h] (diag(w_post) / s_post) W^T`` (``D x R``).

    Maps a vector in the coordinates of ``z_s`` (layer ``layer``'s normalized
    attention input at ``src``) to the readers' coordinates, through the firing's
    value. ``WT`` is the readers' stacked operator on the raw residual at ``dest``.
    """
    attn = model.blocks[layer].attn
    a = cache[f"blocks.{layer}.attn.hook_pattern"][prompt_idx, head, dest, src].float()
    WT = WT.float()
    if config.has_post_attn_ln:
        post = model.blocks[layer].ln1_post.w.float()
        sc = cache[f"blocks.{layer}.ln1_post.hook_scale"][prompt_idx, dest].float()
        WT = post[:, None] * WT / sc
    return a * (attn.W_V[head].float() @ (attn.W_O[head].float() @ WT))


def raw_reader(M: Tensor, ln1_scale: Tensor, center: bool) -> Tensor:
    """The firing as a reader of the raw residual at its source: ``C M / s_ln1``."""
    WT = M / ln1_scale.float()
    return WT - WT.mean(dim=0, keepdim=True) if center else WT


def compress(WT: Tensor, rtol: float = 1e-6) -> Tensor:
    """An operator with the same Gram matrix ``W^T W`` and at most ``rank`` columns.

    The value selection, its shares and Algorithm 1 depend on the reader
    coordinates only through ``W^T W`` (rotations of the coordinates leave
    every norm and inner product unchanged), so stacked readers can be reduced
    to their rank.
    """
    if WT.shape[1] <= 1:
        return WT
    try:
        U, S, _ = torch.linalg.svd(WT, full_matrices=False)
    except torch.linalg.LinAlgError:           # gesdd non-convergence: use the robust gesvd driver
        import scipy.linalg
        u, sv, _ = scipy.linalg.svd(WT.detach().cpu().double().numpy(), full_matrices=False,
                                    lapack_driver="gesvd")
        U = torch.from_numpy(u).to(WT.device, WT.dtype)
        S = torch.from_numpy(sv).to(WT.device, WT.dtype)
    k = int((S > rtol * S[0]).sum().item()) if S[0] > 0 else 1
    return U[:, :k] * S[:k]


def greedy_select(t: Tensor, eps: float, k_max: int) -> tuple[list[int], float]:
    """Forward selection on the rows of ``t`` (``N x R``) to approximate ``T = t.sum(0)``.

    Returns the selected rows in order and the final relative residual
    ``||T - sum_sel t_c|| / ||T||``.
    """
    T = t.sum(0)
    Tn = T.norm().clamp_min(1e-30)
    r = T.clone()
    sel: list[int] = []
    tn = (t * t).sum(1)
    while len(sel) < k_max and r.norm() > eps * Tn:
        gain = 2 * (t @ r) - tn
        if sel:
            gain[sel] = -float("inf")
        i = int(torch.argmax(gain))
        if gain[i] <= 0:
            break
        sel.append(i)
        r = r - t[i]
    return sel, (r.norm() / Tn).item()
