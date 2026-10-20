"""Tracing upstream through MLP nodes (Algorithm 1 of the MLP upstream-tracing report).

ACC++ explains *why an attention head attends* by decomposing its attention
score into upstream signals. An MLP has no decision of its own; when it
contributes a signal to a head, this module finds the small set of upstream
components (head-source pairs, attention biases, the embedding, earlier MLPs)
responsible for the input-dependent part of that signal.

Method (report Sections 2, 7, 8):

- The MLP site ``F`` (the map from the ln2-normalized input ``z`` to
  ``hook_mlp_out``) has a path-average Jacobian ``J_bar = int_0^1 J_F(a z) da``
  with ``F(z) - F(0) = J_bar z`` exactly.
- A *reader operator* ``W`` (``r x D``) gives the coordinates in which the
  downstream reader sees the MLP output. For an attention-head reader these are
  ACC++'s own signal coordinates (RoPE pulled back, divided by the reader's
  ln1 scale); for an MLP reader they are the reader's attribution operator
  divided by its ln2 scale (the "weighted operator" of report Section 8).
- With ``A = W J_bar`` the attributions ``gamma_c = A o_c`` sum exactly to the
  traced target ``t = W (F(z) - F(0))``. Algorithm 1 grows a set greedily,
  applying the row-space edit ``Q Q^T o_c`` (``Q = orth(A^T)``) and re-measuring
  the residual ``W (F(z^-) - F(0))`` in the real MLP after every step.
- The new signal handed upstream for a selected component is its row-space
  part ``Q Q^T o_c``. The constant output ``F(0)`` is a leaf.

The MLP architecture is configured from ``model.cfg`` (see :func:`get_mlp_spec`).
Everything is matrix-free (no ``D x D`` Jacobian), so it runs at Llama scale.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer

from .models import ModelConfig

__all__ = [
    "ComponentSet",
    "MLPSpec",
    "MLPExpansionOptions",
    "MLPSite",
    "MLPTrace",
    "get_mlp_spec",
    "mlp_input_components",
    "run_algorithm1",
]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MLPExpansionOptions:
    """Options for expanding MLP nodes during tracing (report Sections 7-8).

    Attributes:
        mode: How MLPs read by several readers, or reached by several MLP
            paths, are traced. A *reader operator* ``W`` (stored transposed,
            ``WT``, ``D x r``) maps a residual vector written at the MLP's
            position to the ``r`` coordinates its reader measures; an MLP
            selected by a downstream MLP is read through ``W' = W J_bar C / s``
            (the downstream MLP's path-average Jacobian, centering and ln2
            scale). See the README section "Expanding MLP nodes".

            - ``"path_summed"`` (default): one trace per (first-hop reader,
              MLP node). A first-hop reader is an attention-head firing on one
              side (query or key) or the root objective. Within a reader, the
              MLPs at its position form a DAG; each is traced once with the
              *sum* of the operators of all kept paths into it, i.e. its total
              linearized influence on that reader's signal. Nodes are
              ``(layer, "MLP", tok, tok, (reader_step,))``.
            - ``"per_reader"``: one trace per reader *path*: every MLP
              occurrence is traced against the single chain above it. Nodes
              are ``(layer, "MLP", tok, tok, reader_path)``.
            - ``"merged"``: one trace per MLP node against all of its readers
              together (stacked reader operators); keeps the 4-tuple node.
        eps: Algorithm 1's stopping threshold; R_sig <= eps -> expand.
        eps_soft: eps < R_sig <= eps_soft -> expand, marked approximate;
            above -> leaf.
        tau: Prune an upstream MLP child only if BOTH its linear and its
            real-chain share are below ``tau`` in magnitude.
        bias_overshoot_leaf: Keep an MLP as a leaf when R_ideal > 1 (even
            perfect removal of the input-dependent signal would not reduce the
            signal its reader sees).
        expand_mlp0: Expand layer-0 MLPs (default False, matching ACC++'s stop
            at layer-0 heads).
        expand_seed_mlps: Expand MLPs read directly by the root objective
            (seeds) or a probe.
        joint_check: Run the joint-removal check on would-be-pruned siblings.
        max_mlp_nodes: Cap on the number of MLP traces; remaining queued MLPs
            become leaves (with a warning).
        k_max: Optional cap on Algorithm 1's selected-set size.
        quad_points: Quadrature points for path-average integrals that have
            no closed form: Gauss-Legendre for gated MLPs; Gauss-Legendre on
            geometrically graded panels for sites with a post-MLP norm.
        diagnostics: Path-summed and per-reader modes: also run the
            multi-hop check (R_chain, R_floor; report Section 8.2) at every
            MLP traced below the top of its reader's chain.
        prune_head_inputs: Apply the MLP-children pruning rule to selected
            head-source inputs too: a head input whose linear and real shares
            are both below ``tau`` (after the joint-removal check, done jointly
            with the MLP children) keeps its edge but is not handed to the
            ACC++ head recursion. Off by default (every selected head input is
            recursed into, subject only to ACC++'s attention threshold).
        bos_leaf: Treat position 0 (the BOS / attention-sink position) as a
            leaf: MLPs at position 0 are not expanded, and selected head inputs
            reading position 0 are not recursed into.
        min_flow: Flow-ordered expansion (path-summed mode only; default None =
            off). Every node gets a *root flow*: the objective has flow 1, and
            each expanded node splits its flow over its in-edges in proportion
            to ``|weight|``, so a node's flow is the fraction of the objective
            routed through it along the traced graph. Head firings and MLP
            nodes are expanded in one global order (decreasing layer; an MLP
            before the heads of its own layer), so a node's flow is final when
            it is reached; a node whose flow is below ``min_flow`` is not
            expanded (heads: ``flow_pruned`` node attribute; MLPs: decision
            ``"leaf (flow)"``). The flows of any set of nodes no two of which
            lie on one path sum to at most 1 per root, so at most
            ``1 / min_flow`` such nodes are expanded per root. Nodes carry a ``flow`` attribute and
            ``G.graph["flow"]`` summarizes where the objective's flow ends.
            ``tau``, ``prune_head_inputs`` and the other rules still apply.
        min_flow_scope: ``"global"`` (default): a node is expanded if its flow
            is at least ``min_flow``. ``"layer"``: if its flow is at least
            ``min_flow`` times the total flow at its *stage*, a stage being
            the MLP nodes of one layer or the attention-side nodes of one
            layer (heads, bias terms), in the scheduler's order. Flow spreads
            over more nodes, and partly ends at leaves, as it moves away from
            the objective, so a global threshold favours late layers; the
            layer scope compensates. A stage's total is final when the
            scheduler reaches it, and at most ``1 / min_flow`` nodes are
            expanded per stage and root.
    """

    mode: Literal["path_summed", "per_reader", "merged"] = "path_summed"
    eps: float = 0.1
    eps_soft: float = 0.25
    tau: float = 0.01
    bias_overshoot_leaf: bool = True
    expand_mlp0: bool = False
    expand_seed_mlps: bool = True
    joint_check: bool = True
    max_mlp_nodes: int = 1000
    k_max: int | None = None
    quad_points: int = 64
    diagnostics: bool = False
    prune_head_inputs: bool = False
    bos_leaf: bool = False
    min_flow: float | None = None
    min_flow_scope: Literal["global", "layer"] = "global"

    def __post_init__(self):
        if self.mode not in ("path_summed", "per_reader", "merged"):
            raise ValueError(f"mode must be 'path_summed', 'per_reader' or 'merged', got {self.mode!r}")
        if not (0 < self.eps <= self.eps_soft):
            raise ValueError("require 0 < eps <= eps_soft")
        if self.tau < 0:
            raise ValueError("tau must be >= 0")
        if self.max_mlp_nodes < 1:
            raise ValueError("max_mlp_nodes must be >= 1")
        if self.k_max is not None and self.k_max < 1:
            raise ValueError("k_max must be >= 1 or None")
        if self.quad_points < 2:
            raise ValueError("quad_points must be >= 2")
        if self.min_flow is not None:
            if not 0 < self.min_flow < 1:
                raise ValueError("min_flow must be in (0, 1) or None")
            if self.mode != "path_summed":
                raise ValueError("min_flow needs mode='path_summed'")
        if self.min_flow_scope not in ("global", "layer"):
            raise ValueError("min_flow_scope must be 'global' or 'layer'")


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MLPSpec:
    """MLP architecture, derived from ``model.cfg``.

    Attributes:
        kind: ``"standard"`` (``act(z W_in + b_in) W_out + b_out``) or
            ``"gated"`` (``(act(z W_gate) * (z W_in) + b_in) W_out + b_out``).
        act_fn: TransformerLens activation name.
        post_norm: The MLP output passes through a normalization before
            ``hook_mlp_out`` (Gemma-2's ``ln2_post``).
        parallel: Attention and MLP read the same ``resid_pre`` (GPT-J /
            Pythia); MLP ``m`` then cannot read layer ``m``'s attention.
        center: LayerNorm-family normalization (components are mean-centered);
            False for RMSNorm.
    """

    kind: Literal["standard", "gated"]
    act_fn: str
    post_norm: bool
    parallel: bool
    center: bool


def get_mlp_spec(model: HookedTransformer) -> MLPSpec:
    """Detect the MLP architecture from ``model.cfg`` and validate it.

    Raises:
        ValueError: for architectures the method does not support:
            attention-only models, mixture-of-experts, layer-norm activations
            (SoLU-LN), unfolded LayerNorm, or a post-MLP LayerNorm.
    """
    cfg = model.cfg
    name = cfg.model_name
    if cfg.attn_only:
        raise ValueError(f"Model {name} has no MLPs (attn_only=True).")
    if cfg.num_experts is not None:
        raise ValueError(f"MLP tracing does not support mixture-of-experts ({name}).")
    if cfg.is_layer_norm_activation():
        raise ValueError(
            f"MLP tracing does not support layer-norm activations "
            f"(act_fn={cfg.act_fn!r}, {name}): the MLP output is not a fixed "
            f"function of the pre-activation."
        )
    ln2_cls = type(model.blocks[0].ln2).__name__
    if cfg.normalization_type is None:
        raise ValueError(f"MLP tracing needs a normalized MLP input (ln2); {name} has none.")
    if ln2_cls not in ("LayerNormPre", "RMSNormPre"):
        raise ValueError(
            f"MLP tracing needs the ln2 weights folded into the MLP "
            f"(ln2 is {ln2_cls}); load the model with fold_ln=True "
            f"(the TransformerLens default). ACC++ head tracing makes the same "
            f"assumption for ln1."
        )
    post_norm = bool(getattr(cfg, "use_normalization_before_and_after", False))
    if post_norm:
        post_cls = type(model.blocks[0].ln2_post).__name__
        if post_cls not in ("RMSNorm", "RMSNormPre"):
            raise ValueError(
                f"Post-MLP normalization {post_cls} is not supported "
                f"(only RMSNorm, as in Gemma-2)."
            )
    norm = cfg.normalization_type or ""
    return MLPSpec(
        kind="gated" if cfg.gated_mlp else "standard",
        act_fn=str(cfg.act_fn),
        post_norm=post_norm,
        parallel=bool(cfg.parallel_attn_mlp),
        center=norm.startswith("LN"),
    )


# ---------------------------------------------------------------------------
# Input decomposition
# ---------------------------------------------------------------------------

class ComponentSet:
    """The upstream components of one MLP input, held in factored form.

    Row ``i`` is the vector ``o_i`` that component ``ids[i]`` contributes to the
    MLP's normalized input. Head-source rows are ``A[h, t] V[t, h] W_O[h]``
    (times the post-attention norm, when the model has one), so the full
    ``N x d_model`` matrix is never built: ``O @ M`` is computed through
    ``W_O[h] @ M``, and single rows on demand. Rows are mean-centered for
    LayerNorm models (applied to ``M`` instead) and divided by the frozen ln2
    scale.
    """

    def __init__(self, D: int, center: bool, scale: Tensor, device):
        self.D, self.center, self.scale, self.device = D, center, scale.float(), device
        self.ids: list[tuple[int, int, int]] = []
        self._blocks: list[tuple] = []     # ("dense", rows) | ("heads", A, V, W_O, post, sc)
        self._starts: list[int] = []

    # -- construction ----------------------------------------------------
    def _add(self, block, ids):
        self._starts.append(len(self.ids))
        self._blocks.append(block)
        self.ids.extend(ids)

    def add_dense(self, rows: Tensor, ids):
        self._add(("dense", rows.float()), ids)

    def add_heads(self, A, V, W_O, post, sc, layer):
        H, n_tok = A.shape
        self._add(("heads", A.float(), V.float(), W_O, post, sc),
                  [(layer, h, t) for h in range(H) for t in range(n_tok)])

    # -- access ------------------------------------------------------------
    def __len__(self):
        return len(self.ids)

    @property
    def shape(self):
        return (len(self.ids), self.D)

    def _finish(self, v: Tensor) -> Tensor:
        if self.center:
            v = v - v.mean(dim=-1, keepdim=True)
        return v / self.scale

    def _locate(self, i: int):
        import bisect
        b = bisect.bisect_right(self._starts, i) - 1
        return self._blocks[b], i - self._starts[b]

    def row(self, i: int) -> Tensor:
        """Component vector ``o_i`` (``d_model``,)."""
        blk, j = self._locate(int(i))
        if blk[0] == "dense":
            v = blk[1][j]
        else:
            _, A, V, W_O, post, sc = blk
            n_tok = A.shape[1]
            h, t = divmod(j, n_tok)
            v = A[h, t] * (V[t, h] @ W_O[h].float())
            if post is not None:
                v = v * post / sc
        return self._finish(v)

    def __getitem__(self, i):
        if isinstance(i, (list, tuple)):
            return torch.stack([self.row(k) for k in i]) if i else torch.zeros(0, self.D, device=self.device)
        return self.row(i)

    def rows_sum(self, sel) -> Tensor:
        out = torch.zeros(self.D, device=self.device)
        for k in sel:
            out = out + self.row(k)
        return out

    def __matmul__(self, M: Tensor) -> Tensor:
        """``O @ M`` for ``M`` of shape ``(d_model, k)``; returns ``(N, k)``."""
        M = M.float()
        if self.center:
            M = M - M.mean(dim=0, keepdim=True)
        M = M / self.scale
        parts = []
        for blk in self._blocks:
            if blk[0] == "dense":
                parts.append(blk[1] @ M)
            else:
                _, A, V, W_O, post, sc = blk
                Mp = M if post is None else (post[:, None] * M) / sc
                WM = torch.einsum("hdm,mk->hdk", W_O.float(), Mp)            # (H, dh, k)
                g = torch.einsum("ht,thd,hdk->htk", A, V, WM)                # (H, t, k)
                parts.append(g.reshape(-1, M.shape[1]))
        return torch.cat(parts, dim=0)

    def sum(self) -> Tensor:
        """``sum_i o_i`` without materializing the rows."""
        out = torch.zeros(self.D, device=self.device)
        for blk in self._blocks:
            if blk[0] == "dense":
                out = out + blk[1].sum(0)
            else:
                _, A, V, W_O, post, sc = blk
                hd = torch.einsum("ht,thd->hd", A, V)
                v = torch.einsum("hd,hdm->m", hd, W_O.float())
                out = out + (v if post is None else v * post / sc)
        return self._finish(out)

    def dense(self) -> Tensor:
        """The full ``(N, d_model)`` matrix (for tests and small models)."""
        return self @ torch.eye(self.D, device=self.device)


def mlp_input_components(
    model: HookedTransformer,
    config: ModelConfig,
    spec: MLPSpec,
    cache: ActivationCache,
    prompt_idx: int,
    layer: int,
    pos: int,
) -> tuple[list[tuple[int, int, int]], ComponentSet, Tensor]:
    """Decompose MLP ``layer``'s normalized input at ``pos`` into components.

    Component ids follow ACC++'s convention ``(layer, comp_id, src_token)``:
    ``comp_id < n_heads`` is a head reading ``src_token``; ``n_heads`` = MLP,
    ``n_heads + 1`` = AH bias (``b_O``), ``n_heads + 2`` = embedding (the last
    three at ``src_token == pos``). Layer ``layer``'s own attention is included
    unless the model runs attention and MLP in parallel. Components are divided
    by the frozen ``ln2`` scale (and mean-centered for LayerNorm models), so
    they sum exactly to ``ln2.hook_normalized``.

    Returns:
        ``(ids, O, z)``: ``O`` is a :class:`ComponentSet` (factored; supports
        ``O @ M``, ``O[i]``, ``O.dense()``), ``z`` the normalized MLP input.
    """
    H = model.cfg.n_heads
    n_tok = pos + 1
    D = model.cfg.d_model
    z = cache[f"blocks.{layer}.ln2.hook_normalized"][prompt_idx, pos].to(torch.float32)
    O = ComponentSet(D, spec.center, cache[f"blocks.{layer}.ln2.hook_scale"][prompt_idx, pos], z.device)
    O.add_dense(cache["blocks.0.hook_resid_pre"][prompt_idx, pos].unsqueeze(0), [(0, H + 2, pos)])
    last_attn = layer - 1 if spec.parallel else layer
    for l in range(layer + 1):
        if l <= last_attn:
            attn = model.blocks[l].attn
            A = cache[f"blocks.{l}.attn.hook_pattern"][prompt_idx, :, pos, :n_tok]   # (H, t)
            V = cache[f"blocks.{l}.attn.hook_v"][prompt_idx, :n_tok]                # (t, KV, dh)
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
        if l < layer:
            O.add_dense(cache[f"blocks.{l}.hook_mlp_out"][prompt_idx, pos].unsqueeze(0), [(l, H, pos)])
    err = (O.sum() - z).norm() / z.norm().clamp_min(1e-12)
    if err > 1e-3:
        raise RuntimeError(
            f"MLP input decomposition does not reconstruct ln2.hook_normalized "
            f"(layer {layer}, pos {pos}, relative error {err:.2e})."
        )
    return O.ids, O, z


# ---------------------------------------------------------------------------
# The MLP site and its path-average Jacobian
# ---------------------------------------------------------------------------

def _gauss_legendre_01(n: int, device, dtype) -> tuple[Tensor, Tensor]:
    x, w = np.polynomial.legendre.leggauss(n)
    return (torch.tensor((x + 1) / 2, device=device, dtype=dtype),
            torch.tensor(w / 2, device=device, dtype=dtype))


def _graded_gauss_01(n: int, device, dtype, n_panels: int = 8, smallest: float = 1e-7):
    """Gauss-Legendre on geometrically graded panels [0, 1e-7, ..., 1] (about n points).

    Post-MLP normalization makes the path integrand nearly singular at a = 0
    when the raw MLP output vanishes there (no biases, e.g. Gemma-2): the
    norm's gain is ~1/sqrt(eps) inside a boundary layer whose width scales
    with the output size. Graded panels resolve it at any scale.
    """
    edges = [0.0] + list(np.geomspace(smallest, 1.0, n_panels))
    per = max(4, n // n_panels)
    x, w = np.polynomial.legendre.leggauss(per)
    xs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        xs.append(a + (b - a) * (x + 1) / 2)
        ws.append((b - a) / 2 * w)
    return (torch.tensor(np.concatenate(xs), device=device, dtype=dtype),
            torch.tensor(np.concatenate(ws), device=device, dtype=dtype))


def _act_and_prime(act, x: Tensor) -> tuple[Tensor, Tensor]:
    """Elementwise activation and its derivative (autograd; exact for any act)."""
    with torch.enable_grad():
        xr = x.detach().clone().requires_grad_(True)
        y = act(xr)
        (g,) = torch.autograd.grad(y.sum(), xr)
    return y.detach(), g.detach()


class MLPSite:
    """One MLP at ``(layer, pos)``: forward map and path-average Jacobian.

    All Jacobian products are matrix-free. Column convention: ``J_bar`` maps a
    normalized-input vector to an output vector; :meth:`jbar_T` returns
    ``J_bar^T V`` for ``V`` of shape ``(D, k)``.
    """

    def __init__(self, model: HookedTransformer, spec: MLPSpec, layer: int,
                 z: Tensor, quad_points: int = 64):
        self.model, self.spec, self.layer = model, spec, layer
        mlp = model.blocks[layer].mlp   # per-block weights (model.W_in etc. stack all layers)
        self.act = mlp.act_fn
        self.W_in = mlp.W_in.float()          # (D, d_mlp)
        self.b_in = mlp.b_in.float()          # (d_mlp,)
        self.W_out = mlp.W_out.float()        # (d_mlp, D)
        self.b_out = mlp.b_out.float()        # (D,)
        self.W_gate = mlp.W_gate.float() if spec.kind == "gated" else None
        if spec.post_norm:
            pn = model.blocks[layer].ln2_post
            self.pn_w = (pn.w.float() if hasattr(pn, "w")
                         else torch.ones(model.cfg.d_model, device=z.device))
            self.pn_eps = float(pn.eps)
        self.z = z.float()
        self.F0 = self.forward(torch.zeros_like(self.z))
        self.Fz = self.forward(self.z)
        self._prepare(quad_points)
        dz = self.jbar(self.z.unsqueeze(1))[:, 0]
        target = self.Fz - self.F0
        self.completeness_err = ((dz - target).norm() / target.norm().clamp_min(1e-12)).item()
        if self.completeness_err > 1e-2:
            warnings.warn(
                f"MLP {layer}: path-average Jacobian completeness error "
                f"{self.completeness_err:.2e} > 1e-2 (increase quad_points).",
                UserWarning,
            )

    # -- forward ---------------------------------------------------------
    def _raw(self, z: Tensor) -> Tensor:
        if self.spec.kind == "standard":
            h = self.act(z @ self.W_in + self.b_in)
        else:
            h = self.act(z @ self.W_gate) * (z @ self.W_in) + self.b_in
        return h @ self.W_out + self.b_out

    def _norm(self, y: Tensor) -> Tensor:
        return y / (y.pow(2).mean(-1, keepdim=True) + self.pn_eps).sqrt() * self.pn_w

    def forward(self, z: Tensor) -> Tensor:
        """The MLP site output (equals ``hook_mlp_out``) for normalized input ``z``."""
        y = self._raw(z)
        return self._norm(y) if self.spec.post_norm else y

    # -- path-average Jacobian -------------------------------------------
    def _prepare(self, n: int):
        z = self.z
        if self.spec.kind == "standard" and not self.spec.post_norm:
            # Closed form: g_bar_i = (act(h_i + b_i) - act(b_i)) / h_i.
            h = z @ self.W_in
            a_hb, _ = _act_and_prime(self.act, h + self.b_in)
            a_b, da_b = _act_and_prime(self.act, self.b_in)
            safe = h.abs() > 1e-6
            self.g_bar = torch.where(safe, (a_hb - a_b) / torch.where(safe, h, torch.ones_like(h)), da_b)
            self._mode = "closed"
            return
        if self.spec.kind == "gated" and not self.spec.post_norm:
            alphas, weights = _gauss_legendre_01(n, z.device, torch.float32)
            # Per-neuron 1-D integrals: a_bar = u int a act'(a g), b_bar = int act(a g).
            g, u = z @ self.W_gate, z @ self.W_in
            act_v, dact_v = _act_and_prime(self.act, alphas[:, None] * g[None, :])
            self.a_bar = u * (weights[:, None] * alphas[:, None] * dact_v).sum(0)
            self.b_bar = (weights[:, None] * act_v).sum(0)
            self._mode = "gated"
            return
        # Post-norm site: quadrature of the composite Jacobian (graded panels).
        alphas, weights = _graded_gauss_01(n, z.device, torch.float32)
        self._samples = []
        for a, wq in zip(alphas.tolist(), weights.tolist()):
            za = a * z
            if self.spec.kind == "standard":
                pre = za @ self.W_in + self.b_in
                act_v, dact_v = _act_and_prime(self.act, pre)
                d_gate, d_in = None, dact_v
            else:
                g, u = za @ self.W_gate, za @ self.W_in
                act_v, dact_v = _act_and_prime(self.act, g)
                d_gate, d_in = dact_v * u, act_v
            y = self._raw(za)
            s = (y.pow(2).mean() + self.pn_eps).sqrt()
            self._samples.append((wq, d_gate, d_in, y, s))
        self._mode = "quad"

    def _norm_T(self, y: Tensor, s: Tensor, V: Tensor) -> Tensor:
        # Transpose of d/dy [ y / s(y) * w ] applied to V (D, k).
        wV = self.pn_w[:, None] * V
        return wV / s - y[:, None] * ((y @ wV) / (y.numel() * s ** 3))[None, :]

    def _norm_J(self, y: Tensor, s: Tensor, V: Tensor) -> Tensor:
        return self.pn_w[:, None] * (V / s - y[:, None] * ((y @ V) / (y.numel() * s ** 3))[None, :])

    def jbar_T(self, V: Tensor) -> Tensor:
        """``J_bar^T V`` for ``V`` of shape ``(D, k)``; returns ``(D, k)``."""
        V = V.float()
        if self._mode == "closed":
            return self.W_in @ (self.g_bar[:, None] * (self.W_out @ V))
        if self._mode == "gated":
            WoV = self.W_out @ V
            return self.W_gate @ (self.a_bar[:, None] * WoV) + self.W_in @ (self.b_bar[:, None] * WoV)
        out = torch.zeros(self.W_in.shape[0], V.shape[1], device=V.device)
        for wq, d_gate, d_in, y, s in self._samples:
            WoV = self.W_out @ self._norm_T(y, s, V)
            part = self.W_in @ (d_in[:, None] * WoV)
            if d_gate is not None:
                part = part + self.W_gate @ (d_gate[:, None] * WoV)
            out = out + wq * part
        return out

    def jbar(self, V: Tensor) -> Tensor:
        """``J_bar V`` for ``V`` of shape ``(D, k)``; returns ``(D, k)``."""
        V = V.float()
        if self._mode == "closed":
            return self.W_out.T @ (self.g_bar[:, None] * (self.W_in.T @ V))
        if self._mode == "gated":
            return self.W_out.T @ (self.a_bar[:, None] * (self.W_gate.T @ V)
                                   + self.b_bar[:, None] * (self.W_in.T @ V))
        out = torch.zeros(self.W_out.shape[1], V.shape[1], device=V.device)
        for wq, d_gate, d_in, y, s in self._samples:
            hid = d_in[:, None] * (self.W_in.T @ V)
            if d_gate is not None:
                hid = hid + d_gate[:, None] * (self.W_gate.T @ V)
            out = out + wq * self._norm_J(y, s, self.W_out.T @ hid)
        return out


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------

def _orth(M: Tensor, rtol: float = 1e-6) -> Tensor:
    """Orthonormal basis of the column space of ``M`` (``D x k``), rank-revealing."""
    U, s, _ = torch.linalg.svd(M.detach().cpu().double(), full_matrices=False)
    if s.numel() == 0 or s[0] <= 0:
        return torch.zeros(M.shape[0], 0, device=M.device)
    keep = s > rtol * s[0]
    return U[:, keep].float().to(M.device)


@dataclass
class MLPTrace:
    """Result of Algorithm 1 at one MLP (one reader operator).

    Coordinates of all vectors (``gamma``, ``t``) are the reader operator's.
    """

    layer: int
    pos: int
    ids: list                    # component ids (layer, comp_id, src_token)
    selected: list[int]          # indices into ids, in selection order
    gamma: Tensor                # (N, r) attributions
    t: Tensor                    # (r,) traced target W (F(z) - F(0))
    bias: Tensor                 # (r,) constant part W F(0) (a leaf)
    AT: Tensor                   # (D, r) = J_bar^T W^T, the attribution operator
    Q: Tensor                    # (D, q) orthonormal row-space basis of A
    R_sig: float                 # final hop-local residual
    R_ideal: float
    completeness_err: float
    n_candidates: int
    decision: str = ""
    shares: Tensor | None = None  # (N,) linear shares <gamma_c, t> / ||t||^2

    def edit(self, O, sel: list[int]) -> Tensor:
        """Row-space edit removing components ``sel`` (a ``(D,)`` vector)."""
        return self.Q @ (self.Q.T @ _rows_sum(O, sel))


def _rows_sum(O, sel) -> Tensor:
    if isinstance(O, ComponentSet):
        return O.rows_sum(sel)
    if not sel:
        return torch.zeros(O.shape[1], device=O.device)
    return O[sel].sum(0)


def run_algorithm1(
    site: MLPSite,
    O,
    ids: list,
    WT: Tensor,
    eps: float = 0.1,
    k_max: int | None = None,
    pos: int = -1,
) -> MLPTrace:
    """Actual-residual greedy (Algorithm 1) for reader operator ``W`` (``WT = W^T``).

    Each step adds the component with the largest predicted reduction of the
    squared residual, ``2 <gamma_c, r> - ||gamma_c||^2``, applies its row-space
    edit, and recomputes ``r = W (F(z^-) - F(0))`` in the real MLP. Stops when
    ``||r|| / ||t|| < eps``, no positive score remains, or ``k_max`` is reached.
    """
    WT = WT.float()
    AT = site.jbar_T(WT)                         # (D, r)
    gamma = O @ AT                               # (N, r)
    t = WT.T @ (site.Fz - site.F0)
    bias = WT.T @ site.F0
    tot = WT.T @ site.Fz
    R_ideal = (bias.norm() / tot.norm().clamp_min(1e-30)).item()
    Q = _orth(AT)
    t_norm = t.norm().item()
    shares = (gamma @ t) / max(t_norm ** 2, 1e-30)
    out = MLPTrace(site.layer, pos, ids, [], gamma, t, bias, AT, Q, 1.0, R_ideal,
                   site.completeness_err, len(ids), shares=shares)
    if t_norm < 1e-12 or Q.shape[1] == 0:
        out.R_sig = 0.0 if t_norm < 1e-12 else 1.0
        return out
    sel: list[int] = []
    avail = torch.ones(len(ids), dtype=torch.bool, device=gamma.device)
    sq = (gamma * gamma).sum(1)
    edit = torch.zeros_like(site.z)
    r = WT.T @ (site.forward(site.z) - site.F0)
    while r.norm().item() / t_norm >= eps:
        if k_max is not None and len(sel) >= k_max:
            break
        if not bool(avail.any()):
            break
        score = 2 * (gamma @ r) - sq
        score = torch.where(avail, score, torch.full_like(score, -float("inf")))
        i = int(torch.argmax(score))
        if score[i].item() <= 0:
            break
        sel.append(i)
        avail[i] = False
        edit = edit + Q @ (Q.T @ O[i])
        r = WT.T @ (site.forward(site.z - edit) - site.F0)
    out.selected = sel
    out.R_sig = r.norm().item() / t_norm
    return out


# ---------------------------------------------------------------------------
# Reader operators
# ---------------------------------------------------------------------------

def head_reader_operator(
    model: HookedTransformer,
    config: ModelConfig,
    cache: ActivationCache,
    prompt_idx: int,
    U: Tensor,
    VT: Tensor,
    W_Q_pinv: Tensor,
    W_K_pinv: Tensor,
    R_stack: Tensor | None,
    head_layer: int,
    head: int,
    edge_type: str,
    svs_used: list[int],
    pos: int,
    center: bool,
) -> Tensor:
    """``W^T`` (``D x r``) for an MLP read by head ``(head_layer, head)``.

    These are ACC++'s signal coordinates of a residual vector ``x`` written at
    ``pos``: on the query side ``U_svs^T`` of ``(x / s) W_Q R_pos^T W_Q^+``; on
    the key side ``V_svs^T`` of ``(x / s) W_K R_pos^T W_K^+`` (with ``s`` the
    reader's ln1 scale at ``pos``) — i.e. the selected singular vectors pulled
    back through RoPE. Without RoPE they reduce to ``U_svs / s`` (``V_svs / s``).
    For LayerNorm models the vector is centered first, as ln1 does.
    """
    l, h = head_layer, head
    attn = model.blocks[l].attn   # per-block weights (model.W_Q etc. stack all layers)
    if edge_type == "d":
        B = U[l, h][:, svs_used]                     # (D, r)
        Wm, Wp = attn.W_Q[h], W_Q_pinv[l, h]
    else:
        B = VT[l, h][svs_used, :].T
        Wm, Wp = attn.W_K[h], W_K_pinv[l, h]
    if config.has_rope:
        R = R_stack[pos]
        WT = Wm.float() @ (R.T @ (Wp.float() @ B.float()))
    else:
        WT = B.float()
    WT = WT / cache[f"blocks.{l}.ln1.hook_scale"][prompt_idx, pos]
    if center:
        WT = WT - WT.mean(dim=0, keepdim=True)
    return WT


def child_operator(trace: MLPTrace, ln2_scale: Tensor, center: bool) -> Tensor:
    """Reader operator ``W^T`` for an upstream MLP read by the traced MLP.

    The upstream MLP's output enters the traced MLP's normalized input as
    ``center(x) / ln2_scale``; composing with the traced MLP's attribution
    operator ``A`` gives the weighted operator ``W' = A C / s`` of report
    Section 8 (``C`` the centering projector for LayerNorm models).
    """
    WT = trace.AT / ln2_scale
    if center:
        WT = WT - WT.mean(dim=0, keepdim=True)
    return WT
