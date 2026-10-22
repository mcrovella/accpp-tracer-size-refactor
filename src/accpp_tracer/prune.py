"""Frozen-necessity pruning of traced graphs (no ablation reference needed).

A trace's *model nodes* are head outputs at a position ``("H", layer, head, pos)``
(every firing of the head at destination ``pos``) and MLP outputs at a position
``("M", layer, pos)`` (every reader-specific occurrence of the MLP). Attention
biases, the embedding and the objective are always kept.

Every graph edge ``u -> v`` carries a *traced signal*: the projection of
``u``'s clean output (raw residual) onto the column space of the reader
operator ACC++ traced ``v``'s input with: the edge's selected singular vectors
(query / key edges), the firing's value operator (value edges), the MLP node's
path-summed operator (edges into an MLP) or the root reader (edges into the
objective). A source's signals into several readers sharing one input are
removed once, as the projection onto their joint column space.

The *frozen score* of a kept set ``K`` is the objective in one forward pass in
which (a) every model node outside ``K`` is held at its clean output
(``hook_z``, ``hook_mlp_out``) and (b) for every edge ``u -> v`` with ``v``
kept and ``u`` not, ``u``'s traced signal is removed from ``v``'s input: the
query (key) of the head at its destination (source) is recomputed from the live
residual minus the signal, as ACC++'s local intervention; a value term
``A[d, s] v_s`` of ``z_d`` is recomputed from the residual minus the signal; an
MLP's normalised input; the objective's residual. With ``K`` = the trace nothing
changes. The *frozen necessity* of a model node ``n`` is

    nec(n) = (s(M) - s_F(T minus {n})) / norm,

one forward pass per model node, all against the full trace ``T``. The pruned
circuit is ``{n : nec(n) >= theta}``.

Objectives ``s`` and their ``norm``:

- linear seeding from :meth:`Tracer.trace`: the real logit (difference) at the
  last position, through the final norm and any logit soft-cap; norm ``s(M)``;
- linear seeding from :meth:`Tracer.trace_from_cache`: ``ln_final(x) . d``, the
  quantity ACC++ decomposes (``d`` the logit direction); norm ``s(M)``;
- probability-aware seeding: ``J = sum_t q*_t log softmax(z)_t`` on the uncapped
  logits, as the seeding; norm the completeness ``J(M) - J(b_U)``;
- a probe: ``x . w`` at ``blocks.{layer}.hook_resid_post``; norm ``s(M)``. The
  root signals are removed at the probe site and the forward pass stops there.

With several roots (linear seeding with several directions) a node's necessity
is its largest normalised drop over the roots.

Grouped-query attention is ungrouped during the removal passes (each query head
gets its own key and value copy), so a key or value removal changes one head
only; a head's output includes a post-attention norm (weight over the clean
scale) where the model has one (Gemma-2).
"""

from __future__ import annotations

import ast
import time
import types
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace

import networkx as nx
import numpy as np
import torch

from .mlp import head_reader_operator


@dataclass
class PruneOptions:
    """Frozen-necessity pruning (the default for :meth:`Tracer.trace`).

    The trace is first run *tightly* (the four trace settings below), so that
    the untraced remainder of each traced quantity is small and cannot stand in
    for a removed signal; then every model node's frozen necessity is computed
    and the nodes below ``theta`` are removed.

    Attributes:
        theta: Keep model nodes with frozen necessity ``>= theta``.
        kappa: Query/key greedy depth: each side's greedy continues until the
            firing's attention falls to ``u + kappa (thresh - u)``, ``u = 1 /
            (dest + 1)`` the uniform level (ACC++ is ``kappa = 1``). ``None``
            keeps 1.
        mlp_eps: Replaces ``MLPExpansionOptions.eps`` for the trace. ``None``
            keeps the MLP options' value.
        value_eps: Replaces ``ValueInputOptions.eps``. ``None`` keeps it.
        value_k_max: Replaces ``ValueInputOptions.k_max``. ``None`` keeps it.
        return_full: Return the full (tight) trace with every model node's
            ``necessity`` attribute instead of the pruned graph;
            :func:`prune_by_necessity` cuts it at any threshold.
        check_tol: Relative tolerance of the exactness checks (the removal
            pass with nothing removed, and the frozen score of the whole trace,
            must both reproduce the model). A failed check raises.
        baseline: What necessity is a fraction of. ``"model"`` (default): the
            model's objective ``s(M)`` (for probability-aware seeding its
            completeness ``J(M) - J(b_U)``). ``"traced"``: the traced part
            ``s(M) - s_F(empty)``, with ``s_F(empty)`` the frozen score with
            every model node of the trace frozen and every traced signal
            removed; use it when most of the clean objective comes from
            components no traced node controls (biases, embedding), e.g. a
            probe read early in the model.
    """

    theta: float = 0.01
    kappa: float | None = 0.2
    mlp_eps: float | None = 0.05
    value_eps: float | None = 0.02
    value_k_max: int | None = 40
    return_full: bool = False
    check_tol: float = 2e-3
    baseline: str = "model"

    def __post_init__(self):
        if self.baseline not in ("model", "traced"):
            raise ValueError("baseline must be 'model' or 'traced'")
        if self.kappa is not None and not 0 < self.kappa <= 1:
            raise ValueError("kappa must be in (0, 1] or None")
        if self.mlp_eps is not None and not 0 < self.mlp_eps < 1:
            raise ValueError("mlp_eps must be in (0, 1) or None")
        if self.value_eps is not None and not 0 < self.value_eps < 1:
            raise ValueError("value_eps must be in (0, 1) or None")
        if self.value_k_max is not None and self.value_k_max < 1:
            raise ValueError("value_k_max must be >= 1 or None")
        if self.check_tol <= 0:
            raise ValueError("check_tol must be > 0")


def resolve_prune(opt) -> PruneOptions | None:
    """Accept None / False (off), True (defaults), or a PruneOptions."""
    if opt is None or opt is False:
        return None
    if opt is True:
        return PruneOptions()
    if isinstance(opt, PruneOptions):
        return opt
    raise TypeError("prune must be None, a bool, or a PruneOptions instance "
                    f"(got {type(opt).__name__}).")


def tight_options(p: PruneOptions | None, mlp_opts, v_opts):
    """The MLP / value options of the trace that pruning runs on."""
    if p is None:
        return mlp_opts, v_opts
    if mlp_opts is not None and p.mlp_eps is not None:
        mlp_opts = replace(mlp_opts, eps=p.mlp_eps, eps_soft=max(mlp_opts.eps_soft, p.mlp_eps))
    if v_opts is not None:
        kw = {}
        if p.value_eps is not None:
            kw["eps"] = p.value_eps
        if p.value_k_max is not None:
            kw["k_max"] = p.value_k_max
        v_opts = replace(v_opts, **kw)
    return mlp_opts, v_opts


@contextmanager
def greedy_scope(tracer, p: PruneOptions | None):
    """Run the query/key greedy at the pruning depth for the duration of a trace."""
    old = tracer._greedy_kappa, tracer._greedy_stats
    kappa = 1.0 if p is None or p.kappa is None else float(p.kappa)
    tracer._greedy_kappa = kappa
    tracer._greedy_stats = {"kappa": kappa, "firings": 0, "terms": 0, "capped": 0} if kappa != 1.0 else None
    try:
        yield tracer._greedy_stats
    finally:
        tracer._greedy_kappa, tracer._greedy_stats = old


# ---------------------------------------------------------------------- objectives
@dataclass
class Objective:
    """What a removal pass scores (see the module docstring).

    kind: "logit_tokens" (``pairs`` = [(answer_id, wrong_id or None)] per
    root), "linear" (``directions`` per root), "prob" (``support``,
    ``q_star``) or "probe" (``direction``, ``layer``).
    """

    kind: str
    roots: list
    end: int
    pairs: list | None = None
    directions: list | None = None
    support: list | None = None
    q_star: torch.Tensor | None = None
    direction: torch.Tensor | None = None
    layer: int | None = None


def _J(z, support, q_star):
    return (q_star * torch.log_softmax(z, dim=-1)[support]).sum()


class _Scorer:
    def __init__(self, model, obj: Objective):
        self.m, self.obj = model, obj
        L = model.cfg.n_layers
        self.site = L - 1 if obj.kind != "probe" else int(obj.layer)
        dev = model.W_U.device
        if obj.kind == "prob":
            self.support = torch.as_tensor(obj.support, dtype=torch.long, device=dev)
        self.base = None

    def scores(self, x_end: torch.Tensor) -> np.ndarray:
        """Per-root objective from the residual at the site (end position)."""
        m, o = self.m, self.obj
        if o.kind == "probe":
            return np.array([float(x_end.float() @ o.direction.to(x_end.device).float())])
        normed = m.ln_final(x_end[None, None])[0, 0]
        if o.kind == "linear":
            return np.array([float(normed @ d.to(normed.device).to(normed.dtype)) for d in o.directions])
        z = normed @ m.W_U + m.b_U
        if o.kind == "prob":
            q = o.q_star.to(z.device).to(z.dtype)
            return np.array([float(_J(z, self.support, q))])
        cap = getattr(m.cfg, "output_logits_soft_cap", 0.0) or 0.0
        if cap > 0:
            z = cap * torch.tanh(z / cap)
        return np.array([float(z[a] - (z[w] if w is not None else 0.0)) for a, w in o.pairs])

    def norms(self, s_model: np.ndarray) -> np.ndarray:
        if self.obj.kind == "prob":
            m = self.m
            q = self.obj.q_star.to(m.b_U.device).to(m.b_U.dtype)
            self.base = float(_J(m.b_U, self.support, q))
            return s_model - self.base
        return s_model.copy()


# ---------------------------------------------------------------------- helpers
def _is_firing(n):
    return isinstance(n, tuple) and len(n) == 4 and isinstance(n[0], int) and isinstance(n[1], int)


def _is_mlp(n):
    return isinstance(n, tuple) and len(n) >= 4 and n[1] == "MLP"


def _is_root(n):
    return isinstance(n, tuple) and len(n) == 2


def model_key(n, l2p):
    """Model node of a graph node: ("H", layer, head, pos), ("M", layer, pos), or None."""
    if _is_firing(n):
        return ("H", n[0], n[1], l2p[n[2]])
    if _is_mlp(n):
        return ("M", n[0], l2p[n[2]])
    return None


def _orth(W, rtol=1e-6):
    W = W.double().cpu()
    if not torch.isfinite(W).all():
        raise FloatingPointError("non-finite reader operator")
    try:
        U, S, _ = torch.linalg.svd(W, full_matrices=False)
    except torch.linalg.LinAlgError:  # gesdd non-convergence: the robust gesvd driver
        import scipy.linalg
        u, sv, _ = scipy.linalg.svd(W.numpy(), full_matrices=False, lapack_driver="gesvd")
        U, S = torch.from_numpy(u), torch.from_numpy(sv)
    k = int((S > rtol * S[0]).sum()) if S[0] > 0 else 0
    return U[:, :k].float()


@contextmanager
def _ungrouped(model):
    old = getattr(model.cfg, "ungroup_grouped_query_attention", False)
    model.cfg.ungroup_grouped_query_attention = True
    try:
        yield
    finally:
        model.cfg.ungroup_grouped_query_attention = old


def _norm(model, ln, x):
    """An input norm computed directly (calling the module would re-fire its hooks)."""
    nt = model.cfg.normalization_type or ""
    if nt.startswith("LN"):
        x = x - x.mean(-1, keepdim=True)
    eps = getattr(ln, "eps", model.cfg.eps)
    x = x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    if hasattr(ln, "w"):
        x = x * ln.w
    if hasattr(ln, "b"):
        x = x + ln.b
    return x


def readers_stand_in(tracer, cache, prompt_idx, root_readers):
    """Reader source for a trace without MLP expansion or value inputs (plain ACC++)."""
    return types.SimpleNamespace(
        config=tracer.config, cache=cache, prompt_idx=prompt_idx, tr=tracer,
        center=(tracer.model.cfg.normalization_type or "").startswith("LN"),
        root_readers=root_readers, _value_ops={}, _child_ops={})


# ---------------------------------------------------------------------- the frozen pruner
class FrozenPruner:
    """Frozen traced-signal removal passes for one traced graph.

    Args:
        tracer: The :class:`Tracer` that made the trace.
        G: The traced graph.
        X: Reader-operator source: the trace's MLPExpander, or
            :func:`readers_stand_in` for plain ACC++.
        cache: The trace's activation cache; the passes re-run the model
            from its ``blocks.0.hook_resid_pre`` at ``prompt_idx`` (positions
            ``0..end``; left-padded batches are not supported).
        prompt_idx, idx_to_token: As in the trace.
        objective: What to score (:class:`Objective`).
    """

    def __init__(self, tracer, G, X, cache, prompt_idx, idx_to_token, objective: Objective):
        m = tracer.model
        self.m, self.X, self.G, self.obj = m, X, G, objective
        self.end = objective.end
        self.l2p = {v: k for k, v in idx_to_token.items()}
        self.scorer = _Scorer(m, objective)
        self.site = self.scorer.site
        self.dev = m.W_U.device
        self.post = bool(getattr(m.cfg, "use_normalization_before_and_after", False))
        self.parallel = bool(getattr(m.cfg, "parallel_attn_mlp", False))
        self.has_mlp = not m.cfg.attn_only
        try:
            x0 = cache["blocks.0.hook_resid_pre"]
        except KeyError as e:
            raise ValueError("pruning needs blocks.0.hook_resid_pre in the cache (or pass prune=False)") from e
        self.x0 = x0[prompt_idx: prompt_idx + 1, : self.end + 1].clone()
        with torch.no_grad(), _ungrouped(m):
            out, self.cache = m.run_with_cache(self.x0, start_at_layer=0, stop_at_layer=self.site + 1)
        ref = cache[f"blocks.{self.site}.hook_resid_post"][prompt_idx, self.end]
        err = ((out[0, self.end] - ref).norm() / ref.norm().clamp_min(1e-30)).item()
        if err > 1e-3:
            raise ValueError(
                f"pruning re-runs the model from the cache's layer-0 residual and does not reproduce it "
                f"(relative error {err:.2e} at the objective's site); is the batch left-padded? "
                "Pass prune=False to skip pruning.")
        self.s_model = self.scorer.scores(out[0, self.end])
        self.norm = self.scorer.norms(self.s_model)
        self.roots = list(objective.roots)
        self.mkey = {n: model_key(n, self.l2p) for n in G.nodes}
        self.ops, self.edges = [], []
        self.n_by_type, self.skipped = defaultdict(int), defaultdict(int)
        self._src, self._proj = {}, {}
        for u, v, d in G.edges(data=True):
            out_u = self.source(u)
            rd = self.reader(v, d) if out_u is not None else None
            if rd is None:
                self.skipped[f"{d.get('type')}: no {'source' if out_u is None else 'reader'}"] += 1
                continue
            key, op = rd
            self.edges.append((u, v, key, len(self.ops)))
            self.ops.append(op.float().cpu())
            self.n_by_type[key[0]] += 1
        S = self.site + 1
        self.cz = [self.cache[f"blocks.{l}.attn.hook_z"] for l in range(S)]
        self.cm = [self.cache[f"blocks.{l}.hook_mlp_out"] for l in range(S)] if self.has_mlp else None

    # ---- signals
    def _post(self, l, p, x):
        if not self.post:
            return x
        b = self.m.blocks[l]
        return x * b.ln1_post.w / self.cache[f"blocks.{l}.ln1_post.hook_scale"][0, p]

    def source(self, u):
        """Clean output of a graph node in the raw residual (cpu float), or None."""
        if u in self._src:
            return self._src[u]
        c, m, out = self.cache, self.m, None
        if _is_firing(u):
            l, h, p, s = u[0], u[1], self.l2p[u[2]], self.l2p[u[3]]
            a = c[f"blocks.{l}.attn.hook_pattern"][0, h, p, s]
            out = self._post(l, p, a * (c[f"blocks.{l}.attn.hook_v"][0, s, h] @ m.blocks[l].attn.W_O[h]))
        elif isinstance(u, tuple) and len(u) >= 4 and isinstance(u[0], int):
            l, kind, p = u[0], u[1], self.l2p.get(u[2])
            if p is None:
                pass
            elif kind == "MLP":
                out = c[f"blocks.{l}.hook_mlp_out"][0, p]
            elif kind == "AH bias":
                out = self._post(l, p, m.blocks[l].attn.b_O)
            elif kind == "Embedding":
                out = c["blocks.0.hook_resid_pre"][0, p]
        self._src[u] = None if out is None else out.float().cpu()
        return self._src[u]

    def reader(self, v, d):
        X, t = self.X, d.get("type")
        if v in self.roots:
            vec = X.root_readers.get(v)
            return None if vec is None else (("root",), vec.float().reshape(-1, 1))
        if _is_firing(v):
            l, h, dp, sp = v[0], v[1], self.l2p[v[2]], self.l2p[v[3]]
            if t == "v":
                op = X._value_ops.get(v)
                return None if op is None else (("v", (l, h, dp, sp)), op)
            if t in ("d", "s") and d.get("svs_used") is not None:
                svs = d["svs_used"]
                svs = ast.literal_eval(svs) if isinstance(svs, str) else list(svs)
                pos = dp if t == "d" else sp
                op = head_reader_operator(self.m, X.config, X.cache, X.prompt_idx, X.tr.U, X.tr.VT,
                                          X.tr.W_Q_pinv, X.tr.W_K_pinv, X.tr._rotation_stack(pos + 1),
                                          l, h, t, svs, pos, X.center)
                return ("q" if t == "d" else "k", l, h, pos), op
            return None
        if _is_mlp(v):
            op = X._child_ops.get(v)
            return None if op is None else (("mlp", v[0], self.l2p[v[2]]), op)
        return None

    def _signal(self, key, u, oids):
        ck = (key, u, oids)
        if ck not in self._proj:
            Qb = _orth(torch.cat([self.ops[i] for i in oids], dim=1))
            self._proj[ck] = (Qb @ (Qb.T @ self.source(u))).to(self.dev)
        return self._proj[ck]

    # ---- passes
    def model_nodes(self):
        return {k for k in self.mkey.values() if k is not None}

    def keep_graph(self, K):
        """Graph nodes kept for model-node set K (biases, embedding and the objective always)."""
        return {n for n in self.G.nodes if self.mkey[n] is None or self.mkey[n] in K}

    def removals(self, keep):
        grp = defaultdict(list)
        for u, v, key, oid in self.edges:
            if v in keep and u not in keep:
                grp[(key, u)].append(oid)
        R = {}
        for (key, u), oids in grp.items():
            s = self._signal(key, u, tuple(sorted(set(oids))))
            R[key] = s if key not in R else R[key] + s
        return R

    def _freeze_hooks(self, K):
        S, P, H = self.site + 1, self.x0.shape[1], self.m.cfg.n_heads
        kz = torch.zeros(S, P, H, dtype=torch.bool)
        km = torch.zeros(S, P, dtype=torch.bool)
        for k in K:
            if k[1] >= S:
                continue
            if k[0] == "H":
                kz[k[1], k[3], k[2]] = True
            else:
                km[k[1], k[2]] = True
        hooks = []
        for l in range(S):
            hooks.append((f"blocks.{l}.attn.hook_z", lambda x, hook, k=kz[l][None, :, :, None].to(self.dev),
                          c=self.cz[l]: torch.where(k, x, c)))
            if self.has_mlp:
                hooks.append((f"blocks.{l}.hook_mlp_out", lambda x, hook, k=km[l][None, :, None].to(self.dev),
                              c=self.cm[l]: torch.where(k, x, c)))
        return hooks

    def run_F(self, K) -> np.ndarray:
        """Per-root objective with model nodes outside K frozen and the pruned edges' signals removed."""
        K = set(K)
        return self.run_R(self.removals(self.keep_graph(K)), extra=self._freeze_hooks(K))

    def run_R(self, R, extra=None) -> np.ndarray:
        m = self.m
        qk, vv, mm = defaultdict(dict), defaultdict(dict), defaultdict(dict)
        root_dl = None
        for key, dl in R.items():
            if key[0] in ("q", "k"):
                qk[key[1]][(key[0], key[2], key[3])] = dl
            elif key[0] == "v":
                vv[key[1][0]][key[1]] = dl
            elif key[0] == "mlp":
                mm[key[1]][key[2]] = dl
            else:
                root_dl = dl
        live, hooks = {}, list(extra or [])

        def grab(name):
            return lambda x, hook: live.__setitem__(name, x) or x

        for l in set(qk) | set(vv) | (set(mm) if self.parallel else set()):
            hooks.append((f"blocks.{l}.hook_resid_pre", grab(("pre", l))))
        for l, items in qk.items():
            blk = m.blocks[l]
            attn = blk.attn
            for which in ("q", "k"):
                sel = [(h, p, dl) for (w, h, p), dl in items.items() if w == which]
                if not sel:
                    continue
                W, b = (attn.W_Q, attn.b_Q) if which == "q" else (attn.W_K, attn.b_K)

                def hqk(x, hook, l=l, sel=sel, W=W, b=b, ln=blk.ln1):
                    x = x.clone()
                    pre = live[("pre", l)][0]
                    for h, p, dl in sel:
                        x[0, p, h] = _norm(m, ln, pre[p] - dl) @ W[h] + b[h]
                    return x
                hooks.append((f"blocks.{l}.attn.hook_{which}", hqk))
        for l, items in vv.items():
            blk = m.blocks[l]
            attn = blk.attn
            hooks.append((f"blocks.{l}.attn.hook_pattern", grab(("pat", l))))
            hooks.append((f"blocks.{l}.attn.hook_v", grab(("v", l))))

            def hz(x, hook, l=l, items=items, attn=attn, ln=blk.ln1):
                x = x.clone()
                pre, A, V = live[("pre", l)][0], live[("pat", l)][0], live[("v", l)][0]
                for (_, h, dp, sp), dl in items.items():
                    v2 = _norm(m, ln, pre[sp] - dl) @ attn.W_V[h] + attn.b_V[h]
                    x[0, dp, h] = x[0, dp, h] + A[h, dp, sp] * (v2 - V[sp, h])
                return x
            # before the freeze hooks: a frozen head ignores it, a kept head gets it
            hooks.insert(0, (f"blocks.{l}.attn.hook_z", hz))
        for l, items in mm.items():
            src = ("pre", l) if self.parallel else ("mid", l)
            if not self.parallel:
                hooks.append((f"blocks.{l}.hook_resid_mid", grab(src)))

            def hm(x, hook, items=items, src=src, ln=m.blocks[l].ln2):
                x = x.clone()
                for p, dl in items.items():
                    x[:, p] = _norm(m, ln, live[src][:, p] - dl)
                return x
            hooks.append((f"blocks.{l}.ln2.hook_normalized", hm))
        if root_dl is not None:
            def hr(x, hook, dl=root_dl):
                x = x.clone()
                x[:, self.end] = x[:, self.end] - dl
                return x
            hooks.append((f"blocks.{self.site}.hook_resid_post", hr))
        with torch.no_grad(), _ungrouped(m):
            out = m.run_with_hooks(self.x0, start_at_layer=0, stop_at_layer=self.site + 1, fwd_hooks=hooks)
        return self.scorer.scores(out[0, self.end])

    def self_test(self) -> float:
        """Zero removals on every input type, nothing frozen: must reproduce the model."""
        z = torch.zeros(self.m.cfg.d_model, device=self.dev)
        return float(np.max(np.abs(self.run_R({key: z for _, _, key, _ in self.edges}) - self.s_model)))

    def necessity(self, full=None) -> dict:
        """Frozen necessity of every model node (largest normalised drop over the roots)."""
        full = self.model_nodes() if full is None else set(full)
        ok = self.norm > 0
        out = {}
        for n in full:
            drop = (self.s_model - self.run_F(full - {n}))[ok] / self.norm[ok]
            out[n] = float(drop.max())
        return out


# ---------------------------------------------------------------------- graph-level API
def prune_by_necessity(G: nx.MultiDiGraph, theta: float | None = None) -> nx.MultiDiGraph:
    """Cut a full trace annotated with frozen necessities (``PruneOptions(return_full=True)``).

    Keeps the graph nodes whose model node has ``necessity >= theta`` (default:
    the trace's own ``theta``), plus attention biases, embeddings and the
    objective, then drops biases and embeddings left without edges. Kept model
    nodes whose traced paths to the objective all ran through pruned nodes stay
    (``n_kept_disconnected`` counts the kept nodes without a path to it).
    ``G.graph["prune"]`` of the result records the threshold and the kept model
    nodes; its ``frozen_score`` is that of the threshold the trace was pruned at.
    """
    info = G.graph.get("prune")
    if info is None or "necessity" not in info:
        raise ValueError("G has no frozen necessities; trace with prune=PruneOptions(return_full=True)")
    theta = info["theta"] if theta is None else float(theta)
    nec = {tuple(k): v for k, v in info["necessity"]}
    kept = {k for k, v in nec.items() if v >= theta}
    keep = [n for n in G.nodes if G.nodes[n].get("model_node") is None or tuple(G.nodes[n]["model_node"]) in kept]
    H = G.subgraph(keep).copy()
    # drop leaves (biases, embedding) left without edges; kept model nodes stay even if isolated
    H.remove_nodes_from([n for n in list(H.nodes) if H.degree(n) == 0 and not _is_root(n)
                         and H.nodes[n].get("model_node") is None])
    roots = [n for n in H.nodes if _is_root(n)]
    reach = set(roots)
    for r in roots:
        reach |= nx.ancestors(H, r)
    pi = dict(info)
    pi.update(theta=theta, pruned=True, kept=sorted(kept), n_model_nodes_kept=len(kept),
              n_kept_disconnected=sum(1 for n in H.nodes if n not in reach))
    if theta != info["theta"]:
        pi["frozen_score"] = None
    H.graph["prune"] = pi
    return H


def prune_trace(tracer, G, X, cache, prompt_idx, idx_to_token, objective: Objective, p: PruneOptions,
                greedy_stats=None) -> nx.MultiDiGraph:
    """Annotate a tight trace with frozen necessities and prune it (see :class:`PruneOptions`)."""
    t0 = time.time()
    base = dict(theta=p.theta, options=p.__dict__.copy(), objective=objective.kind, qk_greedy=greedy_stats)
    if greedy_stats and greedy_stats.get("capped"):
        warnings.warn(f"tight query/key greedy: {greedy_stats['capped']} firing sides reached the term cap",
                      UserWarning)
    idx = [i for i, r in enumerate(objective.roots) if r in G]
    if G.number_of_edges() == 0 or not idx:
        G.graph["prune"] = dict(base, empty=True, necessity=[], kept=[], n_model_nodes_trace=0,
                                n_model_nodes_kept=0, pruned=not p.return_full)
        return G
    pick = (lambda xs: None if xs is None else [xs[i] for i in idx])
    objective = replace(objective, roots=pick(objective.roots), pairs=pick(objective.pairs),
                        directions=pick(objective.directions))
    SR = FrozenPruner(tracer, G, X, cache, prompt_idx, idx_to_token, objective)
    tol = p.check_tol * max(1.0, float(np.max(np.abs(SR.s_model))))
    st = SR.self_test()
    full = SR.model_nodes()
    s_full = SR.run_F(full)
    err_full = float(np.max(np.abs(s_full - SR.s_model)))
    if st > tol or err_full > tol:
        raise RuntimeError(
            f"frozen pruning: exactness check failed (removal pass with nothing removed differs by {st:.3g}, "
            f"frozen score of the whole trace by {err_full:.3g}; tolerance {tol:.3g}). The model may use an "
            "architecture the removal passes do not support; pass prune=False.")
    s_empty = SR.run_F(set())
    if p.baseline == "traced":
        SR.norm = SR.s_model - s_empty
        SR.scorer.base = None
    if not (SR.norm > 0).any():
        warnings.warn("frozen pruning skipped: the objective is not positive for any root", UserWarning)
        G.graph["prune"] = dict(base, empty=True, skipped="objective not positive", necessity=[], kept=[],
                                pruned=False)
        return G
    nec = SR.necessity(full)
    kept = {n for n, v in nec.items() if v >= p.theta}
    s_kept = SR.run_F(kept)
    ok = SR.norm > 0
    for n in G.nodes:
        k = SR.mkey[n]
        if k is not None:
            G.nodes[n]["model_node"] = k
            G.nodes[n]["necessity"] = nec[k]
    G.graph["prune"] = dict(
        base, empty=False, pruned=False,
        score_model=SR.s_model.tolist(), norm=SR.norm.tolist(), objective_base=SR.scorer.base,
        frozen_score=(1.0 - (SR.s_model[ok] - s_kept[ok]) / SR.norm[ok]).tolist(),
        score_empty=s_empty.tolist(), baseline=p.baseline,
        necessity=sorted(([list(k), v] for k, v in nec.items()), key=lambda x: -x[1]),
        n_model_nodes_trace=len(full), n_model_nodes_kept=len(kept),
        checks=dict(no_removal=st, whole_trace=err_full),
        edges_by_reader=dict(SR.n_by_type), edges_skipped=dict(SR.skipped),
        seconds=time.time() - t0)
    if p.return_full:
        return G
    return prune_by_necessity(G, p.theta)
