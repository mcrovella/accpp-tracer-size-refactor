"""Expanding MLP nodes of a traced ACC++ graph (report Sections 7-8).

Two modes (``MLPExpansionOptions.mode``):

``"per_reader"`` (default)
    An MLP's signal depends on who reads it: a head reads it through its
    selected singular vectors, another MLP through that MLP's attribution
    operator. Every reader edge therefore gets its own expansion, against that
    reader alone. Each expansion is a separate graph node, an *MLP occurrence*
    ``(layer, "MLP", tok, tok, reader_path)``, where ``reader_path`` is a tuple
    of steps from the reader at the top of the MLP chain down to the immediate
    reader:

    - ``("d" | "s", head_layer, head, dest_label, src_label)``: an attention head,
    - ``("root", *root_node)``: the seeding objective (or probe) root,
    - ``("mlp", layer)``: the downstream MLP occurrence that selected it.

    Each occurrence has exactly one outgoing edge. Shares are reported both
    locally and relative to the signal of the occurrence at the top of the
    chain (``*_root``), so they compose along the chain.

``"merged"``
    Each MLP node ``(layer, "MLP", tok, tok)`` is expanded once, against all of
    its readers together: the reader operators are stacked, so the objective
    is the sum over readers of each reader's squared signal. Nodes are
    processed in decreasing layer order, after the head recursion, so all
    readers of an MLP are known when it is expanded. This keeps ACC++'s node
    identity (one node per firing), but each selected input is credited to the
    MLP as a whole, not to a particular reader.

New edges have ``type="m"`` ("MLP input"). Selected head-source pairs are
handed back to the ordinary ACC++ head recursion.
"""

from __future__ import annotations

import ast
import heapq
import warnings
from collections import OrderedDict, deque
from dataclasses import dataclass, field

import networkx as nx
import torch
from torch import Tensor

from .mlp import (
    MLPExpansionOptions,
    MLPSite,
    MLPTrace,
    child_operator,
    get_mlp_spec,
    head_reader_operator,
    mlp_input_components,
    run_algorithm1,
)

M_EDGE_COLOR = "#4DAF4A"
_SITE_CACHE = 32        # MLP sites kept in memory (LRU); contexts keep their own references


def _decision_kind(decision: str) -> tuple[str, str | None]:
    """Stable (kind, reason) for a decision string."""
    if decision in ("expand", "approximate", "pruned"):
        return decision, None
    reasons = {"leaf (MLP 0)": "mlp0", "leaf (bias overshoot)": "bias_overshoot", "leaf (BOS)": "bos",
               "leaf (no input-dependent signal)": "no_signal", "leaf (no reader operator)": "no_reader",
               "leaf (node cap)": "node_cap"}
    if decision in reasons:
        return "leaf", reasons[decision]
    if decision.startswith("leaf (R_sig"):
        return "leaf", "R_sig"
    return "leaf", "other"


def is_mlp_node(n) -> bool:
    """True for MLP nodes: 4-tuples ``(layer, "MLP", tok, tok)`` and 5-tuple occurrences."""
    return isinstance(n, tuple) and len(n) in (4, 5) and n[1] == "MLP" and isinstance(n[0], int)


def mlp_base_node(n):
    """The 4-tuple MLP node of an occurrence (identity for 4-tuples)."""
    return n[:4] if is_mlp_node(n) else n


@dataclass
class _SiteRec:
    ids: list
    O: Tensor
    z: Tensor
    site: MLPSite
    ln2_scale: Tensor
    pos: int


@dataclass
class _OccCtx:
    """Per-reader context of one MLP occurrence."""
    path: tuple
    WT: Tensor                       # reader operator
    root_WT: Tensor                  # operator of the occurrence at the top of the chain
    root_t: Tensor | None            # traced target of the top occurrence (None at the top)
    path_sites: list = field(default_factory=list)   # _SiteRec of downstream MLPs, top first
    parent_Q: Tensor | None = None


class MLPExpander:
    """Expands the MLP nodes of a graph in place. See the module docstring."""

    def __init__(self, tracer, G: nx.MultiDiGraph, is_traced: dict, cache, idx_to_token: dict,
                 prompt_idx: int, attn_weight_thresh, signals, root_readers: dict,
                 opts: MLPExpansionOptions):
        self.tr, self.G, self.is_traced, self.cache = tracer, G, is_traced, cache
        self.model, self.config = tracer.model, tracer.config
        self.idx_to_token, self.prompt_idx = idx_to_token, prompt_idx
        self.thresh, self.signals, self.root_readers, self.opts = attn_weight_thresh, signals, root_readers, opts
        self.spec = get_mlp_spec(self.model)
        self.H = self.model.cfg.n_heads
        labels = list(idx_to_token.values())
        if len(set(labels)) != len(labels):
            raise ValueError("MLP expansion needs unique token labels in idx_to_token "
                             "(Tracer.trace builds them with ' (n)' suffixes).")
        self.token_to_idx = {v: k for k, v in idx_to_token.items()}
        self._sites: OrderedDict = OrderedDict()
        self.n_traced = 0
        self._cap_warned = False

    # ------------------------------------------------------------------ helpers
    def site(self, layer: int, pos: int) -> _SiteRec:
        key = (layer, pos)
        if key in self._sites:
            self._sites.move_to_end(key)
            return self._sites[key]
        ids, O, z = mlp_input_components(self.model, self.config, self.spec, self.cache,
                                         self.prompt_idx, layer, pos)
        site = MLPSite(self.model, self.spec, layer, z, self.opts.quad_points)
        s = self.cache[f"blocks.{layer}.ln2.hook_scale"][self.prompt_idx, pos].float()
        self._sites[key] = _SiteRec(ids, O, z, site, s, pos)
        if len(self._sites) > _SITE_CACHE:
            self._sites.popitem(last=False)
        return self._sites[key]

    def _center(self, v: Tensor) -> Tensor:
        return v - v.mean(dim=0, keepdim=True) if self.spec.center else v

    def _reader(self, src_node, X, data):
        """(step, WT) for the reader edge src_node -> X, or (step, None) if no operator."""
        pos = self.token_to_idx[src_node[2]]
        is_head = (isinstance(X, tuple) and len(X) == 4 and isinstance(X[1], int)
                   and X not in self.root_readers and data.get("svs_used") is not None
                   and data.get("type") in ("d", "s"))
        if is_head:
            et = data.get("type")
            svs = data.get("svs_used")
            svs = ast.literal_eval(svs) if isinstance(svs, str) else list(svs)
            step = (et, X[0], X[1], X[2], X[3])
            WT = head_reader_operator(
                self.model, self.config, self.cache, self.prompt_idx, self.tr.U, self.tr.VT,
                self.tr.W_Q_pinv, self.tr.W_K_pinv, self.tr._rotation_stack(pos + 1),
                X[0], X[1], et, svs, pos, self.spec.center)
            return step, WT
        step = ("root",) + tuple(X)
        vec = self.root_readers.get(X)
        if vec is None or not self.opts.expand_seed_mlps:
            return step, None
        return step, vec.float().reshape(-1, 1)

    def _reader_step(self, X, data):
        if (isinstance(X, tuple) and len(X) == 4 and isinstance(X[1], int) and X not in self.root_readers
                and data.get("svs_used") is not None and data.get("type") in ("d", "s")):
            return (data["type"], X[0], X[1], X[2], X[3])
        return ("root",) + tuple(X)

    def _decide(self, tr: MLPTrace) -> str:
        o = self.opts
        if tr.t.norm().item() < 1e-12:
            return "leaf (no input-dependent signal)"
        if o.bias_overshoot_leaf and tr.R_ideal > 1:
            return "leaf (bias overshoot)"
        if tr.R_sig > o.eps_soft:
            return f"leaf (R_sig {tr.R_sig:.3f})"
        return "expand" if tr.R_sig <= o.eps else "approximate"

    def _record(self, node, tr: MLPTrace | None, decision: str, readers: list, extra: dict | None = None):
        kind, reason = _decision_kind(decision)
        rec = dict(decision=decision, decision_kind=kind, leaf_reason=reason, mode=self.opts.mode,
                   readers=[repr(r) for r in readers])
        if tr is not None:
            tn2 = max(tr.t.norm().item() ** 2, 1e-30)
            rec.update(R_sig=tr.R_sig, R_ideal=tr.R_ideal, n_selected=len(tr.selected),
                       n_candidates=tr.n_candidates, completeness_err=tr.completeness_err,
                       t_norm=tr.t.norm().item(), bias_share=(tr.bias @ tr.t).item() / tn2)
        if extra:
            rec.update(extra)
        self.G.add_node(node)
        self.G.nodes[node]["mlp_trace"] = rec

    def _tick(self) -> bool:
        """Count one trace; False (and warn once) if the node cap is reached."""
        if self.n_traced >= self.opts.max_mlp_nodes:
            if not self._cap_warned:
                warnings.warn(f"MLP expansion reached max_mlp_nodes={self.opts.max_mlp_nodes}; "
                              f"remaining MLPs are kept as leaves.", UserWarning)
                self._cap_warned = True
            return False
        self.n_traced += 1
        return True

    def _add_input_edge(self, u, v, weight: float, rank: int, extra: dict, sig: Tensor | None):
        self.G.add_edge(u, v, weight=float(weight), type="m", color=M_EDGE_COLOR, rank=rank, **extra)
        if self.signals is not None and sig is not None:
            key = self.G.number_of_edges(u, v) - 1
            self.G.edges[u, v, key]["signal"] = sig.detach().cpu()
            self.G.edges[u, v, key]["signal_flavor"] = "mlp_row_space"

    def _component_node(self, cid, pos):
        """Graph node of a selected component, or None if its tokens are not in idx_to_token
        (mirrors _trace_recursive, which skips unmapped positions)."""
        l, c, src = cid
        tok = self.idx_to_token
        if pos not in tok or src not in tok:
            return None
        if c < self.H:
            return (l, c, tok[pos], tok[src])
        if c == self.H + 1:
            return (l, "AH bias", tok[pos], tok[pos])
        if c == self.H + 2:
            return (0, "Embedding", tok[pos], tok[pos])
        return (l, "MLP", tok[pos], tok[pos])

    def _trace_head(self, u, cid, pos):
        l, h, src = cid
        if "attn_weight" not in self.G.nodes[u]:
            self.G.nodes[u]["attn_weight"] = self.cache[f"blocks.{l}.attn.hook_pattern"][
                self.prompt_idx, h, pos, src].item()
        if (l, h, pos, src) not in self.is_traced:
            self.tr._trace_recursive(self.cache, self.idx_to_token, self.G, self.is_traced,
                                     self.prompt_idx, l, h, pos, src, self.thresh, self.signals)

    def _head_input_shares(self, head_inputs, tr: MLPTrace, rec: _SiteRec, ref_t: Tensor, r2: float,
                           removed) -> list:
        """Shares of selected head inputs, for the pruning rule (only if prune_head_inputs)."""
        if not self.opts.prune_head_inputs:
            return []
        out = []
        for h in head_inputs:
            sig = tr.Q @ (tr.Q.T @ rec.O[h["i"]])
            h.update(lin=float(tr.gamma[h["i"]] @ ref_t / r2), real=float(removed(sig) @ ref_t / r2), sig=sig)
            out.append(h)
        return out

    def _finish_head_input(self, h, reader_node, pos):
        """Record the pruning outcome on the head input's edge and recurse unless it is a leaf."""
        u, cid = h["node"], h["cid"]
        status = h.get("status")
        edge = self.G.edges[u, reader_node, max(self.G[u][reader_node])]
        if status is not None:
            edge.update(child_decision=status, share_linear=h["lin"], share_real=h["real"])
        if "attn_weight" not in self.G.nodes[u]:
            l, hd, src = cid
            self.G.nodes[u]["attn_weight"] = self.cache[f"blocks.{l}.attn.hook_pattern"][
                self.prompt_idx, hd, pos, src].item()
        if status == "pruned":
            return
        if self.opts.bos_leaf and cid[2] == 0:
            edge["child_decision"] = "leaf (BOS)"
            return
        self._trace_head(u, cid, pos)

    def _removed_real(self, rec: _SiteRec, sig: Tensor, path_sites: list, out_WT: Tensor) -> Tensor:
        """Signal removed (in out_WT coords) when sig leaves rec's input; real MLPs along path_sites."""
        dF = rec.site.forward(rec.z - sig) - rec.site.Fz
        for ps in reversed(path_sites):
            dF = ps.site.forward(ps.z + self._center(dF.unsqueeze(1))[:, 0] / ps.ln2_scale) - ps.site.Fz
        return -(out_WT.T @ dF)

    def _prune(self, kids: list, tau: float, joint) -> None:
        """Both-below-tau pruning plus the joint-removal check. kids: dicts with lin, real, sig."""
        small = []
        for k in kids:
            big_l, big_r = abs(k["lin"]) >= tau, abs(k["real"]) >= tau
            k["status"] = ("expanded (both)" if big_l and big_r else "expanded (linear)" if big_l
                           else "expanded (real)" if big_r else "pruned")
            if k["status"] == "pruned":
                small.append(k)
        if not small or not self.opts.joint_check:
            return
        small.sort(key=lambda k: -max(abs(k["lin"]), abs(k["real"])))
        while small:
            j = joint(sum(k["sig"] for k in small))
            if abs(j) < tau:
                break
            small.pop(0)["status"] = "expanded (joint check)"

    # ------------------------------------------------------------------ entry
    def run(self) -> None:
        if self.opts.mode == "per_reader":
            self._run_per_reader()
        else:
            self._run_merged()

    # ------------------------------------------------------------------ per-reader
    def _convert_pending(self, queue: deque) -> None:
        G = self.G
        pending = sorted((n for n in G.nodes if is_mlp_node(n) and len(n) == 4),
                         key=lambda n: (-n[0], repr(n)))
        for N in pending:
            for _, X, key, data in sorted(G.out_edges(N, keys=True, data=True), key=lambda e: repr(e[1])):
                leaf = ("leaf (MLP 0)" if N[0] == 0 and not self.opts.expand_mlp0
                        else "leaf (BOS)" if self.opts.bos_leaf and self.token_to_idx[N[2]] == 0 else None)
                if leaf:                                         # no reader operator needed
                    step = self._reader_step(X, data)
                    occ = N + ((step,),)
                    G.add_edge(occ, X, **dict(data))
                    G.remove_edge(N, X, key)
                    self._record(occ, None, leaf, [step], dict(path=repr((step,))))
                    continue
                step, WT = self._reader(N, X, data)
                occ = N + ((step,),)
                G.add_edge(occ, X, **dict(data))
                G.remove_edge(N, X, key)
                if WT is None:
                    self._record(occ, None, "leaf (no reader operator)", [step], dict(path=repr((step,))))
                else:
                    queue.append((occ, _OccCtx(path=(step,), WT=WT, root_WT=WT, root_t=None)))
            if G.degree(N) == 0:
                G.remove_node(N)

    def _run_per_reader(self) -> None:
        queue: deque = deque()
        self._convert_pending(queue)
        while queue:
            occ, ctx = queue.popleft()
            self._expand_occurrence(occ, ctx, queue)
            self._convert_pending(queue)

    def _expand_occurrence(self, occ, ctx: _OccCtx, queue: deque) -> None:
        o, H = self.opts, self.H
        layer, pos = occ[0], self.token_to_idx[occ[2]]
        meta = dict(path=repr(ctx.path))
        if layer == 0 and not o.expand_mlp0:
            self._record(occ, None, "leaf (MLP 0)", [ctx.path[-1]], meta)
            return
        if o.bos_leaf and pos == 0:
            self._record(occ, None, "leaf (BOS)", [ctx.path[-1]], meta)
            return
        if not self._tick():
            self._record(occ, None, "leaf (node cap)", [ctx.path[-1]], meta)
            return
        rec = self.site(layer, pos)
        tr = run_algorithm1(rec.site, rec.O, rec.ids, ctx.WT, o.eps, o.k_max, pos)
        decision = self._decide(tr)
        root_t = tr.t if ctx.root_t is None else ctx.root_t
        r2 = max(root_t.norm().item() ** 2, 1e-30)
        meta.update(t_share_root=(tr.t @ root_t).item() / r2, bias_share_root=(tr.bias @ root_t).item() / r2)
        if o.diagnostics and ctx.path_sites:
            meta.update(self._multi_hop(rec, tr, ctx))
        self._record(occ, tr, decision, [ctx.path[-1]], meta)
        if not (decision == "expand" or decision == "approximate"):
            return
        kids, n_unmapped, head_inputs = [], 0, []
        for rank, i in enumerate(tr.selected):
            cid = rec.ids[i]
            u = self._component_node(cid, pos)
            if u is None:
                n_unmapped += 1
                continue
            sig_i = tr.Q @ (tr.Q.T @ rec.O[i]) if self.signals is not None else None
            extra = dict(share_root=(tr.gamma[i] @ root_t).item() / r2)
            if cid[1] == H:            # upstream MLP: an occurrence of its own
                child = (cid[0], "MLP", self.idx_to_token[pos], self.idx_to_token[pos],
                         ctx.path + (("mlp", layer),))
                c_site = self.site(cid[0], pos).site
                d = self._center((c_site.Fz - c_site.F0).unsqueeze(1))[:, 0] / rec.ln2_scale
                sig = tr.Q @ (tr.Q.T @ d)
                lin = (tr.AT.T @ d) @ root_t / r2
                real = self._removed_real(rec, sig, ctx.path_sites, ctx.root_WT) @ root_t / r2
                kids.append(dict(node=child, layer=cid[0], lin=float(lin), real=float(real), sig=sig))
                u = child
                sig_i = sig      # the signal handed to the child: its traced output's row-space part
                extra.update(traced_share_linear=float(lin), traced_share_real=float(real))
            self._add_input_edge(u, occ, tr.shares[i].item(), rank, extra, sig_i)
            if cid[1] < H:
                head_inputs.append(dict(node=u, cid=cid, i=i))
        if n_unmapped:
            self.G.nodes[occ]["mlp_trace"]["n_unmapped"] = n_unmapped
        joint = (lambda s: float(self._removed_real(rec, s, ctx.path_sites, ctx.root_WT) @ root_t / r2))
        heads = self._head_input_shares(head_inputs, tr, rec, root_t, r2,
                                        lambda s: self._removed_real(rec, s, ctx.path_sites, ctx.root_WT))
        self._prune(kids + heads, o.tau, joint)
        for h in head_inputs:      # updated in place by _head_input_shares / _prune
            self._finish_head_input(h, occ, pos)
        for k in kids:
            self.G.edges[k["node"], occ, 0]["child_decision"] = k["status"]
            if k["status"] == "pruned":
                self._record(k["node"], None, "pruned", [("mlp", layer)],
                             dict(path=repr(k["node"][4]), traced_share_linear=k["lin"],
                                  traced_share_real=k["real"]))
                continue
            queue.append((k["node"], _OccCtx(
                path=k["node"][4], WT=child_operator(tr, rec.ln2_scale, self.spec.center),
                root_WT=ctx.root_WT, root_t=root_t, path_sites=ctx.path_sites + [rec], parent_Q=tr.Q)))

    def _multi_hop(self, rec: _SiteRec, tr: MLPTrace, ctx: _OccCtx) -> dict:
        """R_chain and R_floor (report Section 8.2) for an occurrence below the top."""
        parent = ctx.path_sites[-1]
        d = self._center((rec.site.Fz - rec.site.F0).unsqueeze(1))[:, 0] / parent.ln2_scale
        sig = ctx.parent_Q @ (ctx.parent_Q.T @ d)
        T = self._removed_real(parent, sig, ctx.path_sites[:-1], ctx.root_WT)
        W = self._removed_real(rec, tr.edit(rec.O, tr.selected), ctx.path_sites, ctx.root_WT)
        return dict(R_chain=((T - W).norm() / T.norm().clamp_min(1e-30)).item(),
                    R_floor=((tr.t - T).norm() / tr.t.norm().clamp_min(1e-30)).item())

    # ------------------------------------------------------------------ merged
    def _run_merged(self) -> None:
        G, o, H = self.G, self.opts, self.H
        done: dict = {}            # node -> child operator (WT) for its upstream MLPs
        keep: set = set()          # MLP nodes kept by at least one MLP parent
        heap: list = []
        seen: set = set()

        def push_new():
            for n in G.nodes:
                if is_mlp_node(n) and len(n) == 4 and n not in seen:
                    seen.add(n)
                    heapq.heappush(heap, (-n[0], self.token_to_idx[n[2]], repr(n), n))

        push_new()
        while heap:
            _, pos, _, N = heapq.heappop(heap)
            layer = N[0]
            readers, ops, mlp_only = [], [], True
            for _, X, data in sorted(G.out_edges(N, data=True), key=lambda e: repr(e[1])):
                if data.get("type") == "m":
                    if X not in done:
                        raise RuntimeError(f"merged MLP expansion: reader {X} of {N} not expanded first")
                    if data.get("child_decision") == "pruned":   # that parent found it negligible
                        readers.append(("mlp", X[0], "pruned"))
                        continue
                    readers.append(("mlp", X[0]))
                    ops.append(done[X])
                    continue
                mlp_only = False
                step, WT = self._reader(N, X, data)
                readers.append(step)
                if WT is not None:
                    ops.append(WT)
            if layer == 0 and not o.expand_mlp0:
                self._record(N, None, "leaf (MLP 0)", readers)
                continue
            if o.bos_leaf and pos == 0:
                self._record(N, None, "leaf (BOS)", readers)
                continue
            if mlp_only and N not in keep:
                self._record(N, None, "pruned", readers)
                continue
            if not ops:
                self._record(N, None, "leaf (no reader operator)", readers)
                continue
            if not self._tick():
                self._record(N, None, "leaf (node cap)", readers)
                continue
            rec = self.site(layer, pos)
            WT = torch.cat([op.float() for op in ops], dim=1)
            tr = run_algorithm1(rec.site, rec.O, rec.ids, WT, o.eps, o.k_max, pos)
            decision = self._decide(tr)
            self._record(N, tr, decision, readers, dict(n_reader_rows=WT.shape[1]))
            done[N] = child_operator(tr, rec.ln2_scale, self.spec.center)
            if not (decision == "expand" or decision == "approximate"):
                continue
            t2 = max(tr.t.norm().item() ** 2, 1e-30)
            kids, n_unmapped, head_inputs = [], 0, []
            for rank, i in enumerate(tr.selected):
                cid = rec.ids[i]
                u = self._component_node(cid, pos)
                if u is None:
                    n_unmapped += 1
                    continue
                sig_i = tr.Q @ (tr.Q.T @ rec.O[i]) if self.signals is not None else None
                extra = {}
                if cid[1] == H:
                    c_site = self.site(cid[0], pos).site
                    d = self._center((c_site.Fz - c_site.F0).unsqueeze(1))[:, 0] / rec.ln2_scale
                    sig = tr.Q @ (tr.Q.T @ d)
                    lin = (tr.AT.T @ d) @ tr.t / t2
                    real = self._removed_real(rec, sig, [], WT) @ tr.t / t2
                    kids.append(dict(node=u, lin=float(lin), real=float(real), sig=sig))
                    sig_i = sig
                    # Merged shares are local: relative to this node's own stacked target, with
                    # the "real" share measured through this MLP only.
                    extra.update(local_share_linear=float(lin), local_share_real=float(real))
                self._add_input_edge(u, N, tr.shares[i].item(), rank, extra, sig_i)
                if cid[1] < H:
                    head_inputs.append(dict(node=u, cid=cid, i=i))
            if n_unmapped:
                G.nodes[N]["mlp_trace"]["n_unmapped"] = n_unmapped
            joint = (lambda s: float(self._removed_real(rec, s, [], WT) @ tr.t / t2))
            heads = self._head_input_shares(head_inputs, tr, rec, tr.t, t2,
                                            lambda s: self._removed_real(rec, s, [], WT))
            self._prune(kids + heads, o.tau, joint)
            for h in head_inputs:
                self._finish_head_input(h, N, pos)
            for k in kids:
                key = max(kk for kk in G[k["node"]][N])
                G.edges[k["node"], N, key]["child_decision"] = k["status"]
                if k["status"] != "pruned":
                    keep.add(k["node"])
            push_new()
        # Sanity: no MLP node may have gained a reader after it was expanded.
        for N in done:
            for _, X, data in G.out_edges(N, data=True):
                if data.get("type") == "m" and X not in done:
                    raise RuntimeError(f"merged MLP expansion ordering violated at {N}")


def root_reader_linear(direction: Tensor, ln_final_scale: Tensor, center: bool) -> Tensor:
    """Reader vector for an MLP read by a linear (logit-direction) root."""
    v = direction.float() / direction.float().norm().clamp_min(1e-30) / ln_final_scale.float()
    return v - v.mean() if center else v


def root_reader_prob(model, cache, prompt_idx, target_tokens, end_token_pos, q_star, ig_steps,
                     center: bool) -> Tensor:
    """Reader vector for an MLP read by the probability-aware root.

    The IG attribution of a residual share ``r_c`` is exactly linear,
    ``A_c = r_c . g_bar``, with ``g_bar`` the trapezoid-averaged d_model gradient
    used by ``_ig_attribution_seeds``; the reader is ``g_bar / ||g_bar|| / s_final``.
    """
    W_U, b_U = model.W_U.detach().float(), model.b_U.detach().float()
    support = torch.as_tensor(target_tokens, dtype=torch.long, device=W_U.device)
    z_clean = cache["ln_final.hook_normalized"][prompt_idx, end_token_pos].float() @ W_U
    if q_star is None:
        p = torch.softmax(z_clean + b_U, dim=-1)[support]
        q_star = p / p.sum()
    q_star = q_star.to(W_U.device).float()
    alphas = torch.linspace(0.0, 1.0, ig_steps + 1, device=W_U.device)
    p_path = torch.softmax(b_U.unsqueeze(0) + alphas.unsqueeze(1) * z_clean.unsqueeze(0), dim=-1)
    grad_d = (W_U[:, support] @ q_star).unsqueeze(0) - p_path @ W_U.T
    w = torch.ones(ig_steps + 1, device=W_U.device) / ig_steps
    w[0] = w[-1] = 0.5 / ig_steps
    g_bar = (grad_d * w[:, None]).sum(0)
    return root_reader_linear(g_bar, cache["ln_final.hook_scale"][prompt_idx, end_token_pos], center)


def root_reader_probe(probe_direction: Tensor) -> Tensor:
    """Reader vector for an MLP read by a linear probe on the raw residual stream.

    Not centered: the probe reads the raw residual, and orientation / centering
    of ``probe_direction`` are the caller's responsibility (as in trace_from_probe).
    """
    return probe_direction.float() / probe_direction.float().norm().clamp_min(1e-30)
