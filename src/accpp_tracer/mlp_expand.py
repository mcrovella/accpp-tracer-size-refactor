"""Expanding MLP nodes of a traced ACC++ graph (report Sections 7-8).

Terminology. A *reader operator* ``W`` (``r x D``; the code stores ``WT = W^T``)
maps a residual vector written at an MLP's position to the ``r`` coordinates
in which its reader measures signal. For a head these are ACC++'s selected
singular-vector coordinates (RoPE pulled back, divided by ln1's scale); for the
root objective, the rank-1 logit / IG-gradient / probe direction. An MLP
selected by a downstream MLP ``P`` is read through ``W_P J_bar_P C / s_P``
(``P``'s path-average Jacobian, the centering projector for LayerNorm models,
and ``P``'s ln2 scale): ``child_operator``. Algorithm 1 at an MLP traces the
target ``t = W (F(z) - F(0))``.

Three modes (``MLPExpansionOptions.mode``):

``"path_summed"`` (default)
    One trace per (first-hop reader, MLP node). A *first-hop reader* is one
    side of an attention-head firing (``("d" | "s", layer, head, dest, src)``,
    the query or key side) or the root objective. All MLPs reached from it lie
    at one position, and MLP->MLP edges go from lower to higher layers, so
    they form a DAG. The MLPs are processed in decreasing layer order; each is
    traced once with the *sum* of the operators of all kept edges into it,
    plus the reader's own operator if the reader reads it directly:

        W_M = [W_R if R reads M directly] + sum_{kept parents P} W_P J_bar_P C / s_P

    By linearity this equals the sum over all paths of the per-reader path
    operators, so ``t_M`` is the MLP's total linearized influence on the
    reader's signal along the expanded paths. Nodes are
    ``(layer, "MLP", tok, tok, (reader_step,))``. A node has one outgoing edge
    per kept or pruned parent (``type="m"``) plus the original edge to the
    reader if it is read directly.

    Coordinates. ACC++ chooses singular vectors per upstream edge, so the MLPs
    a head reads directly may use different subsets. The reader's coordinates
    are the union of those subsets; each direct MLP's operator is zero outside
    its own subset (the operator's columns are independent, so this embedding
    is exact).

    Shares are relative to the reader's direct MLP signal
    ``T_R = sum_{direct M} W_R (F_M(z_M) - F_M(0))`` (``*_root`` fields), so a
    child's share is comparable across the DAG. Real-chain shares remove a
    signal from an MLP's input and run the real MLPs on every kept path above
    it (each MLP receives the summed output change of its kept children).

``"per_reader"``
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
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable
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
from .value import (
    ValueInputOptions,
    compress,
    greedy_select,
    head_input_components,
    raw_reader,
    value_operator,
)

M_EDGE_COLOR = "#4DAF4A"
V_EDGE_COLOR = "#984EA3"
_HEAD_INPUT_CACHE = 16  # decomposed head inputs kept in memory (LRU)
_SITE_CACHE = 32        # MLP sites kept in memory (LRU); contexts keep their own references


def _decision_kind(decision: str) -> tuple[str, str | None]:
    """Stable (kind, reason) for a decision string."""
    if decision in ("expand", "approximate", "pruned"):
        return decision, None
    reasons = {"leaf (MLP 0)": "mlp0", "leaf (bias overshoot)": "bias_overshoot", "leaf (BOS)": "bos",
               "leaf (no input-dependent signal)": "no_signal", "leaf (no reader operator)": "no_reader",
               "leaf (node cap)": "node_cap", "leaf (flow)": "flow",
               "leaf (value input, not traced)": "value_not_traced"}
    if decision in reasons:
        return "leaf", reasons[decision]
    if decision.startswith("leaf (R_sig"):
        return "leaf", "R_sig"
    return "leaf", "other"


def _flow_stage(n):
    """(layer, "MLP" | "attn") of a node for min_flow_scope="layer"; None for the objective."""
    if not (isinstance(n, tuple) and len(n) >= 4 and isinstance(n[0], int)):
        return None
    return (n[0], "MLP" if n[1] == "MLP" else "attn")


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


@dataclass
class _Group:
    """Path-summed state of one first-hop reader: the DAG of MLPs at its position (keys: layer)."""
    step: tuple
    pos: int
    tok: str
    direct: dict = field(default_factory=dict)       # layer -> reader operator WT (D x r), direct reads
    WT: dict = field(default_factory=dict)           # layer -> summed operator of the MLP
    parents: dict = field(default_factory=dict)      # layer -> [parent layers that kept it]
    kids: dict = field(default_factory=dict)         # layer -> [kept child layers]
    n_paths: dict = field(default_factory=dict)      # layer -> number of reader paths summed
    recs: dict = field(default_factory=dict)         # layer -> _SiteRec
    Q: dict = field(default_factory=dict)            # layer -> row-space basis of its attribution operator
    pruned_by: dict = field(default_factory=dict)    # layer -> [(parent, lin, real)] for pruning parents
    heap: list = field(default_factory=list)
    queued: set = field(default_factory=set)
    T: Tensor | None = None                          # reference: the reader's direct MLP signal
    r2: float = 1.0
    coords: list | None = None                       # the reader's singular vectors (heads); None for roots
    sink: Callable | None = None                     # flow mode: push into the global schedule instead
    pending: int = 0                                 # flow mode: members still in the global schedule

    def node(self, layer: int) -> tuple:
        return (layer, "MLP", self.tok, self.tok, (self.step,))

    def push(self, layer: int) -> None:
        if layer not in self.queued:
            self.queued.add(layer)
            if self.sink is not None:
                self.pending += 1
                self.sink(self, layer)
            else:
                heapq.heappush(self.heap, -layer)


class MLPExpander:
    """Expands the MLP nodes of a graph in place. See the module docstring."""

    def __init__(self, tracer, G: nx.MultiDiGraph, is_traced: dict, cache, idx_to_token: dict,
                 prompt_idx: int, attn_weight_thresh, signals, root_readers: dict,
                 opts: MLPExpansionOptions | None, deferred: list | None = None,
                 value_opts: ValueInputOptions | None = None):
        self.tr, self.G, self.is_traced, self.cache = tracer, G, is_traced, cache
        self.model, self.config = tracer.model, tracer.config
        self.idx_to_token, self.prompt_idx = idx_to_token, prompt_idx
        self.thresh, self.signals, self.root_readers = attn_weight_thresh, signals, root_readers
        # opts None: MLPs stay leaves (value inputs only); the defaults supply the shared fields.
        self.expand_mlps = opts is not None
        self.opts = opts if opts is not None else MLPExpansionOptions()
        self.vopts = value_opts
        if self.vopts is not None and self.expand_mlps and self.opts.mode != "path_summed":
            raise ValueError("value_inputs with MLP expansion needs mode='path_summed'")
        self.spec = get_mlp_spec(self.model) if self.expand_mlps else None
        self.center = (self.model.cfg.normalization_type or "").startswith("LN")
        self.H = self.model.cfg.n_heads
        labels = list(idx_to_token.values())
        if len(set(labels)) != len(labels):
            raise ValueError("MLP expansion needs unique token labels in idx_to_token "
                             "(Tracer.trace builds them with ' (n)' suffixes).")
        self.token_to_idx = {v: k for k, v in idx_to_token.items()}
        self._sites: OrderedDict = OrderedDict()
        self.n_traced = 0
        self._cap_warned = False
        self.deferred = deferred
        self._phi = self.opts.min_flow if self.expand_mlps else None
        self._flow_mode = self._phi is not None or self.vopts is not None
        if self._flow_mode and deferred is None:
            raise ValueError("min_flow / value_inputs: the tracer did not defer head recursion")
        self._child_ops: dict = {}      # expanded MLP node -> reader operator of its inputs (raw residual)
        self._value_ops: dict = {}      # value-traced firing -> reader operator of its value inputs
        self._head_inputs: OrderedDict = OrderedDict()
        self.n_value_traced = 0
        self._vcap_warned = False

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
        return v - v.mean(dim=0, keepdim=True) if self.center else v

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
                X[0], X[1], et, svs, pos, self.center)
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
        if self._flow_mode:
            self._push_head((l, h, pos, src))
        elif (l, h, pos, src) not in self.is_traced:
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
        if self._flow_mode:
            self._run_flow()
        elif self.opts.mode == "path_summed":
            self._run_path_summed()
        elif self.opts.mode == "per_reader":
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
                path=k["node"][4], WT=child_operator(tr, rec.ln2_scale, self.center),
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

    # ------------------------------------------------------------------ path-summed
    def _head_step_svs(self, X, data):
        """(step, svs) if X is a head firing read through svs on edge data, else None."""
        if (isinstance(X, tuple) and len(X) == 4 and isinstance(X[1], int) and X not in self.root_readers
                and data.get("svs_used") is not None and data.get("type") in ("d", "s")):
            svs = data["svs_used"]
            svs = ast.literal_eval(svs) if isinstance(svs, str) else list(svs)
            return (data["type"], X[0], X[1], X[2], X[3]), [int(i) for i in svs]
        return None

    def _convert_pending_ps(self) -> None:
        """Turn pending 4-tuple MLP nodes into first-hop nodes of reader groups."""
        if not self.expand_mlps:             # value inputs only: MLPs stay 4-tuple leaves
            return
        G, o = self.G, self.opts
        pending = sorted((n for n in G.nodes if is_mlp_node(n) and len(n) == 4),
                         key=lambda n: (-n[0], repr(n)))
        new: dict = {}          # step -> [(N, svs or None)]
        reader_of: dict = {}    # step -> X
        for N in pending:
            for _, X, key, data in sorted(G.out_edges(N, keys=True, data=True), key=lambda e: repr(e[1])):
                hs = self._head_step_svs(X, data)
                is_v = data.get("type") == "v"      # a value input of firing X (reader step ("v", *X))
                step, svs = hs if hs is not None else ((("v",) if is_v else ("root",)) + tuple(X), None)
                # An MLP has no source token; ACC++ can label one with a source slot. Normalize, so
                # one MLP at one position is one node of its reader's DAG.
                occ = (N[0], "MLP", N[2], N[2], (step,))
                G.add_edge(occ, X, **dict(data))
                G.remove_edge(N, X, key)
                leaf = ("leaf (MLP 0)" if N[0] == 0 and not o.expand_mlp0
                        else "leaf (BOS)" if o.bos_leaf and self.token_to_idx[N[2]] == 0
                        else None)
                if leaf is None and is_v and data.get("child_decision") in ("pruned", "not traced"):
                    leaf = "pruned" if data["child_decision"] == "pruned" else "leaf (value input, not traced)"
                if (leaf is None and hs is None and not is_v
                        and (self.root_readers.get(X) is None or not o.expand_seed_mlps)):
                    leaf = "leaf (no reader operator)"
                if leaf:
                    self._record(occ, None, leaf, [step], dict(path=repr((step,)), n_paths=1, direct=True))
                    continue
                new.setdefault(step, []).append((N, svs))
                reader_of[step] = X
            if G.degree(N) == 0:
                G.remove_node(N)
        for step, items in new.items():
            if step in self._ps_steps:
                raise RuntimeError(f"path-summed MLP expansion: reader {step} gained direct MLP reads "
                                   f"after its group was built")
            self._ps_steps.add(step)
            X = reader_of[step]
            pos = self.token_to_idx[items[0][0][2]]
            assert all(self.token_to_idx[N[2]] == pos for N, _ in items)
            g = _Group(step=step, pos=pos, tok=items[0][0][2],
                       sink=self._group_sink if self._flow_mode else None)
            if step[0] in ("d", "s"):
                union = sorted({i for _, svs in items for i in svs})
                g.coords = union
                WU = head_reader_operator(
                    self.model, self.config, self.cache, self.prompt_idx, self.tr.U, self.tr.VT,
                    self.tr.W_Q_pinv, self.tr.W_K_pinv, self.tr._rotation_stack(pos + 1),
                    X[0], X[1], step[0], union, pos, self.center).float()
                col = {sv: j for j, sv in enumerate(union)}
                masks: dict = {}
                for N, svs in items:      # one MLP read twice by one side: union of its svs
                    mask = masks.setdefault(N[0], torch.zeros(len(union), device=WU.device))
                    mask[[col[i] for i in svs]] = 1.0
                for layer, mask in masks.items():
                    g.direct[layer] = WU * mask[None, :]
            elif step[0] == "v":
                op = self._value_ops[X]
                for N, _ in items:
                    g.direct[N[0]] = op
            else:
                vec = self.root_readers[X].float().reshape(-1, 1)
                for N, _ in items:
                    g.direct[N[0]] = vec
            T = None
            for layer, WT in g.direct.items():
                rec = self.site(layer, pos)
                g.recs[layer] = rec
                t = WT.T @ (rec.site.Fz - rec.site.F0)
                T = t if T is None else T + t
                g.WT[layer] = WT.clone()
                g.n_paths[layer] = 1
                g.push(layer)
            g.T = T
            g.r2 = max(T.norm().item() ** 2, 1e-30)
            self._ps_groups.append(g)

    def _run_path_summed(self) -> None:
        self._ps_groups: deque = deque()
        self._ps_steps: set = set()
        self._convert_pending_ps()
        while self._ps_groups:
            g = self._ps_groups.popleft()
            while g.heap:
                self._expand_ps(g, -heapq.heappop(g.heap))
                self._convert_pending_ps()
            self._finish_group(g)

    def _finish_group(self, g: _Group) -> None:
        for layer, pb in g.pruned_by.items():      # children every parent pruned
            node = g.node(layer)
            read_directly = any(d.get("type") != "m" for *_, d in self.G.out_edges(node, data=True))
            if layer not in g.queued and not read_directly:   # a direct leaf keeps its leaf record
                best = max(pb, key=lambda x: max(abs(x[1]), abs(x[2])))
                self._record(node, None, "pruned", [("mlp", p) for p, *_ in pb],
                             dict(path=repr((g.step,)), traced_share_linear=best[1],
                                  traced_share_real=best[2], reader_svs=g.coords))
        g.recs.clear()

    # ------------------------------------------------------------------ flow-ordered (min_flow)
    def _push(self, key: tuple, item: tuple) -> None:
        self._seq += 1
        heapq.heappush(self._heap, key + (self._seq, item))

    def _push_head(self, f: tuple) -> None:
        """Schedule head firing f = (layer, head, dest, src) once."""
        if f in self._queued_heads or f in self.is_traced:
            return
        self._queued_heads.add(f)
        self._push((-f[0], 1), ("H", f))

    def _push_mlp(self, g: _Group, layer: int) -> None:
        self._push((-layer, 0), ("M", g, layer))

    def _head_node(self, f: tuple):
        l, h, d, s = f
        tok = self.idx_to_token
        return (l, h, tok[d], tok[s]) if d in tok and s in tok else None

    def _spread(self, node) -> None:
        """Split node's flow over its in-edges in proportion to |weight| (parallel edges summed).

        A firing with value inputs passes ``ValueInputOptions.flow_share`` of its flow to them
        and the rest to its query/key inputs, each group split in proportion to |weight|."""
        f = self._flow.get(node, 0.0)
        if f == 0.0:
            return
        wq: dict = defaultdict(float)
        wv: dict = defaultdict(float)
        for u, _, d in self.G.in_edges(node, data=True):
            (wv if d.get("type") == "v" else wq)[u] += abs(float(d.get("weight", 0.0)))
        zq, zv = sum(wq.values()), sum(wv.values())
        if zq <= 0 and zv <= 0:
            return
        alpha = (self.vopts.flow_share if zq > 0 else 1.0) if zv > 0 else 0.0
        for w, z, share in ((wq, zq, 1.0 - alpha), (wv, zv, alpha)):
            if z > 0 and share > 0:
                for u, x in w.items():
                    self._flow[u] += f * share * x / z

    # ------------------------------------------------------------------ value inputs
    def _attn_threshold(self, dest: int) -> float:
        t = self.thresh
        if t == "dynamic":
            return min(1.0, self.tr.dynamic_threshold_scale / (dest + 1))
        return float(t(dest)) if callable(t) else float(t)

    def _head_input(self, layer: int, pos: int):
        key = (layer, pos)
        if key in self._head_inputs:
            self._head_inputs.move_to_end(key)
            return self._head_inputs[key]
        O = head_input_components(self.model, self.config, self.center, self.cache, self.prompt_idx, layer, pos)
        self._head_inputs[key] = O
        if len(self._head_inputs) > _HEAD_INPUT_CACHE:
            self._head_inputs.popitem(last=False)
        return O

    def _value_readers(self, node, pos: int) -> list:
        """Operators (raw residual at pos, D x r) of every reader of firing ``node``."""
        ops = []
        for _, X, data in self.G.out_edges(node, data=True):
            kind = data.get("type")
            op = None
            if kind == "v":
                op = self._value_ops.get(X)
            elif kind == "m":
                op = self._child_ops.get(X)
            elif X in self.root_readers:
                op = self.root_readers[X].float().reshape(-1, 1)
            else:
                hs = self._head_step_svs(X, data)
                if hs is not None:
                    (et, Xl, Xh, *_), svs = hs
                    op = head_reader_operator(
                        self.model, self.config, self.cache, self.prompt_idx, self.tr.U, self.tr.VT,
                        self.tr.W_Q_pinv, self.tr.W_K_pinv, self.tr._rotation_stack(pos + 1),
                        Xl, Xh, et, svs, pos, self.center).float()
            if op is not None:
                ops.append(op)
        return ops

    def _trace_value(self, f: tuple, node, defer: list) -> None:
        """Select the value inputs of firing f = (layer, head, dest, src); see accpp_tracer.value."""
        vo, H = self.vopts, self.H
        l, h, d, s = f
        if node is None or node not in self.G or l == 0 or d == 0 or s > d:
            return
        if vo.bos_leaf and s == 0:
            self.G.nodes[node]["value_trace"] = dict(decision="leaf (BOS)")
            return
        a = self.cache[f"blocks.{l}.attn.hook_pattern"][self.prompt_idx, h, d, s].item()
        if a < self._attn_threshold(d):          # ACC++ does not expand this firing either
            return
        ops = self._value_readers(node, d)
        if not ops:
            self.G.nodes[node]["value_trace"] = dict(decision="leaf (no reader operator)")
            return
        if self.n_value_traced >= vo.max_traces:
            if not self._vcap_warned:
                warnings.warn(f"value inputs reached max_traces={vo.max_traces}; remaining firings "
                              f"are not value-traced.", UserWarning)
                self._vcap_warned = True
            self.G.nodes[node]["value_trace"] = dict(decision="leaf (trace cap)")
            return
        self.n_value_traced += 1
        WT = compress(torch.cat(ops, dim=1))
        M = value_operator(self.model, self.config, self.cache, self.prompt_idx, l, h, d, s, WT)
        O = self._head_input(l, s)
        t = O @ M
        T = t.sum(0)
        r2 = max(T.norm().item() ** 2, 1e-30)
        sel, res = greedy_select(t, vo.eps, vo.k_max)
        self._value_ops[node] = raw_reader(M, self.cache[f"blocks.{l}.ln1.hook_scale"][self.prompt_idx, s],
                                           self.center)
        rec = dict(decision="traced", R_value=res, n_selected=len(sel), n_candidates=len(O),
                   n_readers=len(ops), rank=int(WT.shape[1]), T_norm=T.norm().item())
        n_unmapped = 0
        for rank, i in enumerate(sel):
            cid = O.ids[i]
            u = self._component_node(cid, s)
            if u is None:
                n_unmapped += 1
                continue
            share = float(t[i] @ T) / r2
            status = None
            if abs(share) < vo.tau:
                status = "pruned"
            elif not vo.recurse:
                status = "not traced"
            elif vo.bos_leaf and cid[1] < H and cid[2] == 0:
                status = "leaf (BOS)"
            extra = {} if status is None else dict(child_decision=status)
            self.G.add_edge(u, node, weight=share, type="v", color=V_EDGE_COLOR, rank=rank, **extra)
            if cid[1] < H:
                if "attn_weight" not in self.G.nodes[u]:
                    self.G.nodes[u]["attn_weight"] = self.cache[f"blocks.{cid[0]}.attn.hook_pattern"][
                        self.prompt_idx, cid[1], s, cid[2]].item()
                if status is None:
                    defer.append((cid[0], cid[1], s, cid[2]))
        if n_unmapped:
            rec["n_unmapped"] = n_unmapped
        self.G.nodes[node]["value_trace"] = rec

    def _run_flow(self) -> None:
        """Path-summed expansion with every head firing and MLP node in one global schedule:
        decreasing layer, MLP before the heads of its layer. That order is topological for the
        reads (a head at layer l reads layers < l; an MLP at l reads heads <= l and MLPs < l), so
        a node's flow is complete when it is popped; below min_flow it is not expanded."""
        phi = self._phi
        self._ps_groups = deque()
        self._ps_steps = set()
        self._flow: dict = defaultdict(float)
        self._heap: list = []
        self._seq = 0
        self._queued_heads: set = set()
        self._group_sink = self._push_mlp
        roots = [n for n in self.G.nodes if isinstance(n, tuple) and len(n) == 2]
        for r in roots:
            self._flow[r] = 1.0
        self._convert_pending_ps()
        for f in self.deferred:
            self._push_head(f)
        self.deferred.clear()
        for r in roots:
            self._spread(r)
        n_head_pruned = n_mlp_pruned = 0
        layer_scope = phi is not None and self.opts.min_flow_scope == "layer"
        stage, thr, stage_totals = None, (0.0 if phi is None else phi), {}
        while self._heap:
            neg_layer, kind, _, item = heapq.heappop(self._heap)
            if layer_scope and (neg_layer, kind) != stage:
                # entering a stage: every reader of its nodes is expanded, so its total flow is final
                stage = (neg_layer, kind)
                key = (-neg_layer, "MLP" if kind == 0 else "attn")
                total = sum(f for n, f in self._flow.items() if _flow_stage(n) == key)
                stage_totals[f"{key[0]} {key[1]}"] = total
                thr = phi * total
            if item[0] == "H":
                f = item[1]
                node = self._head_node(f)
                if node is None:
                    continue
                if self._flow.get(node, 0.0) < thr:
                    if node in self.G:
                        self.G.nodes[node]["flow_pruned"] = True
                        n_head_pruned += 1
                    continue
                defer: list = []
                self.tr._trace_recursive(self.cache, self.idx_to_token, self.G, self.is_traced,
                                         self.prompt_idx, *f, self.thresh, self.signals, defer=defer)
                if self.vopts is not None:
                    self._trace_value(f, node, defer)
                self._convert_pending_ps()
                for x in defer:
                    self._push_head(x)
                self._spread(node)
            else:
                _, g, layer = item
                g.pending -= 1
                node = g.node(layer)
                if self._flow.get(node, 0.0) < thr:
                    parents = g.parents.get(layer, [])
                    direct = any(d.get("type") != "m" for *_, d in self.G.out_edges(node, data=True))
                    self._record(node, None, "leaf (flow)",
                                 ([g.step] if direct else []) + [("mlp", p) for p in parents],
                                 dict(path=repr((g.step,)), n_paths=g.n_paths[layer],
                                      n_parents=len(parents), direct=direct, reader_svs=g.coords))
                    n_mlp_pruned += 1
                else:
                    self._expand_ps(g, layer)
                    self._convert_pending_ps()
                    self._spread(node)
                if g.pending == 0:
                    self._finish_group(g)
        self._ps_groups.clear()
        self._flow_summary(roots, phi, n_head_pruned, n_mlp_pruned)
        self.G.graph["flow"]["scope"] = self.opts.min_flow_scope if phi is not None else None
        if layer_scope:
            self.G.graph["flow"]["stage_totals"] = stage_totals
        if self.vopts is not None:
            ev = [d for *_, d in self.G.edges(data=True) if d.get("type") == "v"]
            self.G.graph["value_inputs"] = dict(
                firings_value_traced=self.n_value_traced, value_edges=len(ev),
                value_edges_not_followed=sum(1 for d in ev if d.get("child_decision") is not None),
                options=dict(vars(self.vopts)))

    def _flow_summary(self, roots, phi, n_head_pruned, n_mlp_pruned) -> None:
        """Store node flows; summarize where the objective's flow ends (terminal nodes)."""
        G = self.G
        for n, f in self._flow.items():
            if n in G:
                G.nodes[n]["flow"] = f
        ends: dict = defaultdict(float)
        for n in G.nodes:
            f = self._flow.get(n, 0.0)
            if f == 0.0 or any(abs(float(d.get("weight", 0.0))) > 0 for *_, d in G.in_edges(n, data=True)):
                continue                      # expanded nodes pass their flow on
            if G.nodes[n].get("flow_pruned"):
                k = "flow_pruned (head)"
            elif is_mlp_node(n):
                rec = G.nodes[n].get("mlp_trace")
                if rec is None:
                    k = "MLP leaf (not expanded)"
                else:
                    kind, reason = rec.get("decision_kind"), rec.get("leaf_reason")
                    k = ("flow_pruned (MLP)" if reason == "flow"
                         else f"MLP {kind}" + (f" ({reason})" if reason else ""))
            elif isinstance(n, tuple) and len(n) == 4 and isinstance(n[1], int):
                k = "head leaf (layer 0, BOS, attention threshold, or pruned input)"
            else:
                k = f"{n[1]} leaf" if isinstance(n, tuple) and len(n) >= 2 else "other"
            ends[k] += f
        G.graph["flow"] = dict(min_flow=phi, n_roots=len(roots), head_firings_flow_pruned=n_head_pruned,
                               mlp_nodes_flow_pruned=n_mlp_pruned, terminal_flow=dict(ends))

    def _site_ps(self, g: _Group, layer: int) -> _SiteRec:
        if layer not in g.recs:
            g.recs[layer] = self.site(layer, g.pos)
        return g.recs[layer]

    def _dag_removed(self, g: _Group, init: dict) -> Tensor:
        """Signal removed from the reader (its coordinates) when each MLP ``l`` in ``init`` has its
        normalized input moved by ``init[l]``; the real MLPs on every kept path above run, each
        receiving the summed (centered, ln2-scaled) output change of its kept children."""
        todo, stack = set(init), list(init)
        while stack:
            for p in g.parents.get(stack.pop(), []):
                if p not in todo:
                    todo.add(p)
                    stack.append(p)
        dF: dict = {}
        for l in sorted(todo):
            rec = self._site_ps(g, l)
            dz = init.get(l)
            for c in g.kids.get(l, []):
                if c in dF:
                    add = self._center(dF[c].unsqueeze(1))[:, 0] / rec.ln2_scale
                    dz = add if dz is None else dz + add
            if dz is not None:
                dF[l] = rec.site.forward(rec.z + dz) - rec.site.Fz
        out = torch.zeros_like(g.T)
        for l, d in dF.items():
            if l in g.direct:
                out = out + g.direct[l].T @ d
        return -out

    def _multi_hop_ps(self, g: _Group, layer: int, rec: _SiteRec, tr: MLPTrace) -> dict:
        """R_chain and R_floor of an MLP with MLP parents: ideal removal of its row-space output
        into every kept parent (plus its direct read) vs the witness edit run through the DAG."""
        d = rec.site.Fz - rec.site.F0
        T = g.direct[layer].T @ d if layer in g.direct else torch.zeros_like(g.T)
        dc = self._center(d.unsqueeze(1))[:, 0]
        init = {}
        for p in g.parents[layer]:
            Qp = g.Q[p]
            init[p] = -(Qp @ (Qp.T @ (dc / g.recs[p].ln2_scale)))
        T = T + self._dag_removed(g, init)
        W = self._dag_removed(g, {layer: -tr.edit(rec.O, tr.selected)})
        return dict(R_chain=((T - W).norm() / T.norm().clamp_min(1e-30)).item(),
                    R_floor=((tr.t - T).norm() / tr.t.norm().clamp_min(1e-30)).item())

    def _expand_ps(self, g: _Group, layer: int) -> None:
        o, H, pos = self.opts, self.H, g.pos
        occ = g.node(layer)
        parents = g.parents.get(layer, [])
        direct = any(d.get("type") != "m" for *_, d in self.G.out_edges(occ, data=True))
        readers = ([g.step] if direct else []) + [("mlp", p) for p in parents]
        meta = dict(path=repr((g.step,)), n_paths=g.n_paths[layer], n_parents=len(parents), direct=direct,
                    reader_svs=g.coords)
        if layer == 0 and not o.expand_mlp0:
            self._record(occ, None, "leaf (MLP 0)", readers, meta)
            return
        if o.bos_leaf and pos == 0:
            self._record(occ, None, "leaf (BOS)", readers, meta)
            return
        if not self._tick():
            self._record(occ, None, "leaf (node cap)", readers, meta)
            return
        rec = self._site_ps(g, layer)
        tr = run_algorithm1(rec.site, rec.O, rec.ids, g.WT[layer], o.eps, o.k_max, pos)
        decision = self._decide(tr)
        T, r2 = g.T, g.r2
        meta.update(t_share_root=(tr.t @ T).item() / r2, bias_share_root=(tr.bias @ T).item() / r2,
                    t=[float(x) for x in tr.t])
        g.Q[layer] = tr.Q
        if o.diagnostics and parents:
            meta.update(self._multi_hop_ps(g, layer, rec, tr))
        self._record(occ, tr, decision, readers, meta)
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
            extra = dict(share_root=(tr.gamma[i] @ T).item() / r2)
            if cid[1] == H:            # upstream MLP: a node of this reader's DAG
                kl = cid[0]
                child = g.node(kl)
                c_site = self._site_ps(g, kl).site
                d = self._center((c_site.Fz - c_site.F0).unsqueeze(1))[:, 0] / rec.ln2_scale
                sig = tr.Q @ (tr.Q.T @ d)
                lin = (tr.AT.T @ d) @ T / r2
                real = self._dag_removed(g, {layer: -sig}) @ T / r2
                kids.append(dict(node=child, layer=kl, lin=float(lin), real=float(real), sig=sig))
                u = child
                sig_i = sig
                extra.update(traced_share_linear=float(lin), traced_share_real=float(real))
            self._add_input_edge(u, occ, tr.shares[i].item(), rank, extra, sig_i)
            if cid[1] < H:
                head_inputs.append(dict(node=u, cid=cid, i=i))
        if n_unmapped:
            self.G.nodes[occ]["mlp_trace"]["n_unmapped"] = n_unmapped
        removed = (lambda s: self._dag_removed(g, {layer: -s}))
        joint = (lambda s: float(removed(s) @ T / r2))
        heads = self._head_input_shares(head_inputs, tr, rec, T, r2, removed)
        self._prune(kids + heads, o.tau, joint)
        for h in head_inputs:
            self._finish_head_input(h, occ, pos)
        WT_child = child_operator(tr, rec.ln2_scale, self.center)
        self._child_ops[occ] = WT_child
        for k in kids:
            kl = k["layer"]
            self.G.edges[k["node"], occ, max(self.G[k["node"]][occ])]["child_decision"] = k["status"]
            if k["status"] == "pruned":
                g.pruned_by.setdefault(kl, []).append((layer, k["lin"], k["real"]))
                continue
            g.parents.setdefault(kl, []).append(layer)
            g.kids.setdefault(layer, []).append(kl)
            g.WT[kl] = WT_child.clone() if kl not in g.WT else g.WT[kl] + WT_child
            g.n_paths[kl] = g.n_paths.get(kl, 0) + g.n_paths[layer]
            g.push(kl)

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
            done[N] = child_operator(tr, rec.ln2_scale, self.center)
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
