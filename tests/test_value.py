"""Tests for value-input tracing (accpp_tracer.value; ``Tracer.trace(..., value_inputs=...)``).

Covers:
- options and their validation;
- mathematical invariants on gpt2 and tiny Gemma-2 / Llama / parallel models: the
  head-input decomposition reconstructs ``ln1.hook_normalized``; the value operator
  reproduces the readers' coordinates of the firing's real output (up to the
  constant ``b_V`` term); the raw-residual reader is its pullback; compression
  keeps the Gram matrix; the greedy stops at eps or k_max;
- graphs on GPT-2 IOI, with and without MLP expansion: edge structure, shares,
  value records, root-flow conservation with the value / query-key split, the
  ``tau``, ``recurse``, ``bos_leaf`` and ``min_flow`` options, MLP value inputs
  expanded as their own reader group, probe tracing;
- graph utilities on value-traced graphs; a golden graph.
"""

import ast
import warnings

import networkx as nx
import pytest
import torch

from accpp_tracer import MLPExpansionOptions, Tracer, ValueInputOptions, edges_from_graph
from accpp_tracer.graphs import circuit_to_html, combine_prompt_graphs
from accpp_tracer.mlp_expand import is_mlp_node
from accpp_tracer.value import (
    compress,
    greedy_select,
    head_input_components,
    raw_reader,
    resolve_value_inputs,
    value_operator,
)

from conftest import GOLDEN_DIR, build_idx_to_token
from golden_utils import assert_graph_matches
from test_mlp import REPORT_PROMPT, _run, tiny_gemma, tiny_llama, tiny_parallel  # noqa: F401 (fixtures)

torch.set_grad_enabled(False)

KW = dict(answer_token=" Mary", wrong_token=" John", prepend_bos=False, seeding="linear")


def _is_firing(n):
    return isinstance(n, tuple) and len(n) == 4 and isinstance(n[1], int)


def _v_edges(G):
    return [(u, v, d) for u, v, d in G.edges(data=True) if d.get("type") == "v"]


@pytest.fixture(scope="module")
def gpt2_plain_v(gpt2_tracer):
    return gpt2_tracer.trace(REPORT_PROMPT, **KW, value_inputs=True, mlp_expansion=False, prune=False)


@pytest.fixture(scope="module")
def gpt2_mlp_v(gpt2_tracer):
    return gpt2_tracer.trace(REPORT_PROMPT, **KW, mlp_expansion=MLPExpansionOptions(min_flow=1e-3),
                             value_inputs=True, prune=False)


@pytest.fixture(scope="module")
def report_cache(gpt2_model):
    tokens = gpt2_model.to_tokens(REPORT_PROMPT, prepend_bos=False)
    _, cache = gpt2_model.run_with_cache(tokens)
    return tokens, cache


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

def test_options():
    assert resolve_value_inputs(None) is None and resolve_value_inputs(False) is None
    assert resolve_value_inputs(True) == ValueInputOptions()
    o = ValueInputOptions(tau=0.1)
    assert resolve_value_inputs(o) is o
    with pytest.raises(TypeError):
        resolve_value_inputs("yes")
    for bad in (dict(eps=0), dict(eps=1), dict(k_max=0), dict(tau=-1), dict(flow_share=1.5),
                dict(max_traces=0)):
        with pytest.raises(ValueError):
            ValueInputOptions(**bad)


def test_needs_path_summed(gpt2_tracer):
    with pytest.raises(ValueError, match="path_summed"):
        gpt2_tracer.trace(REPORT_PROMPT, **KW, mlp_expansion=MLPExpansionOptions(mode="per_reader", min_flow=None),
                          value_inputs=True, prune=False)


def test_off_flags(gpt2_tracer):
    """value_inputs=None and value_inputs=False both turn value tracing off (on by default since 0.5.0)."""
    G0 = gpt2_tracer.trace(REPORT_PROMPT, **KW, mlp_expansion=False, value_inputs=None, prune=False)
    G1 = gpt2_tracer.trace(REPORT_PROMPT, **KW, value_inputs=False, mlp_expansion=False, prune=False)
    assert not _v_edges(G0) and "value_inputs" not in G0.graph
    assert set(G0.edges(keys=True)) == set(G1.edges(keys=True))


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def _firing_output(model, cfg, cache, l, h, d, s):
    """The firing's contribution to the residual at d: a v_s W_O (times the post-attention norm)."""
    attn = model.blocks[l].attn
    a = cache[f"blocks.{l}.attn.hook_pattern"][0, h, d, s]
    v = cache[f"blocks.{l}.attn.hook_v"][0, s]
    if cfg.has_gqa:
        v = torch.repeat_interleave(v, cfg.gqa_repeats, dim=0)
    out = a * (v[h] @ attn.W_O[h])
    bias = a * (attn.b_V[h] @ attn.W_O[h])
    if cfg.has_post_attn_ln:
        post = model.blocks[l].ln1_post.w
        sc = cache[f"blocks.{l}.ln1_post.hook_scale"][0, d]
        out, bias = out * post / sc, bias * post / sc
    return out.float(), bias.float()


def _check_invariants(model, tracer, cache, n_tok):
    cfg, D = tracer.config, model.cfg.d_model
    center = (model.cfg.normalization_type or "").startswith("LN")
    g = torch.Generator().manual_seed(0)
    for l in range(1, model.cfg.n_layers):
        d, s, h = n_tok - 1, n_tok // 2, l % model.cfg.n_heads
        O = head_input_components(model, cfg, center, cache, 0, l, s)
        z = cache[f"blocks.{l}.ln1.hook_normalized"][0, s]
        assert torch.allclose(O.sum(), z, atol=1e-4 * z.norm())
        WT = torch.randn(D, 3, generator=g)
        M = value_operator(model, cfg, cache, 0, l, h, d, s, WT)
        out, bias = _firing_output(model, cfg, cache, l, h, d, s)
        lhs = WT.T @ out
        rhs = (O @ M).sum(0) + WT.T @ bias
        assert torch.allclose(lhs, rhs, atol=1e-4 * max(lhs.norm().item(), 1e-6)), (l, lhs, rhs)
        # raw reader: a residual vector x at s enters z_s as center(x) / s_ln1
        x = torch.randn(D, generator=g)
        sc = cache[f"blocks.{l}.ln1.hook_scale"][0, s]
        xc = (x - x.mean()) if center else x
        R = raw_reader(M, sc, center)
        assert torch.allclose(R.T @ x, (xc / sc) @ M, atol=1e-5 * (R.T @ x).norm().item())


def test_invariants_gpt2(gpt2_model, gpt2_tracer, report_cache):
    tokens, cache = report_cache
    _check_invariants(gpt2_model, gpt2_tracer, cache, tokens.shape[1])


@pytest.mark.parametrize("fixture", ["tiny_gemma", "tiny_llama", "tiny_parallel"])
def test_invariants_synthetic(request, fixture):
    m = request.getfixturevalue(fixture)
    tracer = Tracer(m, device="cpu")
    tokens, _, cache = _run(m)
    _check_invariants(m, tracer, cache, tokens.shape[1])


def test_compress_and_greedy():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(50, 4, generator=g)
    WT = torch.cat([A, A[:, :2] @ torch.randn(2, 3, generator=g)], dim=1)     # rank 4, 7 columns
    C = compress(WT)
    assert C.shape == (50, 4)
    assert torch.allclose(C @ C.T, WT @ WT.T, atol=1e-4)
    t = torch.randn(40, 3, generator=g)
    sel, res = greedy_select(t, 0.1, 5)
    T = t.sum(0)
    assert len(sel) == len(set(sel)) <= 5
    assert abs(res - ((T - t[sel].sum(0)).norm() / T.norm()).item()) < 1e-5
    if res > 0.1 and len(sel) < 5:                 # stopped because no row reduces the residual
        r = T - t[sel].sum(0)
        gain = 2 * (t @ r) - (t * t).sum(1)
        gain[sel] = -float("inf")
        assert gain.max() <= 0
    sel1, _ = greedy_select(t, 0.1, 1)
    assert sel1 == sel[:1]


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def _check_graph(G, model, idx_to_token, vopts):
    tok2idx = {v: k for k, v in idx_to_token.items()}
    H = model.cfg.n_heads
    ev = _v_edges(G)
    assert ev
    by_target: dict = {}
    for u, v, d in ev:
        assert _is_firing(v) and v[0] >= 1
        assert u[2] == v[3]                      # value inputs sit at the firing's source position
        assert u[0] < v[0] or u[1] == "Embedding"
        by_target.setdefault(v, []).append(d)
    for v, ds in by_target.items():
        rec = G.nodes[v]["value_trace"]
        assert rec["decision"] == "traced"
        assert len(ds) == rec["n_selected"] - rec.get("n_unmapped", 0) <= vopts.k_max
        assert sorted(d["rank"] for d in ds) == list(range(len(ds))) or rec.get("n_unmapped")
        total = sum(d["weight"] for d in ds)
        if not rec.get("n_unmapped"):
            # sum of shares = (sum_sel t_c) . T / |T|^2 = 1 - r . T / |T|^2, and |r . T| <= |r| |T|
            assert abs(total - 1) <= rec["R_value"] + 1e-4
    # every firing that is value-traced passed ACC++'s attention threshold and is not at layer 0
    for n in G.nodes:
        if "value_trace" in G.nodes[n] and G.nodes[n]["value_trace"]["decision"] == "traced":
            assert n[0] >= 1 and tok2idx[n[2]] > 0
    # root flow is conserved: terminal flow sums to one per root
    fl = G.graph["flow"]
    assert abs(sum(fl["terminal_flow"].values()) - fl["n_roots"]) < 1e-6
    # value inputs that are heads (and followed) are traced in turn
    for u, v, d in ev:
        if _is_firing(u) and d.get("child_decision") is None:
            assert G.nodes[u].get("attn_weight") is not None


def test_graph_plain(gpt2_model, gpt2_plain_v, report_cache):
    tokens, _ = report_cache
    G = gpt2_plain_v
    _check_graph(G, gpt2_model, build_idx_to_token(gpt2_model, tokens), ValueInputOptions())
    assert not any(is_mlp_node(n) and len(n) == 5 for n in G.nodes)     # MLPs stay 4-tuple leaves
    assert G.graph["value_inputs"]["firings_value_traced"] > 0
    traced = [n for n in G.nodes if G.nodes[n].get("value_trace", {}).get("decision") == "traced"]
    assert any(any(d.get("type") in ("d", "s") for *_, d in G.in_edges(v, data=True)) for v in traced)


def test_flow_split_exact(gpt2_tracer):
    """flow_share=1: query/key inputs of value-traced firings get no flow from them."""
    G1 = gpt2_tracer.trace(REPORT_PROMPT, **KW, value_inputs=ValueInputOptions(flow_share=1.0), mlp_expansion=False, prune=False)
    G0 = gpt2_tracer.trace(REPORT_PROMPT, **KW, value_inputs=ValueInputOptions(flow_share=0.0), mlp_expansion=False, prune=False)
    for G in (G0, G1):
        fl = G.graph["flow"]
        assert abs(sum(fl["terminal_flow"].values()) - fl["n_roots"]) < 1e-6
    # with flow_share=0 no flow enters a component only through value edges
    for n in G0.nodes:
        outs = list(G0.out_edges(n, data=True))
        if outs and all(d.get("type") == "v" for *_, d in outs):
            assert G0.nodes[n].get("flow", 0.0) == 0.0


def test_graph_mlp(gpt2_model, gpt2_mlp_v, report_cache):
    tokens, _ = report_cache
    G = gpt2_mlp_v
    _check_graph(G, gpt2_model, build_idx_to_token(gpt2_model, tokens), ValueInputOptions())
    v_groups = [n for n in G.nodes if is_mlp_node(n) and len(n) == 5 and n[4][0][0] == "v"]
    assert v_groups, "no MLP value input was expanded as its own reader group"
    kinds = {G.nodes[n]["mlp_trace"]["decision_kind"] for n in v_groups}
    assert kinds <= {"expand", "approximate", "leaf", "pruned"}
    for n in v_groups:      # a v-group MLP read directly by its firing has a v edge to it
        step = n[4][0]
        firing = tuple(step[1:])
        direct = [d for *_, d in G.out_edges(n, data=True) if d.get("type") == "v"]
        if G.nodes[n]["mlp_trace"].get("direct"):
            assert direct and G.has_edge(n, firing)
    # min_flow: every expanded firing has flow >= min_flow
    for n in G.nodes:
        if G.nodes[n].get("value_trace", {}).get("decision") == "traced":
            assert G.nodes[n]["flow"] >= 1e-3 - 1e-12


def test_size_controls(gpt2_tracer, gpt2_plain_v):
    n_full = gpt2_plain_v.graph["value_inputs"]["firings_value_traced"]
    G_tau = gpt2_tracer.trace(REPORT_PROMPT, **KW, value_inputs=ValueInputOptions(tau=0.2), mlp_expansion=False, prune=False)
    G_one = gpt2_tracer.trace(REPORT_PROMPT, **KW, value_inputs=ValueInputOptions(recurse=False), mlp_expansion=False, prune=False)
    G_k = gpt2_tracer.trace(REPORT_PROMPT, **KW, value_inputs=ValueInputOptions(k_max=2), mlp_expansion=False, prune=False)
    for G in (G_tau, G_one, G_k):
        assert G.graph["value_inputs"]["firings_value_traced"] <= n_full
        assert G.number_of_nodes() <= gpt2_plain_v.number_of_nodes()
    for *_, d in _v_edges(G_tau):
        assert (abs(d["weight"]) < 0.2) == (d.get("child_decision") == "pruned")
    assert all(d.get("child_decision") == "not traced" for *_, d in _v_edges(G_one))
    by_t: dict = {}
    for _, v, _ in _v_edges(G_k):
        by_t[v] = by_t.get(v, 0) + 1
    assert max(by_t.values()) <= 2


def test_min_flow_bounds_value(gpt2_tracer):
    Gs = [gpt2_tracer.trace(REPORT_PROMPT, **KW, mlp_expansion=MLPExpansionOptions(min_flow=phi),
                            value_inputs=True, prune=False) for phi in (1e-2, 1e-3)]
    n = [G.graph["value_inputs"]["firings_value_traced"] for G in Gs]
    assert n[0] <= n[1]


def test_bos_leaf(gpt2_tracer):
    G = gpt2_tracer.trace("When Mary and John went to the store, John gave the bottle to",
                          answer_token=" Mary", wrong_token=" John", prepend_bos=True, seeding="linear",
                          value_inputs=ValueInputOptions(bos_leaf=True), mlp_expansion=False, prune=False)
    tok0 = "<|endoftext|>"
    for n in G.nodes:
        rec = G.nodes[n].get("value_trace")
        if rec and rec["decision"] == "traced":
            assert n[3] != tok0
    for u, v, d in _v_edges(G):
        if _is_firing(u) and u[3] == tok0:
            assert d.get("child_decision") == "leaf (BOS)"


@pytest.mark.parametrize("fixture", ["tiny_gemma", "tiny_llama", "tiny_parallel"])
def test_synthetic_pipeline(request, fixture):
    m = request.getfixturevalue(fixture)
    tracer = Tracer(m, device="cpu")
    tokens, logits, cache = _run(m)
    end = tokens.shape[1] - 1
    idx_to_token = {i: f"t{i}" for i in range(end + 1)}
    direction = m.W_U[:, int(logits[0, end].argmax())].detach()
    for mx in (None, MLPExpansionOptions(expand_mlp0=True, min_flow=None)):
        G = tracer.trace_from_cache(cache, direction, end, idx_to_token, ("Logit direction", f"t{end}"),
                                    attn_weight_thresh=0.05, mlp_expansion=mx, value_inputs=True, prune=False)
        fl = G.graph["flow"]
        assert abs(sum(fl["terminal_flow"].values()) - fl["n_roots"]) < 1e-6
        assert _v_edges(G)


def test_probe(gpt2_model, gpt2_tracer, report_cache):
    tokens, cache = report_cache
    end = tokens.shape[1] - 1
    idx = build_idx_to_token(gpt2_model, tokens)
    io, s = gpt2_model.to_single_token(" Mary"), gpt2_model.to_single_token(" John")
    direction = (gpt2_model.W_U[:, io] - gpt2_model.W_U[:, s]).detach()
    G = gpt2_tracer.trace_from_probe(cache, direction, 9, end, idx, ("Probe", idx[end]), value_inputs=True, mlp_expansion=False, prune=False)
    assert _v_edges(G)
    G2 = gpt2_tracer.trace_from_probe(cache, direction, 9, end, idx, ("Probe", idx[end]),
                                      mlp_expansion=MLPExpansionOptions(min_flow=1e-2), value_inputs=True, prune=False)
    assert _v_edges(G2)


def test_utilities(gpt2_model, gpt2_mlp_v, gpt2_plain_v, report_cache, tmp_path):
    tokens, _ = report_cache
    tok2idx = {v: k for k, v in build_idx_to_token(gpt2_model, tokens).items()}
    specs = edges_from_graph(gpt2_mlp_v, tok2idx, gpt2_model.cfg.n_heads)      # value edges are skipped
    assert specs
    out = circuit_to_html(gpt2_mlp_v, list(tok2idx), 12, tmp_path / "c.html")
    html = out.read_text()
    assert "value-input edge (v)" in html and '"type": "v"' in html
    Gc = combine_prompt_graphs([gpt2_plain_v, gpt2_plain_v])
    assert any(d.get("type") == "v" for *_, d in Gc.edges(data=True))


def test_golden_gpt2_linear_value(gpt2_plain_v, update_golden):
    assert_graph_matches(gpt2_plain_v, GOLDEN_DIR / "graph_gpt2_linear_value.json", update_golden)
