"""Tests for frozen-necessity pruning (accpp_tracer.prune; the default of the trace methods since 0.5.0).

Covers every objective the pruner scores: logit targets (linear seeding from ``trace`` = the real logit
difference; from ``trace_from_cache`` = ln_final(x) . d; probability-aware seeding = J) and signal tracing
(``trace_from_probe`` at an intermediate and at the last layer), the tight query/key greedy, the
graph-level API, and the GQA / RoPE / bias (Qwen) and parallel attention-MLP (Pythia) removal paths.
"""

import networkx as nx
import numpy as np
import pytest
import torch

from accpp_tracer import FrozenPruner, PruneOptions, prune_by_necessity, trace_firing
from accpp_tracer.prune import resolve_prune

from conftest import IOI_PROMPT, build_idx_to_token

LIN = dict(answer_token=" Mary", wrong_token=" John", seeding="linear")
FAST = dict(attn_weight_thresh=0.3)           # smaller traces for the slower checks
FULL = PruneOptions(return_full=True)


def _model_nodes(G):
    return {G.nodes[n]["model_node"] for n in G.nodes if "model_node" in G.nodes[n]}


def _check_pruned(G, theta=0.01):
    P = G.graph["prune"]
    assert P["pruned"] and not P["empty"]
    tol = 2e-3 * max(1.0, max(abs(x) for x in P["score_model"]))
    assert P["checks"]["no_removal"] < tol and P["checks"]["whole_trace"] < tol
    nec = {tuple(k): v for k, v in P["necessity"]}
    assert len(nec) == P["n_model_nodes_trace"]
    kept = {tuple(k) for k in P["kept"]}
    assert kept == {k for k, v in nec.items() if v >= theta}
    assert _model_nodes(G) == kept
    assert all(G.nodes[n]["necessity"] >= theta for n in G.nodes if "necessity" in G.nodes[n])
    assert any(isinstance(n, tuple) and len(n) == 2 for n in G.nodes)        # the objective is kept
    return P


# ---------------------------------------------------------------------- options
def test_options():
    assert resolve_prune(True) == PruneOptions()
    assert resolve_prune(False) is None and resolve_prune(None) is None
    o = PruneOptions()
    assert (o.theta, o.kappa, o.mlp_eps, o.value_eps, o.value_k_max) == (0.01, 0.2, 0.05, 0.02, 40)
    for bad in (dict(kappa=0.0), dict(kappa=1.5), dict(mlp_eps=1.0), dict(value_eps=0.0), dict(value_k_max=0),
                dict(check_tol=0.0), dict(baseline="zero")):
        with pytest.raises(ValueError):
            PruneOptions(**bad)
    with pytest.raises(TypeError):
        resolve_prune("yes")


def test_prune_off_has_no_prune_record(gpt2_tracer):
    G = gpt2_tracer.trace(IOI_PROMPT, **LIN, **FAST, mlp_expansion=False, value_inputs=False, prune=False)
    assert "prune" not in G.graph and not any("necessity" in G.nodes[n] for n in G.nodes)


# ---------------------------------------------------------------------- tight greedy
def test_tight_greedy_extends_accpp(gpt2_model, gpt2_tracer, gpt2_ioi):
    """kappa < 1 continues the same term order: ACC++'s selection is a prefix, with the same edge weights."""
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    mary = int((tokens[0] == gpt2_model.to_single_token(" Mary")).nonzero()[0])
    t = gpt2_tracer
    args = (gpt2_model, cache, 0, 9, 9, end, mary, t.U, t.S, t.VT, t.W_Q_pinv, t.W_K_pinv, t.config, "cpu", 0.2)
    stats = {}
    with torch.no_grad():
        d1, w1, s1, ws1 = trace_firing(*args)
        d2, w2, s2, ws2 = trace_firing(*args, kappa=0.2, stats=stats)
    assert stats["firings"] == 2 and stats["capped"] == 0
    for a, b in ((d1, d2), (s1, s2)):
        assert set(a) <= set(b)
        for c in a:
            assert set(a[c]) <= set(b[c])
    assert sum(len(v) for v in d2.values()) > sum(len(v) for v in d1.values())


# ---------------------------------------------------------------------- logit targets
@pytest.fixture(scope="module")
def gpt2_linear_full(gpt2_tracer):
    """MLP expansion + value inputs (defaults), tight trace, all necessities (no cut)."""
    return gpt2_tracer.trace(IOI_PROMPT, **LIN, **FAST, prune=FULL)


def test_logit_objective_is_real_logit_difference(gpt2_model, gpt2_linear_full):
    P = gpt2_linear_full.graph["prune"]
    assert P["objective"] == "logit_tokens" and not P["pruned"]
    logits = gpt2_model(gpt2_model.to_tokens(IOI_PROMPT))[0, -1]
    ld = (logits[gpt2_model.to_single_token(" Mary")] - logits[gpt2_model.to_single_token(" John")]).item()
    assert abs(P["score_model"][0] - ld) < 1e-3
    assert P["norm"] == P["score_model"]
    assert P["qk_greedy"]["kappa"] == 0.2 and P["qk_greedy"]["firings"] > 0
    assert P["edges_by_reader"].get("root", 0) > 0 and P["edges_by_reader"].get("v", 0) > 0
    assert P["edges_by_reader"].get("mlp", 0) > 0
    assert P["options"]["value_eps"] == 0.02


def test_cut_and_thresholds(gpt2_linear_full):
    G = gpt2_linear_full
    heads_and_mlps = [n for n in G.nodes if isinstance(n, tuple) and len(n) >= 4
                      and (isinstance(n[1], int) or n[1] == "MLP")]
    assert heads_and_mlps and all("necessity" in G.nodes[n] for n in heads_and_mlps)
    H = prune_by_necessity(G)
    P = _check_pruned(H)
    H2 = prune_by_necessity(G, 0.05)
    assert _model_nodes(H2) <= _model_nodes(H)
    assert H2.graph["prune"]["frozen_score"] is None and P["frozen_score"] is not None
    assert set(H.nodes) <= set(G.nodes)
    assert not [n for n in H.nodes if H.degree(n) == 0 and not (isinstance(n, tuple) and len(n) == 2)
                and "model_node" not in H.nodes[n]]


def test_default_is_cut_of_full(gpt2_tracer):
    kw = dict(**LIN, mlp_expansion=False, value_inputs=False)
    G = gpt2_tracer.trace(IOI_PROMPT, **kw)                        # prune=True by default
    Gf = gpt2_tracer.trace(IOI_PROMPT, **kw, prune=FULL)
    H = prune_by_necessity(Gf)
    _check_pruned(G)
    assert set(G.nodes) == set(H.nodes) and set(G.edges(keys=True)) == set(H.edges(keys=True))
    assert G.graph["prune"]["frozen_score"] == pytest.approx(H.graph["prune"]["frozen_score"])


def test_linear_from_cache(gpt2_model, gpt2_tracer, gpt2_ioi):
    """trace_from_cache with a logit direction: the objective is ln_final(x) . d."""
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    idx = build_idx_to_token(gpt2_model, tokens)
    d = (gpt2_model.W_U[:, gpt2_model.to_single_token(" Mary")]
         - gpt2_model.W_U[:, gpt2_model.to_single_token(" John")]).detach()
    G = gpt2_tracer.trace_from_cache(cache, d, end, idx, ("Logit direction", idx[end]), **FAST)
    P = _check_pruned(G)
    assert P["objective"] == "linear"
    assert P["score_model"][0] == pytest.approx((cache["ln_final.hook_normalized"][0, end] @ d).item(), abs=1e-3)


def test_prob_objective_default_call(gpt2_model, gpt2_tracer):
    """The plain default call: probability-aware seeding, MLP + value, pruned; the norm is J's completeness."""
    G = gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary")
    P = _check_pruned(G)
    assert P["objective"] == "prob"
    tok = gpt2_model.to_tokens(IOI_PROMPT)
    logits = gpt2_model(tok)[0, -1]
    J = torch.log_softmax(logits, -1)[gpt2_model.to_single_token(" Mary")].item()
    Jb = torch.log_softmax(gpt2_model.b_U, -1)[gpt2_model.to_single_token(" Mary")].item()
    assert P["score_model"][0] == pytest.approx(J, abs=1e-3)
    assert P["norm"][0] == pytest.approx(J - Jb, abs=1e-3)
    assert 0 < P["n_model_nodes_kept"] < P["n_model_nodes_trace"]


# ---------------------------------------------------------------------- signal (probe) tracing
@pytest.mark.parametrize("layer", [8, 11])
def test_probe_objective(gpt2_model, gpt2_tracer, gpt2_ioi, layer):
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    idx = build_idx_to_token(gpt2_model, tokens)
    w = (gpt2_model.W_U[:, gpt2_model.to_single_token(" Mary")]
         - gpt2_model.W_U[:, gpt2_model.to_single_token(" John")]).detach()
    G = gpt2_tracer.trace_from_probe(cache, w, layer, end, idx, ("Probe", idx[end]), **FAST)
    P = _check_pruned(G)
    assert P["objective"] == "probe"
    x = cache[f"blocks.{layer}.hook_resid_post"][0, end]
    assert P["score_model"][0] == pytest.approx((x @ w).item(), rel=1e-4)
    assert P["edges_by_reader"].get("root", 0) > 0
    assert all(k[1] <= layer for k in map(tuple, P["kept"]))          # nothing above the probe site


def test_probe_traced_baseline(gpt2_model, gpt2_tracer, gpt2_ioi):
    """baseline='traced': necessity is a fraction of s(M) - s_F(empty)."""
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    idx = build_idx_to_token(gpt2_model, tokens)
    w = gpt2_model.W_U[:, gpt2_model.to_single_token(" Mary")].detach()
    args = (cache, w, 6, end, idx, ("Probe", idx[end]))
    Gm = gpt2_tracer.trace_from_probe(*args, **FAST, mlp_expansion=False, prune=FULL)
    Gt = gpt2_tracer.trace_from_probe(*args, **FAST, mlp_expansion=False,
                                      prune=PruneOptions(return_full=True, baseline="traced"))
    Pm, Pt = Gm.graph["prune"], Gt.graph["prune"]
    assert Pt["baseline"] == "traced"
    assert Pt["norm"][0] == pytest.approx(Pt["score_model"][0] - Pt["score_empty"][0], rel=1e-6)
    nm = {tuple(k): v for k, v in Pm["necessity"]}
    nt = {tuple(k): v for k, v in Pt["necessity"]}
    assert nm.keys() == nt.keys()
    r = Pm["norm"][0] / Pt["norm"][0]
    assert all(nt[k] == pytest.approx(nm[k] * r, rel=1e-4, abs=1e-7) for k in nm)


# ---------------------------------------------------------------------- other architectures
def test_qwen_gqa_rope_biases(qwen_tracer):
    """Grouped-query attention is ungrouped for the removal passes; RoPE; large query/key biases."""
    _check_pruned(qwen_tracer.trace(IOI_PROMPT, answer_token=" Mary", **FAST))


def test_pythia_parallel_attn_mlp(pythia_tracer):
    """Parallel attention and MLP: an MLP's input is the block's resid_pre."""
    P = _check_pruned(pythia_tracer.trace(IOI_PROMPT, answer_token=" Mary", attn_weight_thresh=0.4, value_inputs=False))
    assert P["edges_by_reader"].get("mlp", 0) > 0


def test_frozen_pruner_direct(gpt2_model, gpt2_tracer, gpt2_ioi):
    """FrozenPruner on a plain trace: the whole trace reproduces the model, the empty set does not."""
    from accpp_tracer.prune import Objective, readers_stand_in
    from accpp_tracer.mlp_expand import root_reader_linear
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    idx = build_idx_to_token(gpt2_model, tokens)
    d = gpt2_model.W_U[:, gpt2_model.to_single_token(" Mary")].detach()
    root = ("Logit direction", idx[end])
    G = gpt2_tracer.trace_from_cache(cache, d, end, idx, root, mlp_expansion=False, value_inputs=False, prune=False)
    X = readers_stand_in(gpt2_tracer, cache, 0, {root: root_reader_linear(d, cache["ln_final.hook_scale"][0, end], True)})
    SR = FrozenPruner(gpt2_tracer, G, X, cache, 0, idx, Objective("linear", [root], end, directions=[d]))
    full = SR.model_nodes()
    assert SR.self_test() < 1e-3
    assert np.allclose(SR.run_F(full), SR.s_model, atol=1e-3)
    assert not np.allclose(SR.run_F(set()), SR.s_model, atol=1e-2)
    assert isinstance(G, nx.MultiDiGraph)
