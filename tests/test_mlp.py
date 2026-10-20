"""Tests for MLP expansion (Algorithm 1; report Sections 7-8).

Covers, on gpt2 (standard GELU MLP, LayerNorm), pythia-160m (parallel
attention+MLP, RoPE), Qwen2.5-0.5B (gated SiLU, RMSNorm, GQA) and tiny random
models for the architectures no fixture has (Gemma-2-style gated MLP with a
post-MLP RMSNorm; a Llama-style gated MLP; a parallel standard MLP):

- architecture detection and its error cases;
- mathematical invariants: the input decomposition reconstructs
  ``ln2.hook_normalized``, the site reproduces ``hook_mlp_out``, the
  path-average Jacobian is complete and its transpose is its adjoint, the
  attributions sum to the target, the head reader operator equals ACC++'s
  own ``rotated_normalized`` signal coordinates;
- parity with the research code (report Table 2: 22 components, R_sig 0.0895);
- per-reader and merged graph expansion: structure, options, edge cases;
- graph utilities (edges_from_graph, run_intervention, combine_prompt_graphs,
  HTML and Cytoscape export) on expanded graphs;
- golden graphs for both modes.
"""

import json
import warnings
from types import SimpleNamespace

import networkx as nx
import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from accpp_tracer import (
    MLPExpansionOptions,
    Tracer,
    edges_from_graph,
    get_mlp_spec,
)
from accpp_tracer.circuit import _compute_residual_shares, _ig_attribution_seeds
from accpp_tracer.graphs import circuit_to_html, combine_prompt_graphs
from accpp_tracer.graphs.visualization import format_graph_cytoscape_by_token_pos
from accpp_tracer.mlp import MLPSite, mlp_input_components, run_algorithm1
from accpp_tracer.mlp_expand import is_mlp_node, root_reader_prob

from conftest import GOLDEN_DIR, IOI_PROMPT, build_idx_to_token
from golden_utils import assert_graph_matches, graph_to_jsonable

torch.set_grad_enabled(False)

# The research repo's prompt and setting (report Section 4: no BOS, linear
# seeding with the IO - S direction).
REPORT_PROMPT = "When Mary and John went to the store, John gave the bottle to"


# ---------------------------------------------------------------------------
# Tiny synthetic models
# ---------------------------------------------------------------------------

def _tiny(**kw):
    base = dict(n_layers=3, d_model=64, n_ctx=32, d_head=16, n_heads=4, d_mlp=96,
                d_vocab=101, attn_only=False, seed=0, device="cpu")
    base.update(kw)
    model = HookedTransformer(HookedTransformerConfig(**base))
    g = torch.Generator().manual_seed(1)
    for blk in model.blocks:   # non-trivial norm weights / biases
        for name in ("ln1_post", "ln2_post"):
            if hasattr(blk, name):
                getattr(blk, name).w.data = 0.5 + torch.rand(base["d_model"], generator=g)
        if hasattr(blk, "mlp") and hasattr(blk.mlp, "b_in"):
            blk.mlp.b_in.data = 0.1 * torch.randn(blk.mlp.b_in.shape, generator=g)
    return model


@pytest.fixture(scope="module")
def tiny_gemma():
    """Gemma-2 style: gated GELU-tanh, RMSNorm with pre+post norms, RoPE, GQA, soft-cap."""
    return _tiny(n_key_value_heads=2, act_fn="gelu_pytorch_tanh", gated_mlp=True,
                 normalization_type="RMSPre", use_normalization_before_and_after=True,
                 positional_embedding_type="rotary", rotary_dim=16, attn_scores_soft_cap=50.0)


@pytest.fixture(scope="module")
def tiny_llama():
    """Llama style: gated SiLU MLP, RMSNorm, RoPE, GQA."""
    return _tiny(n_key_value_heads=2, act_fn="silu", gated_mlp=True,
                 normalization_type="RMSPre", positional_embedding_type="rotary", rotary_dim=16)


@pytest.fixture(scope="module")
def tiny_parallel():
    """GPT-J / Pythia style: parallel attention+MLP, standard GELU MLP, LayerNorm.

    Built with LN and then folded / centered the way TransformerLens loads
    pretrained models (ACC++'s head decomposition assumes centered writes).
    """
    m = _tiny(act_fn="gelu", normalization_type="LN", parallel_attn_mlp=True,
              positional_embedding_type="rotary", rotary_dim=16)
    m.process_weights_(fold_ln=True, center_writing_weights=True, center_unembed=True)
    m.setup()
    return m


def _run(model, n_tok=12, seed=0):
    torch.manual_seed(seed)
    tokens = torch.randint(0, model.cfg.d_vocab, (1, n_tok))
    logits, cache = model.run_with_cache(tokens)
    return tokens, logits, cache


# ---------------------------------------------------------------------------
# Real-model fixtures used by several tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def report_setup(gpt2_model, gpt2_tracer):
    tokens = gpt2_model.to_tokens(REPORT_PROMPT, prepend_bos=False)
    _, cache = gpt2_model.run_with_cache(tokens)
    G = gpt2_tracer.trace(REPORT_PROMPT, answer_token=" Mary", wrong_token=" John",
                          prepend_bos=False, seeding="linear")
    return tokens, cache, G


@pytest.fixture(scope="module")
def gpt2_per_reader(gpt2_tracer):
    return gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", wrong_token=" John",
                             seeding="linear", mlp_expansion=MLPExpansionOptions(mode="per_reader"))


@pytest.fixture(scope="module")
def gpt2_path_summed(gpt2_tracer):
    return gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", wrong_token=" John",
                             seeding="linear", mlp_expansion=True)


@pytest.fixture(scope="module")
def gpt2_ps_deep(gpt2_tracer):
    """Path-summed with the bias rule off and diagnostics on: deeper MLP DAGs."""
    return gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", wrong_token=" John", seeding="linear",
                             mlp_expansion=MLPExpansionOptions(bias_overshoot_leaf=False, diagnostics=True))


@pytest.fixture(scope="module")
def gpt2_pr_deep(gpt2_tracer):
    return gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", wrong_token=" John", seeding="linear",
                             mlp_expansion=MLPExpansionOptions(mode="per_reader", bias_overshoot_leaf=False))


@pytest.fixture(scope="module")
def gpt2_merged(gpt2_tracer):
    return gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", wrong_token=" John",
                             seeding="linear",
                             mlp_expansion=MLPExpansionOptions(mode="merged"))


@pytest.fixture(scope="module")
def gpt2_prob_per_reader(gpt2_tracer):
    return gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary",
                             mlp_expansion=MLPExpansionOptions(mode="per_reader", diagnostics=True))


def _mlp_nodes(G):
    return [n for n in G.nodes if is_mlp_node(n)]


def _decisions(G):
    return {n: G.nodes[n]["mlp_trace"]["decision"] for n in _mlp_nodes(G) if "mlp_trace" in G.nodes[n]}


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def test_spec_real_models(gpt2_model, pythia_model, qwen_model):
    s = get_mlp_spec(gpt2_model)
    assert (s.kind, s.post_norm, s.parallel, s.center) == ("standard", False, False, True)
    s = get_mlp_spec(pythia_model)
    assert (s.kind, s.parallel, s.center) == ("standard", True, True)
    s = get_mlp_spec(qwen_model)
    assert (s.kind, s.act_fn, s.post_norm, s.center) == ("gated", "silu", False, False)


def test_spec_synthetic(tiny_gemma, tiny_llama, tiny_parallel):
    s = get_mlp_spec(tiny_gemma)
    assert (s.kind, s.post_norm, s.center) == ("gated", True, False)
    assert get_mlp_spec(tiny_llama).kind == "gated"
    assert get_mlp_spec(tiny_parallel).parallel


def test_spec_rejects_unsupported():
    with pytest.raises(ValueError, match="layer-norm activations"):
        get_mlp_spec(_tiny(act_fn="solu_ln", normalization_type="LNPre"))
    with pytest.raises(ValueError, match="fold_ln"):
        get_mlp_spec(_tiny(act_fn="gelu", normalization_type="LN"))
    with pytest.raises(ValueError, match="no MLPs"):
        get_mlp_spec(_tiny(act_fn="gelu", normalization_type="LNPre", attn_only=True))
    fake = SimpleNamespace(cfg=SimpleNamespace(model_name="x", attn_only=False, num_experts=8,
                                               is_layer_norm_activation=lambda: False))
    with pytest.raises(ValueError, match="mixture-of-experts"):
        get_mlp_spec(fake)


def test_spec_rejects_post_layernorm(tiny_gemma):
    class LayerNorm(torch.nn.Module):
        pass
    fake = SimpleNamespace(cfg=tiny_gemma.cfg,
                           blocks=[SimpleNamespace(ln2=tiny_gemma.blocks[0].ln2, ln2_post=LayerNorm())])
    with pytest.raises(ValueError, match="Post-MLP normalization"):
        get_mlp_spec(fake)


def test_options_validation():
    with pytest.raises(ValueError):
        MLPExpansionOptions(mode="union")
    with pytest.raises(ValueError):
        MLPExpansionOptions(eps=0.3, eps_soft=0.2)
    with pytest.raises(ValueError):
        MLPExpansionOptions(tau=-1)
    with pytest.raises(ValueError):
        MLPExpansionOptions(max_mlp_nodes=0)
    with pytest.raises(ValueError):
        MLPExpansionOptions(k_max=0)
    with pytest.raises(ValueError):
        MLPExpansionOptions(quad_points=1)


# ---------------------------------------------------------------------------
# Mathematical invariants
# ---------------------------------------------------------------------------

def _models_and_caches(request):
    out = []
    for name in ("gpt2_model", "pythia_model", "qwen_model"):
        m = request.getfixturevalue(name)
        tokens = m.to_tokens(IOI_PROMPT)
        _, cache = m.run_with_cache(tokens)
        out.append((name, m, cache, tokens.shape[1]))
    for name in ("tiny_gemma", "tiny_llama", "tiny_parallel"):
        m = request.getfixturevalue(name)
        tokens, _, cache = _run(m)
        out.append((name, m, cache, tokens.shape[1]))
    return out


@pytest.fixture(scope="module")
def all_models(request):
    return _models_and_caches(request)


def _layers(m):
    L = m.cfg.n_layers
    return sorted({0, 1, L // 2, L - 1})


def test_decomposition_reconstructs_ln2_input(all_models):
    for name, m, cache, T in all_models:
        from accpp_tracer.models import get_model_config
        config = get_model_config(m)
        spec = get_mlp_spec(m)
        H = m.cfg.n_heads
        for layer in _layers(m):
            for pos in (0, T // 2, T - 1):
                ids, O, z = mlp_input_components(m, config, spec, cache, 0, layer, pos)
                Od = O.dense()
                assert torch.allclose(Od.sum(0), z, atol=1e-4, rtol=1e-4), (name, layer, pos)
                assert torch.allclose(O.sum(), z, atol=1e-4, rtol=1e-4)
                assert len(ids) == O.shape[0] == Od.shape[0] == len(set(ids))
                # factored products and rows agree with the dense matrix
                M = torch.randn(m.cfg.d_model, 3, generator=torch.Generator().manual_seed(layer))
                assert torch.allclose(O @ M, Od @ M, atol=1e-4, rtol=1e-3)
                for i in (0, len(ids) // 2, len(ids) - 1):
                    assert torch.allclose(O[i], Od[i], atol=1e-5, rtol=1e-4)
                n_attn = layer if spec.parallel else layer + 1
                assert len(ids) == 1 + n_attn * (H * (pos + 1) + 1) + layer
                if spec.parallel:   # MLP m cannot read layer m's attention
                    assert not any(l == layer and (c < H or c == H + 1) for l, c, _ in ids)


def test_site_matches_hook_and_jbar_is_complete(all_models):
    for name, m, cache, T in all_models:
        from accpp_tracer.models import get_model_config
        config, spec = get_model_config(m), get_mlp_spec(m)
        for layer in _layers(m):
            pos = T - 1
            _, _, z = mlp_input_components(m, config, spec, cache, 0, layer, pos)
            site = MLPSite(m, spec, layer, z, quad_points=64)
            ref = cache[f"blocks.{layer}.hook_mlp_out"][0, pos]
            assert torch.allclose(site.Fz, ref, atol=1e-4, rtol=1e-3), (name, layer)
            tol = 1e-5 if site._mode == "closed" else 2e-3
            assert site.completeness_err < tol, (name, layer, site._mode, site.completeness_err)


def test_jbar_transpose_is_adjoint(all_models):
    g = torch.Generator().manual_seed(0)
    for name, m, cache, T in all_models:
        from accpp_tracer.models import get_model_config
        config, spec = get_model_config(m), get_mlp_spec(m)
        _, _, z = mlp_input_components(m, config, spec, cache, 0, 1, T - 1)
        site = MLPSite(m, spec, 1, z)
        D = m.cfg.d_model
        V, Wv = torch.randn(D, 3, generator=g), torch.randn(D, 2, generator=g)
        lhs = Wv.T @ site.jbar(V)
        rhs = (site.jbar_T(Wv)).T @ V
        assert torch.allclose(lhs, rhs, rtol=1e-3, atol=1e-3 * lhs.abs().max().item()), name


def test_jbar_matches_numerical_path_average(tiny_gemma, tiny_llama, tiny_parallel):
    """J_bar v against a dense midpoint average of autograd JVPs along the ray."""
    from accpp_tracer.models import get_model_config
    g = torch.Generator().manual_seed(3)
    for m in (tiny_gemma, tiny_llama, tiny_parallel):
        tokens, _, cache = _run(m)
        config, spec = get_model_config(m), get_mlp_spec(m)
        _, _, z = mlp_input_components(m, config, spec, cache, 0, 2, 5)
        site = MLPSite(m, spec, 2, z)
        v = torch.randn(m.cfg.d_model, generator=g)
        n = 400
        acc = torch.zeros_like(v)
        with torch.enable_grad():
            for a in ((torch.arange(n) + 0.5) / n).tolist():
                _, jv = torch.autograd.functional.jvp(site.forward, (a * z).detach(), v)
                acc += jv / n
        got = site.jbar(v.unsqueeze(1))[:, 0]
        assert (got - acc).norm() / acc.norm() < 1e-3, m.cfg.act_fn


def test_attributions_complete_and_shares_sum_to_one(report_setup, gpt2_tracer):
    _, cache, _ = report_setup
    tr = gpt2_tracer.trace_mlp_edge(cache, 7, 13, 9, 6, "d", [6, 1, 17, 60, 34])
    assert torch.allclose(tr.gamma.sum(0), tr.t, atol=1e-4 * tr.t.norm().item())
    assert abs(tr.shares.sum().item() - 1.0) < 1e-4
    assert len(set(tr.selected)) == len(tr.selected)


def test_head_reader_matches_rotated_normalized_signal(request):
    """W^T x equals the coordinates of extract_edge_signal(flavor='rotated_normalized')."""
    from accpp_tracer.mlp import head_reader_operator
    cases = [("gpt2_model", "gpt2_tracer", "gpt2_prob_graph"),
             ("pythia_model", "pythia_tracer", "pythia_topk_graph"),
             ("qwen_model", "qwen_tracer", "qwen_prob_graph")]
    n_checked = 0
    for mname, tname, gname in cases:
        m, tr, G = (request.getfixturevalue(x) for x in (mname, tname, gname))
        tokens = m.to_tokens(IOI_PROMPT)
        _, cache = m.run_with_cache(tokens)
        idx_to_token = build_idx_to_token(m, tokens)
        tok2idx = {v: k for k, v in idx_to_token.items()}
        spec = get_mlp_spec(m)
        for u, v, d in G.edges(data=True):
            if not (len(u) == 4 and u[1] == "MLP" and len(v) == 4 and isinstance(v[1], int)):
                continue
            svs = eval(d["svs_used"])
            et = d["type"]
            pos = tok2idx[u[2]]
            dpos, spos = tok2idx[v[2]], tok2idx[v[3]]
            sig = tr.extract_edge_signal(cache, 0, v[0], v[1], dpos, spos, u[0], m.cfg.n_heads,
                                         pos, pos, et, svs, flavor="rotated_normalized")
            B = tr.U[v[0], v[1]][:, svs] if et == "d" else tr.VT[v[0], v[1]][svs].T
            for center in (False, spec.center):   # centering is a no-op on TL-loaded LN models
                WT = head_reader_operator(m, tr.config, cache, 0, tr.U, tr.VT, tr.W_Q_pinv, tr.W_K_pinv,
                                          tr._rotation_stack(pos + 1), v[0], v[1], et, svs, pos, center)
                x = cache[f"blocks.{u[0]}.hook_mlp_out"][0, pos]
                assert torch.allclose(WT.T @ x, B.T @ sig, atol=1e-4, rtol=1e-3), (mname, u, v, center)
            n_checked += 1
    assert n_checked > 5


def test_root_reader_prob_is_exact_for_ig(gpt2_model, gpt2_tracer, gpt2_ioi):
    """IG seed attributions are exactly linear in r_c along the prob root reader."""
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    tgt = [gpt2_model.to_single_token(" Mary")]
    r_c = _compute_residual_shares(gpt2_model, gpt2_tracer.config, cache, 0, end, "cpu")
    W_U, b_U = gpt2_model.W_U, gpt2_model.b_U
    z_clean = cache["ln_final.hook_normalized"][0, end] @ W_U
    sup = torch.tensor(tgt)
    p = torch.softmax(z_clean + b_U, -1)[sup]
    A = _ig_attribution_seeds(r_c, z_clean, b_U, sup, p / p.sum(), W_U, 64)
    reader = root_reader_prob(gpt2_model, cache, 0, tgt, end, None, 64, center=False)
    raw = r_c * cache["ln_final.hook_scale"][0, end]          # undo ln_final: raw outputs
    lin = raw @ reader
    mask = A.abs() > 1e-4 * A.abs().max()
    ratio = (A[mask] / lin[mask])
    assert (ratio.max() - ratio.min()).abs() < 1e-3 * ratio.abs().mean()


# ---------------------------------------------------------------------------
# Parity with the research code (report Table 2)
# ---------------------------------------------------------------------------

TABLE2 = ["MLP 6", "AH 7.9 pos=9", "MLP 4", "MLP 0", "AH 4.7 pos=13", "AH 5.10 pos=13", "MLP 2",
          "AH 2.8 pos=10", "MLP 3", "AH 5.11 pos=0", "AH 1.0 pos=10", "AH 7.3 pos=9", "b_O L6",
          "AH 7.9 pos=8", "AH 6.11 pos=10", "AH 1.10 pos=12", "AH 4.3 pos=12", "AH 3.9 pos=10",
          "MLP 1", "AH 1.10 pos=7", "AH 1.10 pos=10", "AH 1.10 pos=13"]


def _label(cid, H):
    l, c, s = cid
    if c < H:
        return f"AH {l}.{c} pos={s}"
    return {H: f"MLP {l}", H + 1: f"b_O L{l}", H + 2: "Embed"}[c]


def test_parity_edge_a(report_setup, gpt2_tracer):
    _, cache, G = report_setup
    edge = [(u, v, d) for u, v, d in G.edges(data=True)
            if u[:2] == (7, "MLP") and v[:2] == (9, 6) and d["type"] == "d"]
    assert len(edge) == 1
    svs = eval(edge[0][2]["svs_used"])
    tr = gpt2_tracer.trace_mlp_edge(cache, 7, 13, 9, 6, "d", svs)
    assert [_label(tr.ids[i], 12) for i in tr.selected] == TABLE2
    assert abs(tr.R_sig - 0.0895) < 1e-3
    assert abs(tr.shares[tr.selected[0]].item() - 0.148) < 1e-3   # MLP 6's share


def test_trace_mlp_edge_validation(report_setup, gpt2_tracer):
    _, cache, _ = report_setup
    with pytest.raises(ValueError):
        gpt2_tracer.trace_mlp_edge(cache, 9, 13, 9, 6, "d", [0])
    with pytest.raises(ValueError):
        gpt2_tracer.trace_mlp_edge(cache, 7, 13, 9, 6, "d", [])
    with pytest.raises(ValueError):
        gpt2_tracer.trace_mlp_edge(cache, 7, 13, 9, 6, "q", [0])


def test_trace_mlp_edge_key_side_and_position_zero(report_setup, gpt2_tracer):
    _, cache, _ = report_setup
    tr = gpt2_tracer.trace_mlp_edge(cache, 2, 0, 5, 9, "s", [0, 1])     # Edge B side, pos 0
    assert tr.n_candidates == 1 + 3 * (12 * 1 + 1) + 2
    assert tr.R_sig >= 0
    tr2 = gpt2_tracer.trace_mlp_edge(cache, 7, 13, 9, 6, "d", [6, 1], k_max=3)
    assert len(tr2.selected) <= 3


def test_algorithm1_eps_monotone(report_setup, gpt2_tracer):
    _, cache, _ = report_setup
    sizes = [len(gpt2_tracer.trace_mlp_edge(cache, 7, 13, 9, 6, "d", [6, 1, 17, 60, 34],
                                            eps=e).selected) for e in (0.5, 0.2, 0.1)]
    assert sizes == sorted(sizes)


# ---------------------------------------------------------------------------
# Expansion: off is unchanged
# ---------------------------------------------------------------------------

def test_expansion_off_is_unchanged(gpt2_tracer, gpt2_linear_graph):
    G = gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", wrong_token=" John",
                          seeding="linear", mlp_expansion=False)
    assert graph_to_jsonable(G) == graph_to_jsonable(gpt2_linear_graph)


def test_mlp_expansion_type_error(gpt2_tracer):
    with pytest.raises(TypeError):
        gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", mlp_expansion="yes")


def test_expansion_keeps_original_head_edges(gpt2_linear_graph, gpt2_per_reader, gpt2_merged,
                                             gpt2_path_summed):
    def norm(n):
        return n[:4] if is_mlp_node(n) else n
    base = {(u, v, d["type"], d.get("svs_used")) for u, v, d in gpt2_linear_graph.edges(data=True)}
    for G in (gpt2_per_reader, gpt2_merged, gpt2_path_summed):
        got = {(norm(u), v, d["type"], d.get("svs_used")) for u, v, d in G.edges(data=True)}
        assert base <= got


# ---------------------------------------------------------------------------
# Per-reader expansion
# ---------------------------------------------------------------------------

def _valid_step(st):
    if st[0] in ("d", "s"):
        return len(st) == 5 and isinstance(st[1], int) and isinstance(st[2], int)
    if st[0] == "mlp":
        return len(st) == 2 and isinstance(st[1], int)
    return st[0] == "root"


def test_per_reader_structure(gpt2_per_reader):
    G = gpt2_per_reader
    mlps = _mlp_nodes(G)
    assert mlps and all(len(n) == 5 for n in mlps), "per-reader graphs have only MLP occurrences"
    for n in mlps:
        path = n[4]
        assert path and all(_valid_step(st) for st in path)
        outs = list(G.out_edges(n, data=True))
        assert len(outs) == 1, n
        (_, X, d), = outs
        last = path[-1]
        if last[0] in ("d", "s"):
            assert X == (last[1], last[2], last[3], last[4]) and d["type"] == last[0]
        elif last[0] == "mlp":
            assert is_mlp_node(X) and X[0] == last[1] and X[4] == path[:-1] and d["type"] == "m"
            assert n[0] < X[0]
        else:
            assert len(X) == 2
        rec = G.nodes[n]["mlp_trace"]
        n_in = sum(1 for _ in G.in_edges(n))
        if rec["decision"] in ("expand", "approximate"):
            assert n_in == rec["n_selected"]
            assert rec["R_sig"] <= 0.25 + 1e-9
        else:
            assert n_in == 0
    for u, v, d in G.edges(data=True):
        if d["type"] == "m":
            assert is_mlp_node(v)
            assert isinstance(d["rank"], int)


def test_per_reader_policy_and_pruning(gpt2_per_reader):
    G = gpt2_per_reader
    dec = _decisions(G)
    assert any(d == "expand" for d in dec.values())
    for n, d in dec.items():
        if n[0] == 0:
            assert d == "leaf (MLP 0)"
        if d.startswith("leaf (R_sig"):
            assert G.nodes[n]["mlp_trace"]["R_sig"] > 0.25
        if d == "leaf (bias overshoot)":
            assert G.nodes[n]["mlp_trace"]["R_ideal"] > 1
    for u, v, d in G.edges(data=True):
        if d.get("child_decision") == "pruned":
            assert abs(d["traced_share_linear"]) < 0.01 and abs(d["traced_share_real"]) < 0.01
            assert dec[u] == "pruned"
        elif d.get("child_decision", "").startswith("expanded (both"):
            assert abs(d["traced_share_linear"]) >= 0.01 and abs(d["traced_share_real"]) >= 0.01


def test_per_reader_deterministic(gpt2_tracer, gpt2_per_reader):
    G2 = gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", wrong_token=" John",
                           seeding="linear", mlp_expansion=MLPExpansionOptions(mode="per_reader"))
    assert graph_to_jsonable(G2) == graph_to_jsonable(gpt2_per_reader)


def test_per_reader_root_reader_and_diagnostics(gpt2_prob_per_reader):
    G = gpt2_prob_per_reader
    roots = [n for n in _mlp_nodes(G) if n[4][0][0] == "root"]
    assert roots, "the prob root reads at least one MLP seed"
    assert any("R_sig" in G.nodes[n]["mlp_trace"] for n in roots)
    deep = [n for n in _mlp_nodes(G) if len(n[4]) > 1
            and G.nodes[n]["mlp_trace"]["decision"] not in ("pruned", "leaf (MLP 0)")]
    assert deep and all("R_chain" in G.nodes[n]["mlp_trace"] for n in deep)
    assert all(0 <= G.nodes[n]["mlp_trace"]["R_chain"] < 2 for n in deep)


def test_per_reader_options(gpt2_tracer):
    kw = dict(answer_token=" Mary", wrong_token=" John", seeding="linear")
    G = gpt2_tracer.trace(IOI_PROMPT, **kw, mlp_expansion=MLPExpansionOptions(
        mode="per_reader", bias_overshoot_leaf=False, expand_seed_mlps=False, tau=0.0))
    dec = _decisions(G)
    assert "leaf (bias overshoot)" not in dec.values()
    assert "pruned" not in dec.values()
    for n, d in dec.items():
        if n[4][0][0] == "root":
            assert d == "leaf (no reader operator)"
    G = gpt2_tracer.trace(IOI_PROMPT, **kw, mlp_expansion=MLPExpansionOptions(
        mode="per_reader", tau=10.0, joint_check=False))
    kids = [d for *_, d in G.edges(data=True) if "child_decision" in d]
    assert kids and all(d["child_decision"] == "pruned" for d in kids)


def test_per_reader_expand_mlp0_and_cap(gpt2_tracer):
    kw = dict(answer_token=" Mary", wrong_token=" John", seeding="linear")
    G = gpt2_tracer.trace(IOI_PROMPT, **kw, mlp_expansion=MLPExpansionOptions(
        mode="per_reader", expand_mlp0=True, max_mlp_nodes=5))
    dec = _decisions(G)
    assert "leaf (MLP 0)" not in dec.values()
    assert sum("R_sig" in G.nodes[n]["mlp_trace"] for n in dec) <= 5
    with pytest.warns(UserWarning, match="max_mlp_nodes"):
        gpt2_tracer.trace(IOI_PROMPT, **kw, mlp_expansion=MLPExpansionOptions(mode="per_reader",
                                                                              max_mlp_nodes=1))


@pytest.mark.parametrize("mode", ["per_reader", "path_summed"])
def test_per_reader_signals(gpt2_tracer, gpt2_model, mode):
    G = gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", wrong_token=" John",
                          seeding="linear", signals="raw", mlp_expansion=MLPExpansionOptions(mode=mode))
    m_edges = [d for *_, d in G.edges(data=True) if d["type"] == "m"]
    assert m_edges and all(d["signal"].shape == (gpt2_model.cfg.d_model,) for d in m_edges)
    assert all(d["signal_flavor"] == "mlp_row_space" for d in m_edges)
    head_edges = [d for u, v, d in G.edges(data=True) if d["type"] in ("d", "s") and len(v) == 4]
    assert all(d["signal_flavor"] == "raw" for d in head_edges)


def test_k_max_option(gpt2_tracer):
    G = gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", wrong_token=" John", seeding="linear",
                          mlp_expansion=MLPExpansionOptions(k_max=2))
    for n in _mlp_nodes(G):
        rec = G.nodes[n].get("mlp_trace", {})
        if "n_selected" in rec:
            assert rec["n_selected"] <= 2


# ---------------------------------------------------------------------------
# Merged expansion
# ---------------------------------------------------------------------------

def test_merged_structure(gpt2_merged):
    G = gpt2_merged
    mlps = _mlp_nodes(G)
    assert mlps and all(len(n) == 4 for n in mlps)
    for n in mlps:
        rec = G.nodes[n]["mlp_trace"]
        readers = [eval(r) for r in rec["readers"]]
        outs = list(G.out_edges(n, data=True))
        assert len(readers) == len(outs)
        assert rec["decision_kind"] in ("expand", "approximate", "leaf", "pruned")
        if rec["decision"] in ("expand", "approximate"):
            assert sum(1 for _ in G.in_edges(n)) == rec["n_selected"]
            n_rows = 0
            for _, X, d in outs:
                if d["type"] == "m" and d.get("child_decision") == "pruned":
                    continue          # a parent that pruned it does not shape its objective
                if d["type"] == "m":
                    n_rows += G.nodes[X]["mlp_trace"]["n_reader_rows"]
                elif len(X) == 4:
                    n_rows += len(eval(d["svs_used"]))
                else:
                    n_rows += 1
            assert rec["n_reader_rows"] == n_rows
    for u, v, d in G.edges(data=True):
        if d["type"] == "m" and is_mlp_node(u):
            assert u[0] < v[0]


def test_merged_vs_per_reader_single_reader_agree(gpt2_per_reader, gpt2_merged):
    """An MLP with one head reader traces the same set in both modes."""
    Gm, Gp = gpt2_merged, gpt2_per_reader
    n_cmp = 0
    for n in _mlp_nodes(Gm):
        outs = list(Gm.out_edges(n, data=True))
        rec = Gm.nodes[n]["mlp_trace"]
        if len(outs) != 1 or outs[0][2]["type"] not in ("d", "s") or "n_selected" not in rec:
            continue
        (_, X, d), = outs
        occ = n + (((d["type"],) + tuple(X),),)
        if occ in Gp and "n_selected" in Gp.nodes[occ]["mlp_trace"]:
            a = {(repr(u[:4]), dd["rank"]) for u, _, dd in Gm.in_edges(n, data=True)}
            b = {(repr(u[:4]), dd["rank"]) for u, _, dd in Gp.in_edges(occ, data=True)}
            assert a == b, n
            n_cmp += 1
    assert n_cmp > 0


def test_merged_probe(gpt2_model, gpt2_tracer, gpt2_ioi):
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    idx_to_token = build_idx_to_token(gpt2_model, tokens)
    direction = gpt2_model.W_U[:, gpt2_model.to_single_token(" Mary")]
    for mode in ("merged", "per_reader", "path_summed"):
        G = gpt2_tracer.trace_from_probe(cache, direction, 9, end, idx_to_token, ("Probe", idx_to_token[end]),
                                         mlp_expansion=MLPExpansionOptions(mode=mode))
        assert G.number_of_edges() > 0
        assert all("mlp_trace" in G.nodes[n] for n in _mlp_nodes(G))


def test_level2_requires_unique_labels(gpt2_model, gpt2_tracer, gpt2_ioi):
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    labels = {i: gpt2_model.tokenizer.decode(tokens[0, i]) for i in range(end + 1)}   # duplicates
    assert len(set(labels.values())) < len(labels)
    direction = (gpt2_model.W_U[:, gpt2_model.to_single_token(" Mary")]
                 - gpt2_model.W_U[:, gpt2_model.to_single_token(" John")])
    with pytest.raises(ValueError, match="unique token labels"):
        gpt2_tracer.trace_from_cache(cache, direction, end, labels, ("Logit", labels[end]),
                                     mlp_expansion=True)


# ---------------------------------------------------------------------------
# Synthetic architectures: full pipeline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ["tiny_gemma", "tiny_llama", "tiny_parallel"])
@pytest.mark.parametrize("mode", ["path_summed", "per_reader", "merged"])
def test_synthetic_pipeline(request, fixture, mode):
    m = request.getfixturevalue(fixture)
    tracer = Tracer(m, device="cpu")
    tokens, logits, cache = _run(m)
    end = tokens.shape[1] - 1
    idx_to_token = {i: f"t{i}" for i in range(end + 1)}
    direction = m.W_U[:, int(logits[0, end].argmax())].detach()
    G = tracer.trace_from_cache(cache, direction, end, idx_to_token, ("Logit direction", f"t{end}"),
                                attn_weight_thresh=0.05,
                                mlp_expansion=MLPExpansionOptions(mode=mode, expand_mlp0=True,
                                                                  diagnostics=True))
    for n in _mlp_nodes(G):
        rec = G.nodes[n]["mlp_trace"]
        if "completeness_err" in rec:
            assert rec["completeness_err"] < 5e-3
    # Level-1 on a key-side and a query-side edge of the last layer.
    for et in ("d", "s"):
        tr = tracer.trace_mlp_edge(cache, 1, 4, 2, 0, et, [0, 1])
        assert torch.allclose(tr.gamma.sum(0), tr.t, atol=5e-3 * max(tr.t.norm().item(), 1e-6))


# ---------------------------------------------------------------------------
# Graph utilities on expanded graphs
# ---------------------------------------------------------------------------

def test_edges_from_graph_and_intervention(gpt2_model, gpt2_tracer, gpt2_ioi, gpt2_per_reader):
    tokens, logits, cache = gpt2_ioi
    idx_to_token = build_idx_to_token(gpt2_model, tokens)
    tok2idx = {v: k for k, v in idx_to_token.items()}
    specs = edges_from_graph(gpt2_per_reader, tok2idx, gpt2_model.cfg.n_heads)
    n_head_edges = sum(1 for u, v, d in gpt2_per_reader.edges(data=True)
                       if d["type"] in ("d", "s") and len(v) == 4 and isinstance(v[1], int))
    assert len(specs) == n_head_edges
    mlp_specs = [s for s in specs if s.upstream_component_id == gpt2_model.cfg.n_heads]
    assert mlp_specs
    res = gpt2_tracer.run_intervention(tokens, cache, logits, mlp_specs[0], intervention_type="local")
    assert res.attn_scores_interv.shape == (1,)


def test_combine_prompt_graphs(gpt2_per_reader, gpt2_linear_graph):
    # Backward compatible on plain graphs: two copies average to the original.
    C = combine_prompt_graphs([gpt2_linear_graph, gpt2_linear_graph])
    assert C.number_of_edges() == gpt2_linear_graph.number_of_edges()
    for u, v, k, d in C.edges(keys=True, data=True):
        assert d["prompts_appeared"] == [0, 1]
    # Expanded graphs: occurrences kept, or collapsed to MLP nodes.
    C2 = combine_prompt_graphs([gpt2_per_reader, None, gpt2_per_reader])
    assert C2.number_of_edges() == gpt2_per_reader.number_of_edges()
    assert all(C2.nodes[n]["n_appearences"] == 2 for n in C2.nodes)
    C3 = combine_prompt_graphs([gpt2_per_reader], collapse_mlp_occurrences=True)
    assert not any(isinstance(n, tuple) and len(n) == 5 for n in C3.nodes)
    tot = lambda G, t: sum(d["weight"] for *_, d in G.edges(data=True) if d["type"] == t)
    for t in ("d", "s", "m"):
        assert abs(tot(C3, t) - tot(gpt2_per_reader, t)) < 1e-6 * max(1.0, abs(tot(gpt2_per_reader, t)))
    for u, v, k in C3.edges(keys=True):
        assert sum(1 for kk in C3[u][v] if C3[u][v][kk]["type"] == C3.edges[u, v, k]["type"]) == 1


def test_html_and_cytoscape_export(tmp_path, gpt2_model, gpt2_per_reader, gpt2_merged, gpt2_path_summed):
    tokens = gpt2_model.to_tokens(IOI_PROMPT)
    labels = list(build_idx_to_token(gpt2_model, tokens).values())
    for G in (gpt2_per_reader, gpt2_merged, gpt2_path_summed):
        out = circuit_to_html(G, labels, gpt2_model.cfg.n_layers, tmp_path / "c.html")
        html = out.read_text()
        assert "MLP-input edge (m)" in html and "__LEGEND_M__" not in html
    data = json.loads(html.split("const DATA = ")[1].split(";\n")[0])
    assert any(n.get("mlp") for n in data["nodes"])
    G = gpt2_per_reader
    root = [n for n in G.nodes if len(n) == 2][0]
    Gs = nx.relabel_nodes(G.copy(), {n: str(n) for n in G.nodes})
    out = format_graph_cytoscape_by_token_pos(Gs, str(root), gpt2_model.cfg.n_layers,
                                              gpt2_model.cfg.n_heads, labels)
    assert out.number_of_nodes() == Gs.number_of_nodes()   # occurrence labels stay unique


# ---------------------------------------------------------------------------
# Goldens
# ---------------------------------------------------------------------------

def test_golden_gpt2_linear_per_reader(gpt2_per_reader, update_golden):
    assert_graph_matches(gpt2_per_reader, GOLDEN_DIR / "graph_gpt2_linear_mlp_per_reader.json",
                         update_golden)


def test_golden_gpt2_linear_merged(gpt2_merged, update_golden):
    assert_graph_matches(gpt2_merged, GOLDEN_DIR / "graph_gpt2_linear_mlp_merged.json", update_golden)


def test_golden_qwen_prob_per_reader(qwen_tracer, update_golden):
    G = qwen_tracer.trace(IOI_PROMPT, answer_token=" Mary", mlp_expansion=MLPExpansionOptions(mode="per_reader"))
    assert any(G.nodes[n]["mlp_trace"]["decision"] == "expand" for n in _mlp_nodes(G))
    assert_graph_matches(G, GOLDEN_DIR / "graph_qwen_prob_mlp_per_reader.json", update_golden)


# ---------------------------------------------------------------------------
# Additional edge cases (from code review)
# ---------------------------------------------------------------------------

def test_site_matches_tl_off_trajectory(all_models):
    """MLPSite.forward equals TransformerLens's own apply_mlp away from the clean input."""
    from accpp_tracer.models import get_model_config
    g = torch.Generator().manual_seed(7)
    for name, m, cache, T in all_models:
        config, spec = get_model_config(m), get_mlp_spec(m)
        layer = m.cfg.n_layers - 1
        _, _, z = mlp_input_components(m, config, spec, cache, 0, layer, T - 1)
        site = MLPSite(m, spec, layer, z)
        for scale in (0.3, 1.0, 2.0):
            zp = z * scale + 0.2 * z.norm() / m.cfg.d_model ** 0.5 * torch.randn(z.shape, generator=g)
            ref = m.blocks[layer].apply_mlp(zp[None, None, :])[0, 0]
            assert torch.allclose(site.forward(zp), ref, atol=1e-4, rtol=1e-3), (name, scale)


def test_real_chain_matches_manual_two_hop(report_setup, gpt2_model, gpt2_tracer):
    """_removed_real through MLP 6 -> MLP 7 equals a hand-built two-hop pass with TL modules."""
    from accpp_tracer.mlp_expand import MLPExpander
    _, cache, _ = report_setup
    idx = {i: str(i) for i in range(14)}
    ex = MLPExpander(gpt2_tracer, nx.MultiDiGraph(), {}, cache, idx, 0, "dynamic", None, {},
                     MLPExpansionOptions())
    r7, r6 = ex.site(7, 13), ex.site(6, 13)
    WT = torch.randn(768, 3, generator=torch.Generator().manual_seed(0))
    sig = 0.05 * torch.randn(768, generator=torch.Generator().manual_seed(1))
    got = ex._removed_real(r6, sig, [r7], WT)
    blk6, blk7 = gpt2_model.blocks[6], gpt2_model.blocks[7]
    dF6 = blk6.apply_mlp((r6.z - sig)[None, None])[0, 0] - blk6.apply_mlp(r6.z[None, None])[0, 0]
    s7 = cache["blocks.7.ln2.hook_scale"][0, 13]
    dF6c = dF6 - dF6.mean()
    dF7 = blk7.apply_mlp((r7.z + dF6c / s7)[None, None])[0, 0] - blk7.apply_mlp(r7.z[None, None])[0, 0]
    assert torch.allclose(got, -(WT.T @ dF7), atol=1e-4, rtol=1e-3)
    # one hop: equals the plain removal through this MLP
    one = ex._removed_real(r7, sig, [], WT)
    assert torch.allclose(one, -(WT.T @ (r7.site.forward(r7.z - sig) - r7.site.Fz)), atol=1e-5)


def test_prune_and_joint_check_unit(report_setup, gpt2_tracer):
    from accpp_tracer.mlp_expand import MLPExpander
    _, cache, _ = report_setup
    ex = MLPExpander(gpt2_tracer, nx.MultiDiGraph(), {}, cache, {i: str(i) for i in range(14)}, 0,
                     "dynamic", None, {}, MLPExpansionOptions())
    one = torch.ones(3)

    def kids():
        return [dict(lin=0.5, real=0.5, sig=0 * one), dict(lin=0.004, real=0.003, sig=1 * one),
                dict(lin=0.006, real=0.001, sig=2 * one), dict(lin=0.02, real=0.0, sig=4 * one),
                dict(lin=0.0, real=0.03, sig=8 * one)]
    k = kids()
    ex._prune(k, 0.01, joint=lambda s: 0.0)
    assert [x["status"] for x in k] == ["expanded (both)", "pruned", "pruned",
                                        "expanded (linear)", "expanded (real)"]
    # Joint removal of both small siblings is large -> restore the larger one first, then stop.
    calls = []

    def joint(s):
        calls.append(s[0].item())
        return 0.02 if s[0].item() > 2.5 else 0.0
    k = kids()
    ex._prune(k, 0.01, joint)
    assert [x["status"] for x in k][1:3] == ["pruned", "expanded (joint check)"]
    assert calls == [3.0, 1.0]
    k = kids()
    ex._prune(k, 0.01, joint=lambda s: 1.0)       # never small: restores all, stops when none left
    assert [x["status"] for x in k][1:3] == ["expanded (joint check)", "expanded (joint check)"]


def test_partial_idx_to_token(gpt2_model, gpt2_tracer, gpt2_ioi):
    """Positions missing from idx_to_token are skipped, as _trace_recursive does."""
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    idx = build_idx_to_token(gpt2_model, tokens)
    del idx[0]                                    # drop BOS
    direction = (gpt2_model.W_U[:, gpt2_model.to_single_token(" Mary")]
                 - gpt2_model.W_U[:, gpt2_model.to_single_token(" John")])
    labels = set(idx.values())
    for mode in ("per_reader", "merged", "path_summed"):
        G = gpt2_tracer.trace_from_cache(cache, direction, end, idx, ("Logit", idx[end]),
                                         mlp_expansion=MLPExpansionOptions(mode=mode))
        assert G.number_of_edges() > 0
        for n in G.nodes:
            if len(n) >= 4:
                assert n[2] in labels and n[3] in labels
        assert any(G.nodes[n].get("mlp_trace", {}).get("n_unmapped") for n in _mlp_nodes(G))


def test_head_shaped_root_node(gpt2_model, gpt2_tracer, gpt2_ioi):
    tokens, _, cache = gpt2_ioi
    end = tokens.shape[1] - 1
    idx = build_idx_to_token(gpt2_model, tokens)
    direction = gpt2_model.W_U[:, gpt2_model.to_single_token(" Mary")]
    root = (2, 0, idx[end], "probe")             # looks like a head node
    G = gpt2_tracer.trace_from_probe(cache, direction, 9, end, idx, root, mlp_expansion=True)
    seeds = [n for n in _mlp_nodes(G) if n[4][0][0] == "root"]
    assert seeds and all(n[4][0][1:] == root for n in seeds)


def test_root_reader_probe_not_centered():
    from accpp_tracer.mlp_expand import root_reader_probe
    v = torch.arange(1.0, 6.0)
    r = root_reader_probe(v)
    assert torch.allclose(r, v / v.norm())


def test_multi_root_linear_seeding(gpt2_tracer, gpt2_model):
    kw = dict(answer_token=[" Mary", " John"], seeding="linear")
    Gp = gpt2_tracer.trace(IOI_PROMPT, **kw, mlp_expansion=True)
    roots = {n for n in Gp.nodes if len(n) == 2}
    assert len(roots) == 2
    seed_occ = [n for n in _mlp_nodes(Gp) if n[4][0][0] == "root"]
    by_base = {}
    for n in seed_occ:
        by_base.setdefault(n[:4], set()).add(n[4][0][1:])
    assert all(len(v) == len(set(v)) for v in by_base.values())
    labels = list(build_idx_to_token(gpt2_model, gpt2_model.to_tokens(IOI_PROMPT)).values())
    Gs = nx.relabel_nodes(Gp.copy(), {n: str(n) for n in Gp.nodes})
    root = str(sorted(roots)[0])
    n_before = Gs.number_of_nodes() + len(labels)
    out = format_graph_cytoscape_by_token_pos(Gs, root, 12, 12, labels)
    assert out.number_of_nodes() == n_before
    Gm = gpt2_tracer.trace(IOI_PROMPT, **kw, mlp_expansion=MLPExpansionOptions(mode="merged"))
    for n in _mlp_nodes(Gm):
        rec = Gm.nodes[n]["mlp_trace"]
        n_root_readers = sum(1 for r in rec["readers"] if eval(r)[0] == "root")
        n_root_edges = sum(1 for _, X in Gm.out_edges(n) if len(X) == 2)
        assert n_root_readers == n_root_edges


def test_merged_variants(gpt2_tracer, gpt2_model, gpt2_ioi, gpt2_merged):
    kw = dict(answer_token=" Mary", wrong_token=" John", seeding="linear")
    with pytest.warns(UserWarning, match="max_mlp_nodes"):
        G = gpt2_tracer.trace(IOI_PROMPT, **kw, mlp_expansion=MLPExpansionOptions(mode="merged", max_mlp_nodes=2))
    assert sum("R_sig" in G.nodes[n]["mlp_trace"] for n in _mlp_nodes(G)) <= 2
    G2 = gpt2_tracer.trace(IOI_PROMPT, **kw, mlp_expansion=MLPExpansionOptions(mode="merged"))
    assert graph_to_jsonable(G2) == graph_to_jsonable(gpt2_merged)          # deterministic
    G3 = gpt2_tracer.trace(IOI_PROMPT, **kw, signals="raw",
                           mlp_expansion=MLPExpansionOptions(mode="merged", expand_seed_mlps=False))
    assert all(d["signal"].shape == (768,) for *_, d in G3.edges(data=True) if d["type"] == "m")
    for n in _mlp_nodes(G3):
        outs = list(G3.out_edges(n))
        if outs and all(len(X) == 2 for _, X in outs):
            assert G3.nodes[n]["mlp_trace"]["decision"] == "leaf (no reader operator)"
    tokens, _, _ = gpt2_ioi
    tok2idx = {v: k for k, v in build_idx_to_token(gpt2_model, tokens).items()}
    specs = edges_from_graph(gpt2_merged, tok2idx, 12)
    assert any(s.upstream_component_id == 12 for s in specs)
    for *_, d in gpt2_merged.edges(data=True):
        assert "traced_share_linear" not in d     # merged shares are local and named so


def test_zero_signal_and_rank_deficient_reader(report_setup, gpt2_tracer, gpt2_model):
    from accpp_tracer.models import get_model_config
    _, cache, _ = report_setup
    spec = get_mlp_spec(gpt2_model)
    ids, O, z = mlp_input_components(gpt2_model, get_model_config(gpt2_model), spec, cache, 0, 7, 13)
    site = MLPSite(gpt2_model, spec, 7, z)
    tr0 = run_algorithm1(site, O, ids, torch.zeros(768, 1))
    assert tr0.selected == [] and tr0.R_sig == 0.0
    w = torch.randn(768, 1, generator=torch.Generator().manual_seed(2))
    a = run_algorithm1(site, O, ids, w)
    b = run_algorithm1(site, O, ids, torch.cat([w, w], dim=1) / 2 ** 0.5)   # same metric, rank 1
    assert a.selected == b.selected and abs(a.R_sig - b.R_sig) < 1e-4 and b.Q.shape[1] == 1


def test_post_norm_zero_bias_boundary_layer():
    """Gemma-2 regime (no biases, F_raw(0) = 0) with a large output: quadrature stays accurate."""
    from accpp_tracer.models import get_model_config
    m = _tiny(n_key_value_heads=2, act_fn="gelu_pytorch_tanh", gated_mlp=True,
              normalization_type="RMSPre", use_normalization_before_and_after=True,
              positional_embedding_type="rotary", rotary_dim=16)
    for blk in m.blocks:
        blk.mlp.b_in.data.zero_()
        blk.mlp.b_out.data.zero_()
        blk.mlp.W_out.data *= 100.0
    tokens, _, cache = _run(m)
    config, spec = get_model_config(m), get_mlp_spec(m)
    _, _, z = mlp_input_components(m, config, spec, cache, 0, 2, 7)
    site = MLPSite(m, spec, 2, z, quad_points=64)
    assert site.F0.abs().max() == 0
    assert site.completeness_err < 1e-4
    v = torch.randn(m.cfg.d_model, generator=torch.Generator().manual_seed(4))
    n = 4000          # dense midpoint average, with half the cells inside [0, 1e-3]
    ref = torch.zeros_like(v)
    with torch.enable_grad():
        edges = torch.cat([torch.zeros(1), torch.linspace(0, 1e-3, n // 2 + 1)[1:],
                           1e-3 + (1 - 1e-3) * torch.arange(1, n // 2 + 1) / (n // 2)])
        for a0, a1 in zip(edges[:-1].tolist(), edges[1:].tolist()):
            _, jv = torch.autograd.functional.jvp(site.forward, ((a0 + a1) / 2 * z).detach(), v)
            ref += (a1 - a0) * jv
    got = site.jbar(v.unsqueeze(1))[:, 0]
    assert (got - ref).norm() / ref.norm() < 1e-2


def test_golden_qwen_prob_merged(qwen_tracer, update_golden):
    G = qwen_tracer.trace(IOI_PROMPT, answer_token=" Mary",
                          mlp_expansion=MLPExpansionOptions(mode="merged"))
    assert_graph_matches(G, GOLDEN_DIR / "graph_qwen_prob_mlp_merged.json", update_golden)


@pytest.mark.parametrize("mode", ["path_summed", "per_reader"])
def test_prune_head_inputs_and_bos_leaf(gpt2_model, gpt2_tracer, mode):
    """Head inputs obey the same both-below-tau rule as MLP children; position 0 is a leaf."""
    G0 = gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", mlp_expansion=MLPExpansionOptions(mode=mode))
    G = gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", mlp_expansion=MLPExpansionOptions(
        mode=mode, prune_head_inputs=True, bos_leaf=True, tau=0.05))
    bos = build_idx_to_token(gpt2_model, gpt2_model.to_tokens(IOI_PROMPT))[0]
    n_pruned_heads = 0
    for u, v, d in G.edges(data=True):
        if d["type"] != "m" or not (len(u) == 4 and isinstance(u[1], int)):
            continue
        cd = d.get("child_decision")
        assert cd is not None                                 # every head input gets a decision
        if cd == "pruned":
            n_pruned_heads += 1
            assert abs(d["share_linear"]) < 0.05 and abs(d["share_real"]) < 0.05
        elif cd == "leaf (BOS)":
            assert u[3] == bos
    assert n_pruned_heads > 0
    for n in _mlp_nodes(G):
        if n[2] == bos:
            assert G.nodes[n]["mlp_trace"]["decision"] in ("leaf (BOS)", "leaf (MLP 0)")
            assert G.nodes[n]["mlp_trace"]["leaf_reason"] in ("bos", "mlp0")
    assert G.number_of_edges() < G0.number_of_edges()
    # defaults unchanged: no pruning decisions on head inputs without the option
    assert not any("child_decision" in d for u, v, d in G0.edges(data=True)
                   if d["type"] == "m" and len(u) == 4 and isinstance(u[1], int))


def test_prune_head_inputs_merged(gpt2_tracer):
    G = gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", mlp_expansion=MLPExpansionOptions(
        mode="merged", prune_head_inputs=True, bos_leaf=True, tau=0.05))
    heads_in = [d for u, v, d in G.edges(data=True) if d["type"] == "m" and len(u) == 4 and isinstance(u[1], int)]
    assert heads_in and all("child_decision" in d for d in heads_in)


# ---------------------------------------------------------------------------
# Path-summed expansion (the default)
# ---------------------------------------------------------------------------

def _ps_groups(G):
    """step -> {layer: node} for a path-summed graph."""
    groups = {}
    for n in _mlp_nodes(G):
        groups.setdefault(n[4][0], {})[n[0]] = n
    return groups


def _kept_parents(G, n):
    return [v for _, v, d in G.out_edges(n, data=True)
            if d["type"] == "m" and d.get("child_decision") != "pruned"]


def test_path_summed_is_default():
    assert MLPExpansionOptions().mode == "path_summed"


def test_path_summed_structure(gpt2_ps_deep):
    G = gpt2_ps_deep
    mlps = _mlp_nodes(G)
    assert mlps and all(len(n) == 5 and len(n[4]) == 1 and _valid_step(n[4][0]) for n in mlps)
    for n in mlps:
        rec = G.nodes[n]["mlp_trace"]
        direct = [(X, d) for _, X, d in G.out_edges(n, data=True) if d["type"] != "m"]
        assert len(direct) <= 1
        step = n[4][0]
        if direct:
            (X, d), = direct
            if step[0] in ("d", "s"):
                assert X == step[1:] and d["type"] == step[0]
            else:
                assert step == ("root",) + tuple(X)
        for _, v, d in G.out_edges(n, data=True):
            if d["type"] == "m":     # MLP -> MLP edges stay inside one reader's DAG and go upward
                assert is_mlp_node(v) and v[4] == n[4] and v[2] == n[2] and n[0] < v[0]
        n_in = sum(1 for _ in G.in_edges(n))
        if rec["decision"] in ("expand", "approximate"):
            assert n_in == rec["n_selected"] and rec["R_sig"] <= 0.25 + 1e-9
        elif rec["decision"] != "pruned":
            assert n_in == 0
        if "R_sig" in rec:          # n_paths = direct read + paths through kept parents
            want = int(bool(direct)) + sum(G.nodes[p]["mlp_trace"]["n_paths"] for p in _kept_parents(G, n))
            assert rec["n_paths"] == want, n
            assert rec["n_parents"] == len(_kept_parents(G, n))


def test_path_summed_fewer_traces(gpt2_ps_deep, gpt2_pr_deep):
    traced = lambda G: [n for n in _mlp_nodes(G) if "R_sig" in G.nodes[n]["mlp_trace"]]
    ps, pr = traced(gpt2_ps_deep), traced(gpt2_pr_deep)
    assert len({(n[4][0], n[0]) for n in ps}) == len(ps)        # one trace per (reader, MLP)
    assert len(ps) < len(pr)
    assert max(G.nodes[n]["mlp_trace"]["n_paths"] for G in (gpt2_ps_deep,) for n in ps) > 1


def test_path_summed_target_is_sum_over_paths(gpt2_ps_deep, gpt2_model, gpt2_tracer, gpt2_ioi):
    """Each traced MLP's target t equals the sum, over every kept path in its reader's DAG, of
    the path operator W_R J_top C/s ... J_parent C/s applied to F(z) - F(0) (built from scratch)."""
    from accpp_tracer.mlp import head_reader_operator
    from accpp_tracer.models import get_model_config
    G, m = gpt2_ps_deep, gpt2_model
    tokens, _, cache = gpt2_ioi
    tok2idx = {v: k for k, v in build_idx_to_token(m, tokens).items()}
    config, spec = get_model_config(m), get_mlp_spec(m)
    sites = {}

    def site(layer, pos):
        if (layer, pos) not in sites:
            _, _, z = mlp_input_components(m, config, spec, cache, 0, layer, pos)
            sites[(layer, pos)] = MLPSite(m, spec, layer, z)
        return sites[(layer, pos)]

    n_checked = n_multi = 0
    for step, nodes in _ps_groups(G).items():
        if step[0] not in ("d", "s"):
            continue
        X = step[1:]
        svs_of = {}
        for l, n in nodes.items():
            if G.nodes[n]["mlp_trace"].get("leaf_reason") in ("mlp0", "bos"):
                continue          # leaves read at conversion are not part of the reader's coordinates
            for _, v, d in G.out_edges(n, data=True):
                if d["type"] == step[0] and v == X:
                    svs_of[l] = eval(d["svs_used"])
        union = sorted({i for s in svs_of.values() for i in s})
        pos = tok2idx[nodes[next(iter(nodes))][2]]
        WU = head_reader_operator(m, config, cache, 0, gpt2_tracer.U, gpt2_tracer.VT, gpt2_tracer.W_Q_pinv,
                                  gpt2_tracer.W_K_pinv, None, X[0], X[1], step[0], union, pos, spec.center)

        def paths(n):             # list of node chains from n up to a direct node
            out = [[n]] if any(d["type"] != "m" for *_, d in G.out_edges(n, data=True)) else []
            for p in _kept_parents(G, n):
                out += [[n] + c for c in paths(p)]
            return out

        for l, n in nodes.items():
            rec = G.nodes[n]["mlp_trace"]
            if "t" not in rec:
                continue
            ps = paths(n)
            assert len(ps) == rec["n_paths"]
            total = 0
            for chain in ps:
                top = chain[-1]
                mask = torch.zeros(len(union))
                mask[[union.index(i) for i in svs_of[top[0]]]] = 1
                WT = WU * mask[None, :]
                for below in reversed(chain[:-1]):          # pull back through each parent
                    parent = chain[chain.index(below) + 1]
                    WT = site(parent[0], pos).jbar_T(WT) / cache[f"blocks.{parent[0]}.ln2.hook_scale"][0, pos]
                    WT = WT - WT.mean(0, keepdim=True)
                s = site(l, pos)
                total = total + WT.T @ (s.Fz - s.F0)
            t = torch.tensor(rec["t"])
            assert torch.allclose(total, t, atol=1e-4 * max(1.0, t.norm().item()), rtol=1e-3), (n, total, t)
            n_checked += 1
            n_multi += len(ps) > 1
    assert n_checked > 20 and n_multi > 5


def test_path_summed_single_path_matches_per_reader(gpt2_ps_deep, gpt2_pr_deep):
    """An MLP reached by one path only is traced exactly as the per-reader occurrence of that path
    (the zero columns of the union coordinates change nothing)."""
    G, P = gpt2_ps_deep, gpt2_pr_deep
    n_cmp = 0
    for n in _mlp_nodes(G):
        rec = G.nodes[n]["mlp_trace"]
        if rec.get("n_paths") != 1 or "R_sig" not in rec:
            continue
        chain, cur = [], n
        while True:
            ups = _kept_parents(G, cur)
            if not ups:
                break
            cur = ups[0]
            chain.append(("mlp", cur[0]))
        occ = n[:4] + ((n[4][0],) + tuple(reversed(chain)),)
        if occ not in P or "R_sig" not in P.nodes[occ]["mlp_trace"]:
            continue
        a = {(repr(u[:4]), d["rank"]) for u, _, d in G.in_edges(n, data=True)}
        b = {(repr(u[:4]), d["rank"]) for u, _, d in P.in_edges(occ, data=True)}
        assert a == b, n
        assert abs(rec["R_sig"] - P.nodes[occ]["mlp_trace"]["R_sig"]) < 1e-4
        n_cmp += 1
    assert n_cmp > 20


def test_path_summed_union_coordinates(gpt2_ps_deep):
    """A direct-only MLP's target is zero outside the singular vectors of its own edge."""
    G = gpt2_ps_deep
    n_mixed = 0
    for step, nodes in _ps_groups(G).items():
        if step[0] not in ("d", "s"):
            continue
        svs = {}
        for l, n in nodes.items():
            if G.nodes[n]["mlp_trace"].get("leaf_reason") in ("mlp0", "bos"):
                continue
            for _, v, d in G.out_edges(n, data=True):
                if d["type"] == step[0]:
                    svs[l] = eval(d["svs_used"])
        union = sorted({i for s in svs.values() for i in s})
        for n in nodes.values():                      # the stored coordinates are that union
            if "R_sig" in G.nodes[n]["mlp_trace"]:
                assert G.nodes[n]["mlp_trace"]["reader_svs"] == union
        if len({tuple(s) for s in svs.values()}) < 2:
            continue
        for l, n in nodes.items():
            rec = G.nodes[n]["mlp_trace"]
            if rec.get("n_paths") == 1 and rec.get("direct") and "t" in rec:
                assert len(rec["t"]) == len(union)
                off = [x for j, x in enumerate(rec["t"]) if union[j] not in svs[l]]
                assert all(abs(x) < 1e-6 for x in off)
                n_mixed += 1
    assert n_mixed > 0


def test_path_summed_policy_and_pruning(gpt2_tracer):
    kw = dict(answer_token=" Mary", wrong_token=" John", seeding="linear")
    G = gpt2_tracer.trace(IOI_PROMPT, **kw, mlp_expansion=MLPExpansionOptions(tau=0.05, bias_overshoot_leaf=False))
    dec = _decisions(G)
    for n, d in dec.items():
        if n[0] == 0:
            assert d in ("leaf (MLP 0)", "pruned")     # pruned if every parent pruned it
        if d.startswith("leaf (R_sig"):
            assert G.nodes[n]["mlp_trace"]["R_sig"] > 0.25
    n_pruned = 0
    for u, v, d in G.edges(data=True):
        if d.get("child_decision") == "pruned":
            assert abs(d["traced_share_linear"]) < 0.05 and abs(d["traced_share_real"]) < 0.05
            n_pruned += 1
            if not _kept_parents(G, u) and not any(dd["type"] != "m" for *_, dd in G.out_edges(u, data=True)):
                assert dec[u] == "pruned"        # pruned by every parent: not traced
    assert n_pruned > 0
    G = gpt2_tracer.trace(IOI_PROMPT, **kw, mlp_expansion=MLPExpansionOptions(tau=10.0, joint_check=False))
    kids = [d for *_, d in G.edges(data=True) if "child_decision" in d]
    assert kids and all(d["child_decision"] == "pruned" for d in kids)
    assert all(G.nodes[n]["mlp_trace"].get("n_parents", 0) == 0 for n in _mlp_nodes(G))
    # A directly read MLP 0 that every MLP parent pruned keeps its leaf record (regression).
    n_direct_pruned_mlp0 = 0
    for n in _mlp_nodes(G):
        direct = any(d["type"] != "m" for *_, d in G.out_edges(n, data=True))
        if n[0] == 0 and direct:
            assert G.nodes[n]["mlp_trace"]["decision"] == "leaf (MLP 0)", n
            n_direct_pruned_mlp0 += any(d.get("child_decision") == "pruned" for *_, d in G.out_edges(n, data=True))
    assert n_direct_pruned_mlp0 > 0


def test_path_summed_diagnostics(gpt2_ps_deep):
    G = gpt2_ps_deep
    below = [n for n in _mlp_nodes(G) if G.nodes[n]["mlp_trace"].get("n_parents", 0) > 0
             and "R_sig" in G.nodes[n]["mlp_trace"]]
    assert below and all("R_chain" in G.nodes[n]["mlp_trace"] for n in below)
    assert all(0 <= G.nodes[n]["mlp_trace"]["R_chain"] < 2 for n in below)


def test_path_summed_deterministic_and_cap(gpt2_tracer, gpt2_path_summed):
    kw = dict(answer_token=" Mary", wrong_token=" John", seeding="linear")
    G2 = gpt2_tracer.trace(IOI_PROMPT, **kw, mlp_expansion=True)
    assert graph_to_jsonable(G2) == graph_to_jsonable(gpt2_path_summed)
    with pytest.warns(UserWarning, match="max_mlp_nodes"):
        G = gpt2_tracer.trace(IOI_PROMPT, **kw, mlp_expansion=MLPExpansionOptions(max_mlp_nodes=3))
    assert sum("R_sig" in G.nodes[n]["mlp_trace"] for n in _mlp_nodes(G)) == 3


def test_dag_removed_matches_manual(report_setup, gpt2_model, gpt2_tracer):
    """_dag_removed on a 3-MLP DAG (5 -> 6 -> 7 and 5 -> 7; 7 read directly) equals a hand-built
    pass where MLP 7 receives the summed output changes of MLPs 5 and 6."""
    from accpp_tracer.mlp_expand import MLPExpander, _Group
    _, cache, _ = report_setup
    idx = {i: str(i) for i in range(14)}
    ex = MLPExpander(gpt2_tracer, nx.MultiDiGraph(), {}, cache, idx, 0, "dynamic", None, {},
                     MLPExpansionOptions())
    WT = torch.randn(768, 3, generator=torch.Generator().manual_seed(0))
    g = _Group(step=("d", 9, 6, "13", "4"), pos=13, tok="13", direct={7: WT},
               parents={6: [7], 5: [6, 7]}, kids={7: [6, 5], 6: [5]})
    g.T = torch.ones(3)
    sig = 0.05 * torch.randn(768, generator=torch.Generator().manual_seed(1))
    got = ex._dag_removed(g, {5: -sig})
    blk = gpt2_model.blocks
    run = lambda l, z: blk[l].apply_mlp(z[None, None])[0, 0]
    z = {l: ex.site(l, 13).z for l in (5, 6, 7)}
    s = {l: cache[f"blocks.{l}.ln2.hook_scale"][0, 13] for l in (5, 6, 7)}
    c = lambda v: v - v.mean()
    dF5 = run(5, z[5] - sig) - run(5, z[5])
    dF6 = run(6, z[6] + c(dF5) / s[6]) - run(6, z[6])
    dF7 = run(7, z[7] + (c(dF5) + c(dF6)) / s[7]) - run(7, z[7])
    assert torch.allclose(got, -(WT.T @ dF7), atol=1e-4, rtol=1e-3)


def test_golden_gpt2_linear_path_summed(gpt2_path_summed, update_golden):
    assert_graph_matches(gpt2_path_summed, GOLDEN_DIR / "graph_gpt2_linear_mlp_path_summed.json",
                         update_golden)


def test_golden_qwen_prob_path_summed(qwen_tracer, update_golden):
    G = qwen_tracer.trace(IOI_PROMPT, answer_token=" Mary", mlp_expansion=True)
    assert any(G.nodes[n]["mlp_trace"]["decision"] == "expand" for n in _mlp_nodes(G))
    assert_graph_matches(G, GOLDEN_DIR / "graph_qwen_prob_mlp_path_summed.json", update_golden)


# ---------------------------------------------------------------- flow-ordered expansion (min_flow)
def _posthoc_flow(G):
    from collections import defaultdict
    import networkx as nx
    flow = defaultdict(float)
    for r in (n for n in G.nodes if len(n) == 2):
        flow[r] = 1.0
    for v in reversed(list(nx.topological_sort(G))):
        f = flow.get(v, 0.0)
        w = defaultdict(float)
        for u, _, d in G.in_edges(v, data=True):
            w[u] += abs(float(d.get("weight", 0.0)))
        z = sum(w.values())
        if f and z > 0:
            for u, x in w.items():
                flow[u] += f * x / z
    return flow


@pytest.fixture(scope="module")
def gpt2_flow(gpt2_tracer):
    return {phi: gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", wrong_token=" John", seeding="linear",
                                   mlp_expansion=MLPExpansionOptions(min_flow=phi))
            for phi in (1e-2, 1e-3)}


def test_min_flow_validation():
    with pytest.raises(ValueError):
        MLPExpansionOptions(min_flow=0.0)
    with pytest.raises(ValueError):
        MLPExpansionOptions(min_flow=1.0)
    with pytest.raises(ValueError):
        MLPExpansionOptions(min_flow=0.01, mode="per_reader")
    assert MLPExpansionOptions().min_flow is None


def test_min_flow_flows_and_rule(gpt2_flow):
    """Stored flows equal the root flow recomputed on the final graph; every expanded node has flow >= phi,
    every flow-pruned node < phi; the flow ending at terminal nodes sums to 1 per root."""
    for phi, G in gpt2_flow.items():
        post = _posthoc_flow(G)
        assert all(abs(post.get(n, 0.0) - G.nodes[n].get("flow", 0.0)) < 1e-9 for n in G.nodes)
        info = G.graph["flow"]
        assert info["min_flow"] == phi
        assert sum(info["terminal_flow"].values()) == pytest.approx(info["n_roots"], abs=1e-9)
        for n in G.nodes:
            f = G.nodes[n].get("flow", 0.0)
            has_inputs = G.in_degree(n) > 0
            if G.nodes[n].get("flow_pruned"):
                assert f < phi and not has_inputs
            rec = G.nodes[n].get("mlp_trace")
            if rec is not None and rec["leaf_reason"] == "flow":
                assert f < phi and not has_inputs
            if has_inputs and len(n) != 2:
                assert f >= phi


def test_min_flow_shrinks_graph(gpt2_flow, gpt2_path_summed):
    n_tr = lambda G: sum("R_sig" in G.nodes[n]["mlp_trace"] for n in _mlp_nodes(G))
    assert n_tr(gpt2_flow[1e-2]) < n_tr(gpt2_flow[1e-3]) <= n_tr(gpt2_path_summed)
    assert gpt2_flow[1e-2].number_of_nodes() < gpt2_flow[1e-3].number_of_nodes()
    # the high-flow part is traced alike: the top seed heads are expanded in both
    top = sorted(((G.nodes[n].get("flow", 0), n) for G in [gpt2_flow[1e-3]] for n in G.nodes
                  if len(n) == 4 and isinstance(n[1], int)), reverse=True)[:5]
    for _, n in top:
        assert n in gpt2_flow[1e-2] and gpt2_flow[1e-2].in_degree(n) > 0


def test_min_flow_prob_seeding(gpt2_tracer):
    G = gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary",
                          mlp_expansion=MLPExpansionOptions(min_flow=3e-3, prune_head_inputs=True, bos_leaf=True))
    info = G.graph["flow"]
    assert info["n_roots"] == 1
    assert sum(info["terminal_flow"].values()) == pytest.approx(1.0, abs=1e-9)
    assert _mlp_nodes(G)


def test_min_flow_layer_scope(gpt2_tracer):
    """Layer scope: expanded nodes have flow >= phi * (their stage's total), flow-pruned ones below it;
    flows still match the post-hoc root flow."""
    from accpp_tracer.mlp_expand import _flow_stage
    phi = 0.1
    G = gpt2_tracer.trace(IOI_PROMPT, answer_token=" Mary", wrong_token=" John", seeding="linear",
                          mlp_expansion=MLPExpansionOptions(min_flow=phi, min_flow_scope="layer"))
    info = G.graph["flow"]
    assert info["scope"] == "layer" and info["stage_totals"]
    post = _posthoc_flow(G)
    assert all(abs(post.get(n, 0.0) - G.nodes[n].get("flow", 0.0)) < 1e-9 for n in G.nodes)
    totals = {}
    for k, v in info["stage_totals"].items():
        l, kind = k.split()
        totals[(int(l), kind)] = v
    n_checked = 0
    for n in G.nodes:
        st = _flow_stage(n)
        if st is None or st not in totals:
            continue
        f, thr = G.nodes[n].get("flow", 0.0), phi * totals[st]
        rec = G.nodes[n].get("mlp_trace")
        if G.nodes[n].get("flow_pruned") or (rec is not None and rec["leaf_reason"] == "flow"):
            assert f < thr + 1e-12
            n_checked += 1
        elif G.in_degree(n) > 0:
            assert f >= thr - 1e-12
            n_checked += 1
    assert n_checked > 0
    with pytest.raises(ValueError):
        MLPExpansionOptions(min_flow=0.1, min_flow_scope="depth")
