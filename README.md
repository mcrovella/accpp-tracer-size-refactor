# accpp-tracer

A pip-installable library for **ACC++**, a circuit tracing algorithm for mechanistic
interpretability of transformer attention heads.

ACC++ decomposes attention head firings into upstream contributions using SVD of the
bilinear form $\Omega = W_Q W_K^T$, producing per-prompt circuit graphs that reveal how
information flows through the model.

From the paper: ["Finding Interpretable Prompt-Specific Circuits in Language Models"](https://arxiv.org/abs/2602.13483).

## Installation

Requires Python 3.10.

```bash
# Install the library
pip install -e .

# Or with runtime shape checking (beartype + jaxtyping)
pip install -e ".[typecheck]"
```

For exact numerical reproducibility of the paper results, use the pinned environment:

```bash
pip install -r requirements.txt
```

## Quick Start

### Trace a single prompt (Level 3 API)

Since v0.3.0 the default seeding is **probability-aware** (`seeding="prob"`): seeds
are the minimal set of upstream components whose removal from the final residual
stream destroys a `tau` fraction (default 0.8) of the model's log-likelihood of a
target-token support `T`, measured in full-vocabulary probabilities against the
bias-only baseline. The traced graph always has exactly **one root node**.

```python
import torch
from transformer_lens import HookedTransformer
from accpp_tracer import Tracer

torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
tracer = Tracer(model)

# Single-token support: T = {" Mary"}
graph = tracer.trace(
    prompt="When Mary and John went to the store, John gave a drink to",
    answer_token=" Mary",
)

print(f"Circuit: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

The support `T` is chosen via **exactly one** of `answer_token`, `top_p`, or `top_k`:

```python
# Explicit multi-token support T = {" Mary", " John"} — one objective, one root node
graph = tracer.trace(
    prompt="When Mary and John went to the store, John gave a drink to",
    answer_token=[" Mary", " John"],
)
# Root node: ("Prob {' Mary', ' John'}", " to (1)")

# Nucleus support: smallest token set covering 90% of the clean probability mass
graph = tracer.trace(prompt="...", top_p=0.9)

# Top-k support: the k most likely next tokens
graph = tracer.trace(prompt="...", top_k=2)
```

Knobs: `tau` (fraction of the completeness to destroy, default 0.8) and `ig_steps`
(integrated-gradients quadrature intervals, default 64).

> **Note (Gemma-2)**: the seeding objective is built from uncapped logits and ignores
> the final logit soft-cap. Token ranking is unaffected (the cap is monotone), but
> probability and attribution values are computed on the uncapped distribution; a
> `UserWarning` is emitted.

### Linear (contrastive) seeding — pre-0.3.0 behavior

`seeding="linear"` reproduces the pre-0.3.0 behavior exactly: seeds are selected by
linear attribution of a logit direction, and each answer token (or each top-p token)
becomes its own direction and root node. This is the mode used for paper reproduction,
and the only mode that supports `wrong_token` (a contrastive *probability* objective
would be mathematically identical to the linear logit-diff direction, so it is not a
separate method).

```python
# IO - S contrastive direction, root node ("Logit direction", ...)
graph = tracer.trace(
    prompt="When Mary and John went to the store, John gave a drink to",
    answer_token=" Mary",
    wrong_token=" John",
    seeding="linear",
)

# One direction and root node per token
graph = tracer.trace(
    prompt="When Mary and John went to the store, John gave a drink to",
    answer_token=[" Mary", " John"],   # two root nodes
    seeding="linear",
)
```

### Trace from a pre-computed cache (Level 2 API)

For batch processing (paper reproduction), use `trace_from_cache()`. The seeding mode
is selected by which objective argument is given — exactly one of `logit_direction`
(linear) or `target_tokens` (probability-aware):

```python
# Probability-aware at Level 2: pass token ids and a single root node.
# Clean logits are recomputed from the cache; q_star defaults to the clean
# probabilities renormalized within the support (pass q_star to override).
graph = tracer.trace_from_cache(
    cache=cache,
    logit_direction=None,
    end_token_pos=end_pos,
    idx_to_token=idx_to_token,
    root_node=("Prob ' Mary'", idx_to_token[end_pos]),
    prompt_idx=prompt_id,
    target_tokens=[io_token_id],          # or [io_token_id, s_token_id], ...
)
```

Passing a single direction and root node works exactly as before:

```python
from accpp_tracer import Tracer
from accpp_tracer.datasets import IOIDataset

model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
tracer = Tracer(model)

dataset = IOIDataset(
    model_family="gpt2", prompt_type="mixed", N=8,
    tokenizer=model.tokenizer, prepend_bos=False, seed=0, device="cpu",
)

logits, cache = model.run_with_cache(dataset.toks)

prompt_id = 0
logit_dir = model.W_U[:, dataset.toks[prompt_id, dataset.word_idx["IO"][prompt_id]]] \
          - model.W_U[:, dataset.toks[prompt_id, dataset.word_idx["S1"][prompt_id]]]

graph = tracer.trace_from_cache(
    cache=cache,
    logit_direction=logit_dir,
    end_token_pos=dataset.word_idx["end"][prompt_id].item(),
    idx_to_token={i: model.tokenizer.decode(dataset.toks[prompt_id, i])
                  for i in range(dataset.word_idx["end"][prompt_id].item() + 1)},
    root_node=("IO-S direction", "to"),
    prompt_idx=prompt_id,
)
```

Pass lists to `trace_from_cache()` for multi-direction tracing at Level 2:

```python
io_dir = model.W_U[:, dataset.toks[prompt_id, dataset.word_idx["IO"][prompt_id]]]
s1_dir = model.W_U[:, dataset.toks[prompt_id, dataset.word_idx["S1"][prompt_id]]]

graph = tracer.trace_from_cache(
    cache=cache,
    logit_direction=[io_dir, s1_dir],           # two directions
    end_token_pos=dataset.word_idx["end"][prompt_id].item(),
    idx_to_token={i: model.tokenizer.decode(dataset.toks[prompt_id, i])
                  for i in range(dataset.word_idx["end"][prompt_id].item() + 1)},
    root_node=[("IO direction", "to"), ("S1 direction", "to")],  # one label per direction
    prompt_idx=prompt_id,
)
```

### Caching the Omega SVD on disk

The Tracer recomputes the Omega SVD (``U, S, VT``) and weight pseudoinverses
(``W_Q_pinv, W_K_pinv``) on every instantiation — a few seconds for GPT-2 /
Pythia, noticeably longer for Gemma-2-2b. Pass ``cache_dir`` to save these
tensors to disk on the first run and reuse them on subsequent runs:

```python
tracer = Tracer(model, cache_dir="~/.cache/accpp_tracer")
# First call: computes the SVD, writes
#   ~/.cache/accpp_tracer/{model_name}_torch.h5
# Subsequent calls: loads from disk, skips SVD.
```

Cache files are gzip-compressed h5 (fp32) and reusable across processes. One
file per ``(model_name, use_numpy_svd)`` pair. Sizes: ~105 MB for GPT-2 /
Pythia, ~1.8 GB for Gemma-2-2b. Default (``cache_dir=None``) recomputes
every time.

### Causal interventions

Once a circuit is traced, `Tracer.run_intervention()` ablates (or boosts) one edge of
the circuit and returns the perturbed logits, cache, and per-prompt metrics. Two modes:

- `"local"` — modify the LN-normalized Q/K input of the downstream head only.
  Surgical: affects just the targeted edge's downstream head.
- `"global"` — modify the upstream component's output directly in the residual
  stream. Broad: affects every downstream consumer. The AH **offset** component
  has no residual-stream hook point and is not supported here.

Centering is auto-detected from `model.cfg.normalization_type` (LN → center,
RMS → don't); pass `center=True/False` to override.

```python
from accpp_tracer import edges_from_graph, InterventionResult

# Re-use model, tracer, dataset, prompt_id, idx_to_token, graph, logits, cache
# from the `trace_from_cache` example above.
token_to_idx = {label: idx for idx, label in idx_to_token.items()}
edges = edges_from_graph(graph, token_to_idx, n_heads=model.cfg.n_heads)

result: InterventionResult = tracer.run_intervention(
    tokens=dataset.toks,            # (batch, seq)
    cache=cache,                    # clean ActivationCache
    logits=logits,                  # (batch, seq, d_vocab)
    edge=edges[0],                  # one EdgeSpec
    prompt_idx=prompt_id,
    intervention_type="local",      # or "global"
    boost=False,                    # False = ablate, True = boost
)

# result.logits_interv        (batch, seq, d_vocab)
# result.interv_cache         intervention ActivationCache
# result.delta                (batch, seq, d_model) — applied delta
# result.norm_ratio           (batch,) at intervention position
# result.cos_sim              (batch,) at intervention position
# result.attn_scores_clean    (batch,) at downstream (dest, src)
# result.attn_scores_interv   (batch,) at downstream (dest, src)
```

The API is intentionally one-edge-per-call: multi-edge experiments loop over `edges`
and compose at the call site.

## Expanding MLP nodes (v0.4.0)

ACC++ explains why attention heads attend. Its recursion used to stop at MLPs:
an MLP that contributes a signal to a head was a leaf. With `mlp_expansion`,
the tracer continues upstream through MLPs. For each MLP it finds the small
set of upstream components (head-source pairs, attention biases, the
embedding, earlier MLPs) that produce the signal the MLP sends on.

```python
from accpp_tracer import Tracer, MLPExpansionOptions

graph = tracer.trace(prompt, answer_token=" Mary", mlp_expansion=True)   # defaults

graph = tracer.trace(prompt, answer_token=" Mary",
                     mlp_expansion=MLPExpansionOptions(mode="merged", tau=0.02))
```

With `mlp_expansion=None` (the default) the graph is exactly what earlier
versions produced.

**Method** (Algorithm 1 of the MLP-tracing report):
- **Exact attribution.** The MLP's input is an exact sum of upstream component
  outputs. A path-average Jacobian `J̄` (with `F(z) − F(0) = J̄z`) splits the
  input-dependent part of the signal exactly into per-component attributions.
- **Reader coordinates.** The signal is measured in the coordinates the reader
  uses. For a head, these are ACC++'s own signal coordinates (the edge's
  singular vectors pulled back through RoPE, divided by the head's ln1 scale).
- **Greedy selection.** Components are added greedily. After each one, its
  row-space part is removed from the MLP input, and the remaining signal is
  re-measured in the real, nonlinear MLP. Selection stops when that residual
  `R_sig` falls below `eps`.
- **New signals.** The new signal handed upstream for a selected component is
  its row-space part.
- **Leaves.** An MLP's constant output `F(0)` is a leaf, like a bias.

**Policy at each MLP.** The MLP is a leaf if removing its input-dependent
signal would not reduce what the reader sees (bias overshoot). Otherwise:
- `R_sig ≤ eps` (0.1): expand;
- `R_sig ≤ eps_soft` (0.25): expand, marked approximate;
- otherwise: leaf.

**Downstream MLPs.**
- Selected head-source pairs are traced by the ordinary ACC++ recursion.
- A selected upstream MLP is traced in turn. It is read through the downstream
  MLP's attribution operator, so attributions stay in the original reader's
  units and shares compose along the chain.
- It is pruned only if both its linear share and its share measured through
  the real MLPs are below `tau` (default 0.25; 0.01 in 0.4.0).
- A joint-removal check then restores pruned siblings whose combined effect is
  not small.

**Optional pruning:**
- `prune_head_inputs=True` applies the same both-below-`tau` rule, with the
  joint check, to the head-source inputs an MLP trace selects. A pruned head
  input keeps its edge but is not handed to the ACC++ head recursion. By
  default every selected head input is recursed into, subject only to ACC++'s
  attention threshold.
- `bos_leaf=True` makes position 0 (the BOS / attention-sink position) a leaf:
  MLPs there are not expanded, and head inputs reading it are not recursed
  into.

Expansion can grow a graph several-fold. Selected head inputs start new head
recursions, and in per-reader mode an MLP is traced once per reader path. On GPT-2 IOI
these two options cut edges by about 20%. Raising `tau` from 0.01 to 0.10
removes a further 5%. On Gemma-2-2B IOI, `tau=0.25` (the default) keeps at
least 99% of the flow-weighted head firings of the `tau=0.01` graph, while
`tau <= 0.10` hit a 3,000-trace cap.

**Supported MLPs, configured from `model.cfg`:**
- standard MLPs (GPT-2, Pythia, including Pythia's parallel attention+MLP);
- gated MLPs (Llama, Qwen, Gemma);
- Gemma-2's post-MLP RMSNorm.

`get_mlp_spec(model)` reports the detected architecture. It rejects SoLU-LN
activations, mixture-of-experts, and unfolded LayerNorm. Load models with the
TransformerLens defaults (`fold_ln=True`).

### Expansion modes

An attention head can be expanded once, whoever reads it: the question ACC++
asks of a head, why it attends from d to s, is the head's own. An MLP has no
decision of its own, so what counts as its signal depends on who reads it.

**Reader operator.** A reader measures an MLP's output through a linear map
`W` (`r × D`; stored transposed as `WT`). `W` takes a residual-stream vector
written at the MLP's position to the `r` coordinates the reader uses:
- for a head, its selected singular vectors (RoPE pulled back, divided by the
  head's ln1 scale);
- for the root objective, the rank-1 logit, IG-gradient or probe direction;
- for an MLP `M` selected by a downstream MLP `P`, `W_P J̄_P C / s_P` (`P`'s
  path-average Jacobian, centering for LayerNorm models, and `P`'s ln2 scale).
  Attributions therefore stay in the original reader's units.

Algorithm 1 traces the target `t = W (F(z) − F(0))`. Along a chain of MLPs
the operators compose: one *path* `R → P_1 → … → P_k → M` gives
`W_path = W_R J̄_{P_1} C / s_{P_1} ⋯ J̄_{P_k} C / s_{P_k}`.

| | `mode="path_summed"` (default) | `mode="per_reader"` | `mode="merged"` |
|---|---|---|---|
| Question answered | Which inputs produce *this reader's* signal from the MLP, summed over all MLP routes? | Which inputs produce this reader's signal along *one* route? | Which inputs produce what the MLP sends to *all* its readers? |
| Operator of MLP `M` | `Σ_paths W_path` (its total linearized influence on the reader) | `W_path` of one path | all readers' operators stacked |
| Traces | one per (first-hop reader, MLP) | one per reader path | one per MLP |
| Graph node | `(layer, "MLP", tok, tok, (reader_step,))` | `(layer, "MLP", tok, tok, reader_path)` | `(layer, "MLP", tok, tok)` |
| Size | bounded by readers × layers | exponential in chain depth (bounded by pruning and the cap) | bounded by MLPs |

**Path-summed (default).** A *first-hop reader* is one side of an
attention-head firing (query or key) or the root objective. Every MLP it
reaches, directly or through other MLPs, lies at one position. MLP→MLP edges
go upward in layer, so these MLPs form a DAG.
- The DAG is processed in decreasing layer order, so each MLP's parents are
  traced before it.
- Each MLP is traced once, with `W_M = [W_R if R reads M directly] + Σ_{kept
  parents P} W_P J̄_P C / s_P`. By linearity this is the sum of the per-reader
  path operators.
- A node has an `"m"` edge to every MLP parent that selected it, plus the
  original edge to the reader if it is read directly. `n_paths` records how
  many reader paths its trace sums.
- **Coordinates.** ACC++ picks singular vectors per edge, so the MLPs a head
  reads directly can use different subsets. The reader's coordinates are their
  union, and each direct MLP's operator is zero outside its own subset.
- **Shares** (`*_root` fields) are relative to the reader's direct MLP signal
  `T_R = Σ_{direct M} W_R (F_M(z) − F_M(0))`.
- **Real-chain shares** run the real MLPs on every kept path above the MLP.
  Each MLP receives the summed output change of its kept children.

**Per-reader.** `reader_path` lists the readers from the top of the MLP chain
down to the immediate reader:
- `("d" | "s", head_layer, head, dest_label, src_label)` for an attention head;
- `("root", *root_node)` for the seeding objective or a probe;
- `("mlp", layer)` for a downstream MLP occurrence.

Each occurrence has exactly one outgoing edge, to that reader. The same MLP is
traced once per path, so the count grows exponentially with chain depth. On a
Llama-3.2-3B prompt one MLP was traced 346 times.

**Merged.** The readers' operators are stacked, so the objective is the sum of
each reader's squared signal. A merged set can fail individual readers, since
the readers with larger signals dominate.

Use path-summed expansion (the default) for whole circuits. Use per-reader
when a specific route matters. Use merged for a compact per-MLP summary.

### Flow-ordered expansion (`min_flow`)

ACC++ and the MLP expander each prune locally: a head input or MLP child is
kept if it matters to its *immediate* reader. Repeated hop after hop, this lets
the graph grow without bound (the head cascade). `min_flow` adds a global rule.

- **Root flow.** The objective has flow 1. Each expanded node splits its flow
  over its in-edges in proportion to `|weight|` (parallel edges summed), so a
  node's flow is the fraction of the objective routed through it along the
  traced graph. Normalising per node means head-edge weights (ACC++
  contributions) and MLP-input weights (shares) are never compared directly.
- **Schedule.** Head firings and MLP nodes are expanded in one global order
  (decreasing layer; an MLP before the heads of its own layer). That order is
  topological for the reads, so a node's flow is final when it is reached. A
  node with flow below `min_flow` is not expanded.
- **Bound.** The flows of any set of nodes no two of which lie on one path
  sum to at most 1 per root, so at most `1 / min_flow` such nodes are
  expanded per root.

```python
G = tracer.trace(prompt, answer_token=" Mary", wrong_token=" John", seeding="linear",
                 mlp_expansion=MLPExpansionOptions(min_flow=0.003))
G.nodes[n]["flow"]          # every reached node
G.graph["flow"]             # min_flow, counts, and where the objective's flow ends
```

`min_flow_scope="layer"` makes the threshold relative: a node is expanded if
its flow is at least `min_flow` times the total flow at its stage (the MLP
nodes of one layer, or the attention-side nodes of one layer). Flow is
conserved per step back along an edge, not per layer: edges skip layers, so the
flow reaching a layer's nodes varies widely (on GPT-2 IOI, from about 0.01 for
middle-layer MLPs to about 0.6 for the layer-9 heads). The default global
threshold therefore keeps mostly late layers; `"layer"` keeps the largest
shares of whatever reaches each stage, at most `1 / min_flow` nodes per stage
and root.

Flow-pruned head firings have `flow_pruned=True`; flow-pruned MLP nodes have
the decision `"leaf (flow)"`. `tau`, `prune_head_inputs` and the other rules
still apply. Path-summed mode only. Off by default (`None`), in which case
graphs are unchanged. `circuit_to_html` shows a node's flow on hover.

### What an expanded graph contains

- **New edges.** MLP-input edges have `type="m"`.
  - `weight` is the component's linear share of the signal the MLP sends to
    its reader. Shares over all candidates sum to 1.
  - `rank` is the selection order.
  - Upstream-MLP edges also carry `traced_share_linear`,
    `traced_share_real`, and `child_decision`.
  - `share_root` is the share of the reader's signal: in path-summed mode, of
    the reader's direct MLP signal `T_R`; in per-reader mode, of the signal at
    the top of the chain.
- **Node record.** Every MLP node has `G.nodes[n]["mlp_trace"]`, with
  `decision`, `R_sig`, `R_ideal`, `n_selected`, `n_candidates`, `readers`,
  and more. Path-summed nodes add `n_paths`, `n_parents`, `direct`, `t` (the traced
  target in the reader's coordinates) and `reader_svs` (for a head reader,
  the singular vectors that make up those coordinates).
- **Signals.** With `signals=...`, `"m"` edges carry the selected component's
  row-space part (`signal_flavor="mlp_row_space"`).
- **Single edges.** `Tracer.trace_mlp_edge(cache, mlp_layer, pos, head_layer,
  head, edge_type, svs_used)` traces one MLP → head edge and returns an
  `MLPTrace` (Level 1).
- **Graph utilities:**
  - `edges_from_graph` resolves MLP occurrences to their MLP and skips `"m"`
    edges.
  - `combine_prompt_graphs(..., collapse_mlp_occurrences=True)` merges
    occurrences into their MLP node.
  - `circuit_to_html` draws `"m"` edges dotted.

## Value inputs of head firings

ACC++ traces what drives a head firing's attention score: its query input at the
destination `d` and its key input at the source `s`. What the firing *copies*
comes from its value input, the residual stream at `s`. `value_inputs=True`
traces that input too, as a third edge type (`type="v"`):

```python
from accpp_tracer import ValueInputOptions

G = tracer.trace(prompt, answer_token=" Mary", wrong_token=" John", seeding="linear",
                 value_inputs=True)                                 # plain ACC++ + value inputs
G = tracer.trace(prompt, answer_token=" Mary", wrong_token=" John", seeding="linear",
                 mlp_expansion=MLPExpansionOptions(min_flow=0.003),  # bounded by root flow
                 value_inputs=ValueInputOptions(tau=0.05))
```

**Method.** With the attention pattern and the norm scales frozen (as ACC++
freezes them), a firing's output is linear in its normalized source residual:
`A[h,d,s] (z_s W_V + b_V) W_O`.
- *Readers.* The firing's out-edges in the graph: heads (ACC++'s query- or
  key-side signal coordinates), the root objective, MLPs (the MLP trace's
  attribution operator), and the value side of other firings. Their operators
  are stacked into `W^T`.
- *Pull back.* `M = A W_V W_O W^T` maps a vector at `s` to the readers'
  coordinates. `z_s` decomposes exactly into components (embedding, earlier
  heads with their sources, attention biases, earlier MLPs), so each component
  contributes `t_c = o_c M`.
- *Select.* Greedy forward selection picks components until the readers' value
  signal `T = sum_c t_c` is explained to within `eps` (default 0.1), or
  `k_max` (10) are chosen. Edge `weight` is the share `t_c . T / |T|^2`, and
  `rank` is the selection order.
- *Recurse.* Selected heads are traced in turn (query, key and value). With MLP
  expansion on, selected MLPs are expanded with the firing as their reader
  (reader step `("v", *firing)`). The firing reads its value inputs through
  `C M / s_ln1`, so operators compose along chains of value edges, as MLP
  operators do.

**Schedule and flow.** Value tracing runs in the same global, layer-ordered
schedule as `min_flow`, so every reader of a firing is known when it is
traced. Every node gets a root `flow`. A firing passes `flow_share` (default
0.5) of its flow to its value inputs and the rest to its query/key inputs:
it needs both its attention and its value. With `min_flow`, firings below the
threshold are not traced on any side.

**Options** (`ValueInputOptions`):
- `tau` (0): selected inputs with a smaller share keep their edge
  (`child_decision="pruned"`) but are not traced further;
- `recurse` (True): False adds one level of value inputs only;
- `bos_leaf` (False): do not value-trace firings that read position 0, and do
  not follow value inputs that are heads reading position 0;
- `max_traces` (20000): cap on value-traced firings.

**Graph contents.** Value-traced firings have `G.nodes[n]["value_trace"]`
(`R_value`: the unexplained fraction, `n_selected`, `n_readers`, `rank` of the
stacked reader operator). `G.graph["value_inputs"]` counts the traced firings
and the value edges. Without MLP expansion, MLP value inputs are 4-tuple
leaves. With MLP expansion, value inputs need `mode="path_summed"`.
`edges_from_graph` (interventions) skips value edges.

**Size.** Value inputs add the heads that steer other heads through their
values (for IOI, the duplicate-token and induction heads behind the
S-inhibition heads). The graph grows accordingly; use `min_flow` (and `tau`) to
bound it.

## Supported Models

| Model | Positional encoding | Notes |
|-------|---------------------|-------|
| `gpt2-small` | Standard | Attention bias |
| `EleutherAI/pythia-160m` | RoPE | Attention bias, parallel attn+MLP |
| `gemma-2-2b` | RoPE | GQA, attention softcapping, no bias |
| `meta-llama/Llama-3.1-8B` | RoPE (NTK-by-parts) | GQA, no QK bias; fp32 weights ~30 GB — needs a large-memory GPU or high-RAM CPU |
| `Qwen/Qwen2.5-0.5B` … `7B` | RoPE | GQA (up to 7× KV sharing), large QK biases (AH bias / AH offset components are strongly active) |

### Prompt length limit

`trace_firing` keeps a per-singular-vector attention-score breakdown of shape
`(layer, n_heads + 4, d_head, T, T)` for both the query and key sides. This
T² scaling is inherent to the per-SV granularity of the method (the greedy
selection needs the full tensor), and bounds the usable prompt length: at
Llama-3.1-8B scale it is ~1.1 MB x T² at the deepest layer (~1 GB at T=30,
~19 GB at T=128). On an 80 GB device, prompts up to roughly 100-150 tokens
are practical; IOI-style prompts are far below this. If longer prompts are
ever needed, chunk over upstream layers or store per-SV scores sparsely
after thresholding.

## API levels

The library exposes three levels of abstraction:

| Level | Function / class | Description |
|-------|-----------------|-------------|
| 1 | `trace_firing()` | Decomposes one attention firing into upstream contributions (mathematical core) |
| 1 | `Tracer.trace_mlp_edge()` | Traces one MLP → head edge upstream through the MLP (Algorithm 1) |
| 2 | `Tracer.trace_from_cache()` | Full circuit from a pre-computed activation cache |
| 3 | `Tracer.trace()` | End-to-end: string prompt → circuit graph |

## Paper reproduction

The companion repository contains all experiment scripts and shell pipelines that
reproduce the paper's figures and tables:

**https://github.com/gaabrielfranco/finding-highly-interpretable-circuits**

It has a pinned copy of `accpp_tracer` under `lib/accpp_tracer/` and pins
TransformerLens to the version used for the paper. A single `pip install -e .` from
the repo root installs everything. The pipelines cover:

- **Tracing & figures** — paper §2, appendices B–D
- **Causal interventions** — appendix E
- **Clustering & signals** — §3, appendix F
- **Autointerpretation** — quantitative and qualitative tracks

See the companion repo's README for run instructions.

## Interactive circuit visualization

`accpp_tracer.graphs.circuit_to_html` renders a traced graph as a single
self-contained HTML file — no external dependencies or network access, so the
file can be opened locally, attached to a paper, or shared as-is:

```python
from accpp_tracer.graphs import circuit_to_html

graph = tracer.trace(prompt, answer_token=" Mary")

# Token labels must match the graph's node labels — the same construction
# Tracer.trace uses internally (duplicate tokens get " (1)", " (2)", ...).
circuit_to_html(
    graph,
    tokens=tokens,                       # position-ordered token labels
    n_layers=model.cfg.n_layers,
    out_path="circuit.html",
    title="Llama-3.2-3B — traced IOI circuit",
    subtitle="objective: P(' Mary') at the final position",  # HTML allowed
    stats={"P( Mary)": "0.387", "nodes": "63"},              # header chips
)
```

The page lays the circuit out with token positions on the x-axis and layers
on the y-axis (each head sits at its *destination* token; the root objective
sits above the final position). Component kinds get distinct shapes and
colors (attention heads, MLPs, embeddings, attention biases, AH offsets —
only kinds present appear in the legend); query-side (`d`) and key-side
(`s`) edges are distinguished by color *and* dash pattern; line width scales
with edge weight. Hovering any node or edge shows details (attention
weights, per-edge singular-vector channels); clicking a node pins its
incident edges; a collapsible table lists every edge sorted by weight. Light
and dark color schemes are both supported (OS preference, or set
`data-theme="dark"`/`"light"` on the document root).

## Test suite

`tests/` contains a fast regression suite (~3 minutes, CPU-only) that
exercises every public API on two small models from the local HF cache —
**gpt2** (LayerNorm, absolute positions) and **EleutherAI/pythia-160m**
(RoPE) — plus a tiny random-weight Llama-3-style model (GQA + RMSNorm +
NTK-by-parts RoPE) that covers the architecture features the real test
models lack.

```bash
pip install -e ".[dev,typecheck]"
pytest                    # run everything, compare against tests/golden/
pytest --update-golden    # re-baseline golden files (see below)
```

Two kinds of tests:

1. **Invariant tests** — mathematical properties that hold for any model and
   survive any refactor: the Omega SVD reconstructs `W_Q @ W_K^T`, component
   residual shares sum to the LN-normalized stream, RoPE matrices are
   orthogonal and match TransformerLens's own sin/cos tables, and
   `trace_firing`'s internal decomposition assertions pass.
2. **Golden regression tests** — outputs captured in `tests/golden/` from a
   known-good revision: SVD fingerprints (sign-invariant random-projection
   sketches), rotation-matrix stacks, `trace_firing` results, and fully
   serialized traced graphs. Comparisons require exact topology and
   singular-vector selection, with a small float tolerance (5e-3) on weights
   to absorb fp32 contraction-reorder noise.

Workflow: for a behavior-preserving change (memory/speed refactor), all
tests must pass against the existing goldens — do not regenerate them. For
an intended behavioral change, inspect the golden diffs, confirm they match
the intent, then re-baseline with `pytest --update-golden` and commit the
new goldens together with the code. `tests/README.md` has the full details,
including how to add a new model to the suite.

## Citation

```bibtex
@article{franco2026finding,
  title={Finding Interpretable Prompt-Specific Circuits in Language Models},
  author={Franco, Gabriel and Tassis, Lucas M and Rohr, Azalea and Crovella, Mark},
  journal={arXiv preprint arXiv:2602.13483},
  year={2026}
}
```
