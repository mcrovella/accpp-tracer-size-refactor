"""Combine per-prompt circuit graphs into a unified graph."""

from typing import Sequence

import networkx as nx
import numpy as np

from .pruning import remove_isolated_nodes


def _collapse(n):
    """Map an MLP occurrence ``(layer, "MLP", tok, tok, path)`` to its MLP node."""
    if isinstance(n, tuple) and len(n) == 5 and n[1] == "MLP":
        return n[:4]
    return n


def combine_prompt_graphs(
    graphs: Sequence[nx.MultiDiGraph | None],
    collapse_mlp_occurrences: bool = False,
) -> nx.MultiDiGraph:
    """Combine individual per-prompt circuit graphs into a single unified graph.

    Merges edges across prompts, accumulating weights and tracking which
    prompts each edge appeared in. Parallel edges between the same pair of
    nodes are told apart by their ``type`` (``"d"``, ``"s"``, or ``"m"`` for
    MLP-input edges): edges with the same ``(u, v, type)`` are merged.

    MLP occurrences from per-reader MLP expansion are 5-tuple nodes
    ``(layer, "MLP", tok, tok, reader_path)``. By default they are kept: an
    occurrence matches across prompts only if the same MLP is read along the
    same reader path (same heads, same token labels). With
    ``collapse_mlp_occurrences=True`` every occurrence is mapped to its MLP
    node ``(layer, "MLP", tok, tok)``; edges that then coincide are merged
    (weights summed), giving one node per MLP, as merged-mode expansion does.
    Collapsed MLP-input edges keep only ``weight``, ``type`` and ``color``; their
    weights are sums of shares measured against different readers, so treat
    them as a summary, not as shares.

    Args:
        graphs: Sequence of per-prompt graphs (None entries are skipped).
            Each graph should have edges with "weight" and "type" attributes.
        collapse_mlp_occurrences: Merge MLP occurrences into their MLP node.

    Returns:
        Combined MultiDiGraph with accumulated weights and "prompts_appeared"
        lists on each edge, and "n_appearences" counts on each node (each node
        counted at most once per prompt).

    Edges with the same ``(u, v, type)`` inside one prompt's graph (which
    arise when occurrences are collapsed) are summed and the prompt is
    counted once.
    """
    G_combined = nx.MultiDiGraph()
    index: dict[tuple, int] = {}   # (u, v, type) -> key in G_combined
    mapn = _collapse if collapse_mlp_occurrences else (lambda n: n)

    for prompt_id, G_curr in enumerate(graphs):
        if G_curr is None:
            continue
        seen_here: dict[tuple, int] = {}
        for u0, v0, data in G_curr.edges(data=True):
            u, v = mapn(u0), mapn(v0)
            k = (u, v, data.get("type"))
            if k in seen_here:
                # Same (u, v, type) twice in one prompt (collapsed MLP
                # occurrences): sum the weights, count the prompt once.
                G_combined.edges[(u, v, index[k])]["weight"] += data["weight"]
                continue
            seen_here[k] = 1
            if k in index:
                e = G_combined.edges[(u, v, index[k])]
                e["weight"] += data["weight"]
                e["prompts_appeared"].append(prompt_id)
            else:
                attrs = {kk: vv for kk, vv in data.items() if kk != "prompts_appeared"}
                if collapse_mlp_occurrences and data.get("type") == "m":
                    # Per-occurrence attributes do not survive a merge across readers.
                    attrs = {kk: attrs[kk] for kk in ("weight", "type", "color") if kk in attrs}
                index[k] = G_combined.add_edge(u, v, prompts_appeared=[prompt_id], **attrs)

        for node in {mapn(n) for n in G_curr.nodes}:
            if node not in G_combined:
                G_combined.add_node(node)
            G_combined.nodes[node]["n_appearences"] = (
                G_combined.nodes[node].get("n_appearences", 0) + 1
            )

    # Normalize edge weights by number of non-None graphs
    n_graphs = sum(1 for g in graphs if g is not None)
    for edge in G_combined.edges:
        G_combined.edges[edge]["weight"] /= n_graphs

    return G_combined


def prune_by_frequency(
    G: nx.MultiDiGraph,
    threshold: float,
    n_graphs: int,
) -> nx.MultiDiGraph:
    """Prune a combined graph by node appearance frequency.

    Keeps only nodes that appeared in at least threshold * n_graphs prompts,
    then removes isolated nodes.

    Args:
        G: Combined graph from combine_prompt_graphs().
        threshold: Fraction of prompts a node must appear in (0.0 to 1.0).
        n_graphs: Total number of graphs used in combination.

    Returns:
        Pruned copy of the graph.
    """
    n_appearences_threshold = int(round(threshold * n_graphs, 0))
    G_pruned = nx.MultiDiGraph(
        nx.subgraph_view(
            G,
            filter_node=lambda node: G.nodes[node].get("n_appearences", 0)
            >= n_appearences_threshold,
        )
    )
    remove_isolated_nodes(G_pruned)
    return G_pruned
