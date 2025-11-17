from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from emoatlas.resources import _valences


def compare_fmns_patterns(
    fmns: dict,
    language: str = "italian",
    custom_valences=None,
    top_k: int = 20,
    min_edge_freq: int | None = None,
):
    """
    Compare all FMNs' nodes and edges with valences to highlight strongest recurring patterns.

    Parameters
    - fmns: dict[str, FormamentisNetwork]
    - language: 'italian' | 'english' (used when custom_valences is None)
    - custom_valences: list/tuple of 3 sets [positive, negative, neutral] or None (lexicon fallback)
    - top_k: number of top nodes/edges shown in the printed summary
    - min_edge_freq: optional threshold to filter edges by frequency (in how many FMNs they appear)

    Returns
    dict with:
      - nodes_df: per-node presence across FMNs + valence breakdown and dominance
      - edges_df: per-edge presence across FMNs + valence pair type
      - edge_type_dist_df: distribution of edge valence-pair types (counts and shares)
      - core: dict with 'nodes_all_fmns' and 'edges_ge_half' sets
      - node_valence_maps: per-FMN mapping {fmn_title: {node: -1|0|1}}
    """
    # Resolve valence sets
    if custom_valences is not None:
        if not (isinstance(custom_valences, (list, tuple)) and len(custom_valences) == 3 and all(isinstance(s, set) for s in custom_valences)):
            raise ValueError("custom_valences must be a list/tuple of three sets: [positive, negative, neutral]")
        pos_set, neg_set, neu_set = custom_valences
    else:
        pos_set, neg_set, amb_set = _valences(language)
        neu_set = set(amb_set)  # treat ambivalent as neutral

    def val_cat(n: str) -> int:
        if n in pos_set: return 1
        if n in neg_set: return -1
        if n in neu_set: return 0
        return 0  # fallback neutral

    def pair_label(v1: int, v2: int) -> str:
        a, b = sorted((v1, v2))
        lab = {
            (-1, -1): "neg-neg",
            (-1, 0):  "neg-neu",
            (-1, 1):  "pos-neg",
            (0, 0):   "neu-neu",
            (0, 1):   "pos-neu",
            (1, 1):   "pos-pos",
        }
        return lab[(a, b)]

    def collect_edges(edges_attr):
        """Normalize edges from different fmnt formats into a set of undirected tuples (u, v)."""
        out = set()
        if edges_attr is None:
            return out
        if isinstance(edges_attr, dict):
            for lst in edges_attr.values():
                for e in lst:
                    if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0] != e[1]:
                        out.add(tuple(sorted((str(e[0]), str(e[1])))))
        else:
            for e in edges_attr:
                if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0] != e[1]:
                    out.add(tuple(sorted((str(e[0]), str(e[1])))))
        return out

    total_n = len(fmns)
    node_presence = Counter()                       # across FMNs
    node_valence_counts = defaultdict(Counter)      # node -> counts of {-1,0,1} across FMNs
    node_valence_maps = {}                          # per FMN valence map

    edge_presence = Counter()                       # (u,v) across FMNs
    edge_valence_type_counts = Counter()            # pair-type label counts weighted by FMN presence
    edge_valence_pair = {}                          # per edge -> label

    # Per-FMN loop
    for title, fmn in fmns.items():
        vertices = set(map(str, getattr(fmn, "vertices", [])))
        edges = collect_edges(getattr(fmn, "edges", None))

        # Node valence map for this FMN
        vmap = {n: val_cat(n) for n in vertices}
        node_valence_maps[title] = vmap

        # Node presence and valence tallies
        for n in vertices:
            node_presence[n] += 1
            node_valence_counts[n][vmap[n]] += 1

        # Edge presence and valence-pair type tallies
        for (u, v) in edges:
            edge_presence[(u, v)] += 1
            v1, v2 = vmap.get(u, 0), vmap.get(v, 0)
            label = pair_label(v1, v2)
            edge_valence_pair[(u, v)] = label

    # Build nodes_df
    rows_nodes = []
    for n, cnt in node_presence.items():
        vc = node_valence_counts[n]
        pos_c, neg_c, neu_c = vc.get(1, 0), vc.get(-1, 0), vc.get(0, 0)
        dominant_val = max(vc.items(), key=lambda kv: (kv[1], kv[0]))[0] if vc else 0
        rows_nodes.append({
            "node": n,
            "n_fmns": cnt,
            "share": cnt / total_n if total_n else 0,
            "pos_count": pos_c,
            "neg_count": neg_c,
            "neu_count": neu_c,
            "dominant_valence": dominant_val,  # 1 pos, 0 neu, -1 neg
            "dominance_ratio": (vc[dominant_val] / cnt) if cnt else 0.0,
        })
    nodes_df = pd.DataFrame(rows_nodes).sort_values(
        ["n_fmns", "dominance_ratio", "node"], ascending=[False, False, True]
    ).reset_index(drop=True)

    # Build edges_df
    # Optionally filter by min_edge_freq
    edges_iter = ((e, c) for e, c in edge_presence.items()
                  if (min_edge_freq is None or c >= min_edge_freq))
    rows_edges = []
    type_counter = Counter()
    for (u, v), cnt in edges_iter:
        label = edge_valence_pair.get((u, v), "unknown")
        type_counter[label] += cnt
        rows_edges.append({
            "u": u,
            "v": v,
            "edge": f"{u} -- {v}",
            "n_fmns": cnt,
            "share": cnt / total_n if total_n else 0,
            "valence_pair_type": label,
        })
    edges_df = pd.DataFrame(rows_edges).sort_values(
        ["n_fmns", "edge"], ascending=[False, True]
    ).reset_index(drop=True)

    # Edge type distribution (weighted by presence across FMNs)
    if type_counter:
        total_edge_occ = sum(type_counter.values())
        edge_type_dist_df = (
            pd.DataFrame(
                [{"pair_type": t, "count": c, "share": c / total_edge_occ}
                 for t, c in type_counter.items()]
            ).sort_values(["count", "pair_type"], ascending=[False, True]).reset_index(drop=True)
        )
    else:
        edge_type_dist_df = pd.DataFrame(columns=["pair_type", "count", "share"])

    # Core sets
    core_nodes = set(n for n, c in node_presence.items() if c == total_n)
    ge_half = int(np.ceil(total_n * 0.5))
    core_edges_ge_half = set(e for e, c in edge_presence.items() if c >= ge_half)

    # Print concise summary
    print(f"\nCompared {total_n} FMNs")
    print(f"- Unique nodes: {len(node_presence)} | nodes in ALL FMNs: {len(core_nodes)}")
    print(f"- Unique edges: {len(edge_presence)} | edges in >=50% FMNs: {len(core_edges_ge_half)}")
    if not nodes_df.empty:
        print(f"\nTop {min(top_k, len(nodes_df))} nodes by presence:")
        display(nodes_df.head(top_k))
    if not edges_df.empty:
        print(f"\nTop {min(top_k, len(edges_df))} edges by presence:")
        display(edges_df.head(top_k))
    if not edge_type_dist_df.empty:
        print("\nValence-pair edge type distribution (weighted by FMN presence):")
        display(edge_type_dist_df)

    return {
        "nodes_df": nodes_df,
        "edges_df": edges_df,
        "edge_type_dist_df": edge_type_dist_df,
        "core": {
            "nodes_all_fmns": core_nodes,
            "edges_ge_half": core_edges_ge_half,
        },
        "node_valence_maps": node_valence_maps,
    }


def compute_node_aura(
    fmn,
    target_word: str,
    emo=None,
    language: str | None = None,
    custom_valences=None,
    include_target: bool = False,
    tie_priority: tuple = (0, 1, -1),  # tie-break preference: neutral > positive > negative
    return_details: bool = True,
    case_insensitive: bool = True,
):
    """
    Compute the aura (most frequent sentiment polarity) in the semantic frame of target_word.

    Parameters
    - fmn: FormamentisNetwork
    - target_word: node label to extract the semantic frame for
    - emo: EmoScores instance (used to extract the frame)
    - language: optional, defaults to emo.language if None
    - custom_valences: [pos_set, neg_set, neu_set] or None to use lexicon (_valences)
    - include_target: include the center node in the polarity tally
    - tie_priority: tuple of valence categories for tie-breaking, default (0, 1, -1)
    - return_details: if True, include per-category node lists
    - case_insensitive: match target node and valence sets case-insensitively

    Returns
    dict with:
      node, aura_cat (-1/0/1), aura_label, counts, shares, frame_size, considered_size,
      considered_nodes, [optional] pos_nodes/neg_nodes/neu_nodes
    """
    if language is None:
        language = getattr(emo, "language", "italian")

    # Resolve valence sets
    if custom_valences is not None:
        if not (isinstance(custom_valences, (list, tuple)) and len(custom_valences) == 3 and all(isinstance(s, set) for s in custom_valences)):
            raise ValueError("custom_valences must be [positive_set, negative_set, neutral_set]")
        pos_set, neg_set, neu_set = custom_valences
    else:
        pos_set, neg_set, amb_set = _valences(language)
        neu_set = set(amb_set)

    # Normalize helper
    def norm(s: str) -> str:
        return str(s).casefold() if case_insensitive else str(s)

    pos_l = {norm(w) for w in pos_set}
    neg_l = {norm(w) for w in neg_set}
    neu_l = {norm(w) for w in neu_set}

    # Find actual node label in FMN (preserve original casing for reporting)
    vertices = list(getattr(fmn, "vertices", []))
    vmap_lower = {norm(v): v for v in vertices}
    target_norm = norm(target_word)
    if target_norm not in vmap_lower:
        raise ValueError(f"Node '{target_word}' not found in FMN vertices.")
    target_actual = vmap_lower[target_norm]

    # Extract the semantic frame FMN around target
    frame_fmn = emo.extract_word_from_formamentis(fmn, target_actual)
    frame_vertices = list(getattr(frame_fmn, "vertices", []))

    # Build considered node list
    considered = []
    for n in frame_vertices:
        if not include_target and norm(n) == target_norm:
            continue
        considered.append(n)

    # Tally polarities
    counts = Counter({1: 0, 0: 0, -1: 0})
    pos_nodes, neg_nodes, neu_nodes = [], [], []

    for n in considered:
        nl = norm(n)
        if nl in pos_l:
            counts[1] += 1
            pos_nodes.append(n)
        elif nl in neg_l:
            counts[-1] += 1
            neg_nodes.append(n)
        else:
            # explicit neutral set or fallback neutral
            counts[0] += 1
            neu_nodes.append(n)

    total = max(1, len(considered))
    max_count = max(counts.values()) if total > 0 else 0
    # Tie-break using tie_priority order
    top_cats = [cat for cat, c in counts.items() if c == max_count]
    aura_cat = min(top_cats, key=lambda c: tie_priority.index(c)) if top_cats else 0
    label = {1: "positive", 0: "neutral", -1: "negative"}[aura_cat]

    result = {
        "node": target_actual,
        "aura_cat": aura_cat,
        "aura_label": label,
        "counts": {
            "positive": counts[1],
            "neutral": counts[0],
            "negative": counts[-1],
        },
        "shares": {
            "positive": counts[1] / total,
            "neutral": counts[0] / total,
            "negative": counts[-1] / total,
        },
        "frame_size": len(frame_vertices),
        "considered_size": len(considered),
        "considered_nodes": considered,
    }
    if return_details:
        result.update({
            "pos_nodes": pos_nodes,
            "neg_nodes": neg_nodes,
            "neu_nodes": neu_nodes,
        })
    return result


def compute_auras_for_words(
    fmn,
    words: list[str],
    emo=None,
    language: str | None = None,
    custom_valences=None,
    include_target: bool = False,
    case_insensitive: bool = True,
):
    """
    Batch aura computation for a list of words. Returns DataFrame.
    """
    rows = []
    for w in words:
        try:
            r = compute_node_aura(
                fmn=fmn,
                target_word=w,
                emo=None,
                language=language,
                custom_valences=custom_valences,
                include_target=include_target,
                return_details=False,
                case_insensitive=case_insensitive,
            )
            rows.append({
                "node": r["node"],
                "aura_cat": r["aura_cat"],
                "aura_label": r["aura_label"],
                "pos_count": r["counts"]["positive"],
                "neu_count": r["counts"]["neutral"],
                "neg_count": r["counts"]["negative"],
                "pos_share": r["shares"]["positive"],
                "neu_share": r["shares"]["neutral"],
                "neg_share": r["shares"]["negative"],
                "frame_size": r["frame_size"],
                "considered_size": r["considered_size"],
            })
        except Exception as e:
            rows.append({
                "node": w, "aura_cat": None, "aura_label": None,
                "pos_count": None, "neu_count": None, "neg_count": None,
                "pos_share": None, "neu_share": None, "neg_share": None,
                "frame_size": 0, "considered_size": 0, "error": str(e)
            })
    return pd.DataFrame(rows).sort_values(["aura_cat", "node"], ascending=[False, True]).reset_index(drop=True)


def keyword_compound_index(
    fmns: dict,
    keywords: list[str],
    emo=None,
    per_fmn_custom_valences: dict | None = None,
    language: str | None = None,
    include_target: bool = False,
    case_insensitive: bool = True,
    normalize_within_fmn: bool = True,
) -> dict:
    """
    Compute a compound index for each (keyword, FMN) combining:
      - Keyword Relevance Score: closeness centrality (raw and [0,1] scaled per FMN)
      - Connection Count Index: degree (or strength if weighted) (raw and [0,1] scaled per FMN)
      - Aura Valence Mode: mode label of neighbors' valence (positive/neutral/negative)

    compound_index is (closeness_scaled, connection_scaled, aura_label) with the
    two numeric values rounded to 2 decimals.
    """
    import networkx as nx
    import numpy as np
    import pandas as pd

    if language is None:
        language = getattr(emo, "language", "italian")

    def norm(s: str) -> str:
        return str(s).casefold() if case_insensitive else str(s)

    def minmax_scale(d: dict[str, float]) -> dict[str, float]:
        if not d:
            return {}
        vals = np.array(list(d.values()), dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return {k: 0.0 for k in d}
        rng = (vmax - vmin)
        return {k: (float(v) - float(vmin)) / float(rng) for k, v in d.items()}

    def has_weights(G: nx.Graph) -> bool:
        for _, _, data in G.edges(data=True):
            if isinstance(data, dict) and "weight" in data:
                return True
        return False

    def node_strengths(G: nx.Graph) -> dict:
        strengths = {n: 0.0 for n in G.nodes()}
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            for u, v, data in G.edges(data=True):
                w = float(data.get("weight", 1.0))
                strengths[u] += w
                strengths[v] += w
        else:
            for u, v, data in G.edges(data=True):
                w = float(data.get("weight", 1.0))
                strengths[u] += w
                strengths[v] += w
        return strengths

    rows = []
    missing = []
    connection_kind_per_group = {}

    for group, fmn in fmns.items():
        G = emo.formamentis_to_nxgraph(fmn)

        try:
            closeness = nx.closeness_centrality(G)
        except Exception:
            closeness = {n: 0.0 for n in G.nodes()}

        if has_weights(G):
            connection = node_strengths(G)
            connection_kind_per_group[group] = "strength"
        else:
            deg = dict(G.degree())
            connection = {n: float(deg.get(n, 0)) for n in G.nodes()}
            connection_kind_per_group[group] = "degree"

        if normalize_within_fmn:
            closeness_s = minmax_scale(closeness)
            connection_s = minmax_scale(connection)
        else:
            closeness_s = closeness
            connection_s = connection

        custom_sets = None
        if isinstance(per_fmn_custom_valences, dict):
            custom_sets = per_fmn_custom_valences.get(group, None)

        vertices = list(getattr(fmn, "vertices", G.nodes()))
        vmap_lower = {norm(v): v for v in vertices}

        for kw in keywords:
            kn = norm(kw)
            if kn not in vmap_lower:
                missing.append((group, kw))
                continue
            node = vmap_lower[kn]

            aura = compute_node_aura(
                fmn=fmn,
                target_word=node,
                emo=None,
                language=language,
                custom_valences=custom_sets,
                include_target=include_target,
                return_details=False,
                case_insensitive=False,
            )

            closeness_scaled_val = float(closeness_s.get(node, 0.0))
            connection_scaled_val = float(connection_s.get(node, 0.0))

            rows.append({
                "group": group,
                "keyword": kw,
                "node": node,
                "closeness": float(closeness.get(node, 0.0)),
                "closeness_scaled": closeness_scaled_val,
                "connection_kind": connection_kind_per_group[group],
                "connection": float(connection.get(node, 0.0)),
                "connection_scaled": connection_scaled_val,
                "aura_label": aura["aura_label"],
                "aura_cat": aura["aura_cat"],
                "pos_share": float(aura["shares"]["positive"]),
                "neu_share": float(aura["shares"]["neutral"]),
                "neg_share": float(aura["shares"]["negative"]),
                "frame_size": int(aura["frame_size"]),
                "considered_size": int(aura["considered_size"]),
                # Rounded to 2 decimals as requested
                "compound_index": (
                    round(closeness_scaled_val, 2),
                    round(connection_scaled_val, 2),
                    aura["aura_label"],
                ),
            })

    df = pd.DataFrame(rows).sort_values(["keyword", "group"]).reset_index(drop=True)

    try:
        wide = pd.concat(
            [
                df.pivot(index="keyword", columns="group", values="closeness_scaled").add_prefix("closeness|"),
                df.pivot(index="keyword", columns="group", values="connection_scaled").add_prefix("connection|"),
                df.pivot(index="keyword", columns="group", values="aura_label").add_prefix("aura|"),
            ],
            axis=1,
        )
    except Exception:
        wide = pd.DataFrame()

    meta = {
        "connection_kind_per_group": connection_kind_per_group,
        "missing": missing,
    }
    if missing:
        miss_str = ", ".join([f"{g}:{k}" for g, k in missing[:30]])
        extra = " ..." if len(missing) > 30 else ""
        print(f"[info] {len(missing)} (group, keyword) not found and were skipped: {miss_str}{extra}")

    return {"df": df, "wide": wide, "meta": meta}


def keyword_correlometer_bulk(
    fmns: dict,
    keywords: list[str],
    emo=None,
    per_fmn_custom_valences: dict | None = None,
    include_target: bool = False,
    language: str | None = None,
    annotate: bool = True,
    figsize=(8.0, 6.5),
    dpi=300,
):
    """
    Compute 'correlometer' (importance vs valence) for all keywords across all FMNs,
    and plot all points in a single figure. Legends are placed to avoid covering the plot:
      - Valence legend stays small inside the axes (lower right)
      - FMN group legend is moved outside at the bottom, in multiple columns
    """
    import math
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if language is None:
        language = getattr(emo, "language", "italian")

    # Reuse single-node function for aura
    def valence_for_node(fmn, node_label, custom_sets):
        return compute_node_aura(
            fmn=fmn,
            target_word=node_label,
            emo=None,
            language=language,
            custom_valences=custom_sets,
            include_target=include_target,
            return_details=False,
            case_insensitive=False,
        )

    # Min-max scaler for dicts
    def minmax_scale(d):
        vals = list(d.values())
        if not vals:
            return {n: 0.0 for n in d}
        vmin, vmax = min(vals), max(vals)
        if vmax <= vmin:
            return {n: 0.0 for n in d}
        return {n: (v - vmin) / (vmax - vmin) for n, v in d.items()}

    rows = []
    missing = []

    # Pre-assign a marker per FMN to distinguish groups
    marker_cycle = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "h", "8"]
    group_names = list(fmns.keys())
    marker_map = {g: marker_cycle[i % len(marker_cycle)] for i, g in enumerate(group_names)}

    # Color by aura
    color_map = {"positive": "#1f77b4", "neutral": "#7f7f7f", "negative": "#d62728"}

    # Precompute centralities per FMN once
    centralities_cache = {}

    for group, fmn in fmns.items():
        # Build NX Graph and centralities (cache)
        if group not in centralities_cache:
            G = emo.formamentis_to_nxgraph(fmn)
            # Robust centralities
            try:
                deg_c = nx.degree_centrality(G)
            except Exception:
                deg_c = {n: 0.0 for n in G.nodes()}
            try:
                bet_c = nx.betweenness_centrality(G, normalized=True)
            except Exception:
                bet_c = {n: 0.0 for n in G.nodes()}
            try:
                clo_c = nx.closeness_centrality(G)
            except Exception:
                clo_c = {n: 0.0 for n in G.nodes()}
            try:
                eig_c = nx.eigenvector_centrality(G, max_iter=1000)
            except Exception:
                try:
                    eig_c = nx.pagerank(G, alpha=0.85)
                except Exception:
                    eig_c = {n: 0.0 for n in G.nodes()}

            deg_s = minmax_scale(deg_c)
            bet_s = minmax_scale(bet_c)
            clo_s = minmax_scale(clo_c)
            eig_s = minmax_scale(eig_c)

            centralities_cache[group] = {
                "G": G,
                "deg_s": deg_s, "bet_s": bet_s, "clo_s": clo_s, "eig_s": eig_s
            }

        G = centralities_cache[group]["G"]
        deg_s = centralities_cache[group]["deg_s"]
        bet_s = centralities_cache[group]["bet_s"]
        clo_s = centralities_cache[group]["clo_s"]
        eig_s = centralities_cache[group]["eig_s"]

        # Resolve custom valences for this FMN if provided
        custom_sets = None
        if isinstance(per_fmn_custom_valences, dict):
            custom_sets = per_fmn_custom_valences.get(group, None)

        # Case-insensitive lookup of nodes in this FMN
        vertices = list(getattr(fmn, "vertices", G.nodes()))
        def norm(s): return str(s).casefold()
        vmap_lower = {norm(v): v for v in vertices}

        for kw in keywords:
            kn = norm(kw)
            if kn not in vmap_lower:
                missing.append((group, kw))
                continue
            node = vmap_lower[kn]

            # Importance = mean of scaled centralities
            components = {
                "degree": deg_s.get(node, 0.0),
                "betweenness": bet_s.get(node, 0.0),
                "closeness": clo_s.get(node, 0.0),
                "eigen_or_pagerank": eig_s.get(node, 0.0),
            }
            importance = float(np.mean(list(components.values()))) if components else 0.0

            # Valence via aura of node's frame
            aura = valence_for_node(fmn, node, custom_sets)
            valence_score = aura["shares"]["positive"] - aura["shares"]["negative"]

            rows.append({
                "group": group,
                "keyword": kw,
                "node": node,
                "degree": int(G.degree(node)) if node in G else 0,
                "importance": importance,
                "aura_label": aura["aura_label"],
                "aura_cat": aura["aura_cat"],
                "valence_score": float(valence_score),
                "pos_share": aura["shares"]["positive"],
                "neg_share": aura["shares"]["negative"],
                "neu_share": aura["shares"]["neutral"],
                "n_neighbors": int(aura["considered_size"]),
                "marker": marker_map[group],
                "color": color_map.get(aura["aura_label"], "#7f7f7f"),
            })

    df = pd.DataFrame(rows)

    # Plot all points in a single figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axvline(0.5, color="#cccccc", ls="--", lw=0.8)
    ax.axhline(0.0, color="#cccccc", ls="--", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Importance (composite)")
    ax.set_ylabel("Valence score (pos_share − neg_share)")
    ax.set_title("Keyword correlometer across FMNs")

    # Draw by group to vary marker, color by aura
    for group in group_names:
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        ax.scatter(
            sub["importance"], sub["valence_score"],
            s=70,
            c=sub["color"],
            marker=marker_map[group],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.95,
        )

    # Optional annotations
    if annotate:
        for _, r in df.iterrows():
            ax.annotate(
                r["keyword"],
                (r["importance"], r["valence_score"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.65),
                zorder=5,
            )

    # Legends
    # 1) Small valence legend inside axes (doesn't obstruct much)
    aura_handles = [
        Line2D([0], [0], marker='o', color='w', label=lab,
               markerfacecolor=col, markeredgecolor='black', markersize=8)
        for lab, col in {"positive": "#1f77b4", "neutral": "#7f7f7f", "negative": "#d62728"}.items()
    ]
    leg_val = ax.legend(handles=aura_handles, title="Valence",
                        loc="lower right", framealpha=0.9, fontsize=8, title_fontsize=9)
    ax.add_artist(leg_val)

    # 2) Big "FMN group" legend outside at the bottom, in multiple columns
    marker_handles = [
        Line2D([0], [0], marker=marker_map[g], color='w', label=g,
               markerfacecolor='#aaaaaa', markeredgecolor='black', markersize=8)
        for g in group_names
    ]
    n_groups = max(1, len(group_names))
    # Choose number of columns to keep legend compact
    if n_groups <= 8:
        ncol = 2
    elif n_groups <= 16:
        ncol = 4
    else:
        ncol = 6
    rows_needed = math.ceil(n_groups / ncol)
    # Reserve bottom margin proportional to number of rows in the legend
    bottom_margin = min(0.12 + 0.05 * (rows_needed - 1), 0.35)
    fig.subplots_adjust(bottom=bottom_margin)

    fig.legend(handles=marker_handles, title="FMN group",
               loc="lower center", ncol=ncol, framealpha=0.9,
               fontsize=8, title_fontsize=9)

    # Final styling
    ax.grid(True, alpha=0.15)
    # Keep space for bottom legend
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))

    if missing:
        miss_str = ", ".join([f"{g}:{k}" for g, k in missing[:30]])
        extra = " ..." if len(missing) > 30 else ""
        print(f"[info] {len(missing)} (group, keyword) not found and were skipped: {miss_str}{extra}")

    return {"df": df, "fig": fig, "ax": ax, "missing": missing}


# Add labels per facet
def _add_labels(data, color=None, **kws):
    ax = plt.gca()
    for _, r in data.iterrows():
        ax.annotate(
            r["keyword"],
            (r["importance"], r["valence_score"]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
            clip_on=True,
            zorder=6,
        )


def slope_compare_groups(
    result_df,
    g1,
    g2,
    metric="closeness_scaled",
    top_k=20,
    jitter=True,
    jitter_amount: float | None = None,
    jitter_seed: int | None = 0,
    min_sep_factor: float = 0.04,  # fraction of data-range used as minimum vertical separation
):
    """
    Draw a slopegraph comparing metric for keywords between g1 and g2.
    Uses a deterministic label-stacking algorithm to avoid overlaps.
    jitter_amount (data units) if provided overrides heuristic.
    jitter_seed controls any small random perturbation used for tie-breaking.
    """
    sub = result_df[result_df["group"].isin([g1, g2])]
    piv = sub.pivot(index="keyword", columns="group", values=metric).dropna()
    if piv.empty:
        raise ValueError("No keywords found for the two groups with the requested metric.")
    piv["avg"] = piv[[g1, g2]].mean(axis=1)
    piv = piv.sort_values("avg", ascending=False).head(top_k).drop(columns="avg")

    fig, ax = plt.subplots(figsize=(6, 0.35 * len(piv) + 1), dpi=130)

    # data range and jitter / separation heuristics
    vals = piv[[g1, g2]].values.flatten()
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    data_range = max(1e-6, vmax - vmin)
    if jitter_amount is None:
        jitter_amount = data_range * 0.02

    # deterministic RNG for reproducibility
    rng = np.random.default_rng(jitter_seed) if jitter_seed is not None else np.random.default_rng()

    # draw lines and dots
    for i, (kw, row) in enumerate(piv.iterrows()):
        y1, y2 = float(row[g1]), float(row[g2])
        ax.plot([0, 1], [y1, y2], color="#6b7280", lw=1.5, zorder=1)
        ax.scatter([0, 1], [y1, y2], color=["#2563eb", "#16a34a"], s=30, zorder=3)

    # LABEL PLACEMENT: compute non-overlapping y positions for left-side labels
    left_y = np.array([float(v) for v in piv[g1].values], dtype=float)
    n = len(left_y)
    if n == 0:
        return fig, ax

    # small random tie-break to avoid identical values stacking deterministically
    if jitter:
        left_y = left_y + rng.uniform(-jitter_amount*0.2, jitter_amount*0.2, size=n)

    # minimal separation in data units
    min_sep = max(1e-6, (jitter_amount if jitter_amount is not None else data_range * min_sep_factor))
    # stacking algorithm: forward then backward pass
    order = np.argsort(left_y)
    y_sorted = left_y[order].copy()

    # forward pass
    for i in range(1, n):
        if y_sorted[i] - y_sorted[i - 1] < min_sep:
            y_sorted[i] = y_sorted[i - 1] + min_sep
    # backward pass
    for i in range(n - 2, -1, -1):
        if y_sorted[i + 1] - y_sorted[i] < min_sep:
            y_sorted[i] = y_sorted[i + 1] - min_sep

    # Map back to original order
    adjusted_left_y = np.empty_like(y_sorted)
    adjusted_left_y[order] = y_sorted

    # optionally clamp adjusted positions to data min/max plus a small pad
    pad = max(min_sep, data_range * 0.01)
    global_min, global_max = vmin - pad, vmax + pad
    adjusted_left_y = np.clip(adjusted_left_y, global_min, global_max)

    # Draw left-side labels at adjusted positions
    x_label_pos = -0.02
    for (kw, _), ylab in zip(piv.iterrows(), adjusted_left_y):
        ax.text(x_label_pos, float(ylab), kw, ha="right", va="center", fontsize=8, zorder=4)

    # Also draw right-side short labels near the right points to help readability (optional)
    # keep them aligned to the right point but avoid overlap with left labels by small horizontal offset
    x_right = 1.02
    for (kw, row) in piv.iterrows():
        y2 = float(row[g2])
        ax.text(x_right, y2, "", ha="left", va="center", fontsize=8, zorder=4)  # empty by default

    # ticks / limits
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g1, g2])
    ax.set_xlim(-0.45, 1.45)

    # ensure y-limits include adjusted labels
    y_all = np.concatenate([piv[g1].values, piv[g2].values, adjusted_left_y])
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    y_pad = max(0.02 * (y_max - y_min if y_max > y_min else 1.0), min_sep)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # set y-axis label to the metric (make it human readable)
    y_label = str(metric).replace("_", " ").title()
    ax.set_ylabel(y_label)

    ax.set_title(f"{metric} comparison")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    return fig, ax


