import os
import re
import numpy as np
import pandas as pd
import psutil
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
import gc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json
import networkx as nx
from tqdm import tqdm
from scipy import stats
from emoatlas import EmoScores
from emoatlas.resources import _valences
from typing import Any, Dict, List, Optional

from .data_processing import lemmatize_text_it
from .io import export_whole_fmn_custom


def monitor_memory():
    """Monitor current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.2f} MB")
    return memory_mb


def verify_and_report_nodes(fmns, required, case_insensitive=True, strict=True):
    """
    Minimal unified verification:
      - Checks required nodes per FMN
      - Prints missing nodes per network
      - Prints global summary + per-node coverage
      - No DataFrame creation

    Parameters
    ----------
    fmns : dict[str, FormamentisNetwork] or list[FormamentisNetwork]
    required : iterable[str]
    case_insensitive : bool
    strict : bool
        True -> network must contain ALL required nodes
        False -> network must contain ANY required node

    Returns
    -------
    summary : dict
    """
    if not required:
        raise ValueError("required must be a non-empty iterable")
    required = list(required)

    # Print header
    print(f"Presence of nodes {required} in {'all' if strict else 'any'} FMNs:")

    from collections import Counter
    present_counter = Counter()

    def check_fmnt(fmnt):
        nodes = getattr(fmnt, "vertices", [])
        if case_insensitive:
            lower_map = {n.lower(): n for n in nodes}
            present = [lower_map[r.lower()] for r in required if r.lower() in lower_map]
            missing = [r for r in required if r.lower() not in lower_map]
        else:
            node_set = set(nodes)
            present = [r for r in required if r in node_set]
            missing = [r for r in required if r not in node_set]
        if strict:
            result = len(missing) == 0
            mode = "all"
        else:
            result = len(present) > 0
            mode = "any"
        return {
            "result": result,
            "mode": mode,
            "present": present,
            "missing": missing,
            "present_count": len(present),
            "missing_count": len(missing),
            "required_total": len(required),
            "network_total": len(nodes)
        }

    results = {}
    missing_fmnts = {}

    if not isinstance(fmns, dict):
        fmns = {f"fmnt_{i+1}": fmnt for i, fmnt in enumerate(fmns)}

    for title, fmnt in fmns.items():
        res = check_fmnt(fmnt)
        results[title] = res
        # Update per-node presence
        present_req_forms = {r.lower() for r in required} if case_insensitive else set(required)
        for r in required:
            if (r.lower() in {p.lower() for p in res["present"]}) if case_insensitive else (r in res["present"]):
                present_counter[r] += 1
        if not res["result"]:
            missing_fmnts[title] = res["missing"]
            print(f"FMN '{title}' missing: {res['missing']}")

    total = len(fmns)
    satisfying = sum(1 for r in results.values() if r["result"])
    percentage = (satisfying / total * 100) if total else 0.0

    print(f"\nNetworks satisfying ({'ALL' if strict else 'ANY'} of {required}): {satisfying}/{total} ({percentage:.2f}%)")
    for node in required:
        count = present_counter[node]
        print(f"Presence '{node}': {count}/{total} ({(count/total*100 if total else 0):.2f}%)\n")

    summary = {
        "required": required,
        "mode": "all" if strict else "any",
        "total_fmns": total,
        "fmns_with_requirement": satisfying,
        "percentage": percentage,
        "per_node_presence": {
            node: {
                "count": present_counter[node],
                "percentage": (present_counter[node] / total * 100) if total else 0.0
            } for node in required
        },
        "missing_fmnts": missing_fmnts,
        "results": results
    }
    return summary


def build_participants_bfmns(
    df,
    emo,
    directed=False,
    lemmatize=False,
    keep_original_columns=True
):
    """
    Builds one BFMN per participant_id from a long DataFrame with required columns:
        participant_id, cue_word, word, index, valence_cat (already computed globally).

    Notes:
        Rows with index == 0 register the cue node (no edge should be created from cue to itself).
        Rows with index > 0 where word == cue_word are ignored to avoid self-loops.

    Optional lemmatization (italian, spaCy):
        If lemmatize=True:
            - Columns 'cue_word_lemma' and 'word_lemma' are created.
            - Nodes and edges use lemmatized forms.
            - Valence is propagated at lemma level: conflicts resolved by majority class; ties -> 0 (neutral).

    Parameters:
        df : long-format DataFrame.
        emo : an initialized EmoScores class object.
        directed : bool, if True uses nx.DiGraph (directed) else nx.Graph (undirected). Default: False (undirected).
        lemmatize : bool, apply italian lemmatization.
        keep_original_columns : bool, if False drop auxiliary lemma columns at the end.

    Returns:
        all_bfmns : dict {participant_id: FormamentisNetwork}
        node_val_maps : dict {participant_id: {node: valence_cat}}
    """
    if emo is None:
        raise ValueError("Parameter 'emo' (EmoScores) is required.")

    required_cols = {"participant_id", "cue_word", "word", "index", "valence_cat"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Mancano colonne richieste: {missing}")

    work_df = df.copy()

    if lemmatize:
        # secure lemmatization (keeps NaN)
        def _safe_lem(x):
            if pd.isna(x):
                return x
            try:
                return lemmatize_text_it(str(x))
            except Exception:
                return x  # fallback

        work_df["cue_word_lemma"] = work_df["cue_word"].apply(_safe_lem)
        work_df["word_lemma"] = work_df["word"].apply(_safe_lem)

        # Propagazione valenze ai lemmi:
        # Se lo stesso lemma riceve etichette diverse, usa la moda; se parità -> 0 (neutro)
        lemma_groups = (
            work_df.dropna(subset=["word_lemma"])
            .groupby("word_lemma")["valence_cat"]
            .agg(lambda s: _resolve_valence_conflicts(s))
        )
        # Mappa lemma -> valence_cat aggregata
        lemma_val_map = lemma_groups.to_dict()

    def _resolve_valence_conflicts(series):
        # Return a Series containing counts of unique values in descending order
        counts = series.value_counts(dropna=True)
        if counts.empty:
            return 0
        if len(counts) == 1:
            return counts.index[0]
        # If there is tie between different categories => neutral (0)
        top = counts[counts == counts.max()].index.tolist()
        return 0 if len(top) > 1 else top[0]

    all_bfmns = {}
    features_rows = []

    node_val_maps = {}  # store per-participant node valences

    requested_graph_type = nx.DiGraph if directed else nx.Graph

    for participant_id, p_df in tqdm(work_df.groupby("participant_id"), desc="Building BFMNs"):
        G = requested_graph_type()

        if lemmatize:
            # Use lemmatized forms for nodes and edges
            node_df = (
                p_df[["word_lemma", "valence_cat"]]
                .dropna(subset=["word_lemma"])
                .drop_duplicates()
                .rename(columns={"word_lemma": "node"})
            )
            # Replace valence with aggregated one (if present)
            node_df["valence_cat"] = node_df["node"].map(lemma_val_map).fillna(node_df["valence_cat"]).astype(int)
        else:
            node_df = (
                p_df[["word", "valence_cat"]]
                .dropna(subset=["word"])
                .drop_duplicates()
                .rename(columns={"word": "node"})
            )

        for node, vcat in node_df.itertuples(index=False):
            label = "positive" if vcat == 1 else "negative" if vcat == -1 else "neutral"
            G.add_node(node, valence=label, valence_cat=int(vcat))

        # cue edges -> response (esclude index==0 e self-loop)
        assoc_rows = p_df[(p_df["index"] > 0)]
        if lemmatize:
            assoc_rows = assoc_rows.dropna(subset=["cue_word_lemma", "word_lemma"])
            assoc_rows = assoc_rows[assoc_rows["cue_word_lemma"] != assoc_rows["word_lemma"]]
            edge_iter = zip(assoc_rows["cue_word_lemma"], assoc_rows["word_lemma"])
        else:
            assoc_rows = assoc_rows.dropna(subset=["cue_word", "word"])
            assoc_rows = assoc_rows[assoc_rows["cue_word"] != assoc_rows["word"]]
            edge_iter = zip(assoc_rows["cue_word"], assoc_rows["word"])

        G.add_edges_from(edge_iter)

        # Create FMN object from nx graph. All edges are considered as syntactic
        fmnt = emo.nxgraph_to_formamentis(G)

        # Collect node valences separately (cannot attach to namedtuple)
        node_val_maps[participant_id] = {n: d.get("valence_cat") for n, d in G.nodes(data=True)}

        # Store in dict
        all_bfmns[participant_id] = fmnt

    if lemmatize and not keep_original_columns:
        drop_cols = ["cue_word_lemma", "word_lemma"]
        work_df.drop(columns=[c for c in drop_cols if c in work_df.columns], inplace=True)

    return all_bfmns, node_val_maps


# A function that builds aggregated BFMNs from a CSV file (categorized valence)
def build_aggregated_bfmn(df, emo, resolve_strategy="majority"):
    '''Build an aggregated undirected forma mentis network from the association dataframe.
       Removes self-loops and duplicate undirected edges.
       Returns (fmn, node_val_map) where node_val_map is {node: valence_cat_int}.
       resolve_strategy: "majority" (ties -> 0) or "first".
    '''
    if emo is None:
        raise ValueError("Parameter 'emo' (EmoScores) is required.")
    G = nx.Graph()
    edges = []
    skipped_self_loops = 0
    skipped_missing = 0

    required_cols = {"participant_id", "cue_word", "word", "index", "valence_cat"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Mancano colonne richieste: {missing}")

    work_df = df.copy()

    # Collect valence evidence for each node (both cue_word and word)
    cue_df = (
        work_df[["cue_word", "valence_cat"]]
        .dropna(subset=["cue_word"])
        .rename(columns={"cue_word": "node"})
    )
    assoc_df = (
        work_df[["word", "valence_cat"]]
        .dropna(subset=["word"])
        .rename(columns={"word": "node"})
    )
    valence_evidence = pd.concat([cue_df, assoc_df], ignore_index=True)

    def _resolve_valence(series):
        if resolve_strategy == "first":
            return int(series.iloc[0])
        # majority strategy
        counts = series.value_counts(dropna=True)
        if counts.empty:
            return 0
        top = counts[counts == counts.max()].index.tolist()
        return 0 if len(top) > 1 else int(top[0])

    node_val_map = (
        valence_evidence.groupby("node")["valence_cat"]
        .agg(_resolve_valence)
        .to_dict()
    )

    for _, row in work_df.iterrows():
        cue = row.get('cue_word')
        association = row.get('word')

        # Skip missing
        if pd.isna(cue) or pd.isna(association):
            skipped_missing += 1
            continue

        cue = str(cue).strip()
        association = str(association).strip()
        if not cue or not association:
            skipped_missing += 1
            continue

        # Skip self-records: index==0 rows and cue==association
        if ('index' in work_df.columns and row.get('index', None) == 0) or (cue == association):
            skipped_self_loops += 1
            continue

        # Add nodes (attributes assigned later)
        G.add_node(cue)
        G.add_node(association)
        edges.append((cue, association))

    # Deduplicate undirected edges
    undirected_edges = list({tuple(sorted(e)) for e in edges})

    print(f"Skipped self-loops: {skipped_self_loops} | Skipped missing/empty: {skipped_missing}")
    print(f"Candidate edges: {len(edges)} | Unique undirected edges: {len(undirected_edges)}")

    G.add_edges_from(undirected_edges)
    print(f"Number of edges in graph: {G.number_of_edges()}")
    print(f"Number of unique cue words: {work_df['cue_word'].nunique(dropna=True)}")
    print(f"Number of unique associated words: {work_df['word'].nunique(dropna=True)}")

    # Assign valence attributes
    for n in G.nodes():
        vcat = int(node_val_map.get(n, 0))
        label = "positive" if vcat == 1 else "negative" if vcat == -1 else "neutral"
        G.nodes[n]["valence_cat"] = vcat
        G.nodes[n]["valence"] = label

    fmn = emo.nxgraph_to_formamentis(G)
    return fmn, node_val_map


def extract_bfmn_structural_features(fmnts, save_path, emo, batch_size=10, node_valence_maps=None, persist=True):
    """
    Extract, export and persist (if persist) per-participant network structural features.
    Writes one JSON and one .txt per FMN under output_dir/output_subdir.

    Parameters:
    fmnts (dict): {participant_id: FormaMentisNetwork}
    save_path (str): Base output directory
    emo (EmoScores): Initialized EmoScores object
    batch_size (int): Batch size
    node_valence_maps (dict or None): {participant_id: {node: valence_cat_int(-1,0,1)}}

    Returns:
    dict: {participant_id: structural_features_dict}
    """
    if emo is None:
        raise ValueError("Parameter 'emo' (EmoScores) is required.")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_dir = save_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def _sanitize_number(x):
        '''Sanitize number by converting to float and handling NaN/Inf
        '''
        try:
            if x is None:
                return None
            if isinstance(x, (float, int)):
                if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
                    return None
                return float(x)
        except Exception:
            return None
        return x

    def _valence_label(vcat):
        '''Map valence category to label'''
        if vcat == 1:
            return "positive"
        if vcat == -1:
            return "negative"
        return "neutral"

    # Create a list of (participant_id, fmnt) tuples for batching
    items = list(fmnts.items())
    total_items = len(items)
    print(f"Extracting features from {total_items} networks in batches of {batch_size}")
    # Initial memory usage
    monitor_memory()

    net_str_features_dict = {}

    for i in range(0, total_items, batch_size):
        batch_items = items[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(total_items + batch_size - 1)//batch_size}")

        for fmn_title, fmnt in batch_items:
            if not hasattr(fmnt, "vertices") or len(fmnt.vertices) == 0:
                print(f"Skipping {fmn_title}: empty network.")
                continue

            # Build an undirected NX graph from the FMN
            G = nx.Graph()
            G.add_nodes_from(fmnt.vertices)
            G.add_edges_from(fmnt.edges)

            # Apply node valence attributes if provided (fallback neutral)
            node_val_map = {}
            if isinstance(node_valence_maps, dict):
                node_val_map = node_valence_maps.get(fmn_title, {}) or {}

            # Each node ends up with both a numeric (valence_cat) and string (valence)
            # attribute, enabling later feature calculations (counts, assortativity)
            # and readable outputs.
            for n in G.nodes():
                # Get the valence category for the node (default 0 if not found)
                vcat = node_val_map.get(n, 0)
                # Sanitize and assign the valence attributes for the node.
                # Both numeric and string attributes for valence are set.
                G.nodes[n]["valence_cat"] = int(vcat) if vcat is not None else 0
                G.nodes[n]["valence"] = _valence_label(G.nodes[n]["valence_cat"])

            # Basic counts
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            n_pos = sum(1 for _, d in G.nodes(data=True) if d.get("valence_cat") == 1)
            n_neg = sum(1 for _, d in G.nodes(data=True) if d.get("valence_cat") == -1)

            # Degree stats (undirected graph)
            degrees = [d for _, d in G.degree()]
            min_degree = min(degrees) if degrees else 0
            max_degree = max(degrees) if degrees else 0

            # Local graph-level average clustering.
            # It is the mean of the local clustering coefficients of all the nodes in the graph.
            try:
                # Try the standard method first (more accurate)
                avg_clust = nx.average_clustering(G) if n_nodes > 1 else 0.0
            except Exception:
                try:
                    # Fallback to approximation (less accurate but more robust)
                    avg_clust = nx.algorithms.approximation.clustering_coefficient.average_clustering(G)
                except Exception:
                    avg_clust = None

            # Degree assortativity
            try:
                degree_assortativity = nx.degree_assortativity_coefficient(G)
            except Exception:
                degree_assortativity = None

            # Attribute assortativity for 'valence'
            try:
                valence_assortativity = nx.attribute_assortativity_coefficient(G, "valence")
            except Exception:
                valence_assortativity = None

            # Closeness centrality extremes
            try:
                closeness = nx.closeness_centrality(G)
                max_closeness = max(closeness.values()) if closeness else None
                min_closeness = min(closeness.values()) if closeness else None
            except Exception:
                max_closeness = None
                min_closeness = None

            # ASPL, diameter, nodes and edges of the largest connected component
            # LCC + path stats
            if n_nodes == 0:
                lcc_aspl = None
                lcc_diameter = None
                lcc_nodes = 0
                lcc_edges = 0
            elif n_nodes == 1:
                lcc_aspl = 0.0
                lcc_diameter = 0
                lcc_nodes = 1
                lcc_edges = 0
            else:
                try:
                    undirected_G = G.to_undirected() if G.is_directed() else G
                    if nx.is_connected(undirected_G):
                        H = undirected_G
                    else:
                        largest_cc = max(nx.connected_components(undirected_G), key=len)
                        H = undirected_G.subgraph(largest_cc).copy()
                    lcc_nodes = H.number_of_nodes()
                    lcc_edges = H.number_of_edges()
                    lcc_aspl = nx.average_shortest_path_length(H) if lcc_nodes > 1 else 0.0
                    lcc_diameter = nx.diameter(H) if lcc_nodes > 1 else 0
                except Exception:
                    lcc_aspl = None
                    lcc_diameter = None
                    lcc_nodes = H.number_of_nodes() if 'H' in locals() else 0
                    lcc_edges = H.number_of_edges() if 'H' in locals() else 0

            # Compute z-scores using emo module
            try:
                zscores = emo.zscores(fmnt)
            except Exception:
                zscores = {}

            # Compute local clustering
            try:
                local_clustering = nx.clustering(G)
            except Exception:
                local_clustering = {}

            # Compute hubs: top 5 concepts by highest degree
            try:
                # DegreeView is iterable of (node, degree) pairs; convert/sort directly
                degree_pairs = list(G.degree())
                sorted_degrees = sorted(degree_pairs, key=lambda x: x[1], reverse=True)[:5]
            except Exception:
                sorted_degrees = []

            structural_features = {
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "n_positive_nodes": n_pos,
                "n_negative_nodes": n_neg,
                "min_degree": _sanitize_number(min_degree),
                "max_degree": _sanitize_number(max_degree),
                "average_clustering": _sanitize_number(avg_clust),
                "local_clustering": {k: _sanitize_number(v) for k, v in local_clustering.items()},
                "degree_assortativity": _sanitize_number(degree_assortativity),
                "valence_assortativity": _sanitize_number(valence_assortativity),
                "max_closeness_centrality": _sanitize_number(max_closeness),
                "min_closeness_centrality": _sanitize_number(min_closeness),
                "lcc_nodes": _sanitize_number(lcc_nodes),
                "lcc_edges": _sanitize_number(lcc_edges),
                "lcc_average_shortest_path_length": _sanitize_number(lcc_aspl),
                "lcc_diameter": _sanitize_number(lcc_diameter),
                "zscores": {k: _sanitize_number(v) for k, v in zscores.items()},
                "top_5_hubs_by_degree": [{"node": n, "degree": _sanitize_number(d)} for n, d in sorted_degrees],
            }

            # Save structural features in JSON
            out_path_json = os.path.join(output_dir, f"{fmn_title}_structural_features.json")
            with open(out_path_json, "w", encoding="utf-8") as f:
                json.dump(structural_features, f, ensure_ascii=False, indent=2)
            print(f"Saved at {out_path_json}")

            # Save TXT dump
            out_path_txt = os.path.join(output_dir, f"{fmn_title}_structural_features.txt")
            with open(out_path_txt, "w", encoding="utf-8") as f:
                for k, v in structural_features.items():
                    f.write(f"{k}: {v}\n")
            print(f"Saved at {out_path_txt}")

            net_str_features_dict[fmn_title] = structural_features

        monitor_memory()
        gc.collect()

    print(f"Saved structural features for {len(net_str_features_dict)} networks\n")
    return net_str_features_dict


def make_emo_flowers(fmnts: dict, save_path: str, filename: str, emo, batch_size=10, svg: bool = False):
    """
    Generates emotional flowers for each forma mentis network found in fmnts.
    Saves images to save_path without displaying them.
    """
    if emo is None:
        raise ValueError("Parameter 'emo' (EmoScores) is required.")
    os.makedirs(save_path, exist_ok=True)

    items = list(fmnts.items())
    total_items = len(items)
    print(f"Processing {total_items} emotional flowers in batches of {batch_size}")

    for i in range(0, total_items, batch_size):
        batch_items = items[i:i+batch_size]
        print(f"\nProcessing flower batch {i//batch_size + 1}/{(total_items + batch_size - 1)//batch_size}")
        monitor_memory()

        for fmn_title, fmnt in batch_items:
            if not hasattr(fmnt, "vertices") or len(fmnt.vertices) == 0:
                print(f"Skipping {fmn_title}: empty network.")
                continue
            try:
                z_scores = emo.zscores(fmnt)
                emo.draw_plutchik(z_scores, reject_range=(-1.96, 1.96))

                base = filename if filename else fmn_title
                png_path = os.path.join(out_dir, f"{base}.png")
                fig = plt.gcf()
                fig.savefig(png_path, dpi=300)
                print(f"Saved {png_path}")

                if svg:
                    svg_path = os.path.join(out_dir, f"{base}.svg")
                    fig.savefig(svg_path)
                    print(f"Saved {svg_path}")
            except Exception as e:
                print(f"Error processing flower for {fmn_title}: {e}")
            finally:
                plt.close('all')
        gc.collect()


def all_fmns_features_df(features_dict: dict, orient: str = "participants",
                         count_lists: bool = False,
                         list_suffix: str = "_count",
                         drop_nested_dicts: bool = False) -> pd.DataFrame:
    """
    Build a DataFrame from the features_dict returned by extract_fmn_structural_features.

    Parameters
    ----------
    features_dict : dict
        {participant_title: {feature_key: value}}
    orient : str
        "participants" -> rows=participants, columns=features (requested)
        "features"     -> rows=features, columns=participants (original behavior)
    count_lists : bool
        If True store length of list features (e.g. positive_nodes_count).
    list_suffix : str
        Suffix appended when counting list features.
    drop_nested_dicts : bool
        If True skip nested dicts (e.g. layer_metrics). Set False to flatten one level.
    """
    def _safe_col(name):
        # keep it readable but safe for CSV headers
        return str(name).replace(" ", "_")

    rows = []
    for participant, feat_dict in features_dict.items():
        row = {"participant_id": participant}
        for k, v in feat_dict.items():
            # Special flattening
            if k == "local_clustering" and isinstance(v, dict):
                for node, coeff in v.items():
                    row[f"local_clustering.{_safe_col(node)}"] = coeff
                continue
            if k == "zscores" and isinstance(v, dict):
                for emo_k, z in v.items():
                    row[f"zscores.{_safe_col(emo_k)}"] = z
                continue
            if k == "top_5_hubs_by_degree" and isinstance(v, list):
                for i, item in enumerate(v, start=1):
                    if isinstance(item, dict):
                        row[f"hub_{i}_node"] = item.get("node")
                        row[f"hub_{i}_degree"] = item.get("degree")
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        row[f"hub_{i}_node"] = item[0]
                        row[f"hub_{i}_degree"] = item[1]
                # If fewer than 5 hubs, ensure consistent columns
                for i in range(len(v) + 1, 6):
                    row.setdefault(f"hub_{i}_node", None)
                    row.setdefault(f"hub_{i}_degree", None)
                continue

            # Generic handling
            if isinstance(v, dict):
                if drop_nested_dicts:
                    continue
                for sk, sv in v.items():
                    if isinstance(sv, (list, dict)):
                        continue
                    row[f"{k}.{_safe_col(sk)}"] = sv
            elif isinstance(v, list):
                if count_lists:
                    row[f"{k}{list_suffix}"] = len(v)
                else:
                    # Store as JSON-like string to keep content
                    try:
                        row[k] = json.dumps(v, ensure_ascii=False)
                    except Exception:
                        row[k] = str(v)
            else:
                row[k] = v
        rows.append(row)

    if orient == "participants":
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("participant_id").sort_index()
        return df

    # Original wide style (features as rows)
    series_list = []
    for r in rows:
        participant_id = r.pop("participant_id")
        series_list.append(pd.Series(r, name=participant_id))
    if not series_list:
        return pd.DataFrame()
    wide_df = pd.concat(series_list, axis=1)
    return wide_df


def inspect_fmn(fmn, participant = str, all_features = dict, emo=None):
    '''Inspection of a single FMN (also multi-layer). Converts to NetworkX, prints sizes and basic checks.
       Works whether fmnt.edges is a dict (e.g., {'syntactic': [...], 'synonyms': [...]})
       or a flat iterable of 2-tuples.

    Parameters:
        fmn : FormaMentisNetwork (also multi-layer)
        participant : str
            Participant identifier for logging.
        all_features : dict
            {participant: features_dict} for printing features.
        emo : EmoScores
    '''
    if emo is None:
        raise ValueError("Parameter 'emo' (EmoScores) is required.")
    
    def _collect_edges(edges_attr):
        # Normalize edges from different fmnt formats into a list of (u, v)
        out = []
        if edges_attr is None:
            return out
        if isinstance(edges_attr, dict):
            for lst in edges_attr.values():
                for e in lst:
                    if isinstance(e, (list, tuple)) and len(e) >= 2:
                        out.append((e[0], e[1]))
        else:
            for e in edges_attr:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    out.append((e[0], e[1]))
        return out

    print(f"FMN: {participant}")

    # Before conversion: get vertices/edges if available
    n_vertices = len(getattr(fmn, "vertices", [])) if hasattr(fmn, "vertices") else None
    fmnt_edges = _collect_edges(getattr(fmn, "edges", None))
    fmnt_edges = list(set(tuple(map(str, e)) for e in fmnt_edges))  # dedup; ensure strings
    print("Before conversion to nx graph:")
    if n_vertices is not None:
        print(f"{n_vertices} nodes and {len(fmnt_edges)} edges")
    else:
        print(f"(vertices not available) edges: {len(fmnt_edges)}")

    # Convert to NetworkX
    fmn_G = emo.formamentis_to_nxgraph(fmn)
    print("\nAfter conversion to nx graph:")
    print(fmn_G)

    # Basic consistency check (only if we could read vertices)
    if n_vertices is not None and fmn_G.number_of_nodes() != n_vertices:
        print(f"Warning: Node count mismatch after conversion! for {participant}")

    # Compare edges (robust to undirected orientation)
    missing_edges = []
    if fmnt_edges:
        G_edges = set()
        if isinstance(fmn_G, (nx.Graph, nx.DiGraph)):
            if fmn_G.is_directed():
                G_edges = set((str(u), str(v)) for u, v in fmn_G.edges())
            else:
                G_edges = set(tuple(sorted((str(u), str(v)))) for u, v in fmn_G.edges())
        for u, v in fmnt_edges:
            e = (str(u), str(v))
            if fmn_G.is_directed():
                if e not in G_edges:
                    missing_edges.append((u, v))
            else:
                if tuple(sorted(e)) not in G_edges:
                    missing_edges.append((u, v))
    if missing_edges:
        print(f"Warning: {len(missing_edges)} edges from FMN not found in NetworkX graph for {participant}: {missing_edges[:20]}{' ...' if len(missing_edges) > 20 else ''}")

    # Print provided features (if any)
    print("\n--------------------------------------------")
    print(f"Features for {participant}:")
    print("--------------------------------------------")
    features_for_participant = (all_features or {}).get(participant, {})
    if isinstance(features_for_participant, dict):
        for k, v in features_for_participant.items():
            print(k, v)
    else:
        print(features_for_participant)
    print("\n---\n")


def draw_fmnts(
    fmnts,
    out_dir,
    layout,
    highlight,
    thickness,
    ax=None,
    emo=None,
    custom_valences=None,
    filename_base=None,
    batch_size=5,
    figsize=(10, 10),
    dpi=300,
    bbox_pad=0.08,
    margin=0.05,
    fix_center_mask=True,
    bring_edges_to_front=True,
):
    """
    Draw FMNs and save images directly in out_dir (no subfolders).
    If filename_base is given, it is used; otherwise <fmnt_title>.

    Notes
    - For edge_bundling/circle layout, EmoAtlas draws a central white patch to hide the hub.
      Set fix_center_mask=True to make that patch transparent and avoid "missing" edges near the center.
    - bbox_inches='tight' + pad_inches keeps labels/curves fully in-frame.
    """
    if emo is None:
        raise ValueError("Parameter 'emo' (EmoScores) is required.")
    os.makedirs(out_dir, exist_ok=True)

    items = list(fmnts.items())
    total_items = len(items)
    print(f"Drawing {total_items} networks in batches of {batch_size}")

    for i in range(0, total_items, batch_size):
        batch_items = items[i:i + batch_size]
        print(f"\nDrawing batch {i // batch_size + 1}/{(total_items + batch_size - 1) // batch_size}")
        monitor_memory()

        for fmnt_title, fmnt in batch_items:
            if not hasattr(fmnt, "vertices") or len(fmnt.vertices) == 0:
                print(f"Skipping {fmnt_title}: the network has zero vertices.")
                continue

            print(f"Drawing TFMN: {fmnt_title}")

            # Create a figure only if caller didn't pass one
            created_fig = False
            if ax is None:
                fig, ax_ = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
                created_fig = True
            else:
                ax_ = ax
                fig = ax_.figure

            # Draw with EmoAtlas
            emo.draw_formamentis(
                fmn=fmnt,
                layout=layout,
                highlight=highlight,
                thickness=thickness,
                ax=ax_,
                save_path=None,  # prevent internal saving
                custom_valences=custom_valences,
            )

            # Optional: neutralize the central white mask that hides bundled arcs
            if fix_center_mask or bring_edges_to_front:
                try:
                    # Find smallest circle patch (the hub used by the circle layout)
                    circles = [p for p in ax_.patches if isinstance(p, Circle) and hasattr(p, "radius")]
                    if fix_center_mask and circles:
                        hub = min(circles, key=lambda c: getattr(c, "radius", float("inf")))
                        # Make it fully transparent
                        hub.set_facecolor((1, 1, 1, 0))
                        hub.set_edgecolor((1, 1, 1, 0))
                        hub.set_alpha(0.0)
                        hub.set_visible(True)
                        # Push it to the back
                        hub.set_zorder(0)

                    if bring_edges_to_front:
                        # Lines and curves live in collections
                        for coll in ax_.collections:
                            coll.set_zorder(20)
                        # Keep labels on top
                        for txt in ax_.texts:
                            txt.set_zorder(30)
                        # Other patches (outer rings, etc.) under the edges
                        for p in ax_.patches:
                            if not isinstance(p, Circle):
                                p.set_zorder(10)
                except Exception as e:
                    print(f"[WARN] Could not adjust center mask/zorder: {e}")

            # Ensure full content fits
            ax_.set_aspect("equal", adjustable="datalim")
            ax_.relim()
            ax_.autoscale_view()
            ax_.margins(x=margin, y=margin)
            ax_.axis("off")

            base = filename_base if filename_base else fmnt_title
            png_path = os.path.join(out_dir, f"{base}.png")
            svg_path = os.path.join(out_dir, f"{base}.svg")

            # Save tightly with padding to include outer labels/curves
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=bbox_pad, facecolor="white")
            fig.savefig(svg_path, bbox_inches="tight", pad_inches=bbox_pad, facecolor="white")
            print(f"Saved {png_path}")
            print(f"Saved {svg_path}")

            if created_fig:
                plt.close(fig)

        gc.collect()


# For future emoatlas pull request
def analyze_semantic_frame_concreteness(
    self,
    semantic_frame: Any,
    df_brysbaert: pd.DataFrame,
    seed: Optional[int] = None,
    deduplicate: bool = True,
    plot: bool = True,
    save_json_path: Optional[str] = None,
    save_figure_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Concreteness analysis with lemma-level aggregation.

    Averages Conc.M over all lexical forms that lemmatize to the same lemma:
      multiple rows (e.g. 'abbreviate', 'abbreviated', 'abbreviation') -> lemma 'abbreviare' (it) or 'abbreviate' (en)
      stored as single lemma score (mean Conc.M) plus list of contributing forms.

    Multi-word frame terms: direct lemma match first; else average component lemma scores.
    """
    lang = (getattr(self, "language", None) or globals().get("language") or "italian").lower()
    if lang not in {"italian", "it", "english", "en"}:
        raise ValueError(f"Unsupported language: {lang}")
    required_cols = {'english', 'italian', 'Conc.M'}
    miss = required_cols - set(df_brysbaert.columns)
    if miss:
        raise KeyError(f"Missing columns: {miss}")
    if not hasattr(semantic_frame, "vertices"):
        raise ValueError("semantic_frame must have .vertices")

    nlp_model = getattr(self, "nlp", None) or getattr(self, "_nlp", None) or globals().get("nlp")

    def _norm(t: Any) -> str:
        return re.sub(r'\s+', ' ', str(t).strip().lower())

    def _lemmatize_single(token: str) -> str:
        if not token:
            return token
        if lang in {"italian", "it"}:
            if nlp_model is None:
                return token
            try:
                doc = nlp_model(token)
                return " ".join(t.lemma_.lower() for t in doc if not t.is_space and not t.is_punct)
            except Exception:
                return token
        else:
            try:
                from nltk.stem import WordNetLemmatizer
                from nltk import pos_tag, word_tokenize
                from nltk.corpus import wordnet
                def _pos(tb):
                    if tb.startswith('J'): return wordnet.ADJ
                    if tb.startswith('V'): return wordnet.VERB
                    if tb.startswith('N'): return wordnet.NOUN
                    if tb.startswith('R'): return wordnet.ADV
                    return wordnet.NOUN
                lemm = WordNetLemmatizer()
                toks = word_tokenize(token)
                tagged = pos_tag(toks)
                return " ".join(lemm.lemmatize(w, _pos(p)) for w, p in tagged)
            except Exception:
                return token

    def _lemmatize_expr(expr: str) -> str:
        parts = [p for p in re.split(r'\s+', expr) if p]
        return " ".join(_lemmatize_single(p) for p in parts)

    dataset_col = 'italian' if lang in {"italian", "it"} else 'english'
    work = df_brysbaert[[dataset_col, 'Conc.M']].copy()
    work[dataset_col] = work[dataset_col].astype(str)
    work['norm'] = work[dataset_col].apply(_norm)
    work['lemma'] = work['norm'].apply(_lemmatize_expr)

    # Aggregate: mean Conc.M per lemma + list of distinct original lexical forms
    lemma_group = (
        work.groupby('lemma', dropna=True)
            .agg(
                mean_conc=('Conc.M', 'mean'),
                forms=(dataset_col, lambda s: sorted(set(s)))
            )
            .reset_index()
    )
    lemma_conc_map = {r['lemma']: r['mean_conc'] for _, r in lemma_group.iterrows()}
    lemma_forms_map = {r['lemma']: r['forms'] for _, r in lemma_group.iterrows()}

    verts = list(getattr(semantic_frame, "vertices", []))
    if not verts:
        raise ValueError("semantic_frame has zero vertices")
    norm_frame = [_norm(v) for v in verts]
    frame_tokens = list(dict.fromkeys(norm_frame)) if deduplicate else norm_frame
    frame_lemmas = [_lemmatize_expr(t) for t in frame_tokens]

    empirical_scores: List[float] = []
    matched_items: List[Dict[str, Any]] = []
    unmatched_items: List[str] = []

    # Progress bar over tokens
    try:
        iterator = tqdm(zip(frame_tokens, frame_lemmas),
                         total=len(frame_tokens),
                         desc="Concreteness (lemmas)",
                         leave=False)
    except Exception:
        iterator = zip(frame_tokens, frame_lemmas)

    for raw_tok, lemma in iterator:
        score = lemma_conc_map.get(lemma)
        method = 'direct'
        comp_detail = []
        n_forms = len(lemma_forms_map.get(lemma, []))
        forms_used = lemma_forms_map.get(lemma, [])
        if score is None:
            parts = [p for p in lemma.split() if p]
            part_scores = [lemma_conc_map.get(p) for p in parts if lemma_conc_map.get(p) is not None]
            if part_scores:
                score = float(np.mean(part_scores))
                method = 'component_avg'
                comp_detail = [{'component': p, 'score': lemma_conc_map.get(p), 'n_forms': len(lemma_forms_map.get(p, []))} for p in parts if lemma_conc_map.get(p) is not None]
                n_forms = sum(len(lemma_forms_map.get(p, [])) for p in parts if lemma_conc_map.get(p) is not None)
                forms_used = sorted({f for p in parts for f in lemma_forms_map.get(p, [])})
            else:
                method = 'unmatched'
                unmatched_items.append(raw_tok)
        if score is not None:
            empirical_scores.append(score)
        matched_items.append({
            'original': raw_tok,
            'lemma': lemma,
            'concreteness': score,
            'method': method,
            'n_forms_aggregated': n_forms,
            'forms_aggregated': forms_used,
            'components': comp_detail
        })

    if len(empirical_scores) < 3:
        raise ValueError(f"Empirical sample too small (<3). Got {len(empirical_scores)} scores.")

    n_emp = len(empirical_scores)
    random_sample_df = df_brysbaert.sample(n=n_emp, random_state=seed, replace=False)
    random_scores = [float(x) for x in random_sample_df['Conc.M'].dropna().tolist()]
    if len(random_scores) < 3:
        raise ValueError(f"Random baseline too small (<3). Got {len(random_scores)} scores.")

    u_stat, p_val = stats.mannwhitneyu(empirical_scores, random_scores, alternative='two-sided')
    median_emp = float(np.median(empirical_scores))
    median_rand = float(np.median(random_scores))
    median_diff = median_emp - median_rand

    fig = None
    if plot:
        try:
            import seaborn as sns
            use_seaborn = True
        except ImportError:
            use_seaborn = False
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        plot_df = pd.DataFrame({
            'source': ['Semantic Frame'] * n_emp + ['Random Sample'] * n_emp,
            'concreteness': empirical_scores + random_scores
        })
        if use_seaborn:
            sns.boxplot(data=plot_df, x='source', y='concreteness', ax=ax, width=0.5, fliersize=0, color='lightgray')
        else:
            ax.boxplot([empirical_scores, random_scores], labels=['Semantic Frame', 'Random Sample'], patch_artist=True)
        ax.text(0.5, 0.95,
                f"U={u_stat:.2f}\np={p_val:.3g}\nMedian diff={median_diff:.2f}",
                ha='center', va='top', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'), fontsize=10)
        ax.set_title(f"Concreteness ({dataset_col} lemma aggregation)", fontsize=12)
        ax.set_ylabel("Conc.M")
        ax.set_xlabel("")
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()

    out = {
        'empirical_scores': empirical_scores,
        'random_scores': random_scores,
        'u_statistic': float(u_stat),
        'p_value': float(p_val),
        'median_empirical': median_emp,
        'median_random': median_rand,
        'matched_items': matched_items,
        'unmatched_items': unmatched_items,
        'language_used': lang
    }

    json_path_written = None
    figure_path_written = None
    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path) or ".", exist_ok=True)
        with open(save_json_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        json_path_written = save_json_path
    if save_figure_path and fig is not None:
        os.makedirs(os.path.dirname(save_figure_path) or ".", exist_ok=True)
        fig.savefig(save_figure_path, dpi=300)
        figure_path_written = save_figure_path

    return {**out, 'figure': fig, 'json_path': json_path_written, 'figure_path': figure_path_written}


def analyze_semantic_frame_concreteness_fast(
    self,
    semantic_frame,
    seed=None,
    deduplicate=True,
    plot=True,
    save_json_path=None,
    save_figure_path=None,
    lemmatized_csv_path_en="../data/brysbaert_concreteness_dataset_eng_ita/brysbaert_concreteness_dataset_eng_ita_lemmatized_en.csv",
    lemmatized_csv_path_it="../data/brysbaert_concreteness_dataset_eng_ita/brysbaert_concreteness_dataset_eng_ita_lemmatized_it.csv",
):
    """
    Fast concreteness analysis that uses pre-lemmatized Brysbaert datasets.

    Expected CSV columns: ['lemma', 'Conc.M_mean', 'Conc.M_std', 'n_forms', 'forms']
    - 'forms' is a stringified Python list of forms aggregated into the lemma.

    Behavior matches analyze_semantic_frame_concreteness:
      - direct lemma match
      - fallback to form->lemma lookup
      - fallback to component average (splitting on space or underscore)
      - Mann-Whitney U test vs random baseline
      - Optional plot and JSON export
    """
    lang = (getattr(self, "language", None) or globals().get("language") or "italian").lower()
    if lang.startswith("it"):
        csv_path = lemmatized_csv_path_it
        dataset_col = "italian"
    elif lang.startswith("en"):
        csv_path = lemmatized_csv_path_en
        dataset_col = "english"
    else:
        raise ValueError(f"Unsupported language: {lang}")

    # Load pre-lemmatized dataset
    df_lem = pd.read_csv(csv_path)

    required_cols = {"lemma", "Conc.M_mean", "forms"}
    missing = required_cols - set(df_lem.columns)
    if missing:
        raise KeyError(f"Missing columns in lemmatized dataset: {missing}")

    # Helpers
    def _norm(s):
        return re.sub(r"\s+", " ", str(s).strip().lower())

    def _variants(s):
        # variants to bridge space/underscore conventions
        a = _norm(s)
        return list(dict.fromkeys([a, a.replace("_", " "), a.replace(" ", "_")]))

    def _safe_parse_forms(x):
        if isinstance(x, (list, tuple)):
            return [str(t) for t in x]
        if isinstance(x, str):
            try:
                v = ast.literal_eval(x)
                if isinstance(v, (list, tuple)):
                    return [str(t) for t in v]
            except Exception:
                pass
            # naive fallback: comma split
            return [t.strip(" '\"") for t in x.strip("[]").split(",") if t.strip()]
        return []

    # Build canonical maps
    # Canonical lemma representation will use spaces (not underscores)
    lemma_meta = {}              # canonical_lemma -> {'score', 'n_forms', 'forms'}
    lemma_variant_to_canonical = {}  # any lemma variant -> canonical_lemma
    form_variant_to_canonical = {}   # any form variant -> canonical_lemma

    for _, row in df_lem.iterrows():
        lemma_raw = str(row["lemma"])
        conc = row.get("Conc.M_mean")
        try:
            conc = float(conc) if conc is not None and not pd.isna(conc) else None
        except Exception:
            conc = None
        forms = _safe_parse_forms(row.get("forms"))
        n_forms = row.get("n_forms")
        try:
            n_forms = int(n_forms) if n_forms is not None and not pd.isna(n_forms) else None
        except Exception:
            n_forms = None

        # Canonical lemma = with spaces
        lemma_canonical = _norm(lemma_raw).replace("_", " ")
        if lemma_canonical not in lemma_meta:
            lemma_meta[lemma_canonical] = {
                "score": conc,
                "n_forms": n_forms if n_forms is not None else len(forms),
                "forms": sorted({f for f in forms})  # keep raw forms as-is for reporting
            }

        # Map lemma variants -> canonical
        for v in _variants(lemma_raw):
            lemma_variant_to_canonical[_norm(v)] = lemma_canonical

        # Map each form variant -> canonical lemma
        for f in forms:
            for fv in _variants(f):
                form_variant_to_canonical[_norm(fv)] = lemma_canonical

    # Gather frame tokens
    if not hasattr(semantic_frame, "vertices"):
        raise ValueError("semantic_frame must have .vertices")

    verts = list(getattr(semantic_frame, "vertices", []))
    if not verts:
        raise ValueError("semantic_frame has zero vertices")

    norm_tokens = [_norm(v) for v in verts]
    frame_tokens = list(dict.fromkeys(norm_tokens)) if deduplicate else norm_tokens

    empirical_scores = []
    matched_items = []
    unmatched_items = []

    for tok in frame_tokens:
        # Try lemma direct
        lemma = None
        method = "direct"
        if tok in lemma_variant_to_canonical:
            lemma = lemma_variant_to_canonical[tok]
        else:
            # Try space/underscore variants for direct lemma
            hit = False
            for v in _variants(tok):
                if v in lemma_variant_to_canonical:
                    lemma = lemma_variant_to_canonical[v]
                    hit = True
                    break
            if not hit:
                # Try form lookup
                method = "form_lookup"
                if tok in form_variant_to_canonical:
                    lemma = form_variant_to_canonical[tok]
                else:
                    for v in _variants(tok):
                        if v in form_variant_to_canonical:
                            lemma = form_variant_to_canonical[v]
                            break

        if lemma is not None:
            meta = lemma_meta.get(lemma, {})
            score = meta.get("score")
            if score is not None:
                empirical_scores.append(score)
            matched_items.append({
                "original": tok,
                "lemma": lemma,
                "concreteness": score,
                "method": method,
                "n_forms_aggregated": meta.get("n_forms", 0),
                "forms_aggregated": meta.get("forms", []),
                "components": []
            })
            continue

        # Component average: split by space/underscore and average available parts
        parts = re.split(r"[ _]+", tok)
        comp_details = []
        comp_scores = []
        comp_forms_total = 0
        for p in parts:
            p = _norm(p)
            l = lemma_variant_to_canonical.get(p)
            if l is None:
                # try form map
                l = form_variant_to_canonical.get(p)
            if l is None:
                # try variants
                for v in _variants(p):
                    l = lemma_variant_to_canonical.get(v) or form_variant_to_canonical.get(v)
                    if l is not None:
                        break
            if l is None:
                continue
            m = lemma_meta.get(l, {})
            sc = m.get("score")
            if sc is None:
                continue
            comp_scores.append(float(sc))
            comp_details.append({
                "component": p,
                "score": float(sc),
                "n_forms": int(m.get("n_forms", 0))
            })
            comp_forms_total += int(m.get("n_forms", 0))

        if comp_scores:
            score = float(np.mean(comp_scores))
            empirical_scores.append(score)
            matched_items.append({
                "original": tok,
                "lemma": " ".join(parts),
                "concreteness": score,
                "method": "component_avg",
                "n_forms_aggregated": int(comp_forms_total),
                "forms_aggregated": [],  # components already detailed
                "components": comp_details
            })
        else:
            unmatched_items.append(tok)
            matched_items.append({
                "original": tok,
                "lemma": tok,
                "concreteness": None,
                "method": "unmatched",
                "n_forms_aggregated": 0,
                "forms_aggregated": [],
                "components": []
            })

    if len(empirical_scores) < 3:
        raise ValueError(f"Empirical sample too small (<3). Got {len(empirical_scores)} scores.")

    # Random baseline from lemma means
    all_scores = [float(x) for x in df_lem["Conc.M_mean"].dropna().tolist()]
    if len(all_scores) < 3:
        raise ValueError(f"Random baseline too small (<3). Got {len(all_scores)} rows in dataset.")

    n_emp = len(empirical_scores)
    rng = np.random.default_rng(seed)
    if n_emp <= len(all_scores):
        rand_idx = rng.choice(len(all_scores), size=n_emp, replace=False)
    else:
        rand_idx = rng.choice(len(all_scores), size=n_emp, replace=True)
    random_scores = [all_scores[i] for i in rand_idx]

    u_stat, p_val = stats.mannwhitneyu(empirical_scores, random_scores, alternative="two-sided")
    median_emp = float(np.median(empirical_scores))
    median_rand = float(np.median(random_scores))
    median_diff = median_emp - median_rand

    fig = None
    if plot:
        try:
            import seaborn as sns
            use_seaborn = True
        except Exception:
            use_seaborn = False
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        plot_df = pd.DataFrame({
            "source": ["Semantic Frame"] * n_emp + ["Random Sample"] * n_emp,
            "concreteness": empirical_scores + random_scores
        })
        if use_seaborn:
            sns.boxplot(data=plot_df, x="source", y="concreteness", ax=ax, width=0.5, fliersize=0, color="lightgray")
        else:
            ax.boxplot([empirical_scores, random_scores], labels=["Semantic Frame", "Random Sample"], patch_artist=True)
        ax.text(0.5, 0.95,
                f"U={u_stat:.2f}\np={p_val:.3g}\nMedian diff={median_diff:.2f}",
                ha="center", va="top", transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"), fontsize=10)
        ax.set_title(f"Concreteness (lemmatized: {dataset_col})", fontsize=12)
        ax.set_ylabel("Conc.M (mean per lemma)")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

    out = {
        "empirical_scores": empirical_scores,
        "random_scores": random_scores,
        "u_statistic": float(u_stat),
        "p_value": float(p_val),
        "median_empirical": median_emp,
        "median_random": median_rand,
        "matched_items": matched_items,
        "unmatched_items": unmatched_items,
        "language_used": lang
    }

    json_path_written = None
    figure_path_written = None
    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path) or ".", exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        json_path_written = save_json_path
    if save_figure_path and fig is not None:
        os.makedirs(os.path.dirname(save_figure_path) or ".", exist_ok=True)
        fig.savefig(save_figure_path, dpi=300)
        figure_path_written = save_figure_path

    return {**out, "figure": fig, "json_path": json_path_written, "figure_path": figure_path_written}


# attach the method to EmoScores
EmoScores.analyze_semantic_frame_concreteness = analyze_semantic_frame_concreteness
# attach the fast method
EmoScores.analyze_semantic_frame_concreteness_fast = analyze_semantic_frame_concreteness_fast


def make_plot_by_type(
    df,
    features_to_plot: list,
    save_path: str,
    fmn_type: str,
    plot_type: str,
    filename=None,
):
    """
    Create plots of type <plot_type> for desired structural features.
    Saves one .png file per feature.

    Parameters:
    df (pd.DataFrame): DataFrame containing the features to plot.
    features_to_plot (list): List of feature names (columns in df) to plot.
    save_path (str): Directory where to save the plots.
    filename (str): optional filename for the plots. None by default.
    fmn_type (str): Type of FMN (e.g., 'BFMN', 'TFMN') for labeling.
    plot_type (str): Type of plot to create.

    Supported plot types (new types soon):
    - 'distribution': Histogram showing the distribution of feature values.
    """
    # Update if new plot types are added
    supported_plot_types = ['distribution']
    
    if plot_type not in supported_plot_types:
        raise ValueError("Type must be one of the following: 'distribution'")

    # Ensure output dirs exist (base + per-plot-type)
    os.makedirs(save_path, exist_ok=True)
    out_dir = os.path.join(save_path, plot_type)
    os.makedirs(out_dir, exist_ok=True)

    for feature in features_to_plot:
        if feature not in df.columns:
            print(f"[WARN] Skipping missing feature: {feature}")
            continue  # skip instead of raising

        feature_series = df[feature].dropna()
        if feature_series.empty:
            print(f"[WARN] Skipping empty feature: {feature}")
            continue  # skip instead of raising

        unique_count = feature_series.nunique()

        # Handle constant-valued feature explicitly (avoids seaborn/numpy bin errors)
        if unique_count == 1:
            const_val = feature_series.iloc[0]
            fig, ax = plt.subplots()
            ax.bar([const_val], [len(feature_series)], width=0.6, alpha=0.6, color="C0")
            ax.set_xlabel(str(feature).replace('_', ' ').title())
            ax.set_ylabel("Count")
            ax.set_title(
                f"Distribution of '{str(feature).replace('_', ' ').title()}' in {fmn_type}s (constant value {const_val}) (n={len(feature_series)})"
            )
            ax.legend([f"Value = {const_val}"])
            if filename:
                out_path = f"{out_dir}/{filename}.png"
            else:
                out_path = f"{out_dir}/{feature}_{plot_type}_{fmn_type}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Saved {out_path} (constant feature)")
            continue

        # Heuristic bin width
        if pd.api.types.is_integer_dtype(feature_series):
            bin_width = 1
        else:
            value_range = feature_series.max() - feature_series.min()
            bin_width = value_range / 30 if unique_count > 1 else 1
            if bin_width <= 0 or not np.isfinite(bin_width):
                bin_width = 1

        mean_val = feature_series.mean()
        data = feature_series.to_frame(name=feature)
        
        try:
            if pd.api.types.is_numeric_dtype(feature_series):
                print(f"{feature} is all scalar values")

            plot = (
                so.Plot(data=data, x=feature)
                .add(mark=so.Bars(alpha=0.6), stat=so.Hist(binwidth=bin_width))
                .label(
                    x=feature,
                    y="Count",
                    title=f"Distribution of '{str(feature).replace('_', ' ').title()}' in {fmn_type}s (n={len(feature_series)})"
                )
            )
            fig = plot.plot()
            ax = fig.axes[0]
        except ValueError as e:
            print(f"[INFO] Falling back to seaborn.histplot for {feature}: {e}")
            fig, ax = plt.subplots()
            vmin, vmax = feature_series.min(), feature_series.max()
            if vmin == vmax:
                vmax = vmin + 0.5
            sns.histplot(
                data=data,
                x=feature,
                binwidth=bin_width,
                binrange=(vmin, vmax),
                kde=False,
                alpha=0.6,
                ax=ax
            )
            ax.set_title(f"Distribution of '{str(feature).replace('_', ' ').title()}' in {fmn_type}s (n={len(feature_series)})")
            ax.set_xlabel(str(feature).replace('_', ' ').title())
            ax.set_ylabel("Count")

        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1, label=f"Mean = {mean_val:.2f}")
        ax.legend()
        if filename:
            out_path = f"{out_dir}/{filename}.png"
        else:
            out_path = f"{out_dir}/{feature}_{plot_type}_{fmn_type}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved {out_path}\n")