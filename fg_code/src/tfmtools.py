import os
import re
import numpy as np
import gc
import json
import networkx as nx
from emoatlas.resources import _valences
from collections import Counter
from .fmtools import monitor_memory


def verify_and_report_words_in_texts(
    texts_by_label,
    required,
    case_insensitive=True,
    strict=True,
    whole_word=True
):
    """
    Check if required words appear in each text (texts_by_label[label] = text).
    Prints:
      - Missing words per label
      - Global summary (ALL vs ANY)
      - Per-word coverage
    Mirrors style of verify_and_report_nodes (for FMNs).

    Parameters
    ----------
    texts_by_label : dict[str,str]
    required : iterable[str]
    case_insensitive : bool
    strict : bool
        True -> each text must contain ALL required words
        False -> each text must contain AT LEAST ONE required word
    whole_word : bool
        If True, matches whole words using word boundaries; else substring match.

    Returns
    -------
    summary : dict
    """
    if not required:
        raise ValueError("required must be a non-empty iterable")
    required = list(required)


    def _contains(word, text):
        if whole_word:
            pattern = rrf.get(word)  # small cache
            if pattern is None:
                flags = re.IGNORECASE if case_insensitive else 0
                pattern = re.compile(rf"\b{re.escape(word)}\b", flags)
                rrf[word] = pattern
            return bool(pattern.search(text))
        else:
            if case_insensitive:
                return word.lower() in text.lower()
            return word in text

    rrf = {}  # regex cache
    results = {}
    missing_texts = {}
    per_word_counter = Counter()

    for label, text in texts_by_label.items():
        if text is None:
            text = ""
        chk_text = text if not case_insensitive else text.lower()

        present = []
        missing = []
        for w in required:
            w_cmp = w.lower() if case_insensitive else w
            if _contains(w_cmp if not whole_word else w, chk_text):
                present.append(w)
                per_word_counter[w] += 1
            else:
                missing.append(w)

        if strict:
            ok = len(missing) == 0
            mode = "all"
        else:
            ok = len(present) > 0
            mode = "any"

        if not ok:
            missing_texts[label] = missing
            print(f"Text '{label}' missing: {missing}")

        results[label] = {
            "result": ok,
            "mode": mode,
            "present": present,
            "missing": missing,
            "present_count": len(present),
            "missing_count": len(missing),
            "required_total": len(required),
            "text_length_chars": len(text),
        }

    total = len(texts_by_label)
    satisfying = sum(1 for r in results.values() if r["result"])
    percentage = (satisfying / total * 100) if total else 0.0

    print(f"\nTexts satisfying ({'ALL' if strict else 'ANY'} of {required}): {satisfying}/{total} ({percentage:.2f}%)")
    for w in required:
        c = per_word_counter[w]
        print(f"Presence '{w}': {c}/{total} ({(c/total*100 if total else 0):.2f}%)")

    summary = {
        "required": required,
        "mode": "all" if strict else "any",
        "total_texts": total,
        "texts_with_requirement": satisfying,
        "percentage": percentage,
        "per_word_presence": {
            w: {
                "count": per_word_counter[w],
                "percentage": (per_word_counter[w] / total * 100) if total else 0.0
            } for w in required
        },
        "missing_texts": missing_texts,
        "results": results,
        "whole_word": whole_word,
        "case_insensitive": case_insensitive,
        "strict": strict
    }
    return summary


def group_texts(df, col, labels, exclude_pattern=None, preprocess_txt=None):
    """Group texts by labels from the grouping column (col).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the texts.
    col : str
        The name of the column containing the texts data.
    labels : str or list[str]
        The row labels to apply for grouping texts (supports 'startswith').
    exclude_pattern : str | list[str], optional
        Regex pattern(s) removed from each grouped text before preprocessing.
    preprocess_txt : callable(str) -> str, optional
        A function applied to each grouped text AFTER exclude_pattern removal.
        Use this to plug in lowercasing, spelling fixes, lemmatization, etc.

    Returns
    -------
    tuple[dict, str]
        (dict[label] = processed text, concatenated_all_texts)
    """
    texts_by_label = {}
    all_texts = ""

    for label_str in labels:
        mask = df[col].str.startswith(label_str)
        sliced_texts = "".join(df[mask]['text'].fillna(''))

        # Remove instruction / noise patterns first
        if exclude_pattern is not None:
            if isinstance(exclude_pattern, str):
                sliced_texts = re.sub(pattern=exclude_pattern, repl='', string=sliced_texts)
            else:
                for pattern in exclude_pattern:
                    sliced_texts = re.sub(pattern=pattern, repl='', string=sliced_texts)

        # Optional custom preprocessing
        if preprocess_txt is not None:
            try:
                out = preprocess_txt(sliced_texts)
                if isinstance(out, str) and out:
                    sliced_texts = out
                elif out is None:
                    # Leave original text unchanged
                    pass
                else:
                    print(f"[group_texts] preprocess returned non-string for {label_str}, keeping original.")
            except Exception as e:
                print(f"[group_texts] preprocess failed for {label_str}: {e}")

        texts_by_label[label_str] = sliced_texts
        all_texts += sliced_texts

    return texts_by_label, all_texts


def build_tfmns_by_label(texts_by_label, emo, keywords=None, labels=None, batch_size=10):
    '''
    Builds forma mentis networks for given keywords and grouped texts (by label).
    Process in batches to manage memory usage and avoid memory leaks.
    The FMNs are stored in a dictionary where keys are the fmnt names
    ("keyword_label" or just "label" if keywords is None) and values are the
    corresponding FMN objects (instances of the `FormaMentisNetwork` class).

    Args:
        texts_by_label (dict): A dictionary with labels as keys and texts as values.
        emo (EmoScores): An initialized EmoScores object.
        keywords (list): A list of target words for the networks. Defaults to None.
        labels (list): Specific labels to process. Defaults to None.
        batch_size (int): Number of networks to process in each batch.
    
    Returns:
        dict: A dictionary of the generated forma mentis networks (excluding empty networks).
    '''
    # Initialize a dictionary to store all forma mentis networks
    if emo is None:
        raise ValueError("Parameter 'emo' (EmoScores) is required.")
    all_fmnts = {}
    
    # If labels is provided, filter the texts_by_label dictionary
    if labels is not None:
        texts_by_label = {label: texts for label, texts in texts_by_label.items() if label in labels}
    
    # Convert to list for batch processing
    items = list(texts_by_label.items())
    total_items = len(items)
    
    print(f"Processing {total_items} items in batches of {batch_size}")
    monitor_memory()
    
    # Process in batches
    for i in range(0, total_items, batch_size):
        batch_items = items[i:i+batch_size]
        batch_fmnts = {}
        
        print(f"\nProcessing batch {i//batch_size + 1}/{(total_items + batch_size - 1)//batch_size}")
        
        # If keywords are provided, build FMNs for each provided keyword
        # and label
        if keywords is not None:
            for keyword in keywords:
                for label, texts in batch_items:
                    print(f"Processing {keyword} for {label}")
                    fmnt_title = f"{keyword}_{label}"
                    
                    try:
                        # Build the textual forma mentis network
                        fmnt = emo.formamentis_network(text=texts, target_word=keyword, semantic_enrichment=['synonyms'], multiplex=False)
                        
                        # Check if the network has vertices before adding it
                        if hasattr(fmnt, "vertices") and len(fmnt.vertices) > 0:
                            batch_fmnts[fmnt_title] = fmnt
                        else:
                            print(f"Skipping {fmnt_title}: network has zero vertices.")
                            # Remove participant from participant_ids
                            participant_ids.discard(label)

                    except ValueError as e:
                        if "min() arg is an empty sequence" in str(e):
                            print(f"Skipping {fmnt_title}: not enough data to build the network.")
                        else:
                            raise
        else:
            for label, texts in batch_items:
                print(f"Processing {label}")
                fmnt_title = f"{label}"
                
                try:
                    # Build the textual forma mentis network
                    fmnt = emo.formamentis_network(text=texts, target_word=None, semantic_enrichment=['synonyms'], multiplex=False)
                    
                    # Check if the network has vertices before adding it
                    if hasattr(fmnt, "vertices") and len(fmnt.vertices) > 0:
                        batch_fmnts[fmnt_title] = fmnt
                    else:
                        print(f"Skipping {fmnt_title}: network has zero vertices.")
                        
                except ValueError as e:
                    if "min() arg is an empty sequence" in str(e):
                        print(f"Skipping {fmnt_title}: not enough data to build the network.")
                    else:
                        raise
        
        # Add batch to main dictionary
        all_fmnts.update(batch_fmnts)
        
        # Memory management
        monitor_memory()
        gc.collect()
    
    print(f"Successfully built {len(all_fmnts)} networks with vertices > 0")
    return all_fmnts


def extract_tfm_structural_features(
    fmnts: dict,
    save_path: str,
    emo=None,
    batch_size: int = 10,
    compute_layer_metrics: bool = True,
    multiplex_merge_strategy: str = "union",
    include_zscores: bool = True,
    persist: bool = True
):
    """
    Extract structural features for Textual Forma Mentis Networks.
    Supports single-layer and multiplex FMNs (syntactic / synonyms / hypernyms).
    Produces (optionally) per-layer metrics plus merged (union) metrics.

    Parameters
    ----------
    fmnts : dict
        {title: FormamentisNetwork}
    save_path : str
        Base output directory.
    emo : EmoScores
        Initialized EmoScores instance.
    batch_size : int
        Networks per batch.
    compute_layer_metrics : bool
        If True and fmnt is multiplex (edges is a dict), compute metrics per layer.
    multiplex_merge_strategy : str
        'union' (currently implemented) – all edges merged ignoring type.
        (Placeholder for future strategies like 'intersection'.)
    include_zscores : bool
        If True compute emo.zscores(fmnt).
    persist : bool
        If True write JSON + TXT per network.

    Returns
    -------
    dict
        {fmnt_title: feature_dict}
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    pos_set, neg_set, _ = _valences(emo.language)

    def _sanitize_number(x):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
                    return None
                return float(x)
        except Exception:
            return None
        return None

    def _valence_cat(node: str) -> int:
        '''1=positive, -1=negative, 0=neutral
        
        Uses lexicon sets from _valences(emo.language).
        '''
        if node in pos_set: return 1
        if node in neg_set: return -1
        return 0

    def _valence_label(vcat: int) -> str:
        if vcat == 1: return "positive"
        if vcat == -1: return "negative"
        return "neutral"

    def _make_nx_from_edges(edgelist):
        G = nx.Graph()
        for e in edgelist:
            if len(e) >= 2:
                u, v = e[0], e[1]
                G.add_edge(u, v)
        # Add valence category and label attributes to nodes
        for n in G.nodes():
            vcat = _valence_cat(n)
            G.nodes[n]["valence_cat"] = vcat
            G.nodes[n]["valence"] = _valence_label(vcat)
        return G

    def _compute_graph_metrics(G: nx.Graph):
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        n_pos = sum(1 for _, d in G.nodes(data=True) if d.get("valence_cat") == 1)
        n_neg = sum(1 for _, d in G.nodes(data=True) if d.get("valence_cat") == -1)

        # Degree stats
        degrees = [deg for _, deg in G.degree()]
        min_degree = min(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0

        # Local graph-level average clustering.
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

        # Valence assortativity
        try:
            valence_assortativity = nx.attribute_assortativity_coefficient(G, "valence")
        except Exception:
            valence_assortativity = None

        # Closeness extremes
        try:
            closeness = nx.closeness_centrality(G) if n_nodes > 0 else {}
            max_closeness = max(closeness.values()) if closeness else None
            min_closeness = min(closeness.values()) if closeness else None
        except Exception:
            max_closeness = None
            min_closeness = None

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
                    # Extract largest connected component (a set of nodes)
                    largest_cc = max(nx.connected_components(undirected_G), key=len)
                    # Create a subgraph view of the largest connected component.
                    # The graph structure cannot be changed but node/edge attributes
                    # can and are shared with the original graph.
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

        return {
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

    items = list(fmnts.items())
    total = len(items)
    print(f"Extracting TFMN structural features from {total} networks in batches of {batch_size}")
    monitor_memory()

    all_features = {}

    # Process in batches
    for i in range(0, total, batch_size):
        batch = items[i:i+batch_size]
        print(f"\nBatch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")
        
        for title, fmnt in batch:
            if not hasattr(fmnt, "vertices") or len(fmnt.vertices) == 0:
                print(f"Skipping {title}: empty network.")
                continue

            # Check if the network is multiplex
            is_multiplex = isinstance(fmnt.edges, dict)
            feature_record = {
                "title": title,
                "is_multiplex": bool(is_multiplex),
            }

            # Per-layer metrics (if multiplex and requested)
            layer_metrics = {}
            layer_edge_counts = {}
            n_layers = 1

            if is_multiplex:
                # fmnt.edges is assumed to be a mapping (likely dict-like) whose keys
                # denote layers (e.g., syntactic, synonims etc...). Calling fmnt.edges.keys()
                # returns a dynamic view object;
                # wrapping it with the built-in list constructor immediately copies
                # those keys into a concrete list.
                layers = list(fmnt.edges.keys())
                n_layers = len(layers)
                feature_record["layers"] = layers
                if compute_layer_metrics:
                    for layer in layers:
                        G_layer = _make_nx_from_edges(fmnt.edges[layer])
                        layer_metrics[layer] = _compute_graph_metrics(G_layer)
                        layer_edge_counts[layer] = G_layer.number_of_edges()
                # Merge strategy
                if multiplex_merge_strategy == "union":
                    merged_edges = []
                    for layer in layers:
                        merged_edges.extend(fmnt.edges[layer])
                else:
                    # Fallback to union if unsupported strategy
                    merged_edges = []
                    for layer in layers:
                        merged_edges.extend(fmnt.edges[layer])
                G_merged = _make_nx_from_edges(merged_edges)
                merged_metrics = _compute_graph_metrics(G_merged)
                feature_record["merged_strategy"] = multiplex_merge_strategy
                for k, v in merged_metrics.items():
                    feature_record[k] = v
            else:
                # Single-layer network
                G_single = _make_nx_from_edges(fmnt.edges)
                single_metrics = _compute_graph_metrics(G_single)
                for k, v in single_metrics.items():
                    feature_record[k] = v

            feature_record["n_layers"] = n_layers
            feature_record["layer_edge_counts"] = layer_edge_counts if layer_edge_counts else None

            # Global z-scores (EmoAtlas already handles multiplex internally if passed fmnt)
            if include_zscores:
                try:
                    z_scores = emo.zscores(fmnt)
                except Exception:
                    z_scores = {}
                feature_record.update({f"z_{k}": v for k, v in z_scores.items()})

            # Node valence lists (lexicon-based)
            pos_nodes = sorted([n for n in fmnt.vertices if n in pos_set])
            neg_nodes = sorted([n for n in fmnt.vertices if n in neg_set])
            neu_nodes = sorted([n for n in fmnt.vertices if (n not in pos_set and n not in neg_set)])
            feature_record["positive_nodes"] = pos_nodes
            feature_record["negative_nodes"] = neg_nodes
            feature_record["neutral_nodes"] = neu_nodes

            # Include compact per-layer metrics if computed
            if layer_metrics:
                feature_record["layer_metrics"] = layer_metrics

            all_features[title] = feature_record

            if persist:
                json_path = os.path.join(save_path, f"{title}_tfmn_structural_features.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(feature_record, f, ensure_ascii=False, indent=2)
                txt_path = os.path.join(save_path, f"{title}_tfmn_structural_features.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    for k, v in feature_record.items():
                        if k in ("positive_nodes", "negative_nodes", "neutral_nodes") and isinstance(v, list):
                            f.write(f"{k} ({len(v)}): {', '.join(v)}\n")
                        else:
                            f.write(f"{k}: {v}\n")
                print(f"Saved {json_path}")

        monitor_memory()
        gc.collect()

    print(f"Saved structural features for {len(all_features)} TFMNs")
    return all_features