from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Dict

import pandas as pd
import os
import re
import networkx as nx
from emoatlas.resources import _valences
from collections import namedtuple
from .valence import categorize_valence_multiple


def import_fmnt_from_txt(filepath=None, language="english", custom_valences=False):
    '''
    Import a Forma Mentis Network from:
      - export_whole_fmnt_custom(...) or EmoScores.export_whole_fmnt(...) (structured sections), or
      - EmoScores.export_formamentis(...) (plain 'a , b' edgelist).

    Parameters:
    -----------
    filepath : str, optional
        The path of the file from which to import the FMN. If None, defaults to
        "extracted formamentis.txt" in the current working directory.

    custom_valences:
        If True, a list of 3 sets is generated: positive, negative and neutral custom valences
        from the file if present, otherwise no custom valences (all neutral). Retrieved
        from the file sections if present.

    language: str, optional
        Language for valences by lexicon if no custom valences provided.
        Default is 'english'.
    
    Returns a namedtuple with:
    - fmnt: FormamentisNetwork namedtuple with edges and vertices,
    - G: NetworkX graph with nodes and edges,
    - valence_cat_label_map: dict mapping nodes to valence category labels ('positive', 'negative', 'neutral'),
    - valence_cat_map: dict mapping nodes to valence categories (1, -1, 0),
    - custom_valences: list of 3 sets (positive, negative, neutral) if custom_valences is True, else None.
    '''
    ImportedFMN = namedtuple("ImportedFMN", ["fmnt", "G", "valence_cat_label_map", "valence_cat_map", "custom_valences"])
    FormamentisNetwork = namedtuple("FormamentisNetwork", ["edges", "vertices"])

    # Initialize data structures
    edges = []
    vertices = set()
    G = nx.Graph()

    # Handle path and filename
    if filepath == None:
        filepath = "extracted formamentis.txt"
    elif not filepath.endswith(".txt"):
        filepath += ".txt"
    
    # Check file existence
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Print the file being imported
    print(f"Importing Forma Mentis Network from: {filepath}")

    # Read file lines
    with open(filepath, "r", encoding="utf-8") as f:
        # Remove trailing newlines only
        lines = [line.rstrip("\n") for line in f]

    # Detect structured format (sections) vs plain edgelist
    has_sections = any(line.strip().startswith("## ") for line in lines)
    
    # Check if it is structured or plain edgelist format and parse accordingly
    if has_sections:
        # Initialize sets for custom valences
        _positive, _negative, _ambivalent = set(), set(), set()
        current_section = None
        for line in lines:
            line = line.strip()

            # Skip empty lines first
            if not line:
                continue

            # Section headers (start with "## ")
            if line.startswith("## "):
                current_section = line[3:].strip()
                print(f"Parsing section: '{current_section}'")
                continue

            # Skip lines starting with single '# ' anywhere
            if line.startswith("# "):
                continue

            # Edges section
            if current_section and current_section.lower() == "edges":
                # Expect "a -- b"
                if "--" in line:
                    vertex1, vertex2 = map(str.strip, line.split("--", 1))
                    if vertex1 and vertex2:
                        edges.append((vertex1, vertex2))
                        vertices.update([vertex1, vertex2])
                continue

            # Valence sections (case-insensitive)
            if current_section and current_section.lower().startswith("positive"):
                _positive.add(line)
                continue
            if current_section and current_section.lower().startswith("negative"):
                _negative.add(line)
                continue
            if current_section and current_section.lower().startswith("neutral"):
                _ambivalent.add(line)
                continue

    # Plain edgelist format: "a , b" (unstructured)
    else:
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Expect "a , b"
            vertex1, vertex2 = map(str.strip, line.split(",", 1))
            edges.append((vertex1, vertex2))
            vertices.update([vertex1, vertex2])

    # Get positive or negative valences
    if custom_valences:
        custom_valences = [_positive, _negative, _ambivalent]
    else:
        _positive, _negative, _ambivalent = _valences(language)
        custom_valences = None

    # Check for correct custom_valences object creation
    if custom_valences is not None:
        if not isinstance(custom_valences, list) or len(custom_valences) != 3:
            raise ValueError(
                "Custom valences must be a list of 3 sets: positive, negative and neutral"
            )
        if not all(isinstance(val, set) for val in custom_valences):
            raise ValueError(
                "Custom valences must be a list of 3 sets: positive, negative and neutral"
            )

    # Convert the vertices set to a sorted list
    vertices = sorted(vertices)

    # Build the graph
    for vertex in vertices:
        G.add_node(vertex)
    for edge in edges:
        G.add_edge(*edge)
    # Set node attributes for valence categories and build valence maps as dictionaries
    valence_cat_label_map = {}
    valence_cat_map = {}

    # Assign valence labels and categories based on custom or lexicon based valences
    for node in G.nodes:
        if node in _positive:
            G.nodes[node]["valence_label"] = "positive"
            G.nodes[node]["valence"] = 1
        elif node in _negative:
            G.nodes[node]["valence_label"] = "negative"
            G.nodes[node]["valence"] = -1
        else:
            G.nodes[node]["valence_label"] = "neutral"
            G.nodes[node]["valence"] = 0
        valence_cat_label_map[node] = G.nodes[node]["valence_label"]
        valence_cat_map[node] = G.nodes[node]["valence"]

    # Build the FormamentisNetwork namedtuple
    fmnt = FormamentisNetwork(edges=edges, vertices=vertices)

    # Return the ImportedFMN namedtuple (always include custom_valences field)
    return ImportedFMN(
        fmnt=fmnt,
        G=G,
        valence_cat_label_map=valence_cat_label_map,
        valence_cat_map=valence_cat_map,
        custom_valences=custom_valences
    )


def _is_pure_numeric_row(tokens: List[str]) -> bool:
    """
    Heuristic: returns True if all tokens in the row look like integers possibly
    ending with a dot (e.g., '1.').
    """
    if not tokens:
        return False
    for t in tokens:
        if not re.fullmatch(r"\d+\.?", t):
            return False
    return True


def _parse_valence_old_text(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse an old ValenceStudentsUPD-like text file.

    Expected lines (examples):
      - 'acqua    5. 4. 5. 5. 3. ...'
      - Lines that are purely numeric (e.g. '1. 1. 3.') are ignored.

    Returns a long DataFrame with columns: ['cue_word', 'word', 'valence'].
    'cue_word' is set to the word itself to avoid NaNs, but it is not used by the
    Kruskal–Wallis logic.
    """
    file_path = Path(file_path)
    rows = []

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Ignore comment-like lines
            if line.startswith("//") or line.startswith("#") or line.startswith("```"):
                continue

            # Split by whitespace
            tokens = re.findall(r"\S+", line)
            if not tokens:
                continue

            # Skip pure numeric rows
            if _is_pure_numeric_row(tokens):
                continue

            # First token should be the word; the rest should be numeric valences
            word = tokens[0]
            # Some lines may contain weird headers (e.g. 'filepath:'), skip those
            if word.lower().startswith("filepath:"):
                continue

            # Parse numeric values (strip possible trailing dots)
            vals: List[float] = []
            for t in tokens[1:]:
                m = re.fullmatch(r"(\d+)(?:\.)?", t)
                if m:
                    vals.append(float(m.group(1)))
                # else: ignore non-numeric tails on valence rows

            if not vals:
                # Line didn't contain valences; skip
                continue

            # Build rows: each valence value is a separate row
            # We keep cue_word = word (to avoid NaNs downstream)
            for v in vals:
                rows.append({"cue_word": word, "word": word, "valence": v})

    if not rows:
        return pd.DataFrame(columns=["cue_word", "word", "valence"])

    df = pd.DataFrame(rows)
    # Normalize to lowercase to match typical network normalization
    df["word"] = df["word"].astype(str).str.strip().str.lower()
    df["cue_word"] = df["cue_word"].astype(str).str.strip().str.lower()
    return df


def _parse_students_edges_old_text(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse old students edge files (Students_Filtered.txt, Students_Unfiltered.txt).
    Heuristic: each non-empty, non-comment line is treated as '[cue] [assoc1] [assoc2] ...'.
    If there are exactly 2 tokens, it is a single pair.

    Returns a DataFrame with columns: ['source', 'target'] for each pair.
    """
    file_path = Path(file_path)
    pairs: List[Tuple[str, str]] = []

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("//") or line.startswith("#") or line.startswith("```"):
                continue

            tokens = re.findall(r"\S+", line)
            if not tokens:
                continue

            # If this line looks purely numeric (like valence numeric rows), skip
            if _is_pure_numeric_row(tokens):
                continue

            # Heuristic: treat first token as cue, next as associated words
            # Ignore lines that start with 'filepath:' or similar meta info
            if tokens[0].lower().startswith("filepath:"):
                continue

            # Normalize words to lowercase and strip; keep underscores etc.
            norm = [t.strip().lower() for t in tokens if t.strip()]
            if len(norm) < 2:
                continue

            cue = norm[0]
            # Create edges cue -> each follower token
            for target in norm[1:]:
                if not target or target == cue:
                    continue
                # Skip numeric-only tokens (defensive)
                if re.fullmatch(r"\d+(?:\.)?", target):
                    continue
                pairs.append((cue, target))

    if not pairs:
        return pd.DataFrame(columns=["source", "target"])

    df = pd.DataFrame(pairs, columns=["source", "target"])
    return df


def import_fmnt_from_old_text_format(
    files: Union[str, Path, Iterable[Union[str, Path]]],
    alpha: float = 0.1,
    min_occurrences: int = 1,
    direction_stat: str = "mean",
    directed: bool = False,
) -> Dict[str, object]:
    """
    Import a Forma Mentis Network from old text files:
      - Students_Filtered.txt
      - Students_Unfiltered.txt
      - ValenceStudentsUPD.txt

    - Merges edges across any number of student files.
    - Computes custom valences via Kruskal–Wallis (median-based) if a valence file is provided.
    - Attaches 'valence_cat' attribute to nodes when available.

    Returns:
      {
        'graph': nx.Graph or nx.DiGraph,
        'edges': pd.DataFrame(columns=['source','target','weight']),
        'valence_df': pd.DataFrame(columns=['cue_word','word','valence']),
        'custom_valences': [positive_set, negative_set, neutral_set] or None
      }
    """
    # Normalize files to list
    if isinstance(files, (str, Path)):
        files_list = [files]
    else:
        files_list = list(files)

    # Classify files: valence vs. student edges
    edge_files: List[Path] = []
    valence_files: List[Path] = []
    for f in files_list:
        p = Path(f)
        name = p.name.lower()
        if "valence" in name:  # heuristic for ValenceStudentsUPD.txt
            valence_files.append(p)
        else:
            edge_files.append(p)

    # Helper: parse a single edge line like: {"word1", "word2"}
    def _parse_braced_edge_line(line: str) -> Optional[Tuple[str, str]]:
        # Remove inline comments and surrounding whitespace
        line = line.split("//", 1)[0].strip()
        if not line:
            return None
        if not (line.startswith("{") and line.endswith("}")):
            return None

        inner = line[1:-1].strip()
        # Split on the first comma only
        parts = inner.split(",", 1)
        if len(parts) != 2:
            return None

        def _clean_token(tok: str) -> str:
            s = tok.strip()
            # remove surrounding quotes if present and stray trailing commas/braces
            if len(s) >= 2 and s[0] in {'"', "'"} and s[-1] in {'"', "'"}:
                s = s[1:-1]
            else:
                # tolerant cleanup for malformed tokens like '"word",' or 'word"}'
                s = s.strip('",}').strip('"{').strip()
            return s

        a = _clean_token(parts[0])
        b = _clean_token(parts[1])

        # Normalize
        a = a.strip().lower()
        b = b.strip().lower()

        # Skip empties, self-loops, and numeric-only tokens
        if not a or not b or a == b:
            return None
        if re.fullmatch(r"\d+(?:\.\d+)?", a) or re.fullmatch(r"\d+(?:\.\d+)?", b):
            return None

        return a, b

    # Parse and merge edges from all student files
    all_edges: List[Tuple[str, str]] = []
    for ef in edge_files:
        if not ef.exists():
            continue
        with ef.open("r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                pair = _parse_braced_edge_line(raw)
                if pair is not None:
                    all_edges.append(pair)

    # Build edges DataFrame with weights
    if all_edges:
        edges_df = pd.DataFrame(all_edges, columns=["source", "target"])
        if directed:
            agg = (
                edges_df.groupby(["source", "target"], as_index=False)
                .size()
                .rename(columns={"size": "weight"})
            )
        else:
            # undirected: count pairs regardless of order
            agg = (
                edges_df.assign(pair=edges_df.apply(lambda r: tuple(sorted((r["source"], r["target"]))), axis=1))
                .groupby("pair")
                .size()
                .reset_index(name="weight")
            )
            agg[["source", "target"]] = pd.DataFrame(agg["pair"].tolist(), index=agg.index)
            agg = agg.drop(columns=["pair"])
        edges_df = agg[["source", "target", "weight"]]
    else:
        edges_df = pd.DataFrame(columns=["source", "target", "weight"])

    # Build graph
    G = nx.DiGraph() if directed else nx.Graph()
    for _, row in edges_df.iterrows():
        s, t, w = row["source"], row["target"], int(row["weight"])
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)

    # Parse valence files and compute custom valences
    if valence_files:
        val_rows: List[pd.DataFrame] = []
        for vf in valence_files:
            if not vf.exists():
                continue
            df_v = _parse_valence_old_text(vf)
            if not df_v.empty:
                val_rows.append(df_v)
        if val_rows:
            valence_df = pd.concat(val_rows, ignore_index=True)
        else:
            valence_df = pd.DataFrame(columns=["cue_word", "word", "valence"])

        if not valence_df.empty:
            _, custom_valences = categorize_valence_multiple(
                valence_df,
                alpha=alpha,
                min_occurrences=min_occurrences,
                return_custom_valences=True,
                direction_stat=direction_stat,
                normalize=True,
            )
            # Map valence to nodes
            val_map: Dict[str, int] = {}
            for w in custom_valences[0]:
                val_map[w] = 1
            for w in custom_valences[1]:
                val_map[w] = -1
            for w in custom_valences[2]:
                val_map.setdefault(w, 0)

            for n in G.nodes:
                wn = str(n).strip().lower()
                if wn in val_map:
                    G.nodes[n]["valence_cat"] = int(val_map[wn])
        else:
            custom_valences = None
    else:
        valence_df = pd.DataFrame(columns=["cue_word", "word", "valence"])
        custom_valences = None

    return {
        "graph": G,
        "edges": edges_df,
        "valence_df": valence_df,
        "custom_valences": custom_valences,
    }


def export_df(df, export_path, filename: str, overwrite=False, extension='csv'):
    '''Export the DataFrame to a file in the specified export path.

    Parameters:
    - df: pandas DataFrame to be exported.
    - export_path: str, path to the directory where the file will be saved.
    - overwrite: bool, whether to overwrite existing files (default is False).
    - filename: str, name of the output file (default is None).
    - extension: str, file extension (default is 'csv').
    '''
    # ensure the output folder exists else create it
    os.makedirs(export_path, exist_ok=True)

    # check if the file already exists
    if filename and os.path.exists(os.path.join(export_path, f"{filename}.{extension}")):
        if not overwrite:
            print(f"File {filename}.{extension} already exists in {export_path}. Use overwrite=True to replace it.")
            return
        else:
            print(f"Overwriting existing file {filename}.{extension} in {export_path}.")

    # save the df to a file in the output folder
    output_file = os.path.join(export_path, f"{filename}.{extension}")
    if extension == 'csv':
        df.to_csv(output_file, index=False)
    elif extension == 'xlsx':
        df.to_excel(output_file, index=False)
    else:
        raise ValueError("Unsupported file extension. Use 'csv' or 'xlsx'.")
    print(f"Data saved to {output_file}\n")


# An updated version of emo.export_whole_fmnt() that supports custom valence exports.
def export_whole_fmn_custom(
    fmnt,
    save_path: str,
    custom_valences=None,
    language: str = None,
    include_valences: bool = True,
    separate_layers_files: bool = False,
    merge_multiplex: bool = False,
    overwrite: bool = True,
    fmn_type: str = "TFMN"
):
    """
    Export a single Formamentis Network to one or (optionally) multiple .txt file(s).

    Supports:
      - Flat edgelists: [(u,v)] or typed edges [(u,v,type)]
      - Multiplex edgelists: {'syntactic':[(u,v)], 'synonyms':[(u,v)], ...}

    Multiplex modes (when fmnt.edges is a dict):
      separate_layers_files=True -> one file per layer
      separate_layers_files=False & merge_multiplex=True  -> single merged edge list with layer annotations
      separate_layers_files=False & merge_multiplex=False -> single file with sections per layer

    Valence handling:
      - include_valences controls whether Positive / Negative / Neutral sections are written
      - custom_valences: list/tuple of 3 sets [positive_set, negative_set, neutral_set]; if provided overrides lexicon
      - If neutral_set omitted (3rd set empty), neutrality inferred (nodes not in pos or neg)

    Overwrite strategy:
      - If overwrite is False and target file exists, a suffix _1, _2, ... is appended.

    Parameters
    ----------
    fmnt : FormamentisNetwork-like (has .edges and optionally .vertices)
    save_path : str
        Either a directory path or a full file path.
        If separate_layers_files=True it must be a directory (not ending with .txt).
    custom_valences : list[set] | tuple[set] | None
    language : str | None
        If None tries to infer (emo.language) else falls back to 'italian'.
    include_valences : bool
    separate_layers_files : bool
    merge_multiplex : bool
    overwrite : bool
    fmn_type : str
        Used in filename construction when save_path is a directory.

    Returns
    -------
    list[str] : list of written file paths.
    """
    # Infer language if needed
    if language is None:
        try:
            lang_guess = emo.language  # relies on a global EmoScores instance if available
        except Exception:
            lang_guess = "italian"
    else:
        lang_guess = language

    # Collect valence sets (or custom)
    if custom_valences:
        if (not isinstance(custom_valences, (list, tuple))) or len(custom_valences) != 3:
            raise ValueError("custom_valences must be [positive_set, negative_set, neutral_set]")
        pos_set, neg_set, neu_set = custom_valences
        if not all(isinstance(s, set) for s in (pos_set, neg_set, neu_set)):
            raise ValueError("custom_valences elements must be sets")
        def is_positive(n): return n in pos_set
        def is_negative(n): return n in neg_set
        def is_neutral(n): return (n in neu_set) or (n not in pos_set and n not in neg_set)
    else:
        pos_set, neg_set, _ = _valences(lang_guess)
        def is_positive(n): return n in pos_set
        def is_negative(n): return n in neg_set
        def is_neutral(n): return (n not in pos_set and n not in neg_set)

    def ensure_path(path):
        if overwrite or (not os.path.exists(path)):
            return path
        base, ext = os.path.splitext(path)
        k = 1
        new_path = f"{base}_{k}{ext}"
        while os.path.exists(new_path):
            k += 1
            new_path = f"{base}_{k}{ext}"
        return new_path

    def write_valence_sections(fh, nodes):
        if not include_valences:
            return
        pos_nodes = sorted([n for n in nodes if is_positive(n)])
        neg_nodes = sorted([n for n in nodes if is_negative(n)])
        neu_nodes = sorted([n for n in nodes if is_neutral(n)])
        fh.write("\n## Positive Nodes\n")
        for n in pos_nodes: fh.write(f"{n}\n")
        fh.write("\n## Negative Nodes\n")
        for n in neg_nodes: fh.write(f"{n}\n")
        fh.write("\n## Neutral Nodes\n")
        for n in neu_nodes: fh.write(f"{n}\n")

    # Determine if save_path is file or directory
    is_directory_target = not save_path.lower().endswith(".txt")
    if is_directory_target:
        os.makedirs(save_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    written = []

    edges_obj = getattr(fmnt, "edges", None)
    title = getattr(fmnt, "title", "formamentis")

    # Helper to gather all nodes from edge collections
    def gather_nodes_from_edges(edge_iterable):
        nodes = set()
        for e in edge_iterable:
            if len(e) >= 2:
                nodes.update([e[0], e[1]])
        return nodes

    # Multiplex case
    if isinstance(edges_obj, dict):
        if separate_layers_files:
            if not is_directory_target:
                raise ValueError("When separate_layers_files=True, save_path must be a directory (not a .txt file).")
            for layer, edgelist in edges_obj.items():
                filename = f"{title}_{layer}_{fmn_type}.txt"
                out_file = ensure_path(os.path.join(save_path, filename))
                nodes = gather_nodes_from_edges(edgelist)
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write("# Forma Mentis Network Export (Layer)\n")
                    f.write(f"# Layer: {layer}\n\n")
                    f.write("## Edges\n")
                    for e in edgelist:
                        if len(e) >= 2:
                            u, v = e[0], e[1]
                            if len(e) >= 3:
                                etype = e[2]
                                f.write(f"{u} -- {v} [{etype}]\n")
                            else:
                                f.write(f"{u} -- {v}\n")
                    write_valence_sections(f, nodes)
                written.append(out_file)
        else:
            # Single file path
            if is_directory_target:
                out_file = os.path.join(save_path, f"{title}_{fmn_type}.txt")
            else:
                out_file = save_path
            out_file = ensure_path(out_file)

            if merge_multiplex:
                merged = {}
                for layer, edgelist in edges_obj.items():
                    for e in edgelist:
                        if len(e) >= 2:
                            u, v = e[0], e[1]
                            key = tuple(sorted((u, v)))
                            merged.setdefault(key, set()).add(layer)
                all_nodes = set([n for pair in merged for n in pair])
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write("# Forma Mentis Network Export (Merged Multiplex)\n\n")
                    f.write("## Edges (u -- v) [layers]\n")
                    for (u, v), layers in sorted(merged.items()):
                        layers_str = ",".join(sorted(layers))
                        f.write(f"{u} -- {v} [{layers_str}]\n")
                    write_valence_sections(f, all_nodes)
                written.append(out_file)
            else:
                all_nodes = set()
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write("# Forma Mentis Network Export (Layered Single File)\n\n")
                    for layer, edgelist in edges_obj.items():
                        f.write(f"## Layer: {layer}\n")
                        for e in edgelist:
                            if len(e) >= 2:
                                u, v = e[0], e[1]
                                all_nodes.update([u, v])
                                if len(e) >= 3:
                                    etype = e[2]
                                    f.write(f"{u} -- {v} [{etype}]\n")
                                else:
                                    f.write(f"{u} -- {v}\n")
                        f.write("\n")
                    write_valence_sections(f, all_nodes)
                written.append(out_file)
    else:
        # Flat list (possibly typed edges)
        if is_directory_target:
            out_file = os.path.join(save_path, f"{title}_{fmn_type}.txt")
        else:
            out_file = save_path
        out_file = ensure_path(out_file)

        nodes = set()
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("# Forma Mentis Network Export\n\n")
            f.write("## Edges\n")
            for e in edges_obj or []:
                if len(e) >= 2:
                    u, v = e[0], e[1]
                    nodes.update([u, v])
                    if len(e) >= 3:
                        etype = e[2]
                        f.write(f"{u} -- {v} [{etype}]\n")
                    else:
                        f.write(f"{u} -- {v}\n")
            write_valence_sections(f, nodes)
        written.append(out_file)

    for p in written:
        print(f"[✔] Network exported to '{p}'")
    return written


def export_whole_fmn_multiple_custom(fmnts, save_path, filename, custom_valences, language="english"):
    """
    Export multiple FMNs with (optional) custom valences into txt files.
    Accepts dict or list. Produces unique filenames for lists.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if isinstance(fmnts, list):
        if len(fmnts) == 1:
            out_path = os.path.join(save_path, f"{filename}.txt")
            export_whole_fmn_custom(fmnts[0], out_path, custom_valences, language=language)
        else:
            for i, item in enumerate(fmnts, start=1):
                out_path = os.path.join(save_path, f"{filename}_{i}.txt")
                export_whole_fmn_custom(item, out_path, custom_valences, language=language)

    elif isinstance(fmnts, dict):
        for name, fmn in fmnts.items():
            out_path = os.path.join(save_path, f"{name}.txt")
            export_whole_fmn_custom(fmn, out_path, custom_valences, language=language)