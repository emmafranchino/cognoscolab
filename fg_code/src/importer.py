from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Dict, Set

import pandas as pd
from collections import namedtuple, defaultdict, Counter
import os
import numpy as np
import re
import networkx as nx
from emoatlas.resources import _valences
# from now on the packages are for the import_fmnt_from_sets function
# Also defaultdict, Counter are used only there
from scipy import stats
from typing import Tuple, List, Set, Dict



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


def categorize_valence_multiple(df, alpha=0.1, min_occurrences=3, return_custom_valences=False, direction_stat="mean", normalize=False):
    """
    Global (pooled) nonparametric per-word valence categorization (Kruskal-Wallis).
    direction_stat: 'median' (default) or 'mean' to decide if positive, negative or neutral after significance.
    If normalize=True, lowercase/strip the 'word' column inside this function.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'word' and 'valence' columns.
    alpha : float
        Significance level for the Kruskal-Wallis test (default: 0.1).
    min_occurrences : int
        Minimum number of occurrences of a word to perform the test (default: 3).
    return_custom_valences : bool
        If True, return a custom mapping of words to their valence categories.
    direction_stat : str
        Statistic to use for direction ('median' or 'mean').
    normalize : bool
        If True, normalize the 'word' column by lowercasing and stripping whitespace.
    """
    # Do not drop NaNs in valence early because we need as much
    # word occurrences as possible for the statistical test.
    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Ensure 'word' column is string type and normalize if requested
    df['word'] = df['word'].astype("string")
    if normalize:
        df['word'] = df['word'].str.strip().str.lower()

    # Check for possible mojibakes (encoding issues)
    for word in df['word']:
        if isinstance(word, str) and ('Ã' in word or 'ã' in word):
            print(f"\nWarning! Found problematic character in word: {word}\n")

    # Find number of NaNs in each column
    cols = ['cue_word', 'word', 'valence']
    for col in cols:
        if df[col].isna().sum() > 0:
            print(f"Missing values (NaNs) found in '{col}': {df[col].isna().sum()}\n")

    # Get unique words excluding NaNs
    unique_words = list(pd.Series(df['word'].unique().dropna()))

    # Initialize valence_map to store the valence category for each word
    valence_map = {}
    val_series = df['valence']

    for w in unique_words:
        mask = df['word'] == w

        if val_series[mask].isna().sum() > 0:
            print(f"Important! Found NaNs in valence series for word '{w}': {val_series[mask].isna().sum()}")

        # Get the valence scores for the current word (ignoring NaNs) in a Series
        word_valences = val_series[mask].dropna().astype(float)

        # If the word has less than min_occurrences, assign neutral valence
        if len(word_valences) < min_occurrences:
            valence_map[w] = 0
            continue

        other_vals = val_series[~mask].dropna().astype(float)
        if len(other_vals) == 0:
            valence_map[w] = 0
            continue
        
        # Perform Kruskal-Wallis test
        stat, p_value = stats.kruskal(word_valences, other_vals)

        if direction_stat == "median":
            word_v_center = word_valences.median()
            other_v_center = other_vals.median()
        else:
            word_v_center = word_valences.mean()
            other_v_center = other_vals.mean()

        if p_value < alpha:
            if word_v_center < other_v_center:
                valence_map[w] = -1
            elif word_v_center > other_v_center:
                valence_map[w] = 1
            else:
                valence_map[w] = 0
        else:
            valence_map[w] = 0

    # Map the valence categories back to the original column
    df['valence_cat'] = df['word'].map(valence_map).astype('Int64')

    # Create custom valences in the order: [positive, negative, neutral]
    positive, negative, neutral = set(), set(), set()
    custom_valences = [positive, negative, neutral]
    for word, cat in valence_map.items():
        if cat == 1:
            custom_valences[0].add(word)  # positive
        elif cat == -1:
            custom_valences[1].add(word)  # negative
        else:
            custom_valences[2].add(word)  # neutral

    if return_custom_valences:
        return df, custom_valences
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