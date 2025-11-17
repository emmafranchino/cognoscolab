import os
import re
import pandas as pd
from nltk import word_tokenize, pos_tag

def get_wordnet_pos(treebank_tag):
    """Convert POS tag to a format recognized by WordNetLemmatizer."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# Lemmatization function for English text using NLTK
def lemmatize_text(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags
    ]
    return ' '.join(lemmatized_tokens)

# Lemmatization for Italian text using spaCy (NLTK lemmatization is in English)
def lemmatize_text_it(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])


def _normalize_token(x: str):
    '''Normalize a token by lowercasing and removing punctuation/numbers.
    It’s used in build_association_df to normalize cue and association fields.

    Parameters
    ----------
    x : str
        The input token to normalize.

    Returns
    -------
    str or None
        The normalized token, or None if the input is None or results in an empty string.
    '''
    if x is None or pd.isna(x):
        return None
    x = str(x).lower()
    # remove punctuation and numbers
    x = re.sub(r'[\d\W_]+', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x if x else None


def build_association_df(df: pd.DataFrame, lemmatize: bool = False, lang: str = "italian") -> pd.DataFrame:
    '''Convert a wide-format DataFrame into long format with optional lemmatization.
       - No caching.
       - No preservation of original raw columns.
    '''
    assoc_pattern = re.compile(r"^association_(\d+)$")
    association_numbers = sorted({
        int(m.group(1)) for col in df.columns for m in [assoc_pattern.match(col)] if m
    })

    def _lem(s: str) -> str:
        if not isinstance(s, str) or not s:
            return s
        try:
            if lang.lower().startswith("it"):
                return lemmatize_text_it(s)
            else:
                return lemmatize_text(s)
        except Exception:
            return s

    records = []
    for _, row in df.iterrows():
        participant_id = row.get("participant_id")
        cue_raw = row.get("cue_word")
        norm_cue = _normalize_token(cue_raw)
        if norm_cue is not None and lemmatize:
            norm_cue = _lem(norm_cue)

        # Add cue row (index 0) if present
        if norm_cue is not None:
            val_cue = pd.to_numeric(row.get("valence_cue_word"), errors="coerce")
            records.append({
                "participant_id": participant_id,
                "cue_word": norm_cue,
                "index": 0,
                "word": norm_cue,
                "valence": val_cue
            })

        # Add association rows
        for n in association_numbers:
            word_col = f"association_{n}"
            val_col = f"valence_association_{n}"
            if word_col not in df.columns:
                continue
            raw_word = row.get(word_col)
            norm_word = _normalize_token(raw_word)
            if norm_word is None:
                continue
            if lemmatize:
                norm_word = _lem(norm_word)

            val_raw = row.get(val_col) if val_col in df.columns else None
            numeric_valence = pd.to_numeric(val_raw, errors="coerce")
            records.append({
                "participant_id": participant_id,
                "cue_word": norm_cue,
                "index": n,
                "word": norm_word,
                "valence": numeric_valence
            })

    long_df = pd.DataFrame.from_records(records)
    if not long_df.empty:
        long_df["index"] = long_df["index"].astype(int)
        long_df["valence"] = pd.to_numeric(long_df["valence"], errors="coerce")
    return long_df


def subset_by_group_labels(dfs_and_group_labels, col: str) -> pd.DataFrame:
    """Subset and merge DataFrames by group labels.
    Accepts an iterable of (df, group_labels) pairs.
    group_labels can be a list/tuple of prefixes or None/[] to include all rows.
    Prints unique counts before/after per dataset and total after merging.
    """
    if not dfs_and_group_labels:
        raise ValueError("The input iterable is empty.")

    # Validate same columns across all DataFrames
    first_df, _ = dfs_and_group_labels[0]
    base_columns = set(first_df.columns)
    if col not in base_columns:
        raise ValueError(f"Column '{col}' not found in DataFrames.")
    for df, _ in dfs_and_group_labels[1:]:
        if set(df.columns) != base_columns:
            raise ValueError("All DataFrames must have the same columns.")

    # Subset each DataFrame with reporting
    subset_dfs = []
    for i, (df, group_labels) in enumerate(dfs_and_group_labels, start=1):
        before_unique = df[col].astype("string").nunique(dropna=True)

        if group_labels:
            subset = df[df[col].astype("string").str.startswith(tuple(group_labels), na=False)].copy()
            filter_info = f"labels={list(group_labels)}"
        else:
            subset = df.copy()
            filter_info = "no filtering"

        after_unique = subset[col].astype("string").nunique(dropna=True)
        removed = before_unique - after_unique
        print(f"[{i}] unique '{col}' before: {before_unique} | after: {after_unique} | removed: {removed}")
        # Print the filtered labels from each dataframe
        print(f"[{i}] filtered rows from '{col}':")
        if group_labels:
            filtered_labels = subset[col].astype("string").unique()
            for label in filtered_labels:
                print(f"    {label}")
        subset_dfs.append(subset)

    # Merge subsets and report total
    merged_df = pd.concat(subset_dfs, ignore_index=True)
    total_after = merged_df[col].astype("string").nunique(dropna=True)
    print(f"[total] unique '{col}' after subsetting and merging: {total_after}")

    return merged_df


def filter_associations_by_occurrences(
    df: pd.DataFrame,
    min_association_occurrences: int | None = None,
    word_col: str = "word",
    assoc_only: bool = True,
    keep_cue_rows: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter by association pairs (cue_word, word) frequency.

    Keeps only rows where the association pair (cue_word, word) occurs at least
    min_association_occurrences across the dataset. Rows with index==0 (cue rows)
    are always kept if keep_cue_rows is True.

    Notes
    - Only association rows (index > 0) are filtered; cue rows are never part of
      (cue, word) pairs.
    - If min_association_occurrences is None or <= 1, the input is returned unchanged.
    """
    if min_association_occurrences is None or min_association_occurrences <= 1:
        if verbose:
            print("No filtering applied (min_association_occurrences is None or <= 1).")
        return df

    required_cols = {"cue_word", word_col, "index"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns {missing} in df")

    work = df.copy()

    # Association rows are those with index > 0
    assoc_mask = work["index"] > 0

    # Count pair frequencies on association rows only
    pair_counts = (
        work.loc[assoc_mask, ["cue_word", word_col]]
            .dropna(subset=["cue_word", word_col])
            .groupby(["cue_word", word_col], dropna=False)
            .size()
            .rename("pair_count")
            .reset_index()
    )

    # Merge counts back to all rows; cue rows (index==0) will have NaN counts
    work = work.merge(pair_counts, on=["cue_word", word_col], how="left")

    # Allowed only if pair_count >= threshold for association rows
    keep_assoc = work["pair_count"].fillna(0).astype(int) >= int(min_association_occurrences)

    # Build final keep mask
    keep_mask = pd.Series(False, index=work.index)
    if keep_cue_rows:
        keep_mask |= (work["index"] == 0)
    # Filter association rows by pair frequency
    keep_mask |= (assoc_mask & keep_assoc)

    out = work.loc[keep_mask].drop(columns=["pair_count"])

    if verbose:
        # Print top pairs for inspection
        top_n = 50
        sorted_pairs = (
            pair_counts.sort_values("pair_count", ascending=False)
                        .head(top_n)
        )
        total_pairs = int(pair_counts.shape[0])
        print(f"\nTop {min(top_n, total_pairs)} association pairs (of {total_pairs} total):")
        if not sorted_pairs.empty:
            print(sorted_pairs.to_string(index=False))
        else:
            print("(no association pairs found)")

        before_rows = len(df)
        after_rows = len(out)
        before_assoc_rows = int(assoc_mask.sum())
        after_assoc_rows = int((out["index"] > 0).sum())
        kept_unique_pairs = (
            out.loc[out["index"] > 0, ["cue_word", word_col]]
              .dropna()
              .drop_duplicates()
              .shape[0]
        )
        print(f"\nApplied pair frequency filter >= {min_association_occurrences} on (cue_word, {word_col}).")
        print(f"Rows: {before_rows} -> {after_rows} (removed {before_rows - after_rows})")
        print(f"Association rows: {before_assoc_rows} -> {after_assoc_rows} (removed {before_assoc_rows - after_assoc_rows})")
        print(f"Kept unique association pairs: {kept_unique_pairs}")

    return out


def lemmatize_brysbaert(
        input_path="../data/brysbaert_concreteness_dataset_eng_ita/brysbaert_concreteness_dataset_eng_ita.csv",
        output_path="../data/brysbaert_concreteness_dataset_eng_ita/brysbaert_concreteness_dataset_eng_ita_lemmatized_en.csv",
        word_column=None,
        conc_column="Conc.M",
        language="en",
        n_process=1,
        overwrite=False,
        keep_intermediate_forms=True,
        nlp=None
    ):
    """
    Create a lemmatized / aggregated concreteness dataset using the pre-loaded spaCy model.
    Requires that `nlp` is provided or a global `nlp` already initialized via init_spacy_and_emo().
    No fallback or automatic model loading is performed.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if os.path.exists(output_path) and not overwrite:
        return pd.read_csv(output_path)

    lang = language.lower()
    if word_column is None:
        word_column = "english" if lang.startswith("en") else "italian"

    df = pd.read_csv(input_path)
    if word_column not in df.columns:
        raise ValueError(f"word_column '{word_column}' not in columns: {list(df.columns)}")
    if conc_column not in df.columns:
        raise ValueError(f"conc_column '{conc_column}' not in columns: {list(df.columns)}")

    if nlp is None:
        if "nlp" in globals():
            nlp = globals()["nlp"]
        else:
            raise RuntimeError("spaCy 'nlp' not provided and global 'nlp' not found. Initialize first.")

    df = df.dropna(subset=[word_column, conc_column]).copy()
    df[conc_column] = pd.to_numeric(df[conc_column], errors="coerce")
    df = df.dropna(subset=[conc_column])
    surface_forms = df[word_column].astype(str).tolist()
    unique_forms = list(dict.fromkeys(surface_forms))

    def norm_lemma(doc):
        toks = [t.lemma_.lower() for t in doc if t.is_alpha]
        if not toks:
            return doc.text.lower()
        return "_".join(toks)

    lemmas = []
    for doc in nlp.pipe(unique_forms, n_process=n_process, batch_size=500):
        lemmas.append(norm_lemma(doc))
    form2lem = dict(zip(unique_forms, lemmas))
    df["lemma"] = df[word_column].map(form2lem)

    grouped = (
        df.groupby("lemma")[conc_column]
          .agg(['mean', 'std', 'count'])
          .rename(columns={'mean': 'Conc.M_mean', 'std': 'Conc.M_std', 'count': 'n_forms'})
          .reset_index()
          .sort_values("lemma")
          .reset_index(drop=True)
    )

    if keep_intermediate_forms:
        forms_series = (df.groupby("lemma")[word_column]
                          .apply(lambda s: sorted(set(s))))
        grouped = grouped.merge(forms_series.rename("forms"), on="lemma", how="left")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grouped.to_csv(output_path, index=False)
    return grouped