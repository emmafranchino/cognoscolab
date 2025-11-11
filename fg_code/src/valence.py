import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import spacy
from emoatlas import EmoScores
from scipy import stats


# compute the numeric valence categorization means for all words in each BFMN
def fmn_valence_stats(network, custom_valences=None, emo=None):
    """
    Compute nodes valence stats for the entire network using ONLY
    custom_valences.
    custom_valences must be [positive_set, negative_set, neutral_set].

    Mainly used to check if valence categorizations are correctly imported
    from external files (compare against the network you have exported,
    calling this same function with both).

    Parameters:
    network: FormaMentisNetwork or NetworkX graph.
    emo: EmoScores instance required if 'network' is a FormaMentisNetwork.
    """
    if custom_valences is None:
        raise ValueError("custom_valences is required; no lexicon fallback is allowed.")
    if not isinstance(custom_valences, (list, tuple)) or len(custom_valences) != 3:
        raise ValueError("custom_valences must be a list/tuple of three sets: [positive_set, negative_set, neutral_set]")
    pos_set, neg_set, neu_set = custom_valences
    if not all(isinstance(s, set) for s in (pos_set, neg_set, neu_set)):
        raise ValueError("Each element in custom_valences must be a set")

    # If a FormaMentisNetwork is provided, convert it to a NetworkX graph
    if not isinstance(network, nx.Graph):
        if emo is None:
            raise ValueError("Provide 'emo' (EmoScores) when passing a FormaMentisNetwork.")
        G = emo.formamentis_to_nxgraph(network)
    else:
        G = network
    
    # Handle empty graph case
    if G.number_of_nodes() == 0:
        raise ValueError("The provided graph has no nodes.")

    vals = []
    for node in G.nodes():
        if node in pos_set:
            vals.append(1)
        elif node in neg_set:
            vals.append(-1)
        elif node in neu_set:
            vals.append(0)
        else:
            # Not listed in any custom set -> treat as neutral
            vals.append(0)

    arr = np.array(vals, dtype=float)
    total = arr.size
    n_pos = int((arr == 1).sum())
    n_neg = int((arr == -1).sum())
    n_neu = int((arr == 0).sum())

    return {
        "mean_valence": float(arr.mean()) if total else None,
        "median_valence": float(np.median(arr)) if total else None,
        "std_valence": float(arr.std(ddof=0)) if total else None,
        "n_positive_nodes": n_pos,
        "n_negative_nodes": n_neg,
        "n_neutral_nodes": n_neu,
        "total_nodes": total
    }


def categorize_valence_single(df):
    
    # convert to NumPy array
    arr = np.array(df['valence'])

    # compute value range (gamma)
    minv, maxv = arr.min(), arr.max()
    gamma = maxv - minv

    # if values are too close together, return all neutral --> if gamma ≤ 1 (gamma = 0 and gamma < 1 included)
    # return pd.Series([0] * len(arr), index=valence_series.index)
    if gamma <= 1:
        df['valence_cat'] = 0

    # if gamma > 1: use inclusive quartiles to classify
    Q1, Q3 = np.percentile(arr, [25, 75])

    # defines classification logic based on position relative to quartiles
    def cat_score(v):
        if v <= Q1:
            return -1  # negative
        elif v >= Q3:
            return 1   # positive
        else:
            return 0   # neutral

    # apply classification to the new "valence" column of the long form df
    df['valence_cat'] = df['valence'].apply(cat_score)

    return df


# function to plot valence categories distrubution
def plot_valence_distribution(df):
    '''Plot the distribution of valence categories in the DataFrame.
    '''
    plt.figure(figsize=(6,5))

    # count the number of occurrences of each valence category
    # and return the sorted df
    counts = df['valence_cat'].value_counts().sort_index()
    tick_positions = range(len(counts)) 
    counts.plot(kind='bar')

    plt.title('Valence categories distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(tick_positions, labels=['-1 (Negative)', '0 (Neutral)', '+1 (Positive)'], rotation=0)

    plt.tight_layout()
    plt.show()


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