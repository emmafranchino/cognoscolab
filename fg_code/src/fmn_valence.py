import os
import sys
from emoatlas import EmoScores
from emoatlas.resources import _valences
import spacy
import numpy as np
import networkx as nx

# Add project root (parent of this scripts/ folder) to sys.path once
PROJECT_ROOT = os.path.abspath("..")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Retrieve environment specific variables (e.g. for NLTK and spaCy
# data paths). Ensure the env file is in the same directory as this script.
env_file_path = "../env"

# Recommended: install spacy model in
# /home/username/spacy-models or similar and NLTK data in
# /home/user/nltk_data or similar, then set the paths in 
# env and retrieve them here.

# Check if the .env file exists and load its variables
if os.path.exists(env_file_path):
    
    with open(env_file_path) as env_file:
        for line in env_file:
            # Ignore comments and empty lines
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

    SPACY_MODEL_PATH = os.environ.get("SPACY_MODEL_PATH")
    NLTK_DATA = os.environ.get("NLTK_DATA")
    # Print loaded paths for verification
    print(f"Loaded SPACY_MODEL_PATH: {SPACY_MODEL_PATH}")
    print(f"Loaded NLTK_DATA: {NLTK_DATA}\n")

else:
    print(f"Warning: {env_file_path} file not found. Using default paths.")
    # set default paths or handle the absence of the file as needed
    SPACY_MODEL_PATH = None
    NLTK_DATA = None

# Load spacy
if SPACY_MODEL_PATH:
    nlp = spacy.load(SPACY_MODEL_PATH)
else:
    nlp = spacy.load("it_core_news_lg")  # fallback if appropriate

# This initializes the emotional lexicons, spaCy models, and other internal
# resources needed for textual analysis.
emo = EmoScores(language='italian', spacy_model=SPACY_MODEL_PATH)


# compute the numeric valence categorization means for all words in each BFMN
def fmn_valence_stats(network, custom_valences=None, emo=emo):
    """
    Compute nodes valence stats for the entire network using ONLY
    custom_valences.
    custom_valences must be [positive_set, negative_set, neutral_set].

    Mainly used to check if valence categorizations are correctly imported
    from external files (compare against the network you have exported,
    calling this same function with both).

    Parameters:
    network: FormaMentisNetwork or NetworkX graph.
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


# Create a function that shows the valence of a given word in a given network
def show_word_valence(word, G, custom_valences_sets):
    node_data = G.nodes.get(word)
    if node_data is None:
        return {"word": word, "error": "Word not found in the network"}

    custom_label = None
    if custom_valences_sets:
        pos, neg, neu = custom_valences_sets
        if word in pos:
            custom_label = "positive"
        elif word in neg:
            custom_label = "negative"
        elif word in neu:
            custom_label = "neutral"

    return {
        "word": word,
        "valence": node_data.get("valence", "No valence"),
        "custom_valence": custom_label or "No custom valence"
    }