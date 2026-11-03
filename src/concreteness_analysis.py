import ast
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import CONCRETENESS_NORMS

# base seed of the random baselines: each frame's baseline is seeded from it and from the frame's
# (sample, keyword), so reruns give the same results and adding a sample changes no other result
RANDOM_SEED = 12345
# random word samples behind each frame's baseline (as in the rassereno pipelines)
N_NULL_SAMPLES = 5000


def derive_seed(*keys, base=RANDOM_SEED):
    """Seed of one random draw: a stable function of `base` and `keys` (unlike hash(), the same in
    every Python process)."""
    key = zlib.crc32("\x1f".join(map(str, keys)).encode("utf-8"))
    return int(np.random.SeedSequence([base, key]).generate_state(1)[0])


def _norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _variants(s):
    """Spellings of a multi-word expression: with spaces or with underscores."""
    a = _norm(s)
    return list(dict.fromkeys([a, a.replace("_", " "), a.replace(" ", "_")]))


def _parse_forms(x):
    if isinstance(x, (list, tuple)):
        return [str(t) for t in x]
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return [str(t) for t in v]
        except (ValueError, SyntaxError):
            pass
        return [t.strip(" '\"") for t in x.strip("[]").split(",") if t.strip()]
    return []


def load_concreteness_norms(language="ita"):
    """Lookup tables of the lemmatised Brysbaert et al. (2014) concreteness norms (1 = abstract,
    5 = concrete), translated into Italian.

    The norms were lemmatised and the scores of the word forms sharing a lemma averaged (Conc.M_mean).
    Returns a dict with: lemma_score (lemma -> score), lemma_variant (spelling -> lemma),
    form_variant (word form -> lemma), accent_variant (lemma or form with à written as a -> lemma)
    and all_scores (every score, the pool of the random baseline).
    """
    df = pd.read_csv(CONCRETENESS_NORMS[language])

    lemma_score, lemma_variant, form_variant = {}, {}, {}
    for lemma_raw, score, forms in zip(df["lemma"], df["Conc.M_mean"], df["forms"]):
        lemma = _norm(lemma_raw).replace("_", " ")
        if lemma not in lemma_score:
            lemma_score[lemma] = None if pd.isna(score) else float(score)
        for v in _variants(lemma_raw):
            lemma_variant[v] = lemma
        for f in _parse_forms(forms):
            for v in _variants(f):
                form_variant[v] = lemma

    # text_cleaning writes à as a ('universita'): those spellings are looked up last, so they never
    # replace a word that is in the norms as it is (e.g. 'papa' is not 'papà')
    accent_variant = {}
    for table in (lemma_variant, form_variant):
        for spelling, lemma in table.items():
            plain = spelling.replace("à", "a")
            if plain != spelling:
                accent_variant.setdefault(plain, lemma)

    return {
        "lemma_score": lemma_score,
        "lemma_variant": lemma_variant,
        "form_variant": form_variant,
        "accent_variant": accent_variant,
        "all_scores": df["Conc.M_mean"].dropna().to_numpy(dtype=float),
    }


def _lookup_lemma(word, norms):
    """Lemma of `word` in the norms: as a lemma, then as a word form, then with à written as a
    (None if absent)."""
    for table in ("lemma_variant", "form_variant", "accent_variant"):
        for v in _variants(word):
            if v in norms[table]:
                return norms[table][v]
    return None


def score_words(words, norms):
    """Concreteness of each word: its lemma's score or, for a multi-word expression not in the norms,
    the mean score of its components. Words are deduplicated and sorted, so the result does not
    depend on their order.

    Returns (scores, matched words, unmatched words).
    """
    scores, matched, unmatched = [], [], []
    for word in sorted(dict.fromkeys(_norm(w) for w in words)):
        lemma = _lookup_lemma(word, norms)
        score = norms["lemma_score"].get(lemma) if lemma is not None else None
        if lemma is None:
            # multi-word expression treated as one concept: mean of its components' scores
            components = [_lookup_lemma(p, norms) for p in re.split(r"[ _]+", word)]
            comp_scores = [
                norms["lemma_score"][l]
                for l in components
                if l is not None and norms["lemma_score"].get(l) is not None
            ]
            score = float(np.mean(comp_scores)) if comp_scores else None
        if score is None:
            unmatched.append(word)
        else:
            scores.append(score)
            matched.append(word)
    return scores, matched, unmatched


def cliffs_delta(x, y):
    """Cliff's delta: P(x > y) - P(x < y) over every pair (x, y)."""
    y = np.sort(np.asarray(y, dtype=float))
    x = np.asarray(x, dtype=float)
    less = np.searchsorted(y, x, side="left")  # y values below each x
    greater = len(y) - np.searchsorted(y, x, side="right")  # y values above each x
    return float((less.sum() - greater.sum()) / (len(x) * len(y)))


def frame_concreteness(words, norms, seed, n_samples=N_NULL_SAMPLES):
    """Mean concreteness of a frame's words against a random baseline.

    Null distribution: means of `n_samples` random lists of k scores drawn from the norms, k being
    the number of frame words found in the norms. Z = (frame mean - null mean) / null std, two-sided.
    Cohen's d and Cliff's delta compare the frame scores with the pooled scores of `n_samples` more
    random lists, and are absolute values (the direction is the sign of mean_diff). The Mann-Whitney
    U test compares the same two samples.
    """
    scores, matched, unmatched = score_words(words, norms)
    k = len(scores)
    if k < 3:
        raise ValueError(f"only {k} words of the frame are in the norms (at least 3 needed)")

    pool = norms["all_scores"]
    replace = k > len(pool)
    rng = np.random.default_rng(seed)
    null_means = np.array(
        [pool[rng.choice(len(pool), size=k, replace=replace)].mean() for _ in range(n_samples)]
    )
    random_scores = np.concatenate(
        [rng.choice(pool, size=k, replace=replace) for _ in range(n_samples)]
    )

    mean = float(np.mean(scores))
    null_mean = float(null_means.mean())
    null_std = float(null_means.std(ddof=1))
    z = (mean - null_mean) / null_std

    n_rand = len(random_scores)
    s_pooled = np.sqrt(
        ((k - 1) * np.var(scores, ddof=1) + (n_rand - 1) * np.var(random_scores, ddof=1))
        / (k + n_rand - 2)
    )
    u_stat, p_u = stats.mannwhitneyu(scores, random_scores, alternative="two-sided")

    return {
        "n_frame_words": k + len(unmatched),
        "k": k,
        "mean": mean,
        "null_mean": null_mean,
        "null_std": null_std,
        "mean_diff": mean - null_mean,
        "z": float(z),
        "p_z": float(2 * stats.norm.sf(abs(z))),
        "cohen_d": float(abs(mean - random_scores.mean()) / s_pooled),
        "cliffs_delta": abs(cliffs_delta(scores, random_scores)),
        "median": float(np.median(scores)),
        "median_random": float(np.median(random_scores)),
        "u_statistic": float(u_stat),
        "p_u": float(p_u),
        "ci_low": float(np.percentile(null_means, 2.5)),
        "ci_high": float(np.percentile(null_means, 97.5)),
        "seed": seed,
        "n_null_samples": n_samples,
        "unmatched_words": ", ".join(unmatched),
    }


def concreteness_table(frames, norms, n_samples=N_NULL_SAMPLES):
    """One row per (sample, keyword) frame of `frames` ({(sample, keyword): frame words}).

    Each frame's baseline is seeded from (sample, keyword), so a frame gets the same result whatever
    other frames are analysed with it. Frames with fewer than 3 words in the norms are skipped.
    """
    rows = []
    for (sample, keyword), words in frames.items():
        try:
            result = frame_concreteness(
                words, norms, seed=derive_seed(sample, keyword, "concreteness"), n_samples=n_samples
            )
        except ValueError as e:
            print(f"⚠️ Skipping {sample} / {keyword}: {e}")
            continue
        rows.append({"sample": sample, "keyword": keyword, **result})
    return pd.DataFrame(rows)


def concreteness_latex_tables(table, out_dir, alpha=0.1, labels=None, keyword_labels=None):
    """The paper's two tables of frames significantly more and less concrete than random
    (|Z| above the two-sided critical value of `alpha`), ordered by mean_diff.

    labels / keyword_labels: names shown for the samples and keywords (default: as in `table`).
    Written to <out_dir>/concreteness_more_latex.txt and concreteness_less_latex.txt.
    """
    z_crit = stats.norm.ppf(1 - alpha / 2)
    sig = table[table["z"].abs() > z_crit]
    tab = pd.DataFrame(
        {
            "group": sig["sample"].map(lambda s: (labels or {}).get(s, s)),
            "keyword": sig["keyword"].map(lambda k: (keyword_labels or {}).get(k, k)),
            "k": sig["k"],
            r"$\bar{x}-\hat{\mu}_0$": sig["mean_diff"],
            "$Z$": sig["z"],
            "Cohen's $d$": sig["cohen_d"],
            r"$\bar{x}$": sig["mean"],
            r"Cliff's $\delta$": sig["cliffs_delta"],
        }
    )
    for k in ("group", "keyword"):
        tab[k] = tab[k].astype(str).str.replace("_", r"\_")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    effect = r"$\bar{x}-\hat{\mu}_0$"
    for side, title, ascending in (("more", "higher", False), ("less", "lower", True)):
        sub = tab[tab[effect] > 0] if side == "more" else tab[tab[effect] < 0]
        sub = sub.sort_values(effect, ascending=ascending)
        order = "decreasing" if side == "more" else "increasing"
        latex = sub.to_latex(
            index=False,
            float_format="%.2f",
            position="!hbpt",
            column_format="llrrrrrr",
            caption=(
                f"Frames showing significantly {title} concreteness ($|Z| > {z_crit:.4f}$, "
                rf"$\alpha={alpha}$), ordered by {order} $\bar{{x}} - \hat{{\mu}}_0$."
            ),
            label=f"tab:concreteness_{side}",
        )
        latex = latex.replace(r"\begin{table}[!hbpt]", "\\begin{table}[!hbpt]\n\\centering")
        with open(out_dir / f"concreteness_{side}_latex.txt", "w") as f:
            f.write(latex)
        print(f"{len(sub)} frames {side} concrete than random -> concreteness_{side}_latex.txt")
