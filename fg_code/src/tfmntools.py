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

    import re
    from collections import Counter

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