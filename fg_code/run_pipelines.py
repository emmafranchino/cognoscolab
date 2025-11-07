#!/usr/bin/env python3
import sys
import os
import re
import json
import shutil
import tempfile
import gc
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# example run:
# RUN_NOTEBOOKS="scripts/bfmns_*.ipynb:scripts/tfmns_*.ipynb" python3 run_pipelines.py "['matematica','ansia','stress','depressione']"

# --- helpers ---------------------------------------------------------------
KW_PATTERN = re.compile(r'^\s*keywords\s*=\s*\[.*?\]\s*$', flags=re.MULTILINE)
GLOB_CHARS = set("*?[]")
# New: match any assignment to min_association_occurrences (allow inline comments)
MIN_ASSOC_PATTERN = re.compile(r'^min_association_occurrences\s=\s\d+$', flags=re.MULTILINE)

def parse_keywords_arg(arg: str):
    """
    Accepts either:
      - CSV: "a,b,c"
      - JSON-ish Python list string: "['a','b']" or '["a","b"]'
    Returns a list[str] (unique, stripped, lowercased).
    """
    s = arg.strip()
    words: list[str] = []

    # Try JSON first (valid JSON list)
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            arr = json.loads(s.replace("'", '"'))
            if isinstance(arr, (list, tuple)):
                words = [str(x).strip() for x in arr if str(x).strip()]
            else:
                raise ValueError
        except Exception:
            # Fallback: manual parse of a Python-like list
            inner = s[1:-1].strip()
            if inner:
                parts = re.split(r"[,\n]+", inner)
                words = [p.strip().strip("'").strip('"') for p in parts if p.strip()]
    else:
        # CSV or colon/comma separated
        parts = re.split(r"[,\n:]+", s)
        words = [p.strip() for p in parts if p.strip()]

    # Normalize and deduplicate preserving order
    out = []
    seen = set()
    for w in words:
        w2 = w.strip()
        if not w2:
            continue
        if w2 not in seen:
            out.append(w2)
            seen.add(w2)
    return out


# Parse min_association_occurrences from argv or env
def parse_min_assoc_arg(optional_arg: str | None) -> int | None:
    """
    Accepts either the provided string or ENV MIN_ASSOC.
    Returns a positive int or None if not provided.
    """
    s = (optional_arg.strip() if optional_arg else "") or os.environ.get("MIN_ASSOC", "").strip()
    if not s:
        return None
    try:
        v = int(s)
        if v <= 0:
            raise ValueError
        return v
    except Exception:
        raise SystemExit("Error: min_association_occurrences must be a positive integer (2nd arg or MIN_ASSOC env var).")


# New: replace "min_association_occurrences = <...>" lines in code cells
def replace_min_assoc(nb, value: int) -> int:
    """
    Replace every line like:
        min_association_occurrences = ...
    with:
        min_association_occurrences = <value>
    Returns the total number of replaced occurrences.
    """
    replaced = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        if not src:
            continue
        matches = list(MIN_ASSOC_PATTERN.finditer(src))
        if not matches:
            continue
        new_src = MIN_ASSOC_PATTERN.sub(f"min_association_occurrences = {value}", src)
        if new_src != src:
            cell["source"] = new_src
            replaced += len(matches)
    return replaced


def make_keywords_literal(words):
    """
    Produce a Python literal for: keywords = ['a','b',...]
    Ensures safe quoting.
    """
    safe = []
    for w in words:
        # escape single quotes
        safe.append("'" + w.replace("'", "\\'") + "'")
    return f"[{','.join(safe)}]"


def replace_keywords(nb, new_literal: str) -> int:
    """
    Replace every line like:
        keywords = [...]
    in all code cells with:
        keywords = <new_literal>

    Returns the total number of replaced occurrences.
    """
    replaced = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        if not src:
            continue
        matches = list(KW_PATTERN.finditer(src))
        if not matches:
            continue
        new_src = KW_PATTERN.sub(f"keywords = {new_literal}", src)  # replace ALL occurrences
        if new_src != src:
            cell["source"] = new_src
            replaced += len(matches)
    return replaced

def _list_all_notebooks(root: Path):
    # (unused after simplification, can be removed if desired)
    nbs = []
    for p in root.rglob("*.ipynb"):
        if ".ipynb_checkpoints" in p.parts:
            continue
        nbs.append(p)
    return sorted(nbs)

def _expand_patterns(root: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    seen = set()
    for raw in patterns:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("./"):
            s = s[2:]
        if any(c in s for c in GLOB_CHARS):
            matches = sorted(root.rglob(s))
            for m in matches:
                if m.suffix == ".ipynb" and ".ipynb_checkpoints" not in m.parts:
                    if m not in seen:
                        out.append(m)
                        seen.add(m)
        else:
            p = (root / s).resolve()
            if p.is_file() and p.suffix == ".ipynb" and ".ipynb_checkpoints" not in p.parts:
                if p not in seen:
                    out.append(p)
                    seen.add(p)
    return out

def select_notebooks(root: Path) -> list[Path]:
    """
    Selection:
      RUN_NOTEBOOKS env var (CSV, colon-separated, or newline; supports globs)
    """
    env_val = os.environ.get("RUN_NOTEBOOKS", "").strip()
    if not env_val:
        raise SystemExit(
            "Error: set RUN_NOTEBOOKS to a CSV/':'/newline list of notebook paths or globs.\n"
            'Example: RUN_NOTEBOOKS="scripts/a.ipynb:scripts/b*.ipynb"'
        )
    patterns = re.split(r"[,\n:]", env_val)
    nbs = _expand_patterns(root, patterns)
    if not nbs:
        raise SystemExit("Error: RUN_NOTEBOOKS matched 0 notebooks.")
    print("Notebook selection: RUN_NOTEBOOKS env var")
    return nbs

def execute_notebook(nb_path: Path, tmp_nb_path: Path):
    """
    Execute tmp_nb_path and write results in-place.
    Uses the same directory as working directory for relative paths.
    """
    with open(tmp_nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=None, kernel_name="python3", allow_errors=False)
    # Use the notebook's dir as working directory
    wd = str(nb_path.parent)
    resources = {"metadata": {"path": wd}}

    ep.preprocess(nb, resources)

    with open(tmp_nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

# --- main ------------------------------------------------------------------

def main():
    if len(sys.argv) not in (2, 3):
        print("Usage:")
        print('  RUN_NOTEBOOKS="path1.ipynb:path2.ipynb" python3 run_pipelines.py "matematica,ansia" [MIN_ASSOC]')
        print('  RUN_NOTEBOOKS="scripts/*.ipynb" MIN_ASSOC=3 python3 run_pipelines.py "[\'matematica\',\'ansia\']"')
        sys.exit(2)

    keywords_list = parse_keywords_arg(sys.argv[1])
    if not keywords_list:
        print("Error: empty keywords list.")
        sys.exit(2)

    # Optional second arg or env MIN_ASSOC
    min_assoc = parse_min_assoc_arg(sys.argv[2] if len(sys.argv) == 3 else None)

    new_literal = make_keywords_literal(keywords_list)

    repo_root = Path(__file__).resolve().parent
    notebooks = select_notebooks(repo_root)

    print(f"Found {len(notebooks)} notebooks:")
    for p in notebooks:
        print(f" - {p.relative_to(repo_root)}")

    executed = 0
    tmp_dir = Path(tempfile.mkdtemp(prefix="nb_run_"))
    try:
        for nb_path in notebooks:
            print(f"\n[CHECK] {nb_path.name}")
            with open(nb_path, "r", encoding="utf-8") as fh:
                nb = nbformat.read(fh, as_version=4)

            # Perform replacements
            kw_replaced = replace_keywords(nb, new_literal)
            min_assoc_replaced = 0
            if min_assoc is not None:
                min_assoc_replaced = replace_min_assoc(nb, min_assoc)

            if kw_replaced == 0 and min_assoc_replaced == 0:
                print(f"[SKIP]  {nb_path.name}: no 'keywords = [...]' or 'min_association_occurrences = ...' assignment found.")
                try:
                    del nb
                except Exception:
                    pass
                gc.collect()
                continue
            else:
                if kw_replaced > 0:
                    print(f"[REPLACE] {nb_path.name}: keywords -> {new_literal} ({kw_replaced} occurrence(s))")
                if min_assoc_replaced > 0:
                    print(f"[REPLACE] {nb_path.name}: min_association_occurrences -> {min_assoc} ({min_assoc_replaced} occurrence(s))")
                if min_assoc is not None and min_assoc_replaced == 0:
                    print(f"[WARN] {nb_path.name}: no 'min_association_occurrences =' assignment found to replace.")

            tmp_nb = tmp_dir / nb_path.name
            with open(tmp_nb, "w", encoding="utf-8") as fh:
                nbformat.write(nb, fh)

            print(f"[RUN]   {nb_path.name}")
            execute_notebook(nb_path, tmp_nb)
            executed += 1
            print(f"[OK]    {nb_path.name}")

            try:
                del nb
            except Exception:
                pass
            try:
                del tmp_nb
            except Exception:
                pass
            gc.collect()

        print(f"\nExecuted {executed}/{len(notebooks)} notebooks.")
        if executed == 0:
            sys.exit(3)
    except Exception as e:
        print(f"\n[FAIL] Execution error: {e}")
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()