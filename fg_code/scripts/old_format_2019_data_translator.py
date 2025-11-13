#!/usr/bin/env python3
import argparse
import re
import sys
from typing import Dict, Optional

# Add tqdm progress bar (safe fallback if not installed)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

# Optional online translator (Google via deep_translator)
try:
    from deep_translator import GoogleTranslator  # pip install deep-translator
except Exception:
    GoogleTranslator = None  # Fallback to local mapping only

# ...existing code...
# Small built-in fallback mapping (extend with --mapping)
# REMOVED: FALLBACK_MAP
# ...existing code...

PAIR_RE = re.compile(r'^\s*\{\s*"(?P<a>.*?)"\s*,\s*"(?P<b>.*?)"\s*\}\s*$')

def load_mapping(path: Optional[str]) -> Dict[str, str]:
    # Start with empty mapping (no built-in fallback)
    mapping: Dict[str, str] = {}
    if not path:
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"[warn] mapping:{ln}: expected 'source<TAB>italian', got: {line}", file=sys.stderr)
                continue
            src, tgt = parts[0].strip(), parts[1].strip()
            if src:
                mapping[src] = tgt
                mapping[src.lower()] = tgt
    return mapping

def preserve_case(src: str, tgt: str) -> str:
    if not tgt:
        return tgt
    if src.isupper():
        return tgt.upper()
    if src[:1].isupper() and src[1:].islower():
        # Capitalize first letter, keep rest as-is (to preserve accents/casing)
        return tgt[:1].upper() + tgt[1:]
    return tgt

class Translator:
    def __init__(self, mapping: Dict[str, str], backend: str = "auto", src_lang: str = "en", tgt_lang: str = "it"):
        self.mapping = mapping
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.cache: Dict[str, str] = {}
        self.backend = None
        if backend == "none":
            self.backend = None
        elif backend in ("auto", "deep_translator"):
            if GoogleTranslator is not None:
                self.backend = GoogleTranslator(source=self.src_lang, target=self.tgt_lang)
            elif backend == "deep_translator":
                print("[error] deep-translator not available. Install with: pip install deep-translator", file=sys.stderr)
                sys.exit(2)
            else:
                self.backend = None

    def translate_token(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        # mapping: exact
        if token in self.mapping:
            out = self.mapping[token]
            out = out.lower()
            self.cache[token] = out
            return out
        # mapping: lowercase
        low = token.lower()
        if low in self.mapping:
            out = self.mapping[low]
            out = out.lower()
            self.cache[token] = out
            return out
        # backend
        out = token
        if self.backend is not None and token.strip():
            try:
                res = self.backend.translate(token)
                if isinstance(res, str) and res:
                    out = res
            except Exception:
                out = token  # keep original on failure
        out = out.lower()
        self.cache[token] = out
        return out

def _count_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)

def translate_format1(in_path: str, out_path: str, tr: Translator):
    total = _count_lines(in_path)
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total, desc="format1"):
            m = PAIR_RE.match(line.rstrip("\n"))
            if not m:
                # Lowercase unmatched lines too
                fout.write(line.lower())
                continue
            a, b = m.group("a"), m.group("b")
            ta = tr.translate_token(a)
            tb = tr.translate_token(b)
            fout.write(f'{{"{ta}", "{tb}"}}\n')

def translate_format2(in_path: str, out_path: str, tr: Translator):
    total = _count_lines(in_path)
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for raw in tqdm(fin, total=total, desc="format2"):
            line = raw.rstrip("\n")
            if not line.strip():
                fout.write(raw)
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                word = parts[0].strip()
                tword = tr.translate_token(word) if word else word
                fout.write((tword + "\n").lower())
                continue
            word = parts[0]
            rest = parts[1:]  # keep exactly as-is
            tword = tr.translate_token(word)
            # Ensure the whole line is lowercase (numbers/tabs unaffected)
            fout.write(("\t".join([tword] + rest) + "\n").lower())

def main():
    ap = argparse.ArgumentParser(description="Translate two custom text formats to Italian, preserving exact formats.")
    ap.add_argument("--format1", nargs=2, metavar=("IN", "OUT"), help='Format 1: lines like {"word1", "word2"}')
    ap.add_argument("--format2", nargs=2, metavar=("IN", "OUT"), help="Format 2 (valences): first field token, then tab-separated numbers")
    ap.add_argument("--mapping", help="Optional tab-separated mapping file: source<TAB>italian")
    ap.add_argument("--backend", choices=["auto", "deep_translator", "none"], default="auto", help="Translation backend")
    ap.add_argument("--src", default="en", help="Source language code (default: en)")
    ap.add_argument("--tgt", default="it", help="Target language code (default: it)")
    args = ap.parse_args()

    if not args.format1 and not args.format2:
        ap.error("Provide at least one of --format1 or --format2")

    mapping = load_mapping(args.mapping)
    tr = Translator(mapping, backend=args.backend, src_lang=args.src, tgt_lang=args.tgt)

    if args.format1:
        inp, outp = args.format1
        translate_format1(inp, outp, tr)
    if args.format2:
        inp, outp = args.format2
        translate_format2(inp, outp, tr)

if __name__ == "__main__":
    main()