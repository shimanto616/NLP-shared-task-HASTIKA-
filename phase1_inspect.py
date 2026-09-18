#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HASTIKA shared task (Kanglish hate speech) -- PHASE 1
Environment & Data Inspection.

Run  : python phase1_inspect.py
Deps : pandas (pip install pandas); everything else is stdlib.
I/O  : prints a report to the console only; writes nothing to disk.

Paste the ENTIRE console output back to your research lead.
"""

import os
import re
import sys
import platform
from collections import Counter

try:
    import pandas as pd
except ImportError:
    sys.exit("[FATAL] pandas is required -> pip install pandas")

# Make exotic characters safe to print on any console (e.g., Windows cp1252)
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEARCH_DIRS = [".", "data", "../data", "inputs", "./HASTIKA", "./hastika"]

FILES = {
    "A_train": dict(path="binary_train.csv", text="Comment", label="Label",
                    allowed={"Hate", "Non-Hate"}),
    "A_val":   dict(path="binary_validation_inputs.csv", text="Comment",
                    label=None, allowed=None),
    "B_train": dict(path="multiclass_train.csv", text="Comment",
                    label="Hate Category",
                    allowed={"Gender", "Political", "Religion",
                             "Geo-political", "Violence", "Others"}),
    "B_val":   dict(path="multiclass_validation_inputs.csv", text="Comment",
                    label=None, allowed=None),
}

BAR = "=" * 78
SUB = "-" * 78

# Regex toolbox (unicode-escaped so this .py file is encoding-proof)
RE_KANNADA  = re.compile(r"[\u0C80-\u0CFF]")     # native Kannada script
RE_DEVANAG  = re.compile(r"[\u0900-\u097F]")     # Devanagari
RE_LATIN    = re.compile(r"[A-Za-z]")
RE_EMOJI    = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]")
RE_MOJI_KN  = re.compile(r"\u00e0[\u00b2\u00b3]")                    # mojibake Kannada
RE_MOJI_EMO = re.compile(r"\u00f0\u0178|\u00e2\u20ac|\u00ef\u00b8")  # mojibake emoji/quotes
RE_TOKEN    = re.compile(r"[A-Za-z\u0C80-\u0CFF]+")

ARTEFACTS = {
    "html tag (<br>, <a href>)":     r"<[a-zA-Z/][^>]{0,120}>",
    "html entity (&quot; &#39;)":    r"&[a-zA-Z#0-9]{2,8};",
    "url (http/www)":                r"(?:https?://|www\.)\S+",
    "@mention":                      r"@[A-Za-z0-9_.\-]+",
    "#hashtag":                      r"#[A-Za-z0-9_]+",
    "repeated punct (???/!!!)":      r"([!?])\1{2,}",
    "elongated words (aaaa)":        r"\b[A-Za-z]*([a-z])\1{2,}[A-Za-z]*\b",
    "contains digits":               r"[0-9]",
    "no letters (emoji/punct only)": r"^[^A-Za-z\u0C80-\u0CFF]*$",
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def section(title):
    print(f"\n{BAR}\n{title}\n{BAR}")

def resolve_path(fname):
    for d in SEARCH_DIRS:
        cand = os.path.join(d, fname)
        if os.path.isfile(cand):
            return cand
    return None

def smart_load(fname):
    """utf-8 first; cp1252/latin-1 fallback (mojibake-tolerant)."""
    path = resolve_path(fname)
    if path is None:
        return None, None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError:
            continue
    return None, None

def find_col(df, target):
    if target is None:
        return None
    for c in df.columns:
        if str(c).strip().lower() == str(target).strip().lower():
            return c
    return None

def texts(df, tcol):
    return df[tcol].fillna("").astype(str)

def norm_texts(df, tcol):
    return (texts(df, tcol)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.lower())

def label_series(d):
    return d["df"][d["lcol"]].fillna("<<NaN>>").astype(str).str.strip()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # -- 0. environment -------------------------------------------------------
    section("0. ENVIRONMENT PROBE")
    print(f"Python : {platform.python_version()}  on {platform.system()}")
    print(f"pandas : {pd.__version__}")
    for mod in ("numpy", "sklearn", "xgboost", "torch", "transformers"):
        try:
            m = __import__(mod)
            print(f"{mod:<12}: {getattr(m, '__version__', 'installed')}")
        except Exception:
            print(f"{mod:<12}: not installed (OK for Phase 1)")

    data = {}

    # -- 1. inventory ---------------------------------------------------------
    section("1. FILE INVENTORY, SCHEMA, MISSING VALUES, DUPLICATES")
    for key, cfg in FILES.items():
        df, enc = smart_load(cfg["path"])
        print(f"\n{SUB}\n[{key}]  {cfg['path']}")
        if df is None:
            print("  !! NOT FOUND -- searched: " + ", ".join(SEARCH_DIRS))
            continue
        tcol = find_col(df, cfg["text"])
        lcol = find_col(df, cfg["label"])
        data[key] = dict(df=df, tcol=tcol, lcol=lcol, enc=enc, cfg=cfg)
        print(f"  loaded from : {resolve_path(cfg['path'])}")
        print(f"  encoding    : {enc}")
        print(f"  shape       : {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"  columns     : {list(df.columns)}")
        if tcol is None:
            print("  !! text column 'Comment' not found -- check header")
            continue
        s = texts(df, tcol)
        print(f"  null comments      : {int(df[tcol].isna().sum())}")
        print(f"  empty/whitespace   : {int((s.str.strip() == '').sum())}")
        if "id" in df.columns:
            print(f"  duplicate ids      : {int(df['id'].duplicated().sum())}")
        dup_rows = int(len(s) - norm_texts(df, tcol).nunique())
        print(f"  duplicate comments : {dup_rows} (rows beyond 1st occurrence,")
        print(f"                       after lowercasing + whitespace collapse)")
        print(f"  label column       : {lcol!r}"
              + ("" if lcol else "   (inputs-only file)"))

    # -- 2. labels ------------------------------------------------------------
    section("2. LABEL AUDIT & CLASS BALANCE (+ analytic baseline floors)")
    for key in ("A_train", "B_train"):
        if key not in data or data[key]["lcol"] is None:
            continue
        d = data[key]
        lab = label_series(d)
        vc = lab.value_counts()
        n = len(lab)
        print(f"\n{SUB}\n[{key}]  label column {d['lcol']!r}  (n={n})")
        for k, v in vc.items():
            print(f"   {k:<14} {v:>6}  ({100.0 * v / n:5.2f}%)")
        allowed = d["cfg"]["allowed"]
        if allowed:
            bad = sorted(set(lab) - set(allowed))
            print(f"   unexpected label values : {bad if bad else 'none'}")
        p = vc.max() / n
        C = len(vc)
        print(f"   majority class          : {vc.index[0]} ({100.0 * p:.2f}%)")
        print(f"   imbalance ratio max/min : {vc.max() / vc.min():.2f}")
        print(f"   analytic floor -- all-majority macro-F1        : "
              f"{(2 * p / (1 + p)) / C:.4f}")
        print(f"   analytic floor -- train-prior random macro-F1  : "
              f"{float((vc / n).pow(2).sum()):.4f}")
        print(f"   (floors assume the eval set mirrors the train prior)")

    # -- 3. lengths -----------------------------------------------------------
    section("3. TEXT LENGTH STATISTICS")
    for key, d in data.items():
        if d["tcol"] is None:
            continue
        s = texts(d["df"], d["tcol"])
        tmp = pd.DataFrame({
            "chars": s.str.len().astype(int),
            "words": s.map(lambda x: len(x.split())).astype(int),
        })
        print(f"\n{SUB}\n[{key}]")
        print(f"   [all] n={len(tmp)}  mean_chars={tmp['chars'].mean():.1f}"
              f"  mean_words={tmp['words'].mean():.1f}"
              f"  <=2words={100.0 * (tmp['words'] <= 2).mean():.2f}%")
        if d["lcol"]:
            tmp["label"] = label_series(d).values
            g = tmp.groupby("label").agg(
                n=("chars", "size"),
                mean_chars=("chars", "mean"),
                median_chars=("chars", "median"),
                max_chars=("chars", "max"),
                mean_words=("words", "mean"),
                max_words=("words", "max"),
            ).round(2)
            print(g.to_string())

    # -- 4. script / code-mixing ---------------------------------------------
    section("4. SCRIPT & CODE-MIXING PROFILE (% of rows)")
    for key, d in data.items():
        if d["tcol"] is None:
            continue
        s = texts(d["df"], d["tcol"])
        n = len(s)
        rows = {
            "native Kannada script":    s.str.contains(RE_KANNADA).sum(),
            "Devanagari script":        s.str.contains(RE_DEVANAG).sum(),
            "Latin/Roman (Kanglish)":   s.str.contains(RE_LATIN).sum(),
            "true-unicode emoji":       s.str.contains(RE_EMOJI).sum(),
            "mojibake Kannada":         s.str.contains(RE_MOJI_KN).sum(),
            "mojibake emoji/quotes":    s.str.contains(RE_MOJI_EMO).sum(),
        }
        print(f"\n{SUB}\n[{key}]  (n={n})")
        for k, v in rows.items():
            print(f"   {k:<24} {int(v):>6}  ({100.0 * v / n:5.2f}%)")
        if d["lcol"]:
            lab = label_series(d)
            print("   per-label ->  %native-Kannada | %mojibake-KN | %emoji")
            for l in lab.value_counts().index:
                m = (lab == l).values
                kn = 100.0 * s[m].str.contains(RE_KANNADA).mean()
                mj = 100.0 * s[m].str.contains(RE_MOJI_KN).mean()
                em = 100.0 * s[m].str.contains(RE_EMOJI).mean()
                print(f"      {l:<14} {kn:6.2f}%  {mj:6.2f}%  {em:6.2f}%")

    # -- 5. artefacts ----------------------------------------------------------
    section("5. NOISE / ARTEFACT SCAN (% of rows containing pattern)")
    for key, d in data.items():
        if d["tcol"] is None:
            continue
        s = texts(d["df"], d["tcol"])
        n = len(s)
        print(f"\n{SUB}\n[{key}]  (n={n})")
        for name, pat in ARTEFACTS.items():
            cnt = int(s.str.contains(pat, regex=True).sum())
            print(f"   {name:<28} {cnt:>6}  ({100.0 * cnt / n:5.2f}%)")

    # -- 6. vocabulary + lexical peek ------------------------------------------
    section("6. VOCABULARY INFORMALITY & TOP TOKENS PER LABEL")
    for key in ("A_train", "B_train"):
        d = data.get(key)
        if d is None or d["lcol"] is None or d["tcol"] is None:
            continue
        lab = label_series(d)
        s = texts(d["df"], d["tcol"])
        allc = Counter()
        for x in s:
            allc.update(t.lower() for t in RE_TOKEN.findall(x))
        n_tok = sum(allc.values())
        V = len(allc)
        hapax = sum(1 for v in allc.values() if v == 1)
        print(f"\n{SUB}\n[{key}]")
        print(f"   total tokens={n_tok}  vocab={V}  hapax%={100.0 * hapax / V:.1f}"
              f"  type-token-ratio={V / max(n_tok, 1):.3f}")
        print(f"   (high hapax% / low TTR => heavy informal spelling variation,")
        print(f"    i.e., transliteration variants of the same Kannada words)")
        for l in lab.value_counts().index:
            m = (lab == l).values
            c = Counter()
            for x in s[m]:
                c.update(t.lower() for t in RE_TOKEN.findall(x))
            top = ", ".join(f"{w}:{ct}" for w, ct in c.most_common(15))
            print(f"   {l:<14} -> {top}")

    # -- 7. cross-file overlap --------------------------------------------------
    section("7. CROSS-FILE OVERLAP DIAGNOSTICS (report-only; NO label transfer)")
    sets = {}
    for key, d in data.items():
        if d["tcol"] is None:
            continue
        df = d["df"]
        sets[key] = dict(
            ids=set(df["id"].astype(str)) if "id" in df.columns else set(),
            txt=set(norm_texts(df, d["tcol"])),
            n=len(df),
        )

    print("\n(a) unique normalised texts per file")
    for key, v in sets.items():
        print(f"   {key:<8} rows={v['n']:>5}  unique_texts={len(v['txt']):>5}")

    print("\n(b) pairwise overlaps (shared ids / shared normalised texts)")
    pairs = [("A_train", "A_val"), ("B_train", "B_val"),
             ("A_train", "B_train"), ("A_train", "B_val"),
             ("B_train", "A_val"), ("A_val", "B_val")]
    for a, b in pairs:
        if a not in sets or b not in sets:
            continue
        print(f"   {a} vs {b}: ids={len(sets[a]['ids'] & sets[b]['ids']):>5}"
              f"  texts={len(sets[a]['txt'] & sets[b]['txt']):>5}")

    print("\n(c) train comments that also appear in a validation file")
    print("    (snippet only; train labels deliberately withheld)")
    for tr, va in [("A_train", "A_val"), ("B_train", "A_val"),
                   ("A_train", "B_val"), ("B_train", "B_val")]:
        if tr not in sets or va not in sets:
            continue
        common = sorted(sets[tr]["txt"] & sets[va]["txt"])
        print(f"   {tr} & {va}: {len(common)} shared texts")
        for t in common[:3]:
            print(f"       - {t[:70]}")

    print("\n(d) task-structure check: crosstab of binary Label x Hate Category")
    print("    for texts shared between A_train and B_train (train labels only)")
    a, b = data.get("A_train"), data.get("B_train")
    if a and b and a["tcol"] and b["tcol"]:
        ta = pd.DataFrame({"txt": norm_texts(a["df"], a["tcol"]),
                           "bin": label_series(a)})
        tb = pd.DataFrame({"txt": norm_texts(b["df"], b["tcol"]),
                           "cat": label_series(b)})
        m = ta.merge(tb, on="txt")
        if len(m):
            print(f"   shared (text-pair) rows: {len(m)}")
            print(pd.crosstab(m["bin"], m["cat"]).to_string())
        else:
            print("   no shared texts between A_train and B_train")

    print(f"\n{BAR}\nPHASE 1 COMPLETE -- paste this ENTIRE output back.")
    print("Next: interpret results, lock the preprocessing spec, build Phase 2 baselines.")
    print(BAR)


if __name__ == "__main__":
    main()