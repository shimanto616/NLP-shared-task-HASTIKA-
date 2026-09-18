#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HASTIKA shared task -- PHASE 2: TF-IDF linear baselines.

Task A: binary (Hate/Non-Hate)        binary_train.csv        -> binary_validation_inputs.csv
Task B: 6-way hate category           multiclass_train.csv    -> multiclass_validation_inputs.csv

Steps:
  1. Apply preprocessing spec v1.0.
  2. CV: StratifiedGroupKFold(5), grouped by normalised text (no duplicate leakage).
     Configs: word-TFIDF, char-TFIDF, word+char union, each with LR; union + LinearSVC;
     class-balanced variants for Task B.
  3. Report macro-F1 (primary) + accuracy + per-class F1.
  4. Refit best config on FULL train, predict blind inputs, write submission-format CSVs:
       outputs/taskA_predictions_baseline.csv  (id,label in {Hate, Non-Hate})
       outputs/taskB_predictions_baseline.csv  (id,label in 6 categories)
  5. Dump everything to outputs/phase2_metrics.json (paper table source).

Run  : python phase2_baselines.py
Deps : pip install scikit-learn     (pandas assumed present)
Time : ~2-5 minutes on CPU.
"""

import html
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

try:
    import pandas as pd
except ImportError:
    sys.exit("[FATAL] pandas missing -> pip install pandas")

try:
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score, accuracy_score, classification_report
except ImportError:
    sys.exit("[FATAL] scikit-learn missing -> pip install scikit-learn")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEARCH_DIRS = ["data", ".", "../data"]
OUT_DIR = "outputs"
SEED = 42
N_SPLITS = 5

TASKS = {
    "A": dict(train="binary_train.csv", inputs="binary_validation_inputs.csv",
              label="Label", text="Comment",
              allowed=["Hate", "Non-Hate"], min_df=(2, 3)),
    "B": dict(train="multiclass_train.csv", inputs="multiclass_validation_inputs.csv",
              label="Hate Category", text="Comment",
              allowed=["Gender", "Political", "Religion", "Geo-political",
                       "Violence", "Others"], min_df=(2, 2)),
}

BAR = "=" * 78
SUB = "-" * 78

# ---------------------------------------------------------------------------
# PREPROCESSING SPEC v1.0
# ---------------------------------------------------------------------------
MOJI_HINT = re.compile("\u00f0\u0178|\u00e0\u00b2|\u00e0\u00b3|\u00e2\u20ac|\u00ef\u00b8")
RE_BR  = re.compile(r"<br\s*/?>", re.IGNORECASE)
RE_TAG = re.compile(r"<[^>]+>")
RE_WS  = re.compile(r"\s+")

def fix_mojibake(s: str) -> str:
    """Guarded cp1252->utf-8 roundtrip; restores mojibake Kannada/emoji."""
    if not MOJI_HINT.search(s):
        return s
    for enc in ("cp1252", "latin-1"):
        try:
            return s.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s

def clean_text(s: str) -> str:
    s = fix_mojibake(s)
    s = RE_BR.sub(" ", s)
    s = RE_TAG.sub(" ", s)
    s = html.unescape(s)
    s = RE_WS.sub(" ", s).strip()
    return s

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def resolve(fname):
    for d in SEARCH_DIRS:
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            return p
    return None

def load_csv(fname):
    p = resolve(fname)
    if p is None:
        sys.exit(f"[FATAL] {fname} not found (searched {SEARCH_DIRS})")
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    sys.exit(f"[FATAL] could not decode {fname}")

def find_col(df, target):
    for c in df.columns:
        if str(c).strip().lower() == str(target).strip().lower():
            return c
    return None

def feat_word(mdf):
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                           min_df=mdf, sublinear_tf=True)

def feat_char(mdf):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           min_df=mdf, sublinear_tf=True)

def feat_wordchar(mdf_w, mdf_c):
    return FeatureUnion([("w", feat_word(mdf_w)), ("c", feat_char(mdf_c))])

def build_configs(task):
    mdf_w, mdf_c = TASKS[task]["min_df"]
    lr   = lambda **kw: LogisticRegression(max_iter=4000, C=1.0,
                                           random_state=SEED, **kw)
    svc  = lambda **kw: LinearSVC(C=1.0, dual="auto", max_iter=8000,
                                  random_state=SEED, **kw)
    cfgs = [
        ("LR_wordonly",     feat_word(mdf_w),          lr()),
        ("LR_charonly",     feat_char(mdf_c),          lr()),
        ("LR_word+char",    feat_wordchar(mdf_w, mdf_c), lr()),
        ("SVC_word+char",   feat_wordchar(mdf_w, mdf_c), svc()),
    ]
    if task == "B":
        cfgs += [
            ("LR_word+char_bal",  feat_wordchar(mdf_w, mdf_c), lr(class_weight="balanced")),
            ("SVC_word+char_bal", feat_wordchar(mdf_w, mdf_c), svc(class_weight="balanced")),
        ]
    return cfgs

# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------
def run_cv(task, df, tcol, lcol, configs, allowed):
    X = df[tcol].fillna("").astype(str).map(clean_text)
    y = df[lcol].astype(str).str.strip().to_numpy()
    groups = X.str.lower().to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(cv.split(X, y, groups))

    results = {}
    for name, feat, clf in configs:
        t0 = time.time()
        f1s, accs, y_true_all, y_pred_all = [], [], [], []
        for tr, te in folds:
            pipe = Pipeline([("feat", feat), ("clf", clf)])
            pipe.fit(X.iloc[tr], y[tr])
            pred = pipe.predict(X.iloc[te])
            f1s.append(f1_score(y[te], pred, average="macro"))
            accs.append(accuracy_score(y[te], pred))
            y_true_all.extend(y[te])
            y_pred_all.extend(pred)
        results[name] = dict(
            macro_f1_mean=float(np.mean(f1s)), macro_f1_std=float(np.std(f1s)),
            macro_f1_folds=[round(f, 4) for f in f1s],
            accuracy_mean=float(np.mean(accs)), accuracy_std=float(np.std(accs)),
            secs=round(time.time() - t0, 1),
            per_class=classification_report(y_true_all, y_pred_all,
                                            labels=allowed,
                                            output_dict=True, zero_division=0),
        )
    return results, (X, y)

def fit_full_and_predict(task, X, y, inputs_df, itcol, best_cfg):
    name, feat, clf = best_cfg
    Xte = inputs_df[itcol].fillna("").astype(str).map(clean_text)
    pipe = Pipeline([("feat", feat), ("clf", clf)])
    pipe.fit(X, y)
    preds = pipe.predict(Xte)

    allowed = set(TASKS[task]["allowed"])
    bad = sorted(set(preds) - allowed)
    if bad:
        sys.exit(f"[FATAL] model emitted unexpected labels: {bad}")

    out = pd.DataFrame({"id": inputs_df["id"].to_numpy(), "label": preds})
    path = os.path.join(OUT_DIR, f"task{task}_predictions_baseline.csv")
    out.to_csv(path, index=False)
    return path, preds

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(BAR)
    print("PHASE 2 -- TF-IDF linear baselines (HASTIKA)")
    print(BAR)
    print(f"sklearn {sklearn.__version__} | pandas {pd.__version__} | "
          f"numpy {np.__version__} | seed {SEED} | {N_SPLITS}-fold grouped CV")

    # preprocessing self-check
    print("\n[spec v1.0 self-check]")
    print("   mojibake emoji  :", repr(fix_mojibake("\u00f0\u0178\u02dc\u201a")))
    print("   mojibake apost. :", repr(fix_mojibake("Kohli\u00e2\u20ac\u2122s")))
    print("   html clean      :", repr(clean_text("a&lt;b&gt;c &quot;d&quot; <br> e")))

    metrics = dict(
        phase=2, timestamp=time.strftime("%Y-%m-%d %H:%M"),
        env=dict(sklearn=sklearn.__version__, pandas=pd.__version__,
                 numpy=np.__version__, seed=SEED, n_splits=N_SPLITS),
        preprocessing="spec v1.0: mojibake repair; <br>/tag strip; entity "
                      "unescape; ws collapse; no stopword removal/stemming",
        cv_protocol="StratifiedGroupKFold grouped by normalised cleaned text",
    )

    for task in ("A", "B"):
        cfg = TASKS[task]
        print(f"\n{BAR}\nTASK {task}: {cfg['train']} -> {cfg['inputs']}\n{BAR}")

        train_df  = load_csv(cfg["train"])
        inputs_df = load_csv(cfg["inputs"])
        tcol, lcol = find_col(train_df, cfg["text"]), find_col(train_df, cfg["label"])
        itcol = find_col(inputs_df, cfg["text"])
        if None in (tcol, lcol, itcol):
            sys.exit(f"[FATAL] TASK {task}: missing expected columns")
        print(f"train rows={len(train_df)}  blind-input rows={len(inputs_df)}")

        configs = build_configs(task)
        results, (X, y) = run_cv(task, train_df, tcol, lcol, configs, cfg["allowed"])

        print(f"\n{SUB}\nCV results (mean over {N_SPLITS} folds; selection = macro-F1)")
        print(f"{'config':<18}{'macro-F1':>16}{'accuracy':>16}{'sec':>7}")
        for name, r in sorted(results.items(), key=lambda kv: -kv[1]["macro_f1_mean"]):
            print(f"{name:<18}{r['macro_f1_mean']:>10.4f} ±{r['macro_f1_std']:.4f}"
                  f"{r['accuracy_mean']:>10.4f} ±{r['accuracy_std']:.4f}"
                  f"{r['secs']:>7.1f}")

        best_name = max(results, key=lambda k: results[k]["macro_f1_mean"])
        best_cfg = next(c for c in configs if c[0] == best_name)
        print(f"\n{SUB}\nBEST: {best_name} -- per-class F1 (pooled over folds)")
        print(classification_report(
            *[], labels=None, output_dict=False)) if False else None
        # pretty print per-class from stored dict
        pc = results[best_name]["per_class"]
        hdr = f"{'class':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
        print(hdr)
        for lbl in cfg["allowed"]:
            d = pc.get(lbl, {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
            print(f"{lbl:<16}{d['precision']:>10.3f}{d['recall']:>10.3f}"
                  f"{d['f1-score']:>10.3f}{int(d['support']):>10d}")

        # refit on full train -> predict blind inputs
        path, preds = fit_full_and_predict(task, X, y, inputs_df, itcol, best_cfg)
        n = len(preds)
        dist = Counter(preds)
        prior = Counter(y)
        print(f"\n{SUB}\nBlind-input predictions written: {path}  ({n} rows)")
        print(f"{'class':<16}{'pred %':>9}{'train prior %':>16}")
        for lbl in cfg["allowed"]:
            print(f"{lbl:<16}{100 * dist.get(lbl, 0) / n:>8.2f}%"
                  f"{100 * prior.get(lbl, 0) / len(y):>15.2f}%")
        print("   (prediction mix should roughly track the train prior; a big")
        print("    deviation for a small class = over/under-prediction warning)")

        metrics[f"task{task}"] = dict(
            n_train=len(train_df), n_blind=len(inputs_df),
            configs=results, best=best_name,
            prediction_distribution={k: int(v) for k, v in dist.items()},
            submission_file=path,
        )

    out_json = os.path.join(OUT_DIR, "phase2_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n{BAR}\nPHASE 2 COMPLETE.")
    print(f"Metrics JSON        : {out_json}")
    print(f"Safety-net submissions (exact Phase-4 format):")
    print(f"  {OUT_DIR}/taskA_predictions_baseline.csv  -> id,label  (Hate/Non-Hate)")
    print(f"  {OUT_DIR}/taskB_predictions_baseline.csv  -> id,label  (6 categories)")
    print("Paste the ENTIRE console output back. Next: Phase 3 transformers.")
    print(BAR)

if __name__ == "__main__":
    main()