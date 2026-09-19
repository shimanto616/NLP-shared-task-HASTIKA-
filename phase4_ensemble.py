#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HASTIKA shared task -- PHASE 4: ensemble evaluation + final submission files.

Stage eval (CPU only, ~3-6 min):
    python phase4_ensemble.py --stage eval
  - Rebuilds the exact Phase-3 grouped folds (verified against stored indices)
  - Computes TF-IDF OOF probabilities for the Phase-2 winning config (same folds)
  - Loads transformer OOF probs from outputs/phase3_task{A,B}_{muril,xlmr}.json
  - Grid-searches blend weights (probability space, macro-F1)
  - Parsimony rule: simpler candidate within 0.003 macro-F1 of grid-best wins
  - Writes outputs/phase4_eval.json and prints required refit commands

Stage finalize (CPU; needs refit artifacts if transformers are selected):
    python phase4_ensemble.py --stage finalize --task A
    python phase4_ensemble.py --stage finalize --task B
  - Refits TF-IDF on full train (in-script); reads outputs/taskX_probs_{model}.csv
  - Blends with selected weights; writes + validates:
      outputs/taskA_predictions_final.csv  (id,label in {Hate, Non-Hate})
      outputs/taskB_predictions_final.csv  (id,label in 6 categories)
"""

import argparse
import html
import json
import os
import re
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

try:
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score, accuracy_score, classification_report
except ImportError as e:
    sys.exit(f"[FATAL] missing dependency: {e}")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEARCH_DIRS = ["data", ".", "../data"]
OUT_DIR = "outputs"
SEED = 42
N_SPLITS = 5
PARSIMONY_MARGIN = 0.003   # simpler candidate wins if within this of grid-best

TASKS = {
    "A": dict(train="binary_train.csv", inputs="binary_validation_inputs.csv",
              label="Label", text="Comment",
              classes=["Hate", "Non-Hate"], min_df=(2, 3)),
    "B": dict(train="multiclass_train.csv", inputs="multiclass_validation_inputs.csv",
              label="Hate Category", text="Comment",
              classes=["Gender", "Political", "Religion", "Geo-political",
                       "Violence", "Others"], min_df=(2, 2)),
}

BAR = "=" * 78
SUB = "-" * 78

# ---------------------------------------------------------------------------
# PREPROCESSING SPEC v1.0 (identical to Phases 2/3)
# ---------------------------------------------------------------------------
MOJI_HINT = re.compile("\u00f0\u0178|\u00e0\u00b2|\u00e0\u00b3|\u00e2\u20ac|\u00ef\u00b8")
RE_BR  = re.compile(r"<br\s*/?>", re.IGNORECASE)
RE_TAG = re.compile(r"<[^>]+>")
RE_WS  = re.compile(r"\s+")

def fix_mojibake(s: str) -> str:
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
    return RE_WS.sub(" ", s).strip()

# ---------------------------------------------------------------------------
# IO helpers
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

# ---------------------------------------------------------------------------
# Phase-2 pipeline replication (exact hyperparameters)
# ---------------------------------------------------------------------------
def build_phase2_pipeline(task, config_name):
    mdf_w, mdf_c = TASKS[task]["min_df"]
    def word():
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                               min_df=mdf_w, sublinear_tf=True)
    def char():
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               min_df=mdf_c, sublinear_tf=True)
    def union():
        return FeatureUnion([("w", word()), ("c", char())])
    def lr(**kw):
        return LogisticRegression(max_iter=4000, C=1.0, random_state=SEED, **kw)
    def svc(**kw):
        return LinearSVC(C=1.0, dual="auto", max_iter=8000,
                         random_state=SEED, **kw)
    if config_name == "LR_charonly":
        return Pipeline([("feat", char()), ("clf", lr())])
    if config_name == "LR_wordonly":
        return Pipeline([("feat", word()), ("clf", lr())])
    if config_name == "LR_word+char":
        return Pipeline([("feat", union()), ("clf", lr())])
    if config_name == "LR_word+char_bal":
        return Pipeline([("feat", union()), ("clf", lr(class_weight="balanced"))])
    if config_name == "SVC_word+char":
        return Pipeline([("feat", union()), ("clf", svc())])
    if config_name == "SVC_word+char_bal":
        return Pipeline([("feat", union()), ("clf", svc(class_weight="balanced"))])
    sys.exit(f"[FATAL] unknown phase-2 config: {config_name}")

def proba_of(pipe, X):
    """predict_proba if available, else softmax(decision_function)."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)
    z = pipe.decision_function(X)
    if z.ndim == 1:
        z = np.vstack([-z, z]).T
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# ---------------------------------------------------------------------------
# Shared data prep + fold machinery
# ---------------------------------------------------------------------------
def prepare_train(task):
    cfg = TASKS[task]
    df = load_csv(cfg["train"])
    tcol, lcol = find_col(df, cfg["text"]), find_col(df, cfg["label"])
    X = df[tcol].fillna("").astype(str).map(clean_text).to_numpy()
    y_raw = df[lcol].astype(str).str.strip().to_numpy()
    labels = sorted(set(y_raw))
    y_ids = np.array([labels.index(v) for v in y_raw])
    groups = np.array([t.lower() for t in X])
    return X, y_raw, labels, y_ids, groups

def rebuild_folds(X, y_ids, groups):
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(cv.split(X, y_ids, groups))

def verify_folds(folds, stored, src):
    if not stored:
        sys.exit(f"[FATAL] {src} has no fold_test_indices -- rerun phase 3 CV")
    for i, (_, te) in enumerate(folds):
        if set(int(v) for v in te) != set(int(v) for v in stored[i]):
            sys.exit(f"[FATAL] fold {i+1} mismatch vs {src} -- "
                     f"training data or order changed since phase 3")

def tfidf_oof(task, X, y_raw, labels, folds, config_name):
    oof = np.zeros((len(X), len(labels)), dtype=np.float64)
    f1s = []
    for tr, te in folds:
        pipe = build_phase2_pipeline(task, config_name)
        pipe.fit(X[tr], y_raw[tr])
        P = proba_of(pipe, X[te])
        classes = list(pipe.named_steps["clf"].classes_)
        if classes != labels:                      # belt-and-braces column order
            P = P[:, [classes.index(l) for l in labels]]
        oof[te] = P
        pred = [labels[i] for i in P.argmax(1)]
        f1s.append(f1_score(y_raw[te], pred, average="macro"))
    return oof, float(np.mean(f1s))

# ---------------------------------------------------------------------------
# Stage: eval
# ---------------------------------------------------------------------------
def eval_task(task):
    cfg = TASKS[task]
    print(f"\n{BAR}\nPHASE 4 EVAL | task {task}\n{BAR}")
    X, y_raw, labels, y_ids, groups = prepare_train(task)
    folds = rebuild_folds(X, y_ids, groups)
    print(f"n={len(X)}  classes={labels}  folds rebuilt ({N_SPLITS}-fold grouped)")

    # -- transformer OOF probabilities --------------------------------------
    trans = {}
    for mk in ("muril", "xlmr"):
        p = os.path.join(OUT_DIR, f"phase3_task{task}_{mk}.json")
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        verify_folds(folds, rec.get("fold_test_indices"), p)
        if list(rec.get("oof_y", [])) != [int(v) for v in y_ids]:
            sys.exit(f"[FATAL] oof_y row-order mismatch in {p}")
        trans[mk] = dict(P=np.array(rec["oof_probs"], dtype=np.float64),
                         cv_f1=float(rec["macro_f1_mean"]),
                         max_epochs=rec["hyperparams"]["max_epochs"], src=p)
        print(f"  OOF loaded: {mk:<6} CV macro-F1={rec['macro_f1_mean']:.4f} "
              f"(max_epochs={rec['hyperparams']['max_epochs']})")
    if "muril" not in trans:
        sys.exit("[FATAL] no muril OOF for this task -- run phase 3 CV first")

    # -- TF-IDF OOF probabilities (phase-2 winning config, same folds) -------
    with open(os.path.join(OUT_DIR, "phase2_metrics.json"), encoding="utf-8") as f:
        p2 = json.load(f)
    best_cfg = p2[f"task{task}"]["best"]
    p2_f1 = p2[f"task{task}"]["configs"][best_cfg]["macro_f1_mean"]
    P_tf, tf_foldmean = tfidf_oof(task, X, y_raw, labels, folds, best_cfg)
    tf_pooled = float(f1_score(y_ids, P_tf.argmax(1), average="macro"))
    flag = "" if abs(tf_foldmean - p2_f1) < 0.005 else "  [WARN] deviates from phase 2!"
    print(f"  TF-IDF ({best_cfg}) recomputed: fold-mean={tf_foldmean:.4f} "
          f"pooled={tf_pooled:.4f} (phase-2 recorded {p2_f1:.4f}){flag}")

    # -- candidates -----------------------------------------------------------
    cands = []
    def add(name, W, f1v, simplicity):
        cands.append(dict(name=name, weights=W, f1=float(f1v),
                          simplicity=simplicity))

    add("tfidf", {"tfidf": 1.0}, tf_pooled, 0)
    for mk, d in trans.items():
        f1v = f1_score(y_ids, d["P"].argmax(1), average="macro")
        add(mk, {mk: 1.0}, f1v, 1)

    grid_tables = {}
    for mk, d in trans.items():
        rows = []
        for a in np.arange(0.0, 1.0001, 0.05):
            a = round(float(a), 2)
            Pm = a * P_tf + (1.0 - a) * d["P"]
            pred = Pm.argmax(1)
            rows.append([a, float(f1_score(y_ids, pred, average="macro")),
                         float(accuracy_score(y_ids, pred))])
        grid_tables[mk] = rows
        best_a, best_f1, _ = max(rows, key=lambda r: r[1])
        add(f"tfidf*{best_a:.2f}+{mk}*{1-best_a:.2f}",
            {"tfidf": best_a, mk: round(1 - best_a, 2)}, best_f1, 3)
        half = next(r for r in rows if r[0] == 0.5)
        add(f"tfidf*0.5+{mk}*0.5", {"tfidf": 0.5, mk: 0.5}, half[1], 2)

    if "muril" in trans and "xlmr" in trans:
        for ws, nm in [((1/3, 1/3, 1/3), "uniform3(t,m,x)"),
                       ((0.5, 0.25, 0.25), "t0.5+m0.25+x0.25"),
                       ((0.25, 0.5, 0.25), "t0.25+m0.5+x0.25")]:
            Pm = (ws[0] * P_tf + ws[1] * trans["muril"]["P"]
                  + ws[2] * trans["xlmr"]["P"])
            add(nm, {"tfidf": round(ws[0], 3), "muril": round(ws[1], 3),
                     "xlmr": round(ws[2], 3)},
                f1_score(y_ids, Pm.argmax(1), average="macro"), 4)

    # -- selection (xlmr excluded for task B: documented fold-1 collapse) -----
    pool = [c for c in cands if task == "A" or "xlmr" not in c["weights"]]
    top = max(pool, key=lambda c: c["f1"])
    best, parsimony = top, False
    for c in sorted(pool, key=lambda c: (c["simplicity"], -c["f1"])):
        if top["f1"] - c["f1"] <= PARSIMONY_MARGIN:
            best, parsimony = c, (c["name"] != top["name"])
            break

    print(f"\n{SUB}\n  candidates (OOF pooled macro-F1):")
    print(f"  {'name':<32}{'macro-F1':>10}{'simple':>8}")
    for c in sorted(cands, key=lambda c: -c["f1"]):
        mark = "  <- selected" if c["name"] == best["name"] else ""
        print(f"  {c['name']:<32}{c['f1']:>10.4f}{c['simplicity']:>8}{mark}")
    for mk, rows in grid_tables.items():
        print(f"\n  blend grid: P = a*tfidf + (1-a)*{mk}")
        print(f"  {'a':>6}{'macro-F1':>11}{'accuracy':>11}")
        for a, f1v, acc in rows:
            print(f"  {a:>6.2f}{f1v:>11.4f}{acc:>11.4f}")
    print(f"\n  SELECTED: {best['name']}  weights={best['weights']}  "
          f"oof_macro_f1={best['f1']:.4f}")
    if parsimony:
        print(f"  (parsimony rule: within {PARSIMONY_MARGIN} of grid-best "
              f"{top['f1']:.4f}; simpler model preferred)")
    print("  (note: weight selected on OOF -> mild optimism; reported as such)")

    P_sel = np.zeros_like(P_tf)
    for comp, w in best["weights"].items():
        P_sel += w * (P_tf if comp == "tfidf" else trans[comp]["P"])
    print(f"\n  per-class at selected ({best['name']}):")
    rep = classification_report(y_ids, P_sel.argmax(1), target_names=labels,
                                digits=3, zero_division=0)
    print("  " + rep.replace("\n", "\n  "))

    return dict(labels=labels,
                tfidf=dict(config=best_cfg, oof_fold_mean=tf_foldmean,
                           oof_pooled=tf_pooled, phase2_recorded=p2_f1),
                transformers={mk: dict(cv_macro_f1_mean=d["cv_f1"],
                                       max_epochs=d["max_epochs"])
                              for mk, d in trans.items()},
                candidates=cands, grids=grid_tables,
                selected=dict(name=best["name"], weights=best["weights"],
                              oof_macro_f1=best["f1"], parsimony_applied=parsimony))

# ---------------------------------------------------------------------------
# Stage: finalize
# ---------------------------------------------------------------------------
def finalize_task(task):
    cfg = TASKS[task]
    ev_path = os.path.join(OUT_DIR, "phase4_eval.json")
    if not os.path.isfile(ev_path):
        sys.exit("[FATAL] run `python phase4_ensemble.py --stage eval` first")
    with open(ev_path, encoding="utf-8") as f:
        ev = json.load(f)
    sel = ev[f"task{task}"]["selected"]
    W = sel["weights"]
    print(f"{BAR}\nPHASE 4 FINALIZE | task {task} | {sel['name']} | weights={W}\n{BAR}")

    X, y_raw, labels, _, _ = prepare_train(task)
    inputs_df = load_csv(cfg["inputs"])
    itcol = find_col(inputs_df, cfg["text"])
    ids = inputs_df["id"].to_numpy()
    Xte = inputs_df[itcol].fillna("").astype(str).map(clean_text).tolist()

    with open(os.path.join(OUT_DIR, "phase2_metrics.json"), encoding="utf-8") as f:
        best_cfg = json.load(f)[f"task{task}"]["best"]

    P = np.zeros((len(Xte), len(labels)), dtype=np.float64)
    for comp, w in W.items():
        if comp == "tfidf":
            print(f"  [tfidf] full-train refit ({best_cfg}) ...", flush=True)
            t0 = time.time()
            pipe = build_phase2_pipeline(task, best_cfg)
            pipe.fit(X, y_raw)
            Pc = proba_of(pipe, Xte)
            classes = list(pipe.named_steps["clf"].classes_)
            if classes != labels:
                Pc = Pc[:, [classes.index(l) for l in labels]]
            print(f"  [tfidf] done in {time.time()-t0:.1f}s")
        else:
            f = os.path.join(OUT_DIR, f"task{task}_probs_{comp}.csv")
            if not os.path.isfile(f):
                sys.exit(f"[FATAL] {f} missing. Run first:\n"
                         f"    python phase3_finetune.py --mode refit --task {task} --model {comp}")
            dfp = pd.read_csv(f)
            if set(dfp["id"]) != set(ids):
                sys.exit(f"[FATAL] id mismatch between {f} and the inputs file")
            dfp = dfp.set_index("id").loc[list(ids)]
            miss = [l for l in labels if l not in dfp.columns]
            if miss:
                sys.exit(f"[FATAL] {f} missing probability columns: {miss}")
            Pc = dfp[labels].to_numpy(dtype=np.float64)
            print(f"  [{comp}] blind probs loaded from {f}")
        P += w * Pc

    preds = [labels[int(i)] for i in P.argmax(1)]

    # ---- submission sanity checks (competition spec) ----
    allowed = set(cfg["classes"])
    out_csv = os.path.join(OUT_DIR, f"task{task}_predictions_final.csv")
    pd.DataFrame({"id": ids, "label": preds}).to_csv(out_csv, index=False)
    with open(out_csv, encoding="utf-8") as f:
        header = f.readline().strip()
    back = pd.read_csv(out_csv)
    checks = [
        ("row count matches inputs", len(preds) == len(inputs_df)),
        ("all labels in allowed set", set(preds) <= allowed),
        ("no empty predictions", all(isinstance(p, str) and p for p in preds)),
        ("header is 'id,label'", header == "id,label"),
        ("ids identical to inputs order", list(back["id"]) == list(ids)),
        ("no null cells", not back.isna().any().any()),
    ]
    ok = all(v for _, v in checks)
    for name, v in checks:
        print(f"  [{'OK' if v else 'FAIL'}] {name}")
    if not ok:
        sys.exit("[FATAL] submission checks failed -- DO NOT submit")

    dist = {c: int(sum(1 for p in preds if p == c)) for c in cfg["classes"]}
    prior = {c: float(np.mean(y_raw == c)) for c in cfg["classes"]}
    print(f"\n  written: {out_csv}")
    print(f"  {'class':<16}{'pred %':>9}{'train prior %':>16}")
    for c in cfg["classes"]:
        print(f"  {c:<16}{100*dist[c]/len(preds):>8.2f}%{100*prior[c]:>15.2f}%")

    audit = dict(task=task, selected=sel, weights=W, submission_file=out_csv,
                 row_count=len(preds), prediction_distribution=dist,
                 checks={k: v for k, v in checks},
                 timestamp=time.strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join(OUT_DIR, f"phase4_final_task{task}.json"), "w",
              encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["eval", "finalize"])
    ap.add_argument("--task", choices=["A", "B"])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.stage == "eval":
        out = {}
        for t in ("A", "B"):
            out[f"task{t}"] = eval_task(t)
        path = os.path.join(OUT_DIR, "phase4_eval.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n{BAR}\nPHASE 4 EVAL COMPLETE -> {path}")
        print("Required refits before finalize (if any):")
        needed = False
        for t in ("A", "B"):
            for comp in out[f"task{t}"]["selected"]["weights"]:
                if comp != "tfidf":
                    needed = True
                    print(f"  python phase3_finetune.py --mode refit --task {t} --model {comp}")
        if not needed:
            print("  (none -- TF-IDF-only selections)")
        print("Then: python phase4_ensemble.py --stage finalize --task A / --task B")
        print(BAR)
    else:
        if not args.task:
            sys.exit("[FATAL] --task A|B required for finalize")
        finalize_task(args.task)

if __name__ == "__main__":
    main()