#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HASTIKA shared task -- PHASE 3: Transformer fine-tuning (MuRIL / XLM-R).

Usage (run in this order):
  python phase3_finetune.py --mode smoke  --task A --model muril   # 2-min pipeline check
  python phase3_finetune.py --mode cv    --task A --model muril    # 5-fold grouped CV
  python phase3_finetune.py --mode cv    --task A --model xlmr
  python phase3_finetune.py --mode cv    --task B --model muril
  python phase3_finetune.py --mode cv    --task B --model xlmr
  ... (we choose the winner, then:)
  python phase3_finetune.py --mode refit --task A --model muril    # full-train refit + blind preds
  python phase3_finetune.py --mode summary                          # all results incl. Phase 2
  python phase3_finetune.py --mode diag                             # report-only cross-file diagnostic

Deps : torch (XPU build), transformers, pandas, scikit-learn (all already present)
I/O  : outputs/phase3_task{A,B}_{model}.json, outputs/task{A,B}_predictions_{model}.csv,
       outputs/task{A,B}_probs_{model}.csv
"""

import argparse
import contextlib
import copy
import html
import json
import os
import random
import re
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import pandas as pd
    import torch
    import sklearn
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError as e:
    sys.exit(f"[FATAL] missing dependency: {e}")

try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:  # defensive fallback
    from torch.optim.lr_scheduler import LambdaLR
    def get_linear_schedule_with_warmup(opt, warmup, total):
        def fn(step):
            if step < warmup:
                return step / max(1, warmup)
            return max(0.0, (total - step) / max(1, total - warmup))
        return LambdaLR(opt, fn)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEARCH_DIRS = ["data", ".", "../data"]
OUT_DIR = "outputs"
SEED = 42
N_SPLITS = 5
MAX_LEN = 128

MODELS = {
    "muril":   "google/muril-base-cased",
    "xlmr":    "xlm-roberta-base",
    "indicb":  "ai4bharat/IndicBERTv2-MLM-only",   # optional third model
}

TASKS = {
    "A": dict(train="binary_train.csv", inputs="binary_validation_inputs.csv",
              label="Label", text="Comment",
              classes=["Hate", "Non-Hate"], weighted=False,
              max_epochs=5, patience=2),
    "B": dict(train="multiclass_train.csv", inputs="multiclass_validation_inputs.csv",
              label="Hate Category", text="Comment",
              classes=["Gender", "Political", "Religion", "Geo-political",
                       "Violence", "Others"], weighted=True,
              max_epochs=8, patience=3),
}

BAR = "=" * 78
SUB = "-" * 78

# ---------------------------------------------------------------------------
# PREPROCESSING SPEC v1.0 (identical to Phase 2, but case PRESERVED)
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
# Torch plumbing
# ---------------------------------------------------------------------------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def pick_device(pref="auto"):
    if pref == "cpu":
        return "cpu"
    try:
        if pref in ("auto", "xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def clear_dev_cache():
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def amp_supported(device):
    """Probe: can we run bf16 autocast on this device?"""
    try:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            a = torch.ones(8, 8, device=device)
            (a @ a).sum().item()
        return True
    except Exception:
        return False

def amp_ctx(device, enabled):
    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.autocast(device_type=device, dtype=torch.bfloat16)
    except Exception:
        return contextlib.nullcontext()

class TextDS(Dataset):
    def __init__(self, texts, y_ids):
        self.texts, self.y_ids = list(texts), list(y_ids)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        return self.texts[i], self.y_ids[i]

def make_collate(tok, max_len=None):
    ml = MAX_LEN if max_len is None else max_len
    def collate(batch):
        texts, ys = zip(*batch)
        enc = tok(list(texts), truncation=True, max_length=ml,
                  padding=True, return_tensors="pt")
        return enc, torch.tensor(ys, dtype=torch.long)
    return collate

def param_groups(model, wd):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "LayerNorm.bias")
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay if any(nd in n for nd in no_decay) else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": nodecay, "weight_decay": 0.0}]

def train_one_epoch(model, loader, opt, sched, criterion, device, use_amp,
                    tag="", log_every=50):
    model.train()
    tot, seen, n_batches = 0.0, 0, len(loader)
    for enc, y in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        with amp_ctx(device, use_amp):
            loss = criterion(model(**enc).logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        tot += loss.item(); seen += 1
        if log_every and seen % log_every == 0:
            print(f"    [{tag}] step {seen}/{n_batches} "
                  f"avg_loss={tot/seen:.4f}", flush=True)
    return tot / max(1, n_batches)

@torch.no_grad()
def predict_logits(model, loader, device, use_amp):
    model.eval()
    logits_all = []
    for enc, _ in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        with amp_ctx(device, use_amp):
            logits = model(**enc).logits
        logits_all.append(logits.float().cpu())
    return torch.cat(logits_all)

@torch.no_grad()
def predict_texts(model, tok, texts, bs, device, use_amp, max_len=None):
    ml = MAX_LEN if max_len is None else max_len
    model.eval()
    preds = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=ml,
                  padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with amp_ctx(device, use_amp):
            logits = model(**enc).logits
        preds.append(logits.float().cpu())
    return torch.cat(preds)

def build_model(name, n_labels, id2label, label2id, device):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=n_labels, id2label=id2label, label2id=label2id,
        low_cpu_mem_usage=True)
    model.to(device)
    where = next(model.parameters()).device
    assert str(where).startswith(str(device)), (
        f"[FATAL] model is on {where}, expected {device} -- "
        f".to(device) failed or was dropped")
    print(f"  model device check OK: all parameters on {where}")
    return tok, model

# ---------------------------------------------------------------------------
# Core: grouped CV for one (task, model)
# ---------------------------------------------------------------------------
def run_cv(args):
    cfg = TASKS[args.task]
    tcfg = dict(lr=2e-5, bs=args.bs, wd=0.01, warmup=0.1)
    print(BAR); print(f"PHASE 3 CV | task {args.task} | model {args.model} ({MODELS[args.model]})")
    print(BAR)

    train_df = load_csv(cfg["train"])
    tcol, lcol = find_col(train_df, cfg["text"]), find_col(train_df, cfg["label"])
    X = train_df[tcol].fillna("").astype(str).map(clean_text).to_numpy()
    y_raw = train_df[lcol].astype(str).str.strip().to_numpy()
    labels = sorted(set(y_raw))
    label2id = {c: i for i, c in enumerate(labels)}
    y = np.array([label2id[v] for v in y_raw])
    groups = np.array([t.lower() for t in X])
    n_cls = len(labels)

    device = pick_device(getattr(args, "device", "auto"))
    use_amp = amp_supported(device)
    print(f"device={device}  amp_bf16={use_amp}  n={len(X)}  classes={labels}")
    print(f"hp: lr={tcfg['lr']} bs={tcfg['bs']} wd={tcfg['wd']} "
          f"warmup={tcfg['warmup']} max_len={MAX_LEN} "
          f"max_epochs={cfg['max_epochs']} patience={cfg['patience']} "
          f"weighted_loss={cfg['weighted']}")
    print("(note: 'Loading weights' + LOAD REPORT will print once PER FOLD -- "
          "expected: the model is rebuilt fresh from the pretrained checkpoint "
          "every fold to prevent cross-fold weight carryover)")

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(cv.split(X, y, groups))

    fold_results, y_true_all, y_pred_all = [], [], []
    t_start = time.time()

    for k, (tr, te) in enumerate(folds):
        set_seed(SEED + k)
        t0 = time.time()

        # fresh pretrained model EVERY fold (a reused fine-tuned body would
        # have seen this fold's val samples in earlier folds' training -> biased CV)
        tok, model_ = build_model(MODELS[args.model], n_cls,
                                  {i: c for c, i in label2id.items()},
                                  {c: i for c, i in label2id.items()}, device)

        ytr = y[tr]
        if cfg["weighted"]:
            counts = np.bincount(ytr, minlength=n_cls).astype(np.float64)
            w = counts.sum() / (n_cls * np.maximum(counts, 1.0))
            criterion = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(w, dtype=torch.float32, device=device))
        else:
            criterion = torch.nn.CrossEntropyLoss()

        g = torch.Generator(); g.manual_seed(SEED + k)
        train_loader = DataLoader(TextDS(X[tr], ytr), batch_size=tcfg["bs"],
                                  shuffle=True, generator=g, num_workers=0,
                                  collate_fn=make_collate(tok))
        val_loader = DataLoader(TextDS(X[te], y[te]), batch_size=tcfg["bs"] * 2,
                                shuffle=False, num_workers=0,
                                collate_fn=make_collate(tok))

        epochs = 1 if args.mode == "smoke" else cfg["max_epochs"]
        opt = torch.optim.AdamW(param_groups(model_, tcfg["wd"]), lr=tcfg["lr"])
        total_steps = max(1, len(train_loader) * epochs)
        sched = get_linear_schedule_with_warmup(
            opt, int(tcfg["warmup"] * total_steps), total_steps)

        best = dict(f1=-1.0, epoch=-1, yt=None, yp=None)
        bad = 0
        for ep in range(1, epochs + 1):
            tr_loss = train_one_epoch(model_, train_loader, opt, sched, criterion,
                                      device, use_amp, tag=f"fold{k+1}")
            logits = predict_logits(model_, val_loader, device, use_amp)
            yp = logits.argmax(-1).numpy(); yt = y[te]
            f1 = f1_score(yt, yp, average="macro")
            acc = accuracy_score(yt, yp)
            star = ""
            if f1 > best["f1"]:
                best.update(f1=f1, epoch=ep, yt=yt, yp=yp)
                bad, star = 0, "  <-- best"
            else:
                bad += 1
            print(f"  fold {k+1}/{N_SPLITS} epoch {ep}: "
                  f"loss={tr_loss:.4f}  val_macroF1={f1:.4f}  val_acc={acc:.4f}"
                  f"{star}", flush=True)
            if args.mode != "smoke" and bad >= cfg["patience"]:
                print(f"  fold {k+1}: early stop (patience {cfg['patience']})")
                break
        fold_results.append(dict(fold=k + 1, best_epoch=best["epoch"],
                                 macro_f1=float(best["f1"]), secs=round(time.time() - t0, 1)))
        y_true_all.extend(best["yt"]); y_pred_all.extend(best["yp"])
        print(f"  fold {k+1} DONE: best macro-F1={best['f1']:.4f} "
              f"@epoch {best['epoch']}  ({fold_results[-1]['secs']}s)", flush=True)

        # free this fold's model before building the next one
        del model_, opt, sched, criterion, train_loader, val_loader
        clear_dev_cache()

        if args.mode == "smoke":
            break

    if args.mode == "smoke":
        print("\nSMOKE OK -- pipeline, device, AMP and model loading all work.")
        print("Now run the full CV commands.")
        return

    f1s = [r["macro_f1"] for r in fold_results]
    mean_ep = float(np.mean([r["best_epoch"] for r in fold_results]))
    pooled_f1 = f1_score(y_true_all, y_pred_all, average="macro")
    pooled_acc = accuracy_score(y_true_all, y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(n_cls)))

    print(f"\n{SUB}\nCV SUMMARY (task {args.task}, {args.model})")
    print(f"  macro-F1 : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}   "
          f"(folds: {[round(f,4) for f in f1s]})")
    print(f"  accuracy : {pooled_acc:.4f} (pooled)   pooled macro-F1: {pooled_f1:.4f}")
    print(f"  mean best epoch: {mean_ep:.1f}  (used for refit)")
    print(f"  per-class (pooled over folds):")
    rep = classification_report(y_true_all, y_pred_all,
                                labels=list(range(n_cls)),
                                target_names=labels, digits=3, zero_division=0)
    print("  " + rep.replace("\n", "\n  "))
    print(f"  confusion matrix (rows=true, cols=pred), order={labels}")
    for i, row in enumerate(cm):
        print(f"    {labels[i]:<14}{' '.join(f'{v:>5}' for v in row)}")

    out = dict(
        phase=3, task=args.task, model_key=args.model,
        model_name=MODELS[args.model],
        env=dict(torch=torch.__version__, device=device, amp_bf16=use_amp,
                 seed=SEED, n_splits=N_SPLITS),
        hyperparams=dict(lr=tcfg["lr"], bs=tcfg["bs"], wd=tcfg["wd"],
                         warmup=tcfg["warmup"], max_len=MAX_LEN,
                         max_epochs=cfg["max_epochs"], patience=cfg["patience"],
                         weighted_loss=cfg["weighted"],
                         optim="AdamW", schedule="linear warmup"),
        protocol="fresh pretrained model per fold; StratifiedGroupKFold grouped "
                 "by normalised text; early stop on fold macro-F1",
        folds=fold_results,
        macro_f1_mean=float(np.mean(f1s)), macro_f1_std=float(np.std(f1s)),
        pooled_macro_f1=float(pooled_f1), pooled_accuracy=float(pooled_acc),
        mean_best_epoch=mean_ep,
        per_class=classification_report(y_true_all, y_pred_all,
                                        labels=list(range(n_cls)),
                                        target_names=labels,
                                        output_dict=True, zero_division=0),
        confusion_matrix=cm.tolist(), labels=labels,
        runtime_secs=round(time.time() - t_start, 1),
        refit=None,
    )
    path = os.path.join(OUT_DIR, f"phase3_task{args.task}_{args.model}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwritten: {path}")

# ---------------------------------------------------------------------------
# Core: refit on full train + blind predictions
# ---------------------------------------------------------------------------
def run_refit(args):
    global MAX_LEN
    cfg = TASKS[args.task]
    jpath = os.path.join(OUT_DIR, f"phase3_task{args.task}_{args.model}.json")
    if not os.path.isfile(jpath):
        sys.exit(f"[FATAL] run CV first -- missing {jpath}")
    with open(jpath, encoding="utf-8") as f:
        rec = json.load(f)
    MAX_LEN = int(rec["hyperparams"].get("max_len", MAX_LEN))  # match the CV run

    epochs = max(1, int(round(rec["mean_best_epoch"])))
    bs, lr, wd, warmup = rec["hyperparams"]["bs"], rec["hyperparams"]["lr"], \
        rec["hyperparams"]["wd"], rec["hyperparams"]["warmup"]
    print(BAR); print(f"PHASE 3 REFIT | task {args.task} | model {args.model} "
                      f"| epochs={epochs} (fold-mean of early-stopped epochs)")
    print(BAR)

    train_df = load_csv(cfg["train"]); inputs_df = load_csv(cfg["inputs"])
    tcol, lcol = find_col(train_df, cfg["text"]), find_col(train_df, cfg["label"])
    itcol = find_col(inputs_df, cfg["text"])
    X = train_df[tcol].fillna("").astype(str).map(clean_text).to_numpy()
    y_raw = train_df[lcol].astype(str).str.strip().to_numpy()
    labels = sorted(set(y_raw)); label2id = {c: i for i, c in enumerate(labels)}
    y = np.array([label2id[v] for v in y_raw])
    Xte = inputs_df[itcol].fillna("").astype(str).map(clean_text).tolist()
    n_cls = len(labels)

    device = pick_device(getattr(args, "device", "auto"))
    use_amp = amp_supported(device)
    set_seed(SEED)
    tok, model = build_model(MODELS[args.model], n_cls,
                             {i: c for c, i in label2id.items()},
                             {c: i for c, i in label2id.items()}, device)

    if cfg["weighted"]:
        counts = np.bincount(y, minlength=n_cls).astype(np.float64)
        w = counts.sum() / (n_cls * np.maximum(counts, 1.0))
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32, device=device))
    else:
        criterion = torch.nn.CrossEntropyLoss()

    g = torch.Generator(); g.manual_seed(SEED)
    loader = DataLoader(TextDS(X, y), batch_size=bs, shuffle=True,
                        generator=g, num_workers=0, collate_fn=make_collate(tok))
    opt = torch.optim.AdamW(param_groups(model, wd), lr=lr)
    total_steps = max(1, len(loader) * epochs)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * total_steps), total_steps)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, opt, sched, criterion,
                       device, use_amp, tag=f"ep{ep}")
        print(f"  epoch {ep}/{epochs}: train_loss={loss:.4f} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    logits = predict_texts(model, tok, Xte, bs * 2, device, use_amp)
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = [labels[i] for i in logits.argmax(-1).tolist()]

    allowed = set(cfg["classes"])
    bad = sorted(set(preds) - allowed)
    if bad:
        sys.exit(f"[FATAL] unexpected labels emitted: {bad}")

    out_csv = os.path.join(OUT_DIR, f"task{args.task}_predictions_{args.model}.csv")
    pd.DataFrame({"id": inputs_df["id"].to_numpy(), "label": preds}).to_csv(out_csv, index=False)
    prob_csv = os.path.join(OUT_DIR, f"task{args.task}_probs_{args.model}.csv")
    pd.DataFrame(probs, columns=labels).assign(
        id=inputs_df["id"].to_numpy()).to_csv(prob_csv, index=False)

    dist = {c: int(sum(1 for p in preds if p == c)) for c in cfg["classes"]}
    prior = {c: float(np.mean(y_raw == c)) for c in cfg["classes"]}
    print(f"\nwritten: {out_csv}  ({len(preds)} rows)")
    print(f"written: {prob_csv}  (per-class probabilities, for Phase-4 ensembling)")
    print(f"{'class':<16}{'pred %':>9}{'train prior %':>16}")
    for c in cfg["classes"]:
        print(f"{c:<16}{100*dist[c]/len(preds):>8.2f}%{100*prior[c]:>15.2f}%")

    rec["refit"] = dict(epochs=epochs, submission_file=out_csv,
                        probs_file=prob_csv, prediction_distribution=dist)
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(f"updated: {jpath}")

# ---------------------------------------------------------------------------
# Summary across runs
# ---------------------------------------------------------------------------
def run_summary():
    print(BAR); print("PHASE 3 SUMMARY (all runs vs Phase 2 baselines)"); print(BAR)
    p2 = os.path.join(OUT_DIR, "phase2_metrics.json")
    if os.path.isfile(p2):
        with open(p2, encoding="utf-8") as f:
            m = json.load(f)
        for t in ("A", "B"):
            if f"task{t}" in m:
                d = m[f"task{t}"]
                best = d["best"]
                print(f"  [P2 baseline] task {t}: {best:<18} "
                      f"macro-F1={d['configs'][best]['macro_f1_mean']:.4f}"
                      f" +/- {d['configs'][best]['macro_f1_std']:.4f}")
    rows = []
    for t in ("A", "B"):
        for k in MODELS:
            p = os.path.join(OUT_DIR, f"phase3_task{t}_{k}.json")
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    r = json.load(f)
                rows.append((t, k, r["macro_f1_mean"], r["macro_f1_std"],
                             r["pooled_accuracy"],
                             "refit" if r.get("refit") else "cv-only"))
    if not rows:
        print("  (no phase-3 runs yet)")
        return
    print(f"\n  {'task':<5}{'model':<9}{'macro-F1':>16}{'pooled acc':>12}{'status':>10}")
    for t, k, m1, s1, acc, st in sorted(rows, key=lambda r: (r[0], -r[2])):
        print(f"  {t:<5}{k:<9}{m1:>10.4f} +/-{s1:.4f}{acc:>12.4f}{st:>10}")
    print("\n  best per task:")
    for t in ("A", "B"):
        cand = [r for r in rows if r[0] == t]
        if cand:
            b = max(cand, key=lambda r: r[2])
            print(f"    task {t}: {b[1]} (macro-F1 {b[2]:.4f})")

# ---------------------------------------------------------------------------
# Report-only cross-file diagnostic (uses TRAIN labels only; no test labels)
# ---------------------------------------------------------------------------
def run_diag():
    print(BAR); print("CROSS-FILE DIAGNOSTIC (report-only; train labels only)")
    print("Purpose: understand blind-set composition. NO label transfer.")
    print(BAR)

    at = load_csv(TASKS["A"]["train"])
    bt = load_csv(TASKS["B"]["train"])
    av = load_csv(TASKS["A"]["inputs"])
    bv = load_csv(TASKS["B"]["inputs"])

    atc = find_col(at, "Comment")
    btc = find_col(bt, "Comment")
    avc = find_col(av, "Comment")
    bvc = find_col(bv, "Comment")
    alc = find_col(at, TASKS["A"]["label"])
    blc = find_col(bt, TASKS["B"]["label"])

    def norm_series(df, c):
        return df[c].fillna("").astype(str).map(clean_text).str.lower()

    def text_label_map(df, tcol, lcol):
        """normalized text -> majority train label (robust to duplicate texts)."""
        tmp = pd.DataFrame({
            "txt": norm_series(df, tcol),
            "lab": df[lcol].astype(str).str.strip(),
        })
        agg = (tmp.groupby(["txt", "lab"]).size().reset_index(name="n")
                  .sort_values("n", ascending=False)
                  .drop_duplicates("txt"))
        return dict(zip(agg["txt"], agg["lab"]))

    a_map = text_label_map(at, atc, alc)   # text -> Hate/Non-Hate (A_train)
    b_map = text_label_map(bt, btc, blc)   # text -> category   (B_train)

    bval_txt = norm_series(bv, bvc)
    hits_b = [t for t in bval_txt if t in a_map]
    print(f"\n(B) B_val comments also present in A_train: "
          f"{len(hits_b)}/{len(bval_txt)}")
    if hits_b:
        vc = pd.Series([a_map[t] for t in hits_b]).value_counts()
        print("    their A_train binary-label distribution (train labels only):")
        for k, v in vc.items():
            print(f"      {k:<10} {v:>5}  ({100 * v / len(hits_b):.1f}%)")
        print("    -> if Non-Hate appears, Task B's blind set likely contains")
        print("       non-hate comments; we then discuss a two-stage design.")

    aval_txt = norm_series(av, avc)
    hits_a = [t for t in aval_txt if t in b_map]
    print(f"\n(A) A_val comments also present in B_train: "
          f"{len(hits_a)}/{len(aval_txt)}")
    if hits_a:
        vc = pd.Series([b_map[t] for t in hits_a]).value_counts()
        print("    their B_train category distribution (train labels only):")
        for k, v in vc.items():
            print(f"      {k:<14} {v:>5}  ({100 * v / len(hits_a):.1f}%)")

    print("\n(diagnostic ends; these numbers go into the paper's data-analysis section)")

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["smoke", "cv", "refit", "summary", "diag"])
    ap.add_argument("--task", choices=["A", "B"])
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "xpu", "cuda", "cpu"])
    args = ap.parse_args()

    global MAX_LEN
    MAX_LEN = args.max_len

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.mode == "summary":
        run_summary(); return
    if args.mode == "diag":
        run_diag(); return
    if not args.task or not args.model:
        sys.exit("[FATAL] --task and --model required for smoke/cv/refit modes")
    if args.mode in ("smoke", "cv"):
        run_cv(args)
    else:
        run_refit(args)

if __name__ == "__main__":
    main()