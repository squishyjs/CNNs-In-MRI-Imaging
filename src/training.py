""" MAIN TRAINING PRTOCOL:

Weighted BCE, AdamW, cosine schedule, with mixed precision where available, gradient accumulation over examinations,
early stopping on mean tuning AUC, per-exam probabilities (and saved to CSV)
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .config import Config, DEFAULT_HPARAMS, SEARCH_SPACE, PLANES, get_device, amp_context
from .data import (ExamDataset, collate_single, labels_to_dict, load_mrnet_labels,
                   make_transform)
from .models import build_model, count_params


@dataclass
class RunSpec:
    run_id: str
    model_id: str
    plane: str
    seed: int = 0
    hparams: dict = field(default_factory=dict)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# run grids

def build_run_grid(cfg: Config, models: list[str], seeds=None,
                   planes_for: dict | None = None):
    """Models × planes × seeds. `planes_for` restricts individual models to a
    subset of planes (budget cuts, thesis §5.8)."""
    seeds = list(seeds if seeds is not None else cfg.seeds)
    planes_for = planes_for or {}
    return [RunSpec(f"{m}_{plane}_s{s}", m, plane, seed=s,
                    hparams=dict(DEFAULT_HPARAMS[m]))
            for m in models
            for plane in planes_for.get(m, PLANES)
            for s in seeds]


# data resolution

def _read_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run notebook 01 to create the splits.")
    return pd.read_csv(path, dtype=str)["id"].tolist()


def resolve_data(cfg: Config, spec: RunSpec):
    """Returns (train_ds, tune_ds, test_ds, pos_weight, tasks) for a run."""
    tasks = cfg.tasks
    root = cfg.data_root
    labels_tr = load_mrnet_labels(root, "train")
    labels_te = load_mrnet_labels(root, "valid")
    train_ids = _read_ids(cfg.splits_dir / "dev_ids.csv")
    tune_ids = _read_ids(cfg.splits_dir / "tune_ids.csv")
    test_ids = labels_te["id"].tolist()
    lab_tr = labels_to_dict(labels_tr, tasks)
    lab_te = labels_to_dict(labels_te, tasks)
    tr_folder = root / "train" / spec.plane
    te_folder = root / "valid" / spec.plane

    train_ds = ExamDataset(tr_folder, train_ids, lab_tr, make_transform(spec.model_id, True))
    tune_ds = ExamDataset(tr_folder, tune_ids, lab_tr, make_transform(spec.model_id, False))
    test_ds = ExamDataset(te_folder, test_ids, lab_te, make_transform(spec.model_id, False))

    y_train = np.stack([lab_tr[i] for i in train_ids])          # pos_weight = neg/pos per task
    pos = y_train.sum(axis=0)
    pw = torch.tensor((len(y_train) - pos) / np.clip(pos, 1, None), dtype=torch.float32)
    return train_ds, tune_ds, test_ds, pw, tasks


def make_loader(cfg: Config, ds, model_id: str, shuffle: bool):
    if model_id == "D":
        return DataLoader(ds, batch_size=cfg.batch_3d, shuffle=shuffle,
                          num_workers=cfg.num_workers)
    return DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=cfg.num_workers,
                      collate_fn=collate_single)


# core loops

def _forward_loss(model, crit, x, y, device, cfg):
    x, y = x.to(device), y.to(device)
    with amp_context(device, cfg.use_amp):
        logits = model(x)
    return logits, crit(logits.float(), y)


@torch.no_grad()
def predict(cfg: Config, model, ds, model_id: str, device) -> tuple[pd.DataFrame, float]:
    """Per-exam sigmoid probabilities. Returns (frame, seconds_per_exam)."""
    model.eval()
    loader = make_loader(cfg, ds, model_id, shuffle=False)
    ids, ys, ps = [], [], []
    t0 = time.perf_counter()
    for x, y, eid in loader:
        x = x.to(device)
        with amp_context(device, cfg.use_amp):
            logits = model(x)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        if probs.ndim == 1:                     # slice models: one exam per batch
            ids.append(eid); ys.append(y.numpy()); ps.append(probs)
        else:                                   # model D: real mini-batches
            ids.extend(list(eid)); ys.extend(list(y.numpy())); ps.extend(list(probs))
    secs = (time.perf_counter() - t0) / max(len(ds), 1)
    return ids, np.stack(ys), np.stack(ps), secs


def _preds_frame(ids, ys, ps, tasks) -> pd.DataFrame:
    df = pd.DataFrame({"id": ids})
    for j, t in enumerate(tasks):
        df[f"y_{t}"] = ys[:, j]
        df[f"p_{t}"] = ps[:, j]
    return df


def _tuning_auc(ys, ps, tasks):
    aucs = {}
    for j, t in enumerate(tasks):
        aucs[t] = (roc_auc_score(ys[:, j], ps[:, j])
                   if len(np.unique(ys[:, j])) > 1 else np.nan)
    vals = [v for v in aucs.values() if not np.isnan(v)]
    return aucs, (float(np.mean(vals)) if vals else 0.0)


def train_one(cfg: Config, spec: RunSpec, max_epochs: int | None = None,
              save: bool = True, quiet: bool = False) -> dict:
    """Train one configuration; write best checkpoint, predictions and log.json."""
    device = get_device()
    set_seed(spec.seed)
    max_epochs = max_epochs or cfg.max_epochs
    train_ds, tune_ds, test_ds, pw, tasks = resolve_data(cfg, spec)

    hp = {**DEFAULT_HPARAMS[spec.model_id], **spec.hparams}
    model = build_model(spec.model_id, n_out=len(tasks), dropout=hp["dropout"]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    accum = cfg.accum_exams if spec.model_id != "D" else max(1, cfg.accum_exams // cfg.batch_3d)

    best = dict(auc=-1.0, state=None, epoch=-1)
    history, epochs_without = [], 0
    for epoch in range(max_epochs):
        model.train()
        loader = make_loader(cfg, train_ds, spec.model_id, shuffle=True)
        t0, losses = time.perf_counter(), []
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(loader):
            _, loss = _forward_loss(model, crit, x, y, device, cfg)
            (loss / accum).backward()
            losses.append(float(loss.detach()))
            if (i + 1) % accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
        opt.step()                              # flush the final partial accumulation
        opt.zero_grad(set_to_none=True)
        sched.step()
        train_s = time.perf_counter() - t0

        _, ys, ps, _ = predict(cfg, model, tune_ds, spec.model_id, device)
        per_task, mean_auc = _tuning_auc(ys, ps, tasks)
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)),
                            tune_auc_mean=mean_auc, epoch_seconds=train_s,
                            **{f"tune_auc_{t}": v for t, v in per_task.items()}))
        if not quiet:
            print(f"  [{spec.run_id}] epoch {epoch:02d}  loss {np.mean(losses):.4f}  "
                  f"tuning AUC {mean_auc:.4f}  ({train_s/60:.1f} min)")
        if mean_auc > best["auc"]:
            best = dict(auc=mean_auc,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        epoch=epoch)
            epochs_without = 0
        else:
            epochs_without += 1
            if epochs_without >= cfg.patience:
                break

    model.load_state_dict(best["state"])
    peak_mem_mb = None
    if device.type == "mps":
        try:
            peak_mem_mb = torch.mps.current_allocated_memory() / 1e6
        except Exception:
            pass

    ids, ys, ps, _ = predict(cfg, model, tune_ds, spec.model_id, device)
    tune_df = _preds_frame(ids, ys, ps, tasks)
    ids, ys, ps, infer_s = predict(cfg, model, test_ds, spec.model_id, device)
    test_df = _preds_frame(ids, ys, ps, tasks)

    log = dict(run_id=spec.run_id, model_id=spec.model_id,
               plane=spec.plane, seed=spec.seed, hparams=hp,
               tasks=tasks, n_params=count_params(model), best_epoch=best["epoch"],
               epochs_run=len(history), best_tuning_auc=best["auc"],
               epoch_seconds_mean=float(np.mean([h["epoch_seconds"] for h in history])),
               inference_seconds_per_exam=infer_s, peak_memory_mb=peak_mem_mb,
               history=history)
    if save:
        run_dir = cfg.runs_dir / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), run_dir / "best.pt")
        tune_df.to_csv(run_dir / "preds_tuning.csv", index=False)
        test_df.to_csv(run_dir / "preds_test.csv", index=False)
        (run_dir / "log.json").write_text(json.dumps(log, indent=2))
    return log


def run_grid(cfg: Config, specs: list[RunSpec], resume: bool = True):
    """Run every spec, skipping runs whose log.json already exists (safe to re-run)."""
    done, t0 = [], time.time()
    for k, spec in enumerate(specs):
        log_path = cfg.runs_dir / spec.run_id / "log.json"
        if resume and log_path.exists():
            print(f"[{k+1}/{len(specs)}] {spec.run_id}: already done, skipping")
            continue
        print(f"[{k+1}/{len(specs)}] training {spec.run_id} "
              f"(elapsed {(time.time()-t0)/3600:.2f} h)")
        done.append(train_one(cfg, spec))
    print(f"Grid finished in {(time.time()-t0)/3600:.2f} h")
    return done


# diagnostics

def overfit_check(cfg: Config, model_id: str, n: int = 16, epochs: int = 40) -> bool:
    """Guide step 5: a model must overfit a handful of exams to near-zero loss.

    This is a memorization test, so regularization is off (no augmentation,
    dropout or weight decay). Model D batches `batch_3d` exams per step and
    starts from random weights, so it gets proportionally more epochs to see
    the same number of optimizer steps as the slice models."""
    device = get_device()
    set_seed(0)
    spec = RunSpec("sanity", model_id, plane="sagittal")
    train_ds, _, _, pw, tasks = resolve_data(cfg, spec)
    train_ds.ids = train_ds.ids[:n]
    train_ds.tf = make_transform(model_id, train=False)     # no augmentation here
    model = build_model(model_id, n_out=len(tasks), dropout=0.0).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4 if model_id in "BCE" else 1e-3,
                            weight_decay=0.0)
    if model_id == "D":
        epochs *= cfg.batch_3d
    loss_val = float("inf")
    for e in range(epochs):
        model.train()
        losses = []
        for x, y, _ in make_loader(cfg, train_ds, model_id, shuffle=True):
            opt.zero_grad(set_to_none=True)
            _, loss = _forward_loss(model, crit, x, y, device, cfg)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        loss_val = float(np.mean(losses))
        if loss_val < 0.03:                                 # memorized, stop early
            break
    ok = loss_val < 0.1
    print(f"Model {model_id}: final loss on {n} exams = {loss_val:.4f} -> "
          f"{'PASS' if ok else 'FAIL (investigate before training)'}")
    return ok


def time_one_epoch(cfg: Config, model_id: str, plane: str = "sagittal") -> float:
    """Guide step 7: measure one epoch (train + tuning eval) in seconds."""
    spec = RunSpec("timing", model_id, plane=plane)
    log = train_one(cfg, spec, max_epochs=1, save=False, quiet=True)
    s = log["history"][0]["epoch_seconds"]
    print(f"Model {model_id}: {s/60:.1f} min/epoch (training pass only)")
    return s


def random_search(cfg: Config, model_id: str, n_trials: int = 4, epoch_cap: int = 6,
                  plane: str = "sagittal") -> dict:
    """Thesis §3.6 random search (matched budget per model), on sagittal, seed 0."""
    rng = np.random.default_rng(0)
    rows = []
    for t in range(n_trials):
        hp = dict(
            lr=float(10 ** rng.uniform(np.log10(SEARCH_SPACE["lr"][0]),
                                       np.log10(SEARCH_SPACE["lr"][1]))),
            weight_decay=float(rng.choice(SEARCH_SPACE["weight_decay"])),
            dropout=float(rng.choice(SEARCH_SPACE["dropout"])),
        )
        spec = RunSpec(f"search_{model_id}_t{t}", model_id, plane=plane, hparams=hp)
        log = train_one(cfg, spec, max_epochs=epoch_cap, save=False, quiet=True)
        rows.append({**hp, "tuning_auc": log["best_tuning_auc"]})
        print(f"  trial {t}: {hp} -> tuning AUC {log['best_tuning_auc']:.4f}")
    res = pd.DataFrame(rows).sort_values("tuning_auc", ascending=False)
    res.to_csv(cfg.outputs_dir / f"search_{model_id}.csv", index=False)
    return res.iloc[0][["lr", "weight_decay", "dropout"]].to_dict()
