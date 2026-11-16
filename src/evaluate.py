"""Turns the per-run prediction CSVs into every Chapter 4 table.

All functions read from outputs/runs/, so results can be regenerated at any
time without retraining
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config, CONTRASTS, MODEL_NAMES, get_device
from .data import ExamDataset, make_transform
from .fusion import fuse_planes, seed_average
from .metrics import (bh_correct, delong_ci, delong_test, full_metrics,
                      youden_threshold)
from .models import build_model
from .training import predict


# run collection

def collect_runs(cfg: Config) -> pd.DataFrame:
    rows = []
    for log_path in sorted(cfg.runs_dir.glob("*/log.json")):
        log = json.loads(log_path.read_text())
        hist = log.pop("history", [])
        log["final_tuning_auc"] = hist[-1]["tune_auc_mean"] if hist else np.nan
        rows.append(log)
    if not rows:
        raise FileNotFoundError(f"No completed runs in {cfg.runs_dir}; run notebook 02 first.")
    return pd.DataFrame(rows)


def load_history(cfg: Config, run_id: str) -> pd.DataFrame:
    log = json.loads((cfg.runs_dir / run_id / "log.json").read_text())
    return pd.DataFrame(log["history"])


def load_preds(cfg: Config, run_id: str, split: str) -> pd.DataFrame:
    return pd.read_csv(cfg.runs_dir / run_id / f"preds_{split}.csv", dtype={"id": str})


def load_trained(cfg: Config, run_id: str):
    """Rebuild a model and load its best checkpoint."""
    log = json.loads((cfg.runs_dir / run_id / "log.json").read_text())
    model = build_model(log["model_id"], n_out=len(log["tasks"]),
                        dropout=log["hparams"].get("dropout", 0.0))
    state = torch.load(cfg.runs_dir / run_id / "best.pt", map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    return model.to(get_device()).eval(), log


def save_table(cfg: Config, df: pd.DataFrame, name: str) -> Path:
    path = cfg.tables_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"saved -> {path}")
    return path


# figure tables

def plane_level_table(cfg: Config, runs: pd.DataFrame) -> pd.DataFrame:
    """Table 4.1: per-plane test AUC, mean ± sd across seeds."""
    from sklearn.metrics import roc_auc_score
    rows = []
    for (m, plane), grp in runs.groupby(["model_id", "plane"]):
        row = {"Model": m, "Plane": plane, "Seeds": len(grp)}
        for t in cfg.tasks:
            aucs = []
            for rid in grp["run_id"]:
                df = load_preds(cfg, rid, "test")
                if df[f"y_{t}"].nunique() > 1:
                    aucs.append(roc_auc_score(df[f"y_{t}"], df[f"p_{t}"]))
            row[f"AUC {t}"] = (f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}"
                               if aucs else "n/a")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Model", "Plane"]).reset_index(drop=True)


def fused_predictions(cfg: Config, runs: pd.DataFrame) -> dict:
    """Per model and task: seed-averaged fused tuning and test predictions.

    Returns {model: {task: {"tune": df, "test": df}}} with columns id, y, p.
    """
    out = {}
    for m, grp in runs.groupby("model_id"):
        out[m] = {}
        for t in cfg.tasks:
            tune_frames, test_frames = [], []
            for seed, sg in grp.groupby("seed"):
                tune_by_plane = {r.plane: load_preds(cfg, r.run_id, "tuning")
                                 for r in sg.itertuples()}
                test_by_plane = {r.plane: load_preds(cfg, r.run_id, "test")
                                 for r in sg.itertuples()}
                f_tu, f_te = fuse_planes(tune_by_plane, test_by_plane, t)
                tune_frames.append(f_tu)
                test_frames.append(f_te)
            out[m][t] = {"tune": seed_average(tune_frames),
                         "test": seed_average(test_frames)}
    return out


def exam_level_table(cfg: Config, fused: dict) -> pd.DataFrame:
    """Table 4.2: fused test metrics at the tuning-set Youden threshold."""
    rows = []
    for m in sorted(fused):
        for t in cfg.tasks:
            tune, test = fused[m][t]["tune"], fused[m][t]["test"]
            thr = youden_threshold(tune["y"], tune["p"])
            met = full_metrics(test["y"], test["p"], thr)
            rows.append({
                "Model": m, "Task": t,
                "AUC (95% CI)": f"{met['auc']:.3f} ({met['auc_lo']:.3f}–{met['auc_hi']:.3f})",
                "Sensitivity": f"{met['sensitivity']:.3f} "
                               f"({met['sens_ci'][0]:.3f}–{met['sens_ci'][1]:.3f})",
                "Specificity": f"{met['specificity']:.3f} "
                               f"({met['spec_ci'][0]:.3f}–{met['spec_ci'][1]:.3f})",
                "F1": f"{met['f1']:.3f}", "Brier": f"{met['brier']:.3f}",
                "ECE": f"{met['ece']:.3f}", "Threshold": f"{thr:.3f}",
            })
    return pd.DataFrame(rows)


def pairwise_table(cfg: Config, fused: dict) -> pd.DataFrame:
    """Table 4.3: planned paired DeLong contrasts with BH correction (§3.8)."""
    rows = []
    for t in cfg.tasks:
        for a, b in CONTRASTS:
            if a not in fused or b not in fused:
                continue
            fa, fb = fused[a][t]["test"], fused[b][t]["test"]
            merged = fa.merge(fb, on="id", suffixes=("_a", "_b"))
            diff, p = delong_test(merged["y_a"], merged["p_a"], merged["p_b"])
            rows.append(dict(Task=t, Contrast=f"{a} vs {b}", dAUC=diff, p=p))
    df = pd.DataFrame(rows)
    if len(df):
        reject, p_adj = bh_correct(df["p"].tolist())
        df["p (BH-adjusted)"] = p_adj
        df["Significant (FDR 0.05)"] = reject
        df["dAUC"] = df["dAUC"].map(lambda v: f"{v:+.3f}")
        df["p"] = df["p"].map(lambda v: f"{v:.4f}")
        df["p (BH-adjusted)"] = df["p (BH-adjusted)"].map(lambda v: f"{v:.4f}")
    return df


def efficiency_table(cfg: Config, runs: pd.DataFrame) -> pd.DataFrame:
    """Table 4.5: parameters, training time, inference time, memory."""
    rows = []
    for m, grp in runs.groupby("model_id"):
        rows.append({
            "Model": m, "Name": MODEL_NAMES[m],
            "Parameters (M)": f"{grp['n_params'].iloc[0] / 1e6:.2f}",
            "Epoch time (min)": f"{grp['epoch_seconds_mean'].mean() / 60:.1f}",
            "Inference (s/exam)": f"{grp['inference_seconds_per_exam'].mean():.2f}",
            "Peak memory (GB)": (f"{grp['peak_memory_mb'].max() / 1000:.1f}"
                                 if grp["peak_memory_mb"].notna().any() else "n/a"),
            "Runs": len(grp),
        })
    return pd.DataFrame(rows)


# external validation

def external_validation(cfg: Config, runs: pd.DataFrame, models=None) -> tuple[pd.DataFrame, dict]:
    """§4.7: zero-shot ACL transfer of the sagittal MRNet models to KneeMRI.
    Needs the converted KneeMRI arrays and splits/kneemri_labels.csv (notebook 01 §1.3).
    Returns (table, {model: (y, p)}) for ROC plotting."""
    device = get_device()
    acl_idx = cfg.tasks.index("acl")

    lab_df = pd.read_csv(cfg.splits_dir / "kneemri_labels.csv", dtype={"id": str})
    ids = lab_df["id"].tolist()
    labels = {r["id"]: np.array([r["acl"]], dtype=np.float32) for _, r in lab_df.iterrows()}
    folder = cfg.kneemri_npy_dir
    cand = runs[runs["plane"] == "sagittal"]

    rows, preds = [], {}
    for m in sorted((models or cand["model_id"].unique())):
        grp = cand[cand["model_id"] == m]
        if grp.empty:
            continue
        prob_sum, y_ref = None, None
        for rid in grp["run_id"]:
            model, log = load_trained(cfg, rid)
            ds = ExamDataset(folder, ids, labels, make_transform(m, train=False))
            _, ys, ps, _ = predict(cfg, model, ds, m, device)
            prob_sum = ps[:, acl_idx] if prob_sum is None else prob_sum + ps[:, acl_idx]
            y_ref = ys[:, 0]
            del model
        p = prob_sum / len(grp)
        auc, (lo, hi) = delong_ci(y_ref, p)
        preds[m] = (y_ref, p)
        rows.append({"Model": m, "External AUC (95% CI)": f"{auc:.3f} ({lo:.3f}–{hi:.3f})",
                     "Checkpoints averaged": len(grp), "Examinations": len(ids)})
    return pd.DataFrame(rows), preds