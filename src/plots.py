"""Thesis figures: ROC curves, training curves, reliability diagrams, confusion
matrices and dataset montages. One fixed colorblind-safe color per model (A-E),
single axis per chart, recessive grids, 300 dpi output for print.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import MODEL_COLORS, MODEL_NAMES
from .metrics import delong_ci, reliability_bins


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False, "lines.linewidth": 2.0,
    })


def _save(fig, save: Path | None):
    if save:
        fig.savefig(save)
        print(f"saved -> {save}")


def roc_figure(preds_by_model: dict[str, tuple], title: str, save: Path | None = None):
    """preds_by_model: {model_id: (y_true, y_prob)}; one fixed color per model."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for m in sorted(preds_by_model):
        y, p = preds_by_model[m]
        fpr, tpr, _ = roc_curve(y, p)
        auc, _ = delong_ci(y, p)
        ax.plot(fpr, tpr, color=MODEL_COLORS[m], label=f"Model {m} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#999999", zorder=0)
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, save)
    return fig


def roc_grid(fused: dict, tasks: list[str], split: str = "test",
             save: Path | None = None):
    """One ROC panel per task, five models per panel (Chapter 4 figure)."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.0 * len(tasks), 4.6))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, tasks):
        for m in sorted(fused):
            d = fused[m][t][split]
            fpr, tpr, _ = roc_curve(d["y"], d["p"])
            auc, _ = delong_ci(d["y"], d["p"])
            ax.plot(fpr, tpr, color=MODEL_COLORS[m], label=f"{m} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#999999", zorder=0)
        ax.set_title(t)
        ax.set_xlabel("1 − specificity")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Sensitivity")
    fig.suptitle("Receiver operating characteristic — fused examination-level predictions",
                 y=1.02)
    fig.tight_layout()
    _save(fig, save)
    return fig


def reliability_figure(fused: dict, task: str, save: Path | None = None, n_bins: int = 10):
    """Reliability diagram per model for one task (§3.8 calibration)."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for m in sorted(fused):
        d = fused[m][task]["test"]
        bins = reliability_bins(d["y"], d["p"], n_bins)
        ax.plot(bins["confidence"], bins["frequency"], marker="o", ms=4,
                color=MODEL_COLORS[m], label=f"Model {m}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#999999", zorder=0)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration — {task}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, save)
    return fig


def training_curves_figure(histories: dict[str, pd.DataFrame], save: Path | None = None):
    """Tuning AUC and training loss per epoch, one curve per model."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for m in sorted(histories):
        h = histories[m]
        axes[0].plot(h["epoch"], h["tune_auc_mean"], color=MODEL_COLORS[m],
                     label=f"Model {m}")
        axes[1].plot(h["epoch"], h["train_loss"], color=MODEL_COLORS[m])
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Mean tuning AUC")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Training loss")
    axes[0].set_title("Tuning-set discrimination")
    axes[1].set_title("Training loss")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save)
    return fig


def confusion_figure(tp: int, fp: int, tn: int, fn: int, model_id: str, task: str,
                     save: Path | None = None):
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() * 0.6 else "white")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.grid(False)
    ax.set_title(f"Model {model_id} — {task}", fontsize=10)
    _save(fig, save)
    return fig


def exam_montage(vol: np.ndarray, title: str, n: int = 12, save: Path | None = None):
    """Grid of slices from one examination (dataset figure for Chapter 3)."""
    idx = np.linspace(0, vol.shape[0] - 1, min(n, vol.shape[0])).astype(int)
    cols = 6
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 2.1 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, k in zip(np.ravel(axes), idx):
        ax.imshow(vol[k], cmap="gray")
        ax.set_title(f"slice {k}", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(h_pad=1.6, rect=(0, 0, 1, 0.97))
    _save(fig, save)
    return fig


def slice_count_hist(counts_by_plane: dict[str, list[int]], save: Path | None = None):
    """Histogram of slices per examination, one panel per plane (single hue)."""
    fig, axes = plt.subplots(1, len(counts_by_plane), figsize=(10.5, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (plane, counts) in zip(axes, counts_by_plane.items()):
        ax.hist(counts, bins=20, color="#0072B2", edgecolor="white", linewidth=0.5)
        ax.set_title(plane)
        ax.set_xlabel("Slices per exam")
    axes[0].set_ylabel("Examinations")
    fig.tight_layout()
    _save(fig, save)
    return fig


def roi_example_figure(vol: np.ndarray, slice_idx: int, box: tuple, exam_id: str,
                       save: Path | None = None):
    """KneeMRI radiologist region-of-interest drawn on its central slice."""
    x, y, w, h = box
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(vol[slice_idx], cmap="gray")
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=2.0, edgecolor="#E69F00"))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"KneeMRI exam {exam_id}: annotated ACL region (slice {slice_idx})",
                 fontsize=10)
    _save(fig, save)
    return fig