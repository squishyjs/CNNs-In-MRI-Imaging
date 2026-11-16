"""Grad-CAM saliency: class-activation maps from the final
convolutional stage, with a derived bounding box around
the most activated region so figures show where the model found the injury
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .config import MODEL_NAMES
from .models import gradcam_target


class GradCAM:
    def __init__(self, model: torch.nn.Module, model_id: str):
        self.model, self.model_id = model.eval(), model_id
        self._feats = None
        gradcam_target(model, model_id).register_forward_hook(self._save_feats)

    def _save_feats(self, _m, _i, out):
        # keep the pristine tensor; hand a clone downstream so in-place ReLUs
        # (e.g. in MONAI's DenseNet head) cannot touch it.
        self._feats = out
        return out.clone()

    def compute(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """x: (S, C, 224, 224) for slice models or (1, 1, 32, 128, 128) for model D.
        Returns a cam aligned to the input spatial grid, normalised to [0, 1]."""
        with torch.enable_grad():
            logits = self.model(x)
            logit = logits[class_idx] if logits.ndim == 1 else logits[0, class_idx]
            grads, = torch.autograd.grad(logit, self._feats)
        feats, grads = self._feats.detach(), grads.detach()
        dims = tuple(range(2, feats.ndim))
        w = grads.mean(dim=dims, keepdim=True)
        cam = F.relu((w * feats).sum(dim=1, keepdim=True))
        if self.model_id == "D":                      # (1, 1, d, h, w) -> (32, 128, 128)
            cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear",
                                align_corners=False)[0, 0]
        else:                                         # (S, 1, h, w) -> (S, 224, 224)
            cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear",
                                align_corners=False)[:, 0]
        cam = cam.float().cpu().numpy()
        return cam / max(cam.max(), 1e-8)


def cam_bbox(cam2d: np.ndarray, frac: float = 0.5):
    """Bounding box (x, y, w, h) of the largest connected high-activation region."""
    mask = cam2d >= frac * cam2d.max()
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    largest = np.argmax(ndimage.sum(mask, lab, range(1, n + 1))) + 1
    ys, xs = np.where(lab == largest)
    return xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()


def _display_base(vol: np.ndarray, model_id: str) -> np.ndarray:
    """Grayscale slices aligned with the cam grid (crop for 2D, resize for 3D)."""
    v = torch.from_numpy(vol).float().div(255)
    if model_id == "D":
        v = F.interpolate(v[None, None], size=(32, 128, 128), mode="trilinear",
                          align_corners=False)[0, 0]
        return v.numpy()
    off = (v.shape[-1] - 224) // 2
    return v[:, off:off + 224, off:off + 224].numpy()


def gradcam_figure(vol: np.ndarray, cam: np.ndarray, model_id: str, task: str,
                   exam_id: str, y_true: int, prob: float, save: Path | None = None):
    """Three panels for the thesis: slice, Grad-CAM overlay, overlay + detection box."""
    base = _display_base(vol, model_id)
    k = int(cam.reshape(cam.shape[0], -1).sum(axis=1).argmax())   # most activated slice
    sl, csl = base[k], cam[k]
    bbox = cam_bbox(csl)
    kind = "TP" if (y_true == 1 and prob >= 0.5) else ("FN" if y_true == 1 else
            "TN" if prob < 0.5 else "FP")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    for ax in axes:
        ax.imshow(sl, cmap="gray", interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].imshow(csl, cmap="inferno", alpha=0.45, interpolation="bilinear")
    axes[2].imshow(csl, cmap="inferno", alpha=0.45, interpolation="bilinear")
    if bbox:
        x, y, w, h = bbox
        axes[2].add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=2.0,
                                        edgecolor="#56B4E9"))
    axes[0].set_title("MRI slice", fontsize=10)
    axes[1].set_title("Grad-CAM", fontsize=10)
    axes[2].set_title("Detected region", fontsize=10)
    fig.suptitle(f"Model {model_id} ({MODEL_NAMES[model_id]}) — {task} — exam {exam_id} — "
                 f"label {y_true}, p = {prob:.2f} ({kind}), slice {k}", fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


def plane_comparison_figure(vols: dict[str, np.ndarray], cams: dict[str, np.ndarray],
                            model_id: str, task: str, exam_id: str,
                            save: Path | None = None):
    """Same examination viewed in all three planes with Grad-CAM overlays, as a
    visual answer to 'does the model attend to the right anatomy from every angle?'."""
    planes = list(vols)
    fig, axes = plt.subplots(2, len(planes), figsize=(3.5 * len(planes), 7.2))
    for j, plane in enumerate(planes):
        base = _display_base(vols[plane], model_id)
        cam = cams[plane]
        k = int(cam.reshape(cam.shape[0], -1).sum(axis=1).argmax())
        for i in (0, 1):
            axes[i, j].imshow(base[k], cmap="gray", interpolation="bilinear")
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        axes[1, j].imshow(cam[k], cmap="inferno", alpha=0.45, interpolation="bilinear")
        axes[0, j].set_title(f"{plane} (slice {k})", fontsize=10)
    axes[0, 0].set_ylabel("MRI slice", fontsize=10)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=10)
    fig.suptitle(f"Model {model_id} — {task} — exam {exam_id}: plane comparison", fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


def pick_saliency_exams(fused_test: "pd.DataFrame", n_each: int = 2):
    """Choose true positives (highest p) and false negatives (lowest p) among
    positive-label exams, as §3.10 asks for both successes and misses."""
    pos = fused_test[fused_test["y"] == 1].sort_values("p", ascending=False)
    return pos.head(n_each)["id"].tolist(), pos.tail(n_each)["id"].tolist()