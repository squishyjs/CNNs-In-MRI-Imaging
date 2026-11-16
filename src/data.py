"""Datasets, preprocessing and splits for MRNet and KneeMRI

One `ExamDataset` serves both datasets because KneeMRI volumes are converted
once to MRNet-style uint8 arrays of shape (slices, 256, 256)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

from .config import TASKS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# labels and splits

def load_mrnet_labels(root: Path, split: str) -> pd.DataFrame:
    """Merge the three headerless label CSVs into one frame: id, abnormal, acl, meniscus."""
    df = None
    for t in TASKS:
        d = pd.read_csv(Path(root) / f"{split}-{t}.csv", header=None,
                        names=["id", t], dtype={"id": str})
        d["id"] = d["id"].str.zfill(4)
        df = d if df is None else df.merge(d, on="id")
    return df


def make_dev_tuning_split(labels: pd.DataFrame, n_tuning: int, seed: int = 0):
    """Iterative multi-label stratified split of train/ into development and tuning sets."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    y = labels[TASKS].values
    mss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=n_tuning, random_state=seed)
    dev_idx, tune_idx = next(mss.split(labels["id"].values, y))
    return labels.iloc[dev_idx]["id"].tolist(), labels.iloc[tune_idx]["id"].tolist()


def prevalence_table(partitions: dict[str, pd.DataFrame], tasks=TASKS) -> pd.DataFrame:
    """Rows for thesis Table 3.2: examinations and per-task n (%) in each partition."""
    rows = []
    for name, df in partitions.items():
        row = {"Partition": name, "Examinations": len(df)}
        for t in tasks:
            if t in df.columns:
                n = int(df[t].sum())
                row[t] = f"{n} ({100 * n / len(df):.1f}%)"
            else:
                row[t] = "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def labels_to_dict(df: pd.DataFrame, tasks) -> dict[str, np.ndarray]:
    return {r["id"]: np.array([r[t] for t in tasks], dtype=np.float32)
            for _, r in df.iterrows()}


# transforms

def _rand_geometric(stack: torch.Tensor) -> torch.Tensor:
    """MRNet-style augmentation: one rotation (±25°), translation (≤25 px) and
    horizontal flip sampled per examination and applied to every slice."""
    angle = float(torch.empty(1).uniform_(-25, 25))
    tx, ty = [int(torch.randint(-25, 26, (1,))) for _ in range(2)]
    stack = TF.affine(stack, angle=angle, translate=[tx, ty], scale=1.0,
                      shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR)
    if torch.rand(1) < 0.5:
        stack = torch.flip(stack, dims=[-1])
    return stack


class TorchvisionStackTransform:
    """uint8 volume (S, 256, 256) -> model-ready tensor. Modes: rgb224 (B, C), mlp64 (A)."""

    def __init__(self, mode: str, train: bool):
        self.mode, self.train = mode, train

    def __call__(self, vol: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(vol)).float().div(255).unsqueeze(1)
        if self.train:
            x = _rand_geometric(x)
        if self.mode == "mlp64":
            return F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        x = TF.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return TF.center_crop(x, [224, 224])


class MonaiStackTransform:
    """MONAI pipelines: 2D slice stacks for model E, 3D volumes for model D (§3.4)."""

    def __init__(self, mode: str, train: bool):
        from monai import transforms as mt
        self.mode, self.train = mode, train
        if mode == "rgb224":   # model E: slices as channels (shared geometry)
            self.aug = mt.Compose([
                mt.RandAffine(prob=0.8, rotate_range=0.436, translate_range=(25, 25),
                              padding_mode="zeros"),
                mt.RandFlip(prob=0.5, spatial_axis=1),
                mt.RandGaussianNoise(prob=0.15, std=0.02),
                mt.RandBiasField(prob=0.15),
            ])
        else:                  # model D: channel-first 3D volume
            self.aug = mt.Compose([
                mt.RandAffine(prob=0.8, rotate_range=(0.436, 0.0, 0.0),
                              translate_range=(0, 12, 12), padding_mode="zeros"),
                mt.RandFlip(prob=0.5, spatial_axis=2),
                mt.RandGaussianNoise(prob=0.15, std=0.05),
                mt.RandBiasField(prob=0.15),
            ])

    def __call__(self, vol: np.ndarray) -> torch.Tensor:
        if self.mode == "rgb224":
            x = torch.from_numpy(np.ascontiguousarray(vol)).float().div(255)
            if self.train:
                x = torch.as_tensor(self.aug(x)).clamp(0, 1)
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
            x = TF.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            return TF.center_crop(x, [224, 224])
        # vol3d: z-score over non-zero voxels, then resize to (32, 128, 128)
        v = torch.from_numpy(np.ascontiguousarray(vol)).float()
        nz = v[v > 0]
        v = (v - nz.mean()) / nz.std().clamp(min=1e-6) if nz.numel() else v
        v = F.interpolate(v[None, None], size=(32, 128, 128), mode="trilinear",
                          align_corners=False)[0]
        if self.train:
            v = torch.as_tensor(self.aug(v))
        return v  # (1, 32, 128, 128)


def make_transform(model_id: str, train: bool):
    if model_id == "A":
        return TorchvisionStackTransform("mlp64", train)
    if model_id in ("B", "C"):
        return TorchvisionStackTransform("rgb224", train)
    if model_id == "E":
        return MonaiStackTransform("rgb224", train)
    if model_id == "D":
        return MonaiStackTransform("vol3d", train)
    raise ValueError(model_id)


# dataset

class ExamDataset(Dataset):
    """One item = one examination in one plane (MRNet) or one KneeMRI volume."""

    def __init__(self, folder: Path, ids: list[str], labels: dict[str, np.ndarray],
                 transform):
        self.folder, self.ids, self.labels, self.tf = Path(folder), list(ids), labels, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        eid = self.ids[i]
        vol = np.load(self.folder / f"{eid}.npy")
        return self.tf(vol), torch.from_numpy(self.labels[eid]), eid


def collate_single(batch):
    return batch[0]


# KneeMRI (secondary/external validation)

def load_kneemri_meta(kneemri_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Read metadata.csv with defensive column detection; add id and binary label."""
    meta = pd.read_csv(Path(kneemri_dir) / "metadata.csv")
    cols = {c.lower(): c for c in meta.columns}
    pick = lambda key: next((cols[c] for c in cols if key in c), None)
    names = dict(
        diag=pick("diagnos"), file=pick("filename") or pick("volume"),
        exam=pick("examid"), roi_x=pick("roix"), roi_y=pick("roiy"), roi_z=pick("roiz"),
        roi_w=pick("width"), roi_h=pick("height"), roi_d=pick("depth"),
    )
    if names["diag"] is None or names["file"] is None:
        raise ValueError(f"Could not find diagnosis/filename columns in {list(meta.columns)}")
    meta["id"] = meta[names["file"]].astype(str).str.replace(".pck", "", regex=False)
    meta["acl"] = (meta[names["diag"]] > 0).astype(np.float32)  # injured (partial or full) = 1
    meta["group"] = meta[names["exam"]] if names["exam"] else meta.index
    return meta, names


def load_pck(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        try:
            vol = pickle.load(f)
        except UnicodeDecodeError:    
            f.seek(0)
            vol = pickle.load(f, encoding="latin1")
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim == 3 and np.argmin(vol.shape) != 0:   # slices axis must come first
        vol = np.moveaxis(vol, int(np.argmin(vol.shape)), 0)
    return vol


def to_mrnet_like(vol: np.ndarray) -> np.ndarray:
    """Resize in-plane to 256×256 and rescale 12-bit intensities to uint8 (guide §6.2)."""
    v = torch.from_numpy(vol)[None, None]
    v = F.interpolate(v, size=(vol.shape[0], 256, 256), mode="trilinear",
                      align_corners=False)[0, 0].numpy()
    lo, hi = np.percentile(v, [0.5, 99.5])
    return (np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def convert_kneemri(kneemri_dir: Path, out_dir: Path, meta: pd.DataFrame,
                    file_col: str, progress=True) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    it = meta.iterrows()
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(list(it), desc="Converting KneeMRI")
    n = 0
    for _, row in it:
        dst = out_dir / f"{row['id']}.npy"
        if dst.exists():
            continue
        vol = load_pck(Path(kneemri_dir) / "volumetric_data" / str(row[file_col]))
        np.save(dst, to_mrnet_like(vol))
        n += 1
    return n


# USED FOR TESTING RUNNN
# synthetic smoke data

def make_smoke_dataset(root: Path, n_train: int = 24, n_valid: int = 10, seed: int = 0):
    """Tiny synthetic MRNet-style dataset so the whole pipeline runs before the
    real downloads finish. Positives contain a bright blob the models can find."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    if (root / "train-acl.csv").exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    ids = {"train": [f"{i:04d}" for i in range(n_train)],
           "valid": [f"{i:04d}" for i in range(1130, 1130 + n_valid)]}
    for split, split_ids in ids.items():
        y = {t: rng.integers(0, 2, len(split_ids)) for t in TASKS}
        for t in TASKS:
            pd.DataFrame({"id": split_ids, "y": y[t]}).to_csv(
                root / f"{split}-{t}.csv", header=False, index=False)
        for plane in ("sagittal", "coronal", "axial"):
            d = root / split / plane
            d.mkdir(parents=True, exist_ok=True)
            for j, eid in enumerate(split_ids):
                s = int(rng.integers(18, 34))
                vol = rng.normal(90, 25, (s, 256, 256))
                if y["abnormal"][j] or y["acl"][j] or y["meniscus"][j]:
                    cz, cy, cx = s // 2, 128 + int(rng.integers(-30, 30)), 128 + int(rng.integers(-30, 30))
                    zz, yy, xx = np.ogrid[:s, :256, :256]
                    blob = ((zz - cz) / 4) ** 2 + ((yy - cy) / 28) ** 2 + ((xx - cx) / 28) ** 2 < 1
                    vol[blob] += 120 * (0.5 + 0.5 * float(y["acl"][j]))
                np.save(d / f"{eid}.npy", np.clip(vol, 0, 255).astype(np.uint8))
    return root
