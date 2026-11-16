"""Central configuration for the knee MRI CNN experiments

Contains mostly paths, tasks, and five (A-E) model configs, including
default hyperparameters and presets
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

TASKS = ["abnormal", "acl", "meniscus"]
PLANES = ["sagittal", "coronal", "axial"]

MODEL_IDS = ["A", "B", "C", "D", "E"]
MODEL_NAMES = {
    "A": "Fully connected baseline",
    "B": "AlexNet slice CNN (pretrained)",
    "C": "ResNet-18 slice CNN (pretrained)",
    "D": "MONAI 3D DenseNet-121",
    "E": "ResNet-18 + MONAI pipeline",
}
# colorblind-safe (Okabe-Ito) palette, fixed per model, never cycled
MODEL_COLORS = {
    "A": "#0072B2",
    "B": "#E69F00",
    "C": "#009E73",
    "D": "#D55E00",
    "E": "#56B4E9",
}

# (defaults) used when hyperparameter search is skipped
DEFAULT_HPARAMS = {
    "A": dict(lr=1e-3, weight_decay=1e-4, dropout=0.5),
    "B": dict(lr=5e-5, weight_decay=1e-4, dropout=0.0),
    "C": dict(lr=5e-5, weight_decay=1e-4, dropout=0.0),
    "D": dict(lr=3e-4, weight_decay=1e-4, dropout=0.0),
    "E": dict(lr=5e-5, weight_decay=1e-4, dropout=0.0),
}

# random-search space
SEARCH_SPACE = dict(
    lr=(1e-5, 1e-3),    # log-uniform
    weight_decay=[0.0, 1e-4, 1e-3],
    dropout=[0.0, 0.3, 0.5],
)

# planned pairwise contrasts per task: six tests per task
CONTRASTS = [("B", "A"), ("C", "A"), ("C", "B"), ("E", "C"), ("D", "E"), ("D", "C")]


@dataclass
class Config:
    mrnet_dir: Path = REPO_ROOT / "data" / "MRNet-v1.0"
    kneemri_dir: Path = REPO_ROOT / "data" / "kneemri"          
    kneemri_npy_dir: Path = REPO_ROOT / "data" / "kneemri_npy" 
    smoke_dir: Path = REPO_ROOT / "data" / "smoke_mrnet"       

    outputs_dir: Path = REPO_ROOT / "outputs"
    splits_dir: Path = REPO_ROOT / "splits"

    # experiment switches
    smoke_test: bool = False   
    seeds: tuple = (0,)        
    n_tuning: int = 120   # for fine tuning

    # training protocol
    max_epochs: int = 15       # originally 40, but time constaint now! :(
    patience: int = 4          
    accum_exams: int = 8      
    batch_3d: int = 4         
    num_workers: int = 2
    use_amp: bool = True      # mixed precision

    def __post_init__(self):
        for f_ in ("mrnet_dir", "kneemri_dir", "kneemri_npy_dir", "smoke_dir",
                   "outputs_dir", "splits_dir"):
            setattr(self, f_, Path(getattr(self, f_)))
        if self.smoke_test:
            self.outputs_dir = REPO_ROOT / "outputs_smoke"
            self.splits_dir = REPO_ROOT / "splits_smoke"
            self.kneemri_npy_dir = REPO_ROOT / "data" / "smoke_kneemri_npy"

    # locations for output
    @property
    def runs_dir(self) -> Path: return self.outputs_dir / "runs"
    @property
    def figures_dir(self) -> Path: return self.outputs_dir / "figures"
    @property
    def tables_dir(self) -> Path: return self.outputs_dir / "tables"

    @property
    def data_root(self) -> Path:
        """Root of the MRNet-style dataset in use (real MRNet or the smoke set)."""
        return self.smoke_dir if self.smoke_test else self.mrnet_dir

    @property
    def tasks(self) -> list:
        return list(TASKS)

    def make_dirs(self):
        for d in (self.outputs_dir, self.splits_dir, self.runs_dir,
                  self.figures_dir, self.tables_dir, self.figures_dir / "gradcam"):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def save(self, path: Path | None = None):
        path = path or self.outputs_dir / "config.json"
        d = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
        path.write_text(json.dumps(d, indent=2))
        return path


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def amp_context(device: torch.device, enabled: bool = True):
    """Autocast context that degrades gracefully on backends without AMP."""
    import contextlib
    if not enabled:
        return contextlib.nullcontext()
    try:
        if hasattr(torch.amp, "is_autocast_available") and \
                not torch.amp.is_autocast_available(device.type):
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    except Exception:
        return contextlib.nullcontext()


def env_report() -> str:
    """One-paragraph environment summary to paste into thesis §3.11."""
    import platform, subprocess
    import monai, torchvision, sklearn, statsmodels, numpy, pandas
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        chip = platform.processor()
    dev = get_device()
    lines = [
        f"Machine         : {chip}, {platform.system()} {platform.release()}",
        f"Accelerator     : {dev.type.upper()} "
        + ("(Apple Metal Performance Shaders)" if dev.type == "mps" else ""),
        f"Python          : {platform.python_version()}",
        f"PyTorch         : {torch.__version__}",
        f"torchvision     : {torchvision.__version__}",
        f"MONAI           : {monai.__version__}",
        f"scikit-learn    : {sklearn.__version__}",
        f"statsmodels     : {statsmodels.__version__}",
        f"NumPy / pandas  : {numpy.__version__} / {pandas.__version__}",
    ]
    return "\n".join(lines)
