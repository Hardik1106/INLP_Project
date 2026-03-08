"""
Shared utilities: logging, tensor I/O, seeding, device helpers.
"""

import logging
import os
import random
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str = "cot_faithfulness", level: str = "INFO") -> logging.Logger:
    """Create a console + file logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


logger = setup_logger()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device(requested: str = "cuda") -> torch.device:
    """Return a torch device, falling back to CPU when CUDA is unavailable."""
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        if requested == "cuda":
            logger.warning("CUDA requested but not available – falling back to CPU")
        else:
            logger.info("Using device: cpu")
    return device


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype_str, torch.float32)


# ---------------------------------------------------------------------------
# Tensor I/O
# ---------------------------------------------------------------------------

def save_tensor(tensor: torch.Tensor, path: str) -> None:
    """Save a tensor to disk, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.cpu(), p)
    logger.debug(f"Saved tensor {tensor.shape} -> {p}")


def load_tensor(path: str, device: str = "cpu") -> torch.Tensor:
    """Load a tensor from disk."""
    t = torch.load(path, map_location=device, weights_only=True)
    logger.debug(f"Loaded tensor {t.shape} <- {path}")
    return t


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_json(data: Any, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Result aggregation helpers
# ---------------------------------------------------------------------------

def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, min, max for a list of floats."""
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(values),
    }


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
