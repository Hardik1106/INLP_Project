"""Device abstraction for Windows/AMD (DirectML) compatibility."""

import torch

_directml_available = False
_dml_device = None

try:
    import torch_directml
    _directml_available = torch_directml.is_available()
    if _directml_available:
        _dml_device = torch_directml.device()
except ImportError:
    pass


def get_device():
    """Return the best available device for inference (DirectML > CUDA > CPU)."""
    if _directml_available:
        return _dml_device
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_device_for_training():
    """Return device for SAE training. Always CPU because torch.vmap/torch.func.grad
    are incompatible with DirectML."""
    return torch.device("cpu")


def empty_cache():
    """Safely clear GPU cache (no-op on DirectML/CPU)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_pin_memory(tensor):
    """Pin memory only when CUDA is available (required for DMA transfers)."""
    if torch.cuda.is_available():
        return tensor.pin_memory()
    return tensor


def get_model_dtype():
    """Return the dtype to use for model loading.
    DirectML doesn't reliably support bfloat16, so use float32."""
    if _directml_available and not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16
