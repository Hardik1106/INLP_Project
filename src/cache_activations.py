"""
Module 1b – Activation Caching
===============================
Runs language model forward passes via TransformerLens and caches
hidden-state activations at specified transformer layers to disk.
"""

from typing import List, Dict, Optional
from pathlib import Path

import torch
from tqdm import tqdm
import transformer_lens as tl

from src.config import ExperimentConfig
from src.data_pipeline import PromptSample
from src.utils import logger, save_tensor, get_device, get_torch_dtype, ensure_dir


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_hooked_model(
    model_name: str,
    device: str = "cuda",
    dtype: str = "float32",
) -> tl.HookedTransformer:
    """
    Load a TransformerLens HookedTransformer.

    TransformerLens wraps HuggingFace models and exposes clean hook points
    at every layer for activation interception.
    """
    logger.info(f"Loading HookedTransformer: {model_name}")
    torch_dtype = get_torch_dtype(dtype)

    try:
        model = tl.HookedTransformer.from_pretrained(
            model_name,
            device=device,
        )
    except AttributeError as exc:
        # Compatibility shim for newer transformers versions where
        # GPTNeoXConfig may not expose rotary_pct (needed by some TL versions).
        if "rotary_pct" not in str(exc):
            raise

        logger.warning(
            "TransformerLens compatibility fallback: missing GPTNeoXConfig.rotary_pct; "
            "injecting default rotary_pct=0.25 and retrying model load."
        )
        try:
            from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
        except Exception:
            raise

        if not hasattr(GPTNeoXConfig, "rotary_pct"):
            GPTNeoXConfig.rotary_pct = 0.25

        model = tl.HookedTransformer.from_pretrained(
            model_name,
            device=device,
        )

    model.eval()
    logger.info(
        f"  Loaded model with {model.cfg.n_layers} layers, "
        f"d_model={model.cfg.d_model}, device={device}"
    )
    return model


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def get_hook_names_for_layers(layers: List[int]) -> List[str]:
    """
    Return TransformerLens hook names for the residual stream
    at the given layer indices.

    TransformerLens uses names like 'blocks.{layer}.hook_resid_post'
    for the residual stream output of each transformer block.
    """
    return [f"blocks.{layer}.hook_resid_post" for layer in layers]


@torch.no_grad()
def extract_activations_single(
    model: tl.HookedTransformer,
    text: str,
    layers: List[int],
    max_seq_len: int = 512,
) -> Dict[int, torch.Tensor]:
    """
    Run a single forward pass and return residual stream activations
    at the specified layers.

    Returns:
        Dict mapping layer_idx -> tensor of shape [seq_len, d_model]
    """
    # Tokenize with truncation
    tokens = model.to_tokens(text, prepend_bos=True)
    if tokens.shape[1] > max_seq_len:
        tokens = tokens[:, :max_seq_len]

    hook_names = get_hook_names_for_layers(layers)

    # Run with cache — TransformerLens returns activations at all hook points
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: name in hook_names,
    )

    result = {}
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        # cache[hook_name] has shape [batch=1, seq_len, d_model]
        result[layer] = cache[hook_name][0].cpu()  # [seq_len, d_model]

    return result


# @torch.no_grad()
# def cache_activations_for_dataset(
#     model: tl.HookedTransformer,
#     samples: List[PromptSample],
#     layers: List[int],
#     cache_dir: str,
#     dataset_label: str,
#     max_seq_len: int = 512,
#     aggregate: str = "mean",
# ) -> Dict[int, torch.Tensor]:
#     """
#     Run forward passes for all samples and cache per-layer activations.

#     For contrastive analysis we need a single representative activation
#     vector per sample per layer. We aggregate over the sequence dimension.

#     Args:
#         model: The HookedTransformer model.
#         samples: List of PromptSample objects.
#         layers: Which layers to extract.
#         cache_dir: Root directory to save tensors.
#         dataset_label: e.g. "gsm8k_cot", "gsm8k_no_cot", "triviaqa".
#         max_seq_len: Maximum sequence length for truncation.
#         aggregate: "mean" (average over seq) or "last" (last token).

#     Returns:
#         Dict mapping layer_idx -> stacked tensor of shape [num_samples, d_model]
#     """
#     out_dir = ensure_dir(f"{cache_dir}/{dataset_label}")
#     logger.info(
#         f"Caching activations for '{dataset_label}' "
#         f"({len(samples)} samples, layers={layers})"
#     )

#     # Accumulators: layer -> list of [d_model] vectors
#     layer_accum: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}

#     for sample in tqdm(samples, desc=f"Cache [{dataset_label}]"):
#         acts = extract_activations_single(
#             model, sample.prompt_text, layers, max_seq_len
#         )
#         for layer, tensor in acts.items():
#             # tensor shape: [seq_len, d_model]
#             if aggregate == "mean":
#                 vec = tensor.mean(dim=0)       # [d_model]
#             elif aggregate == "last":
#                 vec = tensor[-1]               # [d_model]
#             else:
#                 raise ValueError(f"Unknown aggregate method: {aggregate}")
#             layer_accum[layer].append(vec)

#     # Stack and save
#     result = {}
#     for layer in layers:
#         stacked = torch.stack(layer_accum[layer], dim=0)  # [N, d_model]
#         save_path = f"{out_dir}/layer_{layer}.pt"
#         save_tensor(stacked, save_path)
#         result[layer] = stacked
#         logger.info(
#             f"  Layer {layer}: saved {stacked.shape} -> {save_path}"
#         )

#     return result


@torch.no_grad()
def cache_activations_for_dataset(
    model: tl.HookedTransformer,
    samples: List[PromptSample],
    layers: List[int],
    cache_dir: str,
    dataset_label: str,
    max_seq_len: int = 512,
    aggregate: str = "mean",
    batch_size: int = 8,  # NEW: process multiple samples at once
) -> Dict[int, torch.Tensor]:
    out_dir = ensure_dir(f"{cache_dir}/{dataset_label}")
    logger.info(
        f"Caching activations for '{dataset_label}' "
        f"({len(samples)} samples, layers={layers}, batch_size={batch_size})"
    )

    layer_accum: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    hook_names = get_hook_names_for_layers(layers)

    for batch_start in tqdm(range(0, len(samples), batch_size), desc=f"Cache [{dataset_label}]"):
        batch_samples = samples[batch_start : batch_start + batch_size]

        # Tokenize and pad batch
        batch_tokens = [
            model.to_tokens(s.prompt_text, prepend_bos=True)[0] for s in batch_samples
        ]
        # Truncate
        batch_tokens = [t[:max_seq_len] for t in batch_tokens]
        # Pad to same length
        max_len = max(t.shape[0] for t in batch_tokens)
        padded = torch.zeros(len(batch_tokens), max_len, dtype=torch.long, device=model.cfg.device)
        attention_mask = torch.zeros(len(batch_tokens), max_len, dtype=torch.bool, device=model.cfg.device)
        for i, t in enumerate(batch_tokens):
            padded[i, :t.shape[0]] = t
            attention_mask[i, :t.shape[0]] = True

        _, cache = model.run_with_cache(
            padded,
            names_filter=lambda name: name in hook_names,
        )

        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            acts = cache[hook_name]  # [batch, seq_len, d_model]
            for i in range(len(batch_samples)):
                # Only consider non-padded positions
                valid_len = batch_tokens[i].shape[0]
                act = acts[i, :valid_len, :]  # [valid_seq_len, d_model]
                if aggregate == "mean":
                    vec = act.mean(dim=0).cpu()
                elif aggregate == "last":
                    vec = act[-1].cpu()
                else:
                    raise ValueError(f"Unknown aggregate: {aggregate}")
                layer_accum[layer].append(vec)

        del cache
        torch.cuda.empty_cache()

    # Stack and save
    result = {}
    for layer in layers:
        stacked = torch.stack(layer_accum[layer], dim=0)
        save_path = f"{out_dir}/layer_{layer}.pt"
        save_tensor(stacked, save_path)
        result[layer] = stacked
        logger.info(f"  Layer {layer}: saved {stacked.shape} -> {save_path}")

    return result


# ---------------------------------------------------------------------------
# Full caching pipeline
# ---------------------------------------------------------------------------

# def run_activation_caching(
#     config: ExperimentConfig,
#     prompt_sets: Dict[str, List[PromptSample]],
#     model: Optional[tl.HookedTransformer] = None,
# ) -> Dict[str, Dict[int, torch.Tensor]]:
#     """
#     Cache activations for all prompt sets across all configured layers.

#     Returns:
#         Nested dict:  dataset_label -> layer_idx -> [N, d_model] tensor
#     """
#     if model is None:
#         model = load_hooked_model(
#             config.model.transformer_lens_name,
#             config.model.device,
#             config.model.dtype,
#         )

#     all_layers = config.layers.all_layers
#     results = {}

#     for label, samples in prompt_sets.items():
#         results[label] = cache_activations_for_dataset(
#             model=model,
#             samples=samples,
#             layers=all_layers,
#             cache_dir=config.data.cache_dir,
#             dataset_label=label,
#             max_seq_len=config.data.max_seq_len,
#         )

#     return results


def run_activation_caching(
    config: ExperimentConfig,
    prompt_sets: Dict[str, List[PromptSample]],
    model: Optional[tl.HookedTransformer] = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    if model is None:
        model = load_hooked_model(
            config.model.transformer_lens_name,
            config.model.device,
            config.model.dtype,
        )

    all_layers = config.layers.all_layers
    results = {}

    # Determine batch size based on available VRAM
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    # batch_size = 16 if vram_gb > 20 else (8 if vram_gb > 10 else 4)
    batch_size = config.performance.activation_cache_batch_size
    logger.info(f"Auto batch size: {batch_size} (VRAM: {vram_gb:.1f} GB)")

    for label, samples in prompt_sets.items():
        results[label] = cache_activations_for_dataset(
            model=model,
            samples=samples,
            layers=all_layers,
            cache_dir=config.data.cache_dir,
            dataset_label=label,
            max_seq_len=config.data.max_seq_len,
            batch_size=batch_size,  # pass auto batch size
        )

    return results