"""
Module 3 – Causal Intervention Engine
=======================================
Intercepts the language model during text generation to ablate or amplify
targeted SAE features at specific transformer layers.

Uses TransformerLens hook functions to modify residual stream activations
during the forward pass.
"""

from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import partial

import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformer_lens as tl
import time

from src.sae_encoder import SparseAutoencoder, load_sae_from_saelens
from src.contrastive_analysis import ContrastiveResult, get_top_k_features, get_random_k_features
from src.data_pipeline import PromptSample
from src.config import InterventionConfig, SAEConfig
from src.utils import logger, save_json, ensure_dir


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------

class InterventionType(Enum):
    ABLATION = "ablation"
    AMPLIFICATION = "amplification"


class SamplingStrategy(Enum):
    TOP_K = "top_k"
    RANDOM_K = "random_k"


@dataclass
class InterventionSpec:
    """Specification for a single intervention experiment."""
    intervention_type: InterventionType
    sampling_strategy: SamplingStrategy
    layer: int
    feature_indices: List[int]    # Which SAE features to intervene on
    k: int                        # How many features
    ablation_value: float = 0.0   # Value for ablation (typically 0)
    amplification_scale: float = 1.0  # Multiplier for amplification
    random_seed: Optional[int] = None
    label: str = ""               # Human-readable label for logging


@dataclass
class InterventionResult:
    """Output of running one intervention on one sample."""
    sample_idx: int
    spec_label: str
    original_prompt: str
    generated_text: str
    ground_truth: str
    intervention_type: str
    sampling_strategy: str
    layer: int
    k: int
    amplification_scale: float = 1.0
    random_seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------

def make_ablation_hook(
    sae: SparseAutoencoder,
    feature_indices: List[int],
    ablation_value: float = 0.0,
) -> Callable:
    """
    Create a TransformerLens hook that ablates specific SAE features.

    The hook:
      1. Encodes the residual stream through the SAE to get sparse features
      2. Sets the targeted features to ablation_value (typically 0)
      3. Decodes back to the residual stream space
      4. Replaces the original activation with the modified reconstruction

    This effectively removes the contribution of those features.
    """
    feature_mask = torch.zeros(sae.d_sae, device=sae.W_enc.device)
    feature_mask[feature_indices] = 1.0

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        # activation: [batch, seq_len, d_model]
        original_shape = activation.shape
        flat = activation.reshape(-1, sae.d_model)

        # Encode to sparse features
        z = sae.encode(flat.to(sae.W_enc.device))

        # Ablate: set targeted features to ablation_value
        z = z * (1 - feature_mask) + ablation_value * feature_mask

        # Decode back
        reconstructed = sae.decode(z)

        return reconstructed.reshape(original_shape).to(activation.device)

    return hook_fn


def make_amplification_hook(
    sae: SparseAutoencoder,
    feature_indices: List[int],
    scale: float = 2.0,
) -> Callable:
    """
    Create a TransformerLens hook that amplifies specific SAE features.

    The hook:
      1. Encodes the residual stream through the SAE
      2. Multiplies the targeted features by the amplification scale
      3. Decodes back and replaces the activation

    This artificially increases the contribution of those features,
    acting as a steering vector in the SAE feature space.
    """
    feature_mask = torch.zeros(sae.d_sae, device=sae.W_enc.device)
    feature_mask[feature_indices] = 1.0

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        original_shape = activation.shape
        flat = activation.reshape(-1, sae.d_model)

        # Encode
        z = sae.encode(flat.to(sae.W_enc.device))

        # Amplify targeted features
        amplified_z = z * (1 + (scale - 1) * feature_mask)

        # Decode
        reconstructed = sae.decode(amplified_z)

        return reconstructed.reshape(original_shape).to(activation.device)

    return hook_fn


def make_steering_vector_hook(
    sae: SparseAutoencoder,
    feature_indices: List[int],
    scale: float = 1.0,
) -> Callable:
    """
    Alternative amplification approach: add a steering vector derived
    from the SAE decoder directions of the target features.

    Instead of encode -> modify -> decode, this directly adds the
    decoder directions (scaled) to the residual stream.
    """
    # Build steering vector as weighted sum of decoder directions
    steering_vec = torch.zeros(sae.d_model, device=sae.W_dec.device)
    for idx in feature_indices:
        steering_vec += sae.W_dec[idx]
    steering_vec = steering_vec * scale / max(len(feature_indices), 1)

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        return activation + steering_vec.to(activation.device)

    return hook_fn


# ---------------------------------------------------------------------------
# Generation with interventions
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_intervention(
    model: tl.HookedTransformer,
    prompt: str,
    hook_fn: Callable,
    hook_point: str,
    max_new_tokens: int = 256,
) -> str:
    """
    Generate text from a prompt with a hook active at a specific layer.

    Uses TransformerLens's hook mechanism to intercept activations during
    every forward pass of autoregressive generation.

    Args:
        model: The HookedTransformer.
        prompt: Input prompt string.
        hook_fn: The hook function (ablation or amplification).
        hook_point: TransformerLens hook point name (e.g., "blocks.12.hook_resid_post").
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The full generated text (prompt + completion).
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)

    # TransformerLens generate with hooks
    # We use run_with_hooks for each forward step via the generate method
    with model.hooks(fwd_hooks=[(hook_point, hook_fn)]):
        output_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # Greedy decoding for reproducibility
            temperature=1.0,
            top_k=0,
            top_p=1.0,
        )

    generated_text = model.to_string(output_tokens[0])
    return generated_text


@torch.no_grad()
def generate_baseline(
    model: tl.HookedTransformer,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate text without any intervention (baseline)."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    output_tokens = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )
    return model.to_string(output_tokens[0])


# ---------------------------------------------------------------------------
# Intervention experiment runner
# ---------------------------------------------------------------------------

def build_intervention_specs(
    contrastive_results: Dict[int, ContrastiveResult],
    intervention_cfg: InterventionConfig,
    layers: List[int],
) -> List[InterventionSpec]:
    """
    Build the full list of InterventionSpec objects for all combinations of:
      - layers x k_values x {ablation, amplification} x {top_k, random_k}
      - For amplification: also sweep amplification_scales
      - For random_k: also sweep random seeds (num_random_trials)
    """
    specs = []

    for layer in layers:
        if layer not in contrastive_results:
            continue
        cr = contrastive_results[layer]

        for k in intervention_cfg.k_values:
            if k > len(cr.reasoning_features):
                logger.warning(
                    f"Layer {layer}: k={k} > {len(cr.reasoning_features)} "
                    f"reasoning features. Skipping."
                )
                continue

            # --- Top-K features ---
            top_k_indices = get_top_k_features(cr, k)

            # Ablation + Top-K
            specs.append(InterventionSpec(
                intervention_type=InterventionType.ABLATION,
                sampling_strategy=SamplingStrategy.TOP_K,
                layer=layer,
                feature_indices=top_k_indices,
                k=k,
                ablation_value=intervention_cfg.ablation_value,
                label=f"ablation_topk_L{layer}_k{k}",
            ))

            # Amplification + Top-K (sweep scales)
            for scale in intervention_cfg.amplification_scales:
                specs.append(InterventionSpec(
                    intervention_type=InterventionType.AMPLIFICATION,
                    sampling_strategy=SamplingStrategy.TOP_K,
                    layer=layer,
                    feature_indices=top_k_indices,
                    k=k,
                    amplification_scale=scale,
                    label=f"amplify_topk_L{layer}_k{k}_s{scale}",
                ))

            # --- Random-K features (multiple trials) ---
            for trial in range(intervention_cfg.num_random_trials):
                seed = 42 + trial
                random_k_indices = get_random_k_features(cr, k, seed=seed)

                # Ablation + Random-K
                specs.append(InterventionSpec(
                    intervention_type=InterventionType.ABLATION,
                    sampling_strategy=SamplingStrategy.RANDOM_K,
                    layer=layer,
                    feature_indices=random_k_indices,
                    k=k,
                    ablation_value=intervention_cfg.ablation_value,
                    random_seed=seed,
                    label=f"ablation_randomk_L{layer}_k{k}_s{seed}",
                ))

                # Amplification + Random-K
                for scale in intervention_cfg.amplification_scales:
                    specs.append(InterventionSpec(
                        intervention_type=InterventionType.AMPLIFICATION,
                        sampling_strategy=SamplingStrategy.RANDOM_K,
                        layer=layer,
                        feature_indices=random_k_indices,
                        k=k,
                        amplification_scale=scale,
                        random_seed=seed,
                        label=f"amplify_randomk_L{layer}_k{k}_s{scale}_seed{seed}",
                    ))

    logger.info(f"Built {len(specs)} intervention specs")
    return specs


def run_single_intervention(
    model: tl.HookedTransformer,
    sae: SparseAutoencoder,
    sample: PromptSample,
    spec: InterventionSpec,
    max_new_tokens: int = 256,
) -> InterventionResult:
    """
    Run a single intervention on a single sample.
    """
    hook_point = f"blocks.{spec.layer}.hook_resid_post"

    # Build the hook
    if spec.intervention_type == InterventionType.ABLATION:
        hook_fn = make_ablation_hook(sae, spec.feature_indices, spec.ablation_value)
    elif spec.intervention_type == InterventionType.AMPLIFICATION:
        hook_fn = make_amplification_hook(sae, spec.feature_indices, spec.amplification_scale)
    else:
        raise ValueError(f"Unknown intervention type: {spec.intervention_type}")

    # Generate
    generated = generate_with_intervention(
        model, sample.prompt_text, hook_fn, hook_point, max_new_tokens
    )

    return InterventionResult(
        sample_idx=sample.original_idx,
        spec_label=spec.label,
        original_prompt=sample.prompt_text,
        generated_text=generated,
        ground_truth=sample.ground_truth,
        intervention_type=spec.intervention_type.value,
        sampling_strategy=spec.sampling_strategy.value,
        layer=spec.layer,
        k=spec.k,
        amplification_scale=spec.amplification_scale,
        random_seed=spec.random_seed,
    )


# def run_intervention_experiment(
#     model: tl.HookedTransformer,
#     sae_config: SAEConfig,
#     samples: List[PromptSample],
#     specs: List[InterventionSpec],
#     max_new_tokens: int = 256,
#     output_dir: str = "outputs/interventions",
#     device: str = "cuda",
# ) -> List[InterventionResult]:
#     """
#     Run all intervention experiments: for each spec, run on all samples.

#     SAEs are loaded per-layer to save memory.
#     """
#     ensure_dir(output_dir)

#     # Group specs by layer so we only load each SAE once
#     specs_by_layer: Dict[int, List[InterventionSpec]] = {}
#     for spec in specs:
#         specs_by_layer.setdefault(spec.layer, []).append(spec)

#     all_results: List[InterventionResult] = []

#     for layer, layer_specs in sorted(specs_by_layer.items()):
#         logger.info(f"=== Layer {layer}: {len(layer_specs)} specs ===")

#         sae = load_sae_from_saelens(
#             release=sae_config.release,
#             sae_id=sae_config.sae_id,
#             layer=layer,
#             device=device,
#         )

#         for spec in tqdm(layer_specs, desc=f"Layer {layer} interventions"):
#             for sample in samples:
#                 result = run_single_intervention(
#                     model, sae, sample, spec, max_new_tokens
#                 )
#                 all_results.append(result)

#             # Save intermediate results per spec
#             spec_results = [r for r in all_results if r.spec_label == spec.label]
#             save_json(
#                 [vars(r) for r in spec_results],
#                 f"{output_dir}/{spec.label}.json",
#             )

#         del sae
#         torch.cuda.empty_cache()

#     logger.info(f"Completed {len(all_results)} total intervention runs")
#     return all_results

def run_intervention_experiment(
    model: tl.HookedTransformer,
    sae_config: SAEConfig,
    samples: List[PromptSample],
    specs: List[InterventionSpec],
    max_new_tokens: int = 256,
    output_dir: str = "outputs/interventions",
    device: str = "cuda",
    batch_size: int = 16,  # NEW: batch size parameter
) -> List[InterventionResult]:
    """
    Run all intervention experiments: for each spec, run on all samples.

    SAEs are loaded per-layer to save memory.
    """
    ensure_dir(output_dir)

    specs_by_layer: Dict[int, List[InterventionSpec]] = {}
    for spec in specs:
        specs_by_layer.setdefault(spec.layer, []).append(spec)

    all_results: List[InterventionResult] = []

    total_steps = len(specs) * len(samples)
    step_count = 0
    start_time = time.time()

    with tqdm(total=total_steps, desc="Total experiment progress") as pbar:
        for layer, layer_specs in sorted(specs_by_layer.items()):
            logger.info(f"=== Layer {layer}: {len(layer_specs)} specs ===")

            try:
                sae = load_sae_from_saelens(
                    release=sae_config.release,
                    sae_id=sae_config.sae_id,
                    layer=layer,
                    device=device,
                )
            except (ValueError, FileNotFoundError, KeyError) as e:
                msg = str(e)
                logger.warning(
                    "Skipping intervention stage because SAE release '%s' could not be loaded. "
                    "Error: %s\n"
                    "Baseline outputs are still valid.",
                    sae_config.release,
                    msg,
                )
                logger.info(
                    "To enable interventions: (1) use a release available in sae_lens, "
                    "(2) update sae_lens with upgraded pretrained_saes.yaml, or "
                    "(3) download SAE weights manually."
                )
                return all_results
            except Exception as e:
                logger.warning(
                    "Unexpected error loading SAE for interventions: %s. "
                    "Skipping intervention stage but keeping baseline outputs.",
                    str(e),
                )
                return all_results

            for spec in layer_specs:
                subdir = f"{output_dir}/{spec.intervention_type.value}/{spec.sampling_strategy.value}"
                ensure_dir(subdir)
                for batch_start in range(0, len(samples), batch_size):
                    batch_samples = samples[batch_start : batch_start + batch_size]
                    batch_prompts = [s.prompt_text for s in batch_samples]
                    batch_tokens = model.to_tokens(batch_prompts, prepend_bos=True)

                    hook_point = f"blocks.{spec.layer}.hook_resid_post"
                    # Build the hook
                    if spec.intervention_type == InterventionType.ABLATION:
                        hook_fn = make_ablation_hook(sae, spec.feature_indices, spec.ablation_value)
                    elif spec.intervention_type == InterventionType.AMPLIFICATION:
                        hook_fn = make_amplification_hook(sae, spec.feature_indices, spec.amplification_scale)
                    else:
                        raise ValueError(f"Unknown intervention type: {spec.intervention_type}")

                    prompt_len = batch_tokens.shape[1]  # NEW: track prompt length

                    with model.hooks(fwd_hooks=[(hook_point, hook_fn)]):
                        output_tokens = model.generate(
                            batch_tokens,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=1.0,
                            stop_at_eos=True,                        # NEW: stop at eos
                            eos_token_id=model.tokenizer.eos_token_id,
                        )

                    for i, sample in enumerate(batch_samples):
                        # NEW: only decode completion, not prompt
                        completion_tokens = output_tokens[i][prompt_len:]
                        text = model.to_string(completion_tokens)
                        result = InterventionResult(
                            sample_idx=sample.original_idx,
                            spec_label=spec.label,
                            original_prompt=sample.prompt_text,
                            generated_text=text,
                            ground_truth=sample.ground_truth,
                            intervention_type=spec.intervention_type.value,
                            sampling_strategy=spec.sampling_strategy.value,
                            layer=spec.layer,
                            k=spec.k,
                            amplification_scale=spec.amplification_scale,
                            random_seed=spec.random_seed,
                        )
                        all_results.append(result)
                        step_count += 1
                        pbar.update(1)
                        elapsed = time.time() - start_time
                        avg_time = elapsed / step_count if step_count > 0 else 0
                        remaining = avg_time * (total_steps - step_count)
                        pbar.set_postfix({
                            "ETA": f"{remaining/60:.1f} min"
                        })

                # Save intermediate results per spec
                spec_results = [r for r in all_results if r.spec_label == spec.label]
                save_json(
                    [vars(r) for r in spec_results],
                    f"{subdir}/{spec.label}.json",
                )

            del sae
            torch.cuda.empty_cache()

    logger.info(f"Completed {len(all_results)} total intervention runs")
    return all_results

# ---------------------------------------------------------------------------
# Baseline generation (no intervention)
# ---------------------------------------------------------------------------

# def run_baseline_generation(
#     model: tl.HookedTransformer,
#     samples: List[PromptSample],
#     max_new_tokens: int = 256,
#     output_dir: str = "outputs/baseline",
# ) -> List[Dict]:
#     """Generate baseline (no intervention) outputs for comparison."""
#     ensure_dir(output_dir)
#     results = []

#     for sample in tqdm(samples, desc="Baseline generation"):
#         text = generate_baseline(model, sample.prompt_text, max_new_tokens)
#         results.append({
#             "sample_idx": sample.original_idx,
#             "prompt": sample.prompt_text,
#             "generated_text": text,
#             "ground_truth": sample.ground_truth,
#         })

#     save_json(results, f"{output_dir}/baseline_outputs.json")
#     logger.info(f"Generated {len(results)} baseline outputs")
#     return results


def run_baseline_generation(
    model: tl.HookedTransformer,
    samples: List[PromptSample],
    max_new_tokens: int = 256,
    output_dir: str = "outputs/baseline",
    batch_size: int = 16,
) -> List[Dict]:
    """Generate baseline (no intervention) outputs for comparison."""
    ensure_dir(output_dir)
    results = []

    for batch_start in tqdm(range(0, len(samples), batch_size), desc="Baseline generation"):
        batch_samples = samples[batch_start : batch_start + batch_size]
        batch_prompts = [s.prompt_text for s in batch_samples]
        batch_tokens = model.to_tokens(batch_prompts, prepend_bos=True)  # [batch, seq_len]
        prompt_len = batch_tokens.shape[1]  # length of prompt tokens

        # Generate for all prompts in batch
        output_tokens = model.generate(
            batch_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            stop_at_eos=True,                        # stop at <|endoftext|>
            eos_token_id=model.tokenizer.eos_token_id,
        )  # [batch, prompt_len + max_new_tokens]

        for i, sample in enumerate(batch_samples):
            # Only decode the newly generated tokens, NOT the prompt
            completion_tokens = output_tokens[i][prompt_len:]
            text = model.to_string(completion_tokens)
            results.append({
                "sample_idx": sample.original_idx,
                "prompt": sample.prompt_text,
                "generated_text": text,
                "ground_truth": sample.ground_truth,
            })

    save_json(results, f"{output_dir}/baseline_outputs.json")
    logger.info(f"Generated {len(results)} baseline outputs")
    return results
