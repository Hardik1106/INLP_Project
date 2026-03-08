"""
Module 2b – Contrastive Analysis
==================================
Performs contrastive filtering to identify candidate reasoning features:
  1. Intra-Dataset Contrast: GSM8K(CoT) - GSM8K(No-CoT)
  2. Inter-Dataset Contrast: GSM8K - TriviaQA
  3. Intersection of the two to find robust reasoning features
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import torch
import numpy as np

from src.config import ContrastiveConfig
from src.utils import logger, save_tensor, save_json, ensure_dir


# ---------------------------------------------------------------------------
# Data structures for results
# ---------------------------------------------------------------------------

@dataclass
class ContrastiveResult:
    """Holds the output of contrastive analysis for one layer."""
    layer: int
    # Raw difference vectors (mean activation differences)
    intra_diff: torch.Tensor         # [d_sae] — CoT minus No-CoT
    inter_diff: torch.Tensor         # [d_sae] — GSM8K minus TriviaQA
    # Candidate feature indices from each contrast
    intra_candidates: List[int] = field(default_factory=list)
    inter_candidates: List[int] = field(default_factory=list)
    # Final intersection of candidate reasoning features
    reasoning_features: List[int] = field(default_factory=list)
    # Statistics
    stats: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Contrastive computation
# ---------------------------------------------------------------------------

def compute_mean_activations(features: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean activation value per SAE feature across all samples.

    Args:
        features: [N, d_sae] sparse feature activations

    Returns:
        mean_act: [d_sae] mean activation per feature
    """
    return features.float().mean(dim=0)


def compute_activation_diff(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean(A) - mean(B) per feature.

    Positive values indicate features more active in A than B.
    """
    mean_a = compute_mean_activations(features_a)
    mean_b = compute_mean_activations(features_b)
    return mean_a - mean_b


def select_candidates_by_threshold(
    diff: torch.Tensor,
    threshold: float,
) -> List[int]:
    """
    Select feature indices where the absolute activation difference
    exceeds a threshold.
    """
    mask = diff.abs() > threshold
    return mask.nonzero(as_tuple=False).squeeze(-1).tolist()


def select_candidates_by_percentile(
    diff: torch.Tensor,
    percentile: float,
) -> List[int]:
    """
    Select feature indices in the top percentile of absolute differences.
    """
    abs_diff = diff.abs()
    cutoff = float(np.percentile(abs_diff.numpy(), percentile))
    mask = abs_diff >= cutoff
    return mask.nonzero(as_tuple=False).squeeze(-1).tolist()


# ---------------------------------------------------------------------------
# Full contrastive pipeline for one layer
# ---------------------------------------------------------------------------

def run_contrastive_for_layer(
    gsm8k_cot_features: torch.Tensor,
    gsm8k_no_cot_features: torch.Tensor,
    triviaqa_features: Optional[torch.Tensor],
    layer: int,
    cfg: ContrastiveConfig,
) -> ContrastiveResult:
    """
    Run the full contrastive analysis for a single layer.

    Steps:
        1. Intra-dataset contrast: CoT - No-CoT  (isolates reasoning state)
        2. Inter-dataset contrast: GSM8K_CoT - TriviaQA  (if TriviaQA available)
        3. Intersect the two candidate sets (or use intra-only if no TriviaQA)

    Args:
        gsm8k_cot_features: [N, d_sae] from GSM8K with CoT prompting
        gsm8k_no_cot_features: [N, d_sae] from GSM8K with direct-answer prompting
        triviaqa_features: [N, d_sae] from TriviaQA (or None if disabled)
        layer: The transformer layer index
        cfg: ContrastiveConfig with thresholds

    Returns:
        ContrastiveResult with all intermediate and final outputs
    """
    logger.info(f"--- Contrastive analysis: Layer {layer} ---")

    # 1. Intra-dataset contrast: CoT minus No-CoT
    intra_diff = compute_activation_diff(gsm8k_cot_features, gsm8k_no_cot_features)
    intra_candidates = select_candidates_by_percentile(
        intra_diff, cfg.top_percentile
    )
    logger.info(
        f"  Intra-dataset (CoT - No-CoT): "
        f"{len(intra_candidates)} candidates above {cfg.top_percentile}th percentile"
    )

    # 2. Inter-dataset contrast (only if TriviaQA is available)
    if triviaqa_features is not None:
        inter_diff = compute_activation_diff(gsm8k_cot_features, triviaqa_features)
        inter_candidates = select_candidates_by_percentile(
            inter_diff, cfg.top_percentile
        )
        logger.info(
            f"  Inter-dataset (GSM8K - TriviaQA): "
            f"{len(inter_candidates)} candidates above {cfg.top_percentile}th percentile"
        )
    else:
        inter_diff = torch.zeros_like(intra_diff)
        inter_candidates = []
        logger.info("  Inter-dataset contrast: skipped (no TriviaQA)")

    # 3. Intersect (or use intra-only when no TriviaQA)
    intra_set = set(intra_candidates)
    inter_set = set(inter_candidates)

    if triviaqa_features is None:
        # No inter-dataset contrast — use intra candidates directly
        reasoning_features = sorted(intra_set)
    elif cfg.intersection_method == "strict":
        reasoning_features = sorted(intra_set & inter_set)
    elif cfg.intersection_method == "union":
        reasoning_features = sorted(intra_set | inter_set)
    else:
        raise ValueError(f"Unknown intersection method: {cfg.intersection_method}")

    logger.info(
        f"  Candidate reasoning features ({cfg.intersection_method} intersection): "
        f"{len(reasoning_features)}"
    )

    # Stats
    stats = {
        "intra_diff_mean": float(intra_diff.abs().mean()),
        "intra_diff_max": float(intra_diff.abs().max()),
        "inter_diff_mean": float(inter_diff.abs().mean()),
        "inter_diff_max": float(inter_diff.abs().max()),
        "n_intra_candidates": len(intra_candidates),
        "n_inter_candidates": len(inter_candidates),
        "n_reasoning_features": len(reasoning_features),
    }

    return ContrastiveResult(
        layer=layer,
        intra_diff=intra_diff,
        inter_diff=inter_diff,
        intra_candidates=intra_candidates,
        inter_candidates=inter_candidates,
        reasoning_features=reasoning_features,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Multi-layer contrastive pipeline
# ---------------------------------------------------------------------------

def run_contrastive_analysis(
    sae_features: Dict[str, Dict[int, torch.Tensor]],
    layers: List[int],
    cfg: ContrastiveConfig,
    output_dir: str = "outputs/contrastive",
) -> Dict[int, ContrastiveResult]:
    """
    Run contrastive analysis across all specified layers.

    Args:
        sae_features: nested dict  dataset_label -> layer -> [N, d_sae]
        layers: list of layer indices to analyze
        cfg: ContrastiveConfig
        output_dir: where to save results

    Returns:
        Dict mapping layer -> ContrastiveResult
    """
    ensure_dir(output_dir)
    results = {}

    for layer in layers:
        cot = sae_features["gsm8k_cot"].get(layer)
        no_cot = sae_features["gsm8k_no_cot"].get(layer)
        trivia = sae_features.get("triviaqa", {}).get(layer)

        if cot is None or no_cot is None:
            logger.warning(f"Skipping layer {layer}: missing GSM8K features")
            continue

        if trivia is None:
            logger.info(f"Layer {layer}: no TriviaQA features, using intra-dataset contrast only")

        result = run_contrastive_for_layer(cot, no_cot, trivia, layer, cfg)
        results[layer] = result

        # Save per-layer outputs
        layer_dir = ensure_dir(f"{output_dir}/layer_{layer}")
        save_tensor(result.intra_diff, f"{layer_dir}/intra_diff.pt")
        save_tensor(result.inter_diff, f"{layer_dir}/inter_diff.pt")
        save_json(result.reasoning_features, f"{layer_dir}/reasoning_features.json")
        save_json(result.stats, f"{layer_dir}/stats.json")

    # Summary
    logger.info("=== Contrastive Analysis Summary ===")
    for layer, res in sorted(results.items()):
        logger.info(
            f"  Layer {layer}: {res.stats['n_reasoning_features']} reasoning features"
        )

    return results


# ---------------------------------------------------------------------------
# Feature ranking utilities (used by intervention module)
# ---------------------------------------------------------------------------

def rank_features_by_diff(
    contrastive_result: ContrastiveResult,
    use_intra: bool = True,
) -> List[Tuple[int, float]]:
    """
    Rank the reasoning features by their absolute activation difference.

    Returns a sorted list of (feature_idx, abs_diff) tuples, descending.
    """
    diff = contrastive_result.intra_diff if use_intra else contrastive_result.inter_diff
    features = contrastive_result.reasoning_features

    ranked = [(idx, float(diff[idx].abs())) for idx in features]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def get_top_k_features(
    contrastive_result: ContrastiveResult,
    k: int,
) -> List[int]:
    """
    Get the K reasoning features with the highest absolute activation differences.
    These are typically shallow formatting tokens (e.g., "Step").
    """
    ranked = rank_features_by_diff(contrastive_result)
    top_k = [idx for idx, _ in ranked[:k]]
    logger.debug(f"Top-{k} features: {top_k[:5]}...")
    return top_k


def get_random_k_features(
    contrastive_result: ContrastiveResult,
    k: int,
    seed: Optional[int] = None,
    exclude_top_n: int = 0,
) -> List[int]:
    """
    Randomly sample K features from the candidate reasoning pool.

    Optionally excludes the top-N highest-diff features to avoid shallow
    formatting features, focusing on distributed reasoning features instead.

    Args:
        contrastive_result: The contrastive result for the layer.
        k: Number of features to sample.
        seed: Random seed for reproducibility.
        exclude_top_n: Exclude the top N highest-difference features.

    Returns:
        List of K randomly sampled feature indices.
    """
    ranked = rank_features_by_diff(contrastive_result)

    # Exclude top-N if requested
    pool = [idx for idx, _ in ranked[exclude_top_n:]]

    if k > len(pool):
        logger.warning(
            f"Requested k={k} but pool only has {len(pool)} features. "
            f"Using all {len(pool)}."
        )
        k = len(pool)

    rng = np.random.RandomState(seed)
    sampled = rng.choice(pool, size=k, replace=False).tolist()
    return sorted(sampled)
