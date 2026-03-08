"""
Module 4c – Language Fluency Evaluation (Perplexity)
=====================================================
Measures the perplexity of generated text using the base language model
to verify that interventions did not destroy surface-level language quality.
"""

from typing import List, Dict, Optional
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.interventions import InterventionResult
from src.config import EvaluationConfig
from src.utils import logger, compute_stats


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model,
    text: str,
    stride: int = 256,
    max_length: Optional[int] = None,
) -> float:
    """
    Compute the perplexity of a text string using a TransformerLens model.

    Uses a sliding window approach for longer texts.

    Args:
        model: A TransformerLens HookedTransformer (or HuggingFace model).
        text: The text to evaluate.
        stride: Sliding window stride.
        max_length: Maximum context length (defaults to model's context size).

    Returns:
        Perplexity (float). Lower = more fluent according to the model.
    """
    tokens = model.to_tokens(text, prepend_bos=True)
    seq_len = tokens.shape[1]

    if max_length is None:
        max_length = getattr(model.cfg, "n_ctx", 2048)

    nlls = []
    prev_end = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        target_len = end_loc - prev_end

        input_ids = tokens[:, begin_loc:end_loc]

        # Get logits from the model
        logits = model(input_ids)  # [batch, seq, vocab]

        # Compute cross-entropy loss for the target portion only
        # Shift: logits[:-1] predicts tokens[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Only consider the new tokens (not the overlapping context)
        if begin_loc > 0:
            # Skip the overlapping portion
            offset = max_length - target_len
            shift_logits = shift_logits[:, offset:, :]
            shift_labels = shift_labels[:, offset:]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="mean",
        )
        nlls.append(loss.item() * shift_labels.numel())

        prev_end = end_loc
        if end_loc >= seq_len:
            break

    total_nll = sum(nlls)
    total_tokens = seq_len - 1  # Exclude BOS
    if total_tokens <= 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)

    return perplexity


@torch.no_grad()
def compute_perplexity_simple(
    model,
    text: str,
) -> float:
    """
    Simplified perplexity computation for shorter texts that fit in context.

    Args:
        model: TransformerLens HookedTransformer
        text: Input text

    Returns:
        Perplexity (float)
    """
    tokens = model.to_tokens(text, prepend_bos=True)
    seq_len = tokens.shape[1]

    if seq_len <= 1:
        return float("inf")

    logits = model(tokens)  # [1, seq_len, vocab]

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="mean",
    )

    return math.exp(loss.item())


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_fluency(
    results: List[InterventionResult],
    model,
    eval_cfg: EvaluationConfig,
) -> Dict[str, Dict]:
    """
    Evaluate language fluency (perplexity) for all intervention results.

    Groups by spec_label and computes per-spec perplexity stats.

    Args:
        results: List of InterventionResult objects
        model: The base language model (for perplexity computation)
        eval_cfg: Evaluation config

    Returns:
        Dict mapping spec_label -> {
            "mean_perplexity": float,
            "perplexity_stats": {...},
            "details": [...]
        }
    """
    # Group by spec
    by_spec: Dict[str, List[InterventionResult]] = {}
    for r in results:
        by_spec.setdefault(r.spec_label, []).append(r)

    eval_results = {}

    for label, spec_results in sorted(by_spec.items()):
        perplexities = []
        details = []

        for r in tqdm(spec_results, desc=f"Fluency [{label}]", leave=False):
            try:
                ppl = compute_perplexity_simple(model, r.generated_text)
                if math.isinf(ppl) or math.isnan(ppl):
                    ppl = -1.0  # Sentinel for invalid
            except Exception as e:
                logger.warning(f"Perplexity failed for sample {r.sample_idx}: {e}")
                ppl = -1.0

            perplexities.append(ppl)
            details.append({
                "sample_idx": r.sample_idx,
                "perplexity": ppl,
            })

        valid_ppls = [p for p in perplexities if p > 0]
        stats = compute_stats(valid_ppls) if valid_ppls else {"mean": -1}

        eval_results[label] = {
            "mean_perplexity": stats["mean"],
            "perplexity_stats": stats,
            "details": details,
        }

        logger.info(
            f"  [{label}] Perplexity: mean={stats['mean']:.2f} "
            f"(n={len(valid_ppls)})"
        )

    return eval_results


def evaluate_fluency_baseline(
    baseline_results: List[Dict],
    model,
    eval_cfg: EvaluationConfig,
) -> Dict:
    """Evaluate fluency for baseline (no-intervention) results."""
    perplexities = []
    details = []

    for r in tqdm(baseline_results, desc="Fluency [Baseline]"):
        try:
            ppl = compute_perplexity_simple(model, r["generated_text"])
            if math.isinf(ppl) or math.isnan(ppl):
                ppl = -1.0
        except Exception:
            ppl = -1.0

        perplexities.append(ppl)
        details.append({
            "sample_idx": r["sample_idx"],
            "perplexity": ppl,
        })

    valid_ppls = [p for p in perplexities if p > 0]
    stats = compute_stats(valid_ppls) if valid_ppls else {"mean": -1}

    logger.info(f"  [Baseline] Perplexity: mean={stats['mean']:.2f}")

    return {
        "mean_perplexity": stats["mean"],
        "perplexity_stats": stats,
        "details": details,
    }
