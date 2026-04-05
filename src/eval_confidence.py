"""
Module 4b – Confidence Evaluation (Answer Log-Probability)
============================================================
Measures model confidence by computing log-probability of the predicted
answer tokens conditioned on the original prompt.
"""

from typing import List, Dict, Optional, Tuple
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.interventions import InterventionResult
from src.eval_accuracy import extract_numerical_answer
from src.utils import logger, compute_stats


@torch.no_grad()
def answer_logprob(
    model,
    prompt_ids: torch.Tensor,
    answer: str,
    hooks=None,
) -> Tuple[float, float, int]:
    """
    Compute log-probability of answer tokens only (not prompt tokens).

    Args:
        model: TransformerLens HookedTransformer.
        prompt_ids: Tokenized prompt tensor with shape [1, prompt_len].
        answer: Answer string whose token log-probability is measured.
        hooks: Optional TransformerLens forward hooks.

    Returns:
        (sum_logprob, mean_logprob_per_token, n_answer_tokens)
    """
    ans_ids = model.tokenizer.encode(" " + answer, add_special_tokens=False)
    if len(ans_ids) == 0:
        return float("nan"), float("nan"), 0

    ans_tensor = torch.tensor(ans_ids, device=prompt_ids.device, dtype=prompt_ids.dtype)
    full_ids = torch.cat([prompt_ids.squeeze(0), ans_tensor], dim=0)

    with model.hooks(fwd_hooks=hooks or []):
        logits = model(full_ids[:-1].unsqueeze(0), return_type="logits")

    logprobs = F.log_softmax(logits.float(), dim=-1)
    tgt = full_ids[1:].unsqueeze(0)
    token_logprobs = logprobs.gather(2, tgt.unsqueeze(-1)).squeeze(-1)

    # Keep only answer positions at the tail.
    answer_token_logprobs = token_logprobs[:, -len(ans_ids):]
    sum_lp = answer_token_logprobs.sum().item()
    mean_lp = answer_token_logprobs.mean().item()
    return sum_lp, mean_lp, len(ans_ids)


def _pick_answer_text(
    generated_text: str,
    patterns: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Choose answer text for confidence scoring.

    Priority:
      1) Extracted numerical answer (for GSM8K-style outputs)
      2) First non-empty generated line as fallback
    """
    extracted = extract_numerical_answer(generated_text, patterns)
    if extracted is not None and extracted.strip():
        return extracted.strip()

    for line in generated_text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return None


def evaluate_confidence(
    results: List[InterventionResult],
    model,
    patterns: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Evaluate answer-token confidence for intervention outputs.

    Returns:
        Dict mapping spec_label -> {
            "mean_answer_logprob": float,
            "mean_answer_logprob_per_token": float,
            "logprob_stats": {...},
            "logprob_per_token_stats": {...},
            "details": [...]
        }
    """
    by_spec: Dict[str, List[InterventionResult]] = {}
    for r in results:
        by_spec.setdefault(r.spec_label, []).append(r)

    eval_results: Dict[str, Dict] = {}

    for label, spec_results in sorted(by_spec.items()):
        sum_lps: List[float] = []
        mean_lps: List[float] = []
        details: List[Dict] = []

        for r in tqdm(spec_results, desc=f"Confidence [{label}]", leave=False):
            answer_text = _pick_answer_text(r.generated_text, patterns)
            if not answer_text:
                details.append(
                    {
                        "sample_idx": r.sample_idx,
                        "answer_text": None,
                        "answer_logprob": -1.0,
                        "answer_logprob_per_token": -1.0,
                        "n_answer_tokens": 0,
                    }
                )
                continue

            try:
                prompt_ids = model.to_tokens(r.original_prompt, prepend_bos=True)
                lp_sum, lp_mean, n_tok = answer_logprob(model, prompt_ids, answer_text)

                if math.isnan(lp_sum) or math.isnan(lp_mean):
                    lp_sum, lp_mean, n_tok = -1.0, -1.0, 0
            except Exception as e:
                logger.warning(f"Confidence failed for sample {r.sample_idx}: {e}")
                lp_sum, lp_mean, n_tok = -1.0, -1.0, 0

            if lp_sum <= 0 and n_tok > 0:
                sum_lps.append(lp_sum)
                mean_lps.append(lp_mean)

            details.append(
                {
                    "sample_idx": r.sample_idx,
                    "answer_text": answer_text,
                    "answer_logprob": lp_sum,
                    "answer_logprob_per_token": lp_mean,
                    "n_answer_tokens": n_tok,
                }
            )

        stats_sum = compute_stats(sum_lps) if sum_lps else {"mean": -1}
        stats_mean = compute_stats(mean_lps) if mean_lps else {"mean": -1}

        eval_results[label] = {
            "mean_answer_logprob": stats_sum["mean"],
            "mean_answer_logprob_per_token": stats_mean["mean"],
            "logprob_stats": stats_sum,
            "logprob_per_token_stats": stats_mean,
            "details": details,
        }

        logger.info(
            f"  [{label}] Confidence logP/token: "
            f"mean={stats_mean['mean']:.4f} (n={len(mean_lps)})"
        )

    return eval_results


def evaluate_confidence_baseline(
    baseline_results: List[Dict],
    model,
    patterns: Optional[List[str]] = None,
) -> Dict:
    """Evaluate answer-token confidence for baseline outputs."""
    sum_lps: List[float] = []
    mean_lps: List[float] = []
    details: List[Dict] = []

    for r in tqdm(baseline_results, desc="Confidence [Baseline]"):
        answer_text = _pick_answer_text(r["generated_text"], patterns)
        if not answer_text:
            details.append(
                {
                    "sample_idx": r["sample_idx"],
                    "answer_text": None,
                    "answer_logprob": -1.0,
                    "answer_logprob_per_token": -1.0,
                    "n_answer_tokens": 0,
                }
            )
            continue

        try:
            prompt_ids = model.to_tokens(r["prompt"], prepend_bos=True)
            lp_sum, lp_mean, n_tok = answer_logprob(model, prompt_ids, answer_text)
            if math.isnan(lp_sum) or math.isnan(lp_mean):
                lp_sum, lp_mean, n_tok = -1.0, -1.0, 0
        except Exception as e:
            logger.warning(f"Confidence failed for sample {r['sample_idx']}: {e}")
            lp_sum, lp_mean, n_tok = -1.0, -1.0, 0

        if lp_sum <= 0 and n_tok > 0:
            sum_lps.append(lp_sum)
            mean_lps.append(lp_mean)

        details.append(
            {
                "sample_idx": r["sample_idx"],
                "answer_text": answer_text,
                "answer_logprob": lp_sum,
                "answer_logprob_per_token": lp_mean,
                "n_answer_tokens": n_tok,
            }
        )

    stats_sum = compute_stats(sum_lps) if sum_lps else {"mean": -1}
    stats_mean = compute_stats(mean_lps) if mean_lps else {"mean": -1}

    logger.info(f"  [Baseline] Confidence logP/token: mean={stats_mean['mean']:.4f}")

    return {
        "mean_answer_logprob": stats_sum["mean"],
        "mean_answer_logprob_per_token": stats_mean["mean"],
        "logprob_stats": stats_sum,
        "logprob_per_token_stats": stats_mean,
        "details": details,
    }
