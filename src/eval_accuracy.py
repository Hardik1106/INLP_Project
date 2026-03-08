"""
Module 4a – Answer Accuracy Evaluation
========================================
Parses and compares the final numerical answer from generated text
against the ground truth.
"""

import re
from typing import List, Dict, Optional, Tuple

from src.interventions import InterventionResult
from src.utils import logger, compute_stats


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_numerical_answer(
    text: str,
    patterns: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Extract the final numerical answer from generated text.

    Tries multiple regex patterns in order, returning the first match.
    Handles common GSM8K formats:
      - "#### 42"
      - "The answer is 42"
      - "= 42"
      - Just the last number in the text
    """
    if patterns is None:
        patterns = [
            r"####\s*(\-?[\d,]+\.?\d*)",          # GSM8K format: #### 42
            r"[Tt]he\s+answer\s+is\s*:?\s*(\-?[\d,]+\.?\d*)",  # "The answer is 42"
            r"[Tt]herefore,?\s+the\s+answer\s+is\s*:?\s*(\-?[\d,]+\.?\d*)",
            r"=\s*(\-?[\d,]+\.?\d*)\s*$",         # Trailing "= 42"
            r"(\-?[\d,]+\.?\d*)\s*$",              # Last number in text
        ]

    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            # Clean commas from numbers
            return match.group(1).replace(",", "").strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize a numerical answer for comparison."""
    # Remove commas, leading/trailing whitespace
    ans = answer.replace(",", "").strip()
    # Remove trailing .0 for integer comparisons
    try:
        val = float(ans)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return ans


def check_answer_correct(
    generated_text: str,
    ground_truth: str,
    patterns: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str], str]:
    """
    Check if the generated answer matches ground truth.

    Returns:
        (is_correct, extracted_answer, normalized_ground_truth)
    """
    extracted = extract_numerical_answer(generated_text, patterns)
    normalized_gt = normalize_answer(ground_truth)

    if extracted is None:
        return False, None, normalized_gt

    normalized_ext = normalize_answer(extracted)
    is_correct = normalized_ext == normalized_gt

    return is_correct, normalized_ext, normalized_gt


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_accuracy(
    results: List[InterventionResult],
    patterns: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Evaluate answer accuracy for a list of intervention results.

    Groups results by intervention spec and computes per-spec accuracy.

    Returns:
        Dict mapping spec_label -> {
            "accuracy": float,
            "correct": int,
            "total": int,
            "details": [...]
        }
    """
    # Group by spec_label
    by_spec: Dict[str, List[InterventionResult]] = {}
    for r in results:
        by_spec.setdefault(r.spec_label, []).append(r)

    eval_results = {}

    for label, spec_results in sorted(by_spec.items()):
        details = []
        correct_count = 0

        for r in spec_results:
            is_correct, extracted, gt = check_answer_correct(
                r.generated_text, r.ground_truth, patterns
            )
            if is_correct:
                correct_count += 1

            details.append({
                "sample_idx": r.sample_idx,
                "is_correct": is_correct,
                "extracted_answer": extracted,
                "ground_truth": gt,
            })

        accuracy = correct_count / len(spec_results) if spec_results else 0.0

        eval_results[label] = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(spec_results),
            "details": details,
        }

        logger.info(
            f"  [{label}] Accuracy: {accuracy:.3f} "
            f"({correct_count}/{len(spec_results)})"
        )

    return eval_results


def evaluate_accuracy_baseline(
    baseline_results: List[Dict],
    patterns: Optional[List[str]] = None,
) -> Dict:
    """Evaluate accuracy for baseline (no-intervention) results."""
    correct = 0
    details = []

    for r in baseline_results:
        is_correct, extracted, gt = check_answer_correct(
            r["generated_text"], r["ground_truth"], patterns
        )
        if is_correct:
            correct += 1
        details.append({
            "sample_idx": r["sample_idx"],
            "is_correct": is_correct,
            "extracted_answer": extracted,
            "ground_truth": gt,
        })

    accuracy = correct / len(baseline_results) if baseline_results else 0.0
    logger.info(f"  [Baseline] Accuracy: {accuracy:.3f} ({correct}/{len(baseline_results)})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(baseline_results),
        "details": details,
    }
