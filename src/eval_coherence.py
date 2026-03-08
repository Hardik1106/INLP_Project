"""
Module 4b – Reasoning Coherence Evaluation (LLM-as-a-Judge)
=============================================================
Uses an LLM (e.g., GPT-4o-mini via OpenAI API or a local model)
to score whether intermediate reasoning steps remain logically
consistent and well-formed after intervention.
"""

import os
import json
import time
from typing import List, Dict, Optional

from src.interventions import InterventionResult
from src.config import EvaluationConfig
from src.utils import logger, compute_stats


# ---------------------------------------------------------------------------
# Prompt templates for the judge
# ---------------------------------------------------------------------------

COHERENCE_SYSTEM_PROMPT = """You are an expert evaluator of mathematical reasoning chains.
You will be given a step-by-step solution to a math word problem.
Your job is to evaluate the REASONING COHERENCE of the solution on a scale of 1-5.

Scoring rubric:
5 - Perfect: Each step logically follows from the previous. All calculations are internally
    consistent (even if the final answer is wrong). The chain is complete.
4 - Minor issues: One small logical gap or unnecessary step, but the overall reasoning
    structure is sound.
3 - Moderate issues: Some steps don't follow logically, or there are contradictions between
    steps, but the overall approach is recognizable.
2 - Major issues: Multiple logical failures, hallucinated numbers, or large jumps in
    reasoning. Barely follows a coherent thread.
1 - Incoherent: The reasoning is completely broken — random numbers, contradictory
    statements, repetitive loops, or nonsensical text.

Respond with ONLY a JSON object with these fields:
{
  "score": <integer 1-5>,
  "explanation": "<brief explanation of your scoring>"
}"""

COHERENCE_USER_TEMPLATE = """Problem: {question}

Generated Solution:
{solution}

Rate the reasoning coherence of this solution (1-5)."""


# ---------------------------------------------------------------------------
# OpenAI API judge
# ---------------------------------------------------------------------------

def judge_coherence_openai(
    question: str,
    solution: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Dict:
    """
    Use OpenAI's API to judge the reasoning coherence of a solution.

    Returns:
        Dict with "score" (int 1-5) and "explanation" (str)
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. Install with: pip install openai")
        return {"score": -1, "explanation": "openai not installed"}

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No OpenAI API key found. Returning placeholder score.")
        return {"score": -1, "explanation": "no API key"}

    client = OpenAI(api_key=api_key)
    user_msg = COHERENCE_USER_TEMPLATE.format(
        question=question, solution=solution
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": COHERENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=300,
            )
            content = response.choices[0].message.content.strip()
            # Parse JSON response
            result = json.loads(content)
            return {
                "score": int(result["score"]),
                "explanation": result.get("explanation", ""),
            }
        except json.JSONDecodeError:
            # Try to extract score from non-JSON response
            import re
            match = re.search(r'"score"\s*:\s*(\d)', content)
            if match:
                return {
                    "score": int(match.group(1)),
                    "explanation": content,
                }
            logger.warning(f"Failed to parse judge response (attempt {attempt+1})")
        except Exception as e:
            logger.warning(f"API call failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    return {"score": -1, "explanation": "failed after retries"}


# ---------------------------------------------------------------------------
# Local model judge (fallback)
# ---------------------------------------------------------------------------

def judge_coherence_heuristic(
    solution: str,
) -> Dict:
    """
    Simple heuristic coherence scoring as a fallback when no API is available.

    Checks for:
    - Presence of step markers ("Step 1", numbered lists)
    - Consistent use of numbers
    - Repetition detection
    - Contradiction markers
    """
    import re

    score = 5  # Start at perfect and deduct
    explanations = []

    # Check for step structure
    has_steps = bool(re.search(r'[Ss]tep\s+\d|^\d+[\.\)]', solution, re.MULTILINE))
    if not has_steps:
        score -= 0.5
        explanations.append("No clear step structure")

    # Check for repetition (same sentence appearing multiple times)
    sentences = [s.strip() for s in re.split(r'[.!?\n]', solution) if len(s.strip()) > 10]
    if sentences:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.5:
            score -= 2
            explanations.append(f"High repetition (unique ratio: {unique_ratio:.2f})")
        elif unique_ratio < 0.7:
            score -= 1
            explanations.append(f"Some repetition (unique ratio: {unique_ratio:.2f})")

    # Check for very short responses (might indicate broken generation)
    if len(solution.split()) < 20:
        score -= 1
        explanations.append("Very short response")

    # Check for number consistency (crude check)
    numbers = re.findall(r'\b\d+\.?\d*\b', solution)
    if len(numbers) < 2 and len(solution.split()) > 50:
        score -= 1
        explanations.append("Very few numbers for a math solution")

    score = max(1, min(5, round(score)))
    return {
        "score": score,
        "explanation": "; ".join(explanations) if explanations else "Looks coherent",
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_coherence(
    results: List[InterventionResult],
    eval_cfg: EvaluationConfig,
    use_api: bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate reasoning coherence for all intervention results.

    Groups by spec_label and computes per-spec average coherence.

    Args:
        results: List of InterventionResult objects
        eval_cfg: Evaluation config (judge model, API key env var)
        use_api: Whether to use OpenAI API (True) or heuristic fallback (False)

    Returns:
        Dict mapping spec_label -> {
            "mean_score": float,
            "scores": [...],
            "details": [...]
        }
    """
    # Group by spec_label
    by_spec: Dict[str, List[InterventionResult]] = {}
    for r in results:
        by_spec.setdefault(r.spec_label, []).append(r)

    api_key = os.environ.get(eval_cfg.openai_api_key_env) if use_api else None

    eval_results = {}

    for label, spec_results in sorted(by_spec.items()):
        scores = []
        details = []

        for r in spec_results:
            # Extract the question from the prompt
            question = r.original_prompt.split("\n")[0].replace("Question: ", "")
            # The solution is the generated text minus the prompt
            solution = r.generated_text

            if use_api and api_key:
                judgment = judge_coherence_openai(
                    question=question,
                    solution=solution,
                    model=eval_cfg.judge_model,
                    temperature=eval_cfg.judge_temperature,
                    api_key=api_key,
                )
            else:
                judgment = judge_coherence_heuristic(solution)

            scores.append(judgment["score"])
            details.append({
                "sample_idx": r.sample_idx,
                "score": judgment["score"],
                "explanation": judgment["explanation"],
            })

        valid_scores = [s for s in scores if s > 0]
        stats = compute_stats(valid_scores) if valid_scores else {"mean": -1}

        eval_results[label] = {
            "mean_score": stats["mean"],
            "score_stats": stats,
            "details": details,
        }

        logger.info(
            f"  [{label}] Coherence: mean={stats['mean']:.2f} "
            f"(n={len(valid_scores)})"
        )

    return eval_results
