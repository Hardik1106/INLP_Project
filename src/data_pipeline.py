"""
Module 1a – Data Pipeline
=========================
Handles loading GSM8K and TriviaQA datasets, formatting prompts into
CoT (step-by-step) and No-CoT (direct answer) variants.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from datasets import load_dataset
from tqdm import tqdm

from src.config import DataConfig, PromptsConfig
from src.utils import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PromptSample:
    """A single formatted prompt with metadata."""
    dataset: str           # "gsm8k" or "triviaqa"
    condition: str         # "cot" or "no_cot"
    prompt_text: str       # The full prompt string fed to the model
    ground_truth: str      # Expected answer (string)
    original_idx: int      # Index in the source dataset
    metadata: Dict = None  # Extra info (e.g., full question, full solution)


# ---------------------------------------------------------------------------
# GSM8K loading & formatting
# ---------------------------------------------------------------------------

def load_gsm8k(
    split: str = "test",
    num_samples: Optional[int] = None,
) -> List[Dict]:
    """Load GSM8K dataset from HuggingFace."""
    print(f"[DATA PIPELINE] Downloading GSM8K split='{split}'...")
    logger.info(f"Loading GSM8K split='{split}'...")
    ds = load_dataset(
        "gsm8k",
        "main",
        split=split,
        verification_mode="no_checks",
    )
    samples = list(ds)
    if num_samples is not None:
        samples = samples[:num_samples]
    logger.info(f"  Loaded {len(samples)} GSM8K samples")
    return samples


def extract_gsm8k_answer(answer_text: str) -> str:
    """
    GSM8K answers have the format:
        ... reasoning ...
        #### <number>
    Extract the final numeric answer.
    """
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


def format_gsm8k_cot(
    sample: Dict,
    prompts_cfg: PromptsConfig,
    idx: int,
) -> PromptSample:
    """Format a GSM8K sample as a CoT prompt."""
    question = sample["question"]
    answer_text = sample["answer"]
    ground_truth = extract_gsm8k_answer(answer_text)

    prompt = (
        f"Question: {question}\n"
        f"{prompts_cfg.cot_prefix}"
    )

    return PromptSample(
        dataset="gsm8k",
        condition="cot",
        prompt_text=prompt,
        ground_truth=ground_truth,
        original_idx=idx,
        metadata={"question": question, "full_answer": answer_text},
    )


def format_gsm8k_no_cot(
    sample: Dict,
    prompts_cfg: PromptsConfig,
    idx: int,
) -> PromptSample:
    """Format a GSM8K sample as a No-CoT (direct answer) prompt."""
    question = sample["question"]
    answer_text = sample["answer"]
    ground_truth = extract_gsm8k_answer(answer_text)

    prompt = (
        f"Question: {question}\n"
        f"{prompts_cfg.no_cot_prefix}"
    )

    return PromptSample(
        dataset="gsm8k",
        condition="no_cot",
        prompt_text=prompt,
        ground_truth=ground_truth,
        original_idx=idx,
        metadata={"question": question, "full_answer": answer_text},
    )


# ---------------------------------------------------------------------------
# TriviaQA loading & formatting
# ---------------------------------------------------------------------------

def load_triviaqa(
    split: str = "validation",
    num_samples: Optional[int] = None,
) -> List[Dict]:
    """Load TriviaQA dataset from HuggingFace."""
    print(f"[DATA PIPELINE] Downloading TriviaQA split='{split}'...")
    logger.info(f"Loading TriviaQA split='{split}'...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split=split)
    samples = list(ds)
    if num_samples is not None:
        samples = samples[:num_samples]
    logger.info(f"  Loaded {len(samples)} TriviaQA samples")
    return samples


def format_triviaqa(
    sample: Dict,
    prompts_cfg: PromptsConfig,
    idx: int,
) -> PromptSample:
    """Format a TriviaQA sample (single-step factual recall)."""
    question = sample["question"]
    # TriviaQA has multiple valid answers; take the first alias
    answer_obj = sample["answer"]
    if isinstance(answer_obj, dict):
        aliases = answer_obj.get("aliases", [])
        ground_truth = answer_obj.get("value", aliases[0] if aliases else "")
    else:
        ground_truth = str(answer_obj)

    prompt = (
        f"Question: {question}\n"
        f"{prompts_cfg.no_cot_prefix}"
    )

    return PromptSample(
        dataset="triviaqa",
        condition="factual",
        prompt_text=prompt,
        ground_truth=ground_truth,
        original_idx=idx,
        metadata={"question": question, "answer_obj": str(answer_obj)},
    )


# ---------------------------------------------------------------------------
# Master data pipeline
# ---------------------------------------------------------------------------

def build_prompt_sets(
    data_cfg: DataConfig,
    prompts_cfg: PromptsConfig,
) -> Dict[str, List[PromptSample]]:
    """
    Build all prompt sets needed for the experiments.

    Returns a dict with keys:
        - "gsm8k_cot"     : GSM8K with CoT prompting
        - "gsm8k_no_cot"  : GSM8K with direct-answer prompting
        - "triviaqa"       : TriviaQA factual recall (only if use_triviaqa=True)
    """
    gsm8k_raw = load_gsm8k(data_cfg.gsm8k_split, data_cfg.num_samples)

    gsm8k_cot = [
        format_gsm8k_cot(s, prompts_cfg, i)
        for i, s in enumerate(tqdm(gsm8k_raw, desc="Formatting GSM8K CoT"))
    ]
    gsm8k_no_cot = [
        format_gsm8k_no_cot(s, prompts_cfg, i)
        for i, s in enumerate(tqdm(gsm8k_raw, desc="Formatting GSM8K No-CoT"))
    ]

    result = {
        "gsm8k_cot": gsm8k_cot,
        "gsm8k_no_cot": gsm8k_no_cot,
    }

    if data_cfg.use_triviaqa:
        trivia_raw = load_triviaqa(data_cfg.triviaqa_split, data_cfg.num_samples)
        triviaqa = [
            format_triviaqa(s, prompts_cfg, i)
            for i, s in enumerate(tqdm(trivia_raw, desc="Formatting TriviaQA"))
        ]
        result["triviaqa"] = triviaqa
        logger.info(
            f"Prompt sets built: gsm8k_cot={len(gsm8k_cot)}, "
            f"gsm8k_no_cot={len(gsm8k_no_cot)}, triviaqa={len(triviaqa)}"
        )
    else:
        logger.info(
            f"Prompt sets built (no TriviaQA): gsm8k_cot={len(gsm8k_cot)}, "
            f"gsm8k_no_cot={len(gsm8k_no_cot)}"
        )

    return result
