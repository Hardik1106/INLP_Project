#!/usr/bin/env python3
"""
Module 5 – Full Pipeline Orchestrator
=======================================
Runs the complete experimental pipeline end-to-end:
  1. Load data & format prompts
  2. Cache activations across layers
  3. Encode through SAEs & run contrastive analysis
  4. Build intervention specs & run causal interventions
  5. Evaluate on all three axes (accuracy, confidence, fluency)
  6. Aggregate & save results

Can be run for a single model or as part of a cross-model sweep.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import time

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.utils import logger, setup_logger, set_seed, save_json, ensure_dir
from src.data_pipeline import build_prompt_sets
from src.cache_activations import load_hooked_model, run_activation_caching
from src.sae_encoder import encode_and_cache_all
from src.contrastive_analysis import run_contrastive_analysis
from src.visualizations import generate_all_visualizations
from src.interventions import (
    build_intervention_specs,
    run_intervention_experiment,
    run_baseline_generation,
)
from src.eval_accuracy import evaluate_accuracy, evaluate_accuracy_baseline
from src.eval_confidence import evaluate_confidence, evaluate_confidence_baseline
from src.eval_fluency import evaluate_fluency, evaluate_fluency_baseline


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_1_data_and_caching(config: ExperimentConfig, model=None):
    """Stage 1: Load datasets, format prompts, cache activations."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Pipeline & Activation Caching")
    logger.info("=" * 60)

    # Build prompt sets
    prompt_sets = build_prompt_sets(config.data, config.prompts)

    # Load model if not provided
    if model is None:
        model = load_hooked_model(
            config.model.transformer_lens_name,
            config.model.device,
            config.model.dtype,
        )

    # Cache activations
    cached_acts = run_activation_caching(config, prompt_sets, model)

    return prompt_sets, cached_acts, model


def stage_2_sae_and_contrastive(
    config: ExperimentConfig,
    cached_acts: Dict[str, Dict[int, torch.Tensor]],
    results_dir: str,  # ADD THIS PARAMETER
):
    """Stage 2: SAE encoding and contrastive feature analysis."""
    logger.info("=" * 60)
    logger.info("STAGE 2: SAE Encoding & Contrastive Analysis")
    logger.info("=" * 60)

    all_layers = config.layers.all_layers

    # Encode through SAEs
    sae_features = encode_and_cache_all(
        sae_config=config.sae,
        cached_activations=cached_acts,
        layers=all_layers,
        cache_dir=config.data.cache_dir,
        device=config.model.device,
        batch_size=config.performance.sae_encode_batch_size,
    )

    # Run contrastive analysis - USE results_dir HERE
    contrastive_results = run_contrastive_analysis(
        sae_features=sae_features,
        layers=all_layers,
        cfg=config.contrastive,
        output_dir=f"{results_dir}/contrastive",  # CHANGED
    )

    return sae_features, contrastive_results

def stage_3_interventions(
    config: ExperimentConfig,
    prompt_sets: Dict,
    contrastive_results: Dict,
    model,
    results_dir: str,  # ADD THIS
):
    """Stage 3: Run causal interventions (ablation + amplification)."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Causal Interventions")
    logger.info("=" * 60)

    # Build intervention specs
    specs = build_intervention_specs(
        contrastive_results=contrastive_results,
        intervention_cfg=config.interventions,
        layers=config.layers.intervention_layers,
    )

    # Use GSM8K CoT samples for interventions
    samples = prompt_sets["gsm8k_cot"]

    baseline_results_cot = run_baseline_generation(
        model=model,
        samples=prompt_sets["gsm8k_cot"],
        max_new_tokens=config.interventions.max_new_tokens,
        output_dir=f"{results_dir}/baseline_cot",  # CHANGED
        batch_size=config.performance.intervention_batch_size,
    )

    baseline_results_no_cot = run_baseline_generation(
        model=model,
        samples=prompt_sets["gsm8k_no_cot"],
        max_new_tokens=config.interventions.max_new_tokens,
        output_dir=f"{results_dir}/baseline_no_cot",  # CHANGED
        batch_size=config.performance.intervention_batch_size,
    )

    # Run interventions
    intervention_results = run_intervention_experiment(
        model=model,
        sae_config=config.sae,
        samples=samples,
        specs=specs,
        max_new_tokens=config.interventions.max_new_tokens,
        output_dir=f"{results_dir}/interventions",  # CHANGED
        device=config.model.device,
        batch_size=config.performance.intervention_batch_size,
    )

    return baseline_results_cot, baseline_results_no_cot, intervention_results, specs

# def stage_4_evaluation(
#     config: ExperimentConfig,
#     baseline_results: List[Dict],
#     intervention_results: List,
#     model,
# ):
#     """Stage 4: Two-axis evaluation (accuracy + fluency)."""
#     logger.info("=" * 60)
#     logger.info("STAGE 4: Evaluation (Accuracy + Fluency)")
#     logger.info("=" * 60)

#     eval_dir = ensure_dir(f"{config.output.results_dir}/evaluation")

#     # --- Axis 1: Answer Accuracy ---
#     logger.info("--- Axis 1: Answer Accuracy ---")
#     accuracy_baseline = evaluate_accuracy_baseline(baseline_results)
#     accuracy_intervention = evaluate_accuracy(intervention_results)

#     save_json(accuracy_baseline, f"{eval_dir}/accuracy_baseline.json")
#     save_json(accuracy_intervention, f"{eval_dir}/accuracy_intervention.json")

#     # --- Axis 2: Language Fluency (Perplexity) ---
#     logger.info("--- Axis 2: Language Fluency ---")
#     fluency_baseline = evaluate_fluency_baseline(
#         baseline_results, model, config.evaluation,
#     )
#     fluency_intervention = evaluate_fluency(
#         intervention_results, model, config.evaluation,
#     )

#     save_json(fluency_baseline, f"{eval_dir}/fluency_baseline.json")
#     save_json(fluency_intervention, f"{eval_dir}/fluency_intervention.json")

#     return {
#         "accuracy_baseline": accuracy_baseline,
#         "accuracy_intervention": accuracy_intervention,
#         "fluency_baseline": fluency_baseline,
#         "fluency_intervention": fluency_intervention,
#     }

def stage_4_evaluation(
    config: ExperimentConfig,
    baseline_results: List[Dict],
    intervention_results: List,
    model,
    results_dir: str,  # ADD THIS
):
    """Stage 4: Evaluation (accuracy + confidence + fluency)."""
    logger.info("=" * 60)
    logger.info("STAGE 4: Evaluation (Accuracy + Confidence + Fluency)")
    logger.info("=" * 60)

    eval_dir = ensure_dir(f"{results_dir}/evaluation")  # CHANGED

    # # --- Axis 1: Answer Accuracy ---
    # logger.info("--- Axis 1: Answer Accuracy ---")
    # accuracy_baseline = evaluate_accuracy_baseline(baseline_results)
    # accuracy_intervention = evaluate_accuracy(intervention_results)

    # save_json(accuracy_baseline, f"{eval_dir}/accuracy_baseline.json")
    # save_json(accuracy_intervention, f"{eval_dir}/accuracy_intervention.json")

    # --- Axis 2: Confidence (Answer Log-Probability) ---
    logger.info("--- Axis 2: Confidence (Answer Log-Probability) ---")
    confidence_baseline = evaluate_confidence_baseline(
        baseline_results,
        model,
    )
    confidence_intervention = evaluate_confidence(
        intervention_results,
        model,
    )

    save_json(confidence_baseline, f"{eval_dir}/confidence_baseline.json")
    save_json(confidence_intervention, f"{eval_dir}/confidence_intervention.json")

    # # --- Axis 3: Language Fluency (Perplexity) ---
    # logger.info("--- Axis 3: Language Fluency ---")
    # fluency_baseline = evaluate_fluency_baseline(
    #     baseline_results, model, config.evaluation,
    # )
    # fluency_intervention = evaluate_fluency(
    #     intervention_results, model, config.evaluation,
    # )

    # save_json(fluency_baseline, f"{eval_dir}/fluency_baseline.json")
    # save_json(fluency_intervention, f"{eval_dir}/fluency_intervention.json")

    return {
        # "accuracy_baseline": accuracy_baseline,
        # "accuracy_intervention": accuracy_intervention,
        "confidence_baseline": confidence_baseline,
        "confidence_intervention": confidence_intervention,
        # "fluency_baseline": fluency_baseline,
        # "fluency_intervention": fluency_intervention,
    }

# ---------------------------------------------------------------------------
# Aggregation & summary
# ---------------------------------------------------------------------------

def generate_summary(
    eval_results: Dict,
    output_dir: str,
):
    """Generate a human-readable summary of all results."""
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("EXPERIMENT RESULTS SUMMARY")
    summary_lines.append("=" * 70)

    # Baseline accuracy
    bl_acc = eval_results["accuracy_baseline"]
    summary_lines.append(
        f"\nBaseline Accuracy: {bl_acc['accuracy']:.3f} "
        f"({bl_acc['correct']}/{bl_acc['total']})"
    )

    # Baseline fluency
    bl_flu = eval_results["fluency_baseline"]
    summary_lines.append(
        f"Baseline Perplexity: {bl_flu['mean_perplexity']:.2f}"
    )

    # Baseline confidence
    bl_conf = eval_results.get("confidence_baseline", {})
    if bl_conf:
        summary_lines.append(
            "Baseline Confidence (mean logP/token): "
            f"{bl_conf.get('mean_answer_logprob_per_token', -1):.4f}"
        )

    # Intervention results table
    summary_lines.append("\n--- Intervention Results ---")
    summary_lines.append(
        f"{'Spec Label':<50} {'Accuracy':>8} {'LogP/tok':>10} {'Perplexity':>11}"
    )
    summary_lines.append("-" * 70)

    acc_inv = eval_results["accuracy_intervention"]
    conf_inv = eval_results.get("confidence_intervention", {})
    flu_inv = eval_results["fluency_intervention"]

    for label in sorted(acc_inv.keys()):
        acc = acc_inv[label]["accuracy"]
        conf = conf_inv.get(label, {}).get("mean_answer_logprob_per_token", -1)
        ppl = flu_inv.get(label, {}).get("mean_perplexity", -1)
        summary_lines.append(
            f"{label:<50} {acc:>8.3f} {conf:>10.4f} {ppl:>11.2f}"
        )

    summary_text = "\n".join(summary_lines)
    logger.info("\n" + summary_text)

    save_path = f"{output_dir}/summary.txt"
    with open(save_path, "w") as f:
        f.write(summary_text)
    logger.info(f"Summary saved to {save_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# def run_full_pipeline(config: ExperimentConfig):
#     """Run the complete experimental pipeline."""
#     set_seed(42)
#     ensure_dir(config.output.results_dir)

#     # Save config for reproducibility
#     save_json(config.to_dict(), f"{config.output.results_dir}/config.json")

#     # --- Add timer ---
#     start_time = time.time()

#     # Estimate total steps (samples × specs)
#     num_samples = config.data.num_samples
#     specs = None
#     total_steps = 0

#     # Stage 1
#     prompt_sets, cached_acts, model = stage_1_data_and_caching(config)

#     # Stage 2
#     sae_features, contrastive_results = stage_2_sae_and_contrastive(
#         config, cached_acts
#     )

#     # Stage 3
#     baseline_results, intervention_results, specs = stage_3_interventions(
#         config, prompt_sets, contrastive_results, model
#     )

#     # --- Calculate total steps ---
#     if specs is not None:
#         total_steps = len(specs) * num_samples
#     else:
#         total_steps = num_samples

#     # Stage 4
#     eval_results = stage_4_evaluation(
#         config, baseline_results, intervention_results, model
#     )

#     # --- End timer ---
#     end_time = time.time()
#     elapsed = end_time - start_time

#     # --- Print summary with ETA ---
#     logger.info(f"Total experiment steps: {total_steps}")
#     logger.info(f"Total elapsed time: {elapsed:.1f} seconds ({elapsed/60:.1f} min)")

#     # Optionally, save to summary.txt
#     summary_path = f"{config.output.results_dir}/summary.txt"
#     with open(summary_path, "a") as f:
#         f.write(f"\nTotal experiment steps: {total_steps}\n")
#         f.write(f"Total elapsed time: {elapsed:.1f} seconds ({elapsed/60:.1f} min)\n")

#     logger.info("Pipeline complete!")
#     return eval_results

def construct_results_dir(base_results_dir: str, config: ExperimentConfig) -> str:
    """
    Construct a model and layer-specific results directory.
    
    Structure: {base_results_dir}/{model_name}/layers_{layer_ids}/
    Example: outputs/pythia-70M-deduped/layers_2_4/
    
    Args:
        base_results_dir: Base output directory from config
        config: Experiment configuration
        
    Returns:
        Full path to model/layer-specific results directory
    """
    # Extract model name (use transformer_lens_name, fallback to name)
    model_name = config.model.transformer_lens_name or config.model.name
    # Clean up model name for filesystem
    model_name = model_name.replace("/", "_").replace(".", "_")
    
    # Get layers as string
    layers_str = "_".join(str(l) for l in sorted(config.layers.all_layers))
    
    # Construct full path
    results_dir = f"{base_results_dir}/{model_name}/layers_{layers_str}"
    
    return results_dir


def run_full_pipeline(config: ExperimentConfig):
    """Run the complete experimental pipeline."""
    set_seed(42)
    
    # Construct model and layer-specific results directory
    results_dir = construct_results_dir(config.output.results_dir, config)
    ensure_dir(results_dir)
    
    # Save config for reproducibility
    save_json(config.to_dict(), f"{results_dir}/config.json")
    
    logger.info(f"Results will be saved to: {results_dir}")

    start_time = time.time()

    # Stage 1
    prompt_sets, cached_acts, model = stage_1_data_and_caching(config)

    # Stage 2
    sae_features, contrastive_results = stage_2_sae_and_contrastive(
        config, cached_acts, results_dir  # PASS HERE
    )

    # Stage 3
    baseline_results_cot, baseline_results_nocot, intervention_results, specs = stage_3_interventions(
        config, prompt_sets, contrastive_results, model, results_dir  # PASS HERE
    )

    # Stage 4
    eval_results = stage_4_evaluation(
        config, baseline_results_cot, intervention_results, model, results_dir  # PASS HERE
    )

    # --- NEW: Stage 5 - Generate Visualizations ---
    logger.info("=" * 60)
    logger.info("STAGE 5: Generate Publication Visualizations")
    logger.info("=" * 60)
    generate_all_visualizations(
        contrastive_results=contrastive_results,
        sae_features=sae_features,
        baseline_results=baseline_results_cot,
        intervention_results=intervention_results,
        model=model,
        confidence_baseline=eval_results.get("confidence_baseline"),
        confidence_intervention=eval_results.get("confidence_intervention"),
        output_dir=f"{results_dir}/visualizations",
        layers=config.layers.all_layers,
        generate_accuracy_plots=False,        # Skip accuracy plots
        generate_fluency_plots=False,         # Skip fluency plots
        generate_confidence_plots=True,       # Keep confidence plots
    )

    end_time = time.time()
    elapsed = end_time - start_time

    logger.info(f"Total elapsed time: {elapsed:.1f} seconds ({elapsed/60:.1f} min)")
    logger.info("Pipeline complete!")
    logger.info(f"All results saved to: {results_dir}")
    return eval_results

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the CoT Mechanistic Faithfulness pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["all", "data", "contrastive", "interventions", "evaluation"],
        help="Run a specific stage or all stages"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model name from config"
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Override number of samples from config"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda/cpu)"
    )
    parser.add_argument(
        "--no-triviaqa", action="store_true", default=False,
        help="Skip TriviaQA dataset (only use GSM8K intra-dataset contrast)"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(PROJECT_ROOT) / args.config
    config = ExperimentConfig.from_yaml(str(config_path))

    # Apply CLI overrides
    if args.model:
        config.model.name = args.model
        config.model.transformer_lens_name = args.model
    if args.num_samples:
        config.data.num_samples = args.num_samples
    if args.device:
        config.model.device = args.device
    if args.no_triviaqa:
        config.data.use_triviaqa = False

    # Setup logging
    setup_logger("cot_faithfulness", config.output.log_level)

    # Run
    run_full_pipeline(config)


if __name__ == "__main__":
    main()
