"""
Module 4d – Visualization & Analysis
=====================================
Creates publication-ready figures for mechanistic interpretability findings:
  1. Feature Isolation Filter (Venn/Scatter) – PER LAYER
  2. Sparsity Distribution (Violin plots) – PER LAYER
  3. Intervention Impact (Comparison charts) – PER LAYER
  4. Cross-Layer Cognitive Evolution (Line chart) – CROSS LAYER
  5. Qualitative "Lobotomy" table – PER LAYER
"""

import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import seaborn as sns
import pandas as pd
import torch

from src.contrastive_analysis import ContrastiveResult
from src.interventions import InterventionResult
from src.eval_accuracy import evaluate_accuracy, check_answer_correct, evaluate_accuracy_baseline
from src.eval_fluency import compute_perplexity_simple, evaluate_fluency
from src.eval_confidence import evaluate_confidence
from src.utils import logger, ensure_dir, save_json


# ---------------------------------------------------------------------------
# Helper Functions: Filter by Layer
# ---------------------------------------------------------------------------

def filter_intervention_results_by_layer(
    intervention_results: List[InterventionResult],
    layer: int,
) -> List[InterventionResult]:
    """
    Filter intervention results to only include those for the specified layer.
    
    Args:
        intervention_results: List of InterventionResult objects
        layer: Layer number to filter by
        
    Returns:
        Filtered list of InterventionResult objects
    """
    return [r for r in intervention_results if r.layer == layer]


def filter_sae_features_by_layer(
    sae_features: Dict[str, Dict[int, torch.Tensor]],
    layer: int,
) -> Dict[str, torch.Tensor]:
    """
    Extract SAE features for a specific layer.
    
    Args:
        sae_features: Dict mapping condition -> layer dict -> features
        layer: Layer number to extract
        
    Returns:
        Dict mapping condition -> features tensor for this layer
    """
    result = {}
    for condition, layer_dict in sae_features.items():
        if layer in layer_dict:
            result[condition] = layer_dict[layer]  # ADD THIS LINE
    return result

def compute_sparsity_stats_per_layer(
    sae_features: Dict[str, torch.Tensor],
) -> Dict[str, List[float]]:
    """
    Compute L0 norm (active feature count) for a specific layer's features.
    
    Args:
        sae_features: Dict mapping condition -> features tensor [N, d_sae]
        
    Returns:
        Dict mapping condition -> list of L0 values across samples
    """
    sparsity_data = {}
    for condition, features in sae_features.items():
        # L0 = count of non-zero features per sample
        l0 = (features > 0).sum(dim=1).float().cpu().numpy()
        sparsity_data[condition] = l0.tolist()
    return sparsity_data


# ---------------------------------------------------------------------------
# 1. Feature Isolation Filter – Scatter Plot (GSM8K vs TriviaQA)
# ---------------------------------------------------------------------------

def plot_feature_isolation_scatter(
    contrastive_results: Dict[int, ContrastiveResult],
    output_dir: str = "outputs/visualizations",
    layer: Optional[int] = None,
) -> None:
    """
    Scatter plot: X=GSM8K feature strength, Y=TriviaQA feature strength.
    Highlights bottom-right (high math, low trivia) as reasoning features.
    Per-layer visualization.
    
    Args:
        contrastive_results: Dict of layer -> ContrastiveResult
        output_dir: Where to save the figure (should be layer-specific already)
        layer: Layer to plot (required for per-layer)
    """
    ensure_dir(output_dir)
    
    if layer is None:
        logger.error("layer must be specified for per-layer feature isolation plot")
        return
    
    if layer not in contrastive_results:
        logger.warning(f"Layer {layer} not in results. Skipping scatter plot.")
        return
    
    result = contrastive_results[layer]
    
    # X: GSM8K-CoT - No-CoT (intra-dataset contrast)
    # Y: GSM8K-CoT - TriviaQA (inter-dataset contrast)
    intra_diff = result.intra_diff.cpu().numpy()
    inter_diff = result.inter_diff.cpu().numpy()
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # All features as gray points
    ax.scatter(intra_diff, inter_diff, alpha=0.3, s=20, c='gray', label='All features')
    
    # Highlight reasoning features (intersection)
    reasoning_idx = np.array(result.reasoning_features)
    if len(reasoning_idx) > 0:
        ax.scatter(
            intra_diff[reasoning_idx],
            inter_diff[reasoning_idx],
            alpha=0.8,
            s=50,
            c='red',
            label='Candidate reasoning features',
            edgecolors='darkred',
            linewidth=0.5,
        )
    
    # Add quadrant lines
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Annotations for quadrants
    ax.text(0.98, 0.98, 'Math\n(Low Trivia)', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.text(0.02, 0.02, 'General\nLanguage', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=10, alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    ax.set_xlabel('GSM8K Intra-Dataset Contrast (CoT - No-CoT)', fontsize=12)
    ax.set_ylabel('Inter-Dataset Contrast (GSM8K - TriviaQA)', fontsize=12)
    ax.set_title(f'Feature Isolation Filter – Layer {layer}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.2)
    
    save_path = f"{output_dir}/01_feature_isolation_scatter.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 2. Sparsity Distribution – Violin Plots (L0 norm comparison)
# ---------------------------------------------------------------------------

def plot_sparsity_distribution(
    sparsity_data: Dict[str, List[float]],
    output_dir: str = "outputs/visualizations",
    layer: Optional[int] = None,
) -> None:
    """
    Violin plot comparing L0 sparsity between CoT and No-CoT for a specific layer.
    Per-layer visualization.
    
    Args:
        sparsity_data: Dict mapping condition -> list of L0 values
        output_dir: Where to save the figure (should be layer-specific already)
        layer: Layer number for title/logging
    """
    ensure_dir(output_dir)
    
    # Prepare dataframe for seaborn
    rows = []
    for label, values in sparsity_data.items():
        # Check "no_cot" BEFORE "cot" to avoid substring match
        if "no_cot" in label.lower():
            print("="*70)
            print(label)
            print("="*70)
            condition = "No-CoT"
        elif "cot" in label.lower():
            condition = "CoT"
        else:
            continue  # Skip other conditions like TriviaQA
        
        for val in values:
            rows.append({"Condition": condition, "L0 Norm (Active Features)": val})
    
    if not rows:
        logger.warning("No sparsity data to plot")
        return
    
    df = pd.DataFrame(rows)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="Condition", y="L0 Norm (Active Features)",
                   palette=['#2ecc71', '#e74c3c'], ax=ax)
    
    # Add mean markers
    means = df.groupby("Condition")["L0 Norm (Active Features)"].mean()
    for i, (cond, mean_val) in enumerate(means.items()):
        ax.plot(i, mean_val, 'D', color='black', markersize=8, zorder=3)
        ax.text(i, mean_val + 50, f'μ={mean_val:.0f}', ha='center', fontsize=9)
    
    ax.set_ylabel('L0 Norm (Number of Active Features)', fontsize=12)
    ax.set_xlabel('Prompting Strategy', fontsize=12)
    
    if layer is not None:
        ax.set_title(f'Sparsity Distribution: CoT vs No-CoT – Layer {layer}', 
                     fontsize=14, fontweight='bold')
    else:
        ax.set_title('Sparsity Distribution: CoT vs No-CoT', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.2, axis='y')
    
    save_path = f"{output_dir}/02_sparsity_distribution.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 3. Intervention Impact – Line Charts (Per-Layer)
# ---------------------------------------------------------------------------

def aggregate_metrics_by_strategy(
    intervention_results: List[InterventionResult],
    acc_results: Dict,
    conf_results: Dict,
    flu_results: Dict,
    layer: Optional[int] = None,
) -> pd.DataFrame:
    """
    Aggregate intervention metrics grouped by k, strategy, and intervention type.
    Optionally filtered by layer.
    
    Returns DataFrame with columns:
        - k (int): number of features
        - strategy (str): "top-k" or "random-k"
        - intervention_type (str): "ablation" or "amplification"
        - accuracy (float)
        - confidence (float)
        - fluency (float)
    """
    rows = []
    
    for spec_label, metrics in acc_results.items():
        # Parse spec label: e.g., "ablation_topk_L2_k16_seed1"
        parts = spec_label.lower().split("_")
        
        # Extract layer from spec label if filtering
        spec_layer = None
        for part in parts:
            if part.startswith("l"):
                try:
                    spec_layer = int(part[1:])
                except:
                    pass
        
        # Skip if filtering and layer doesn't match
        if layer is not None and spec_layer != layer:
            continue
        
        # Extract intervention type
        if "ablation" in spec_label.lower():
            int_type = "ablation"
        elif "amplif" in spec_label.lower():
            int_type = "amplification"
        else:
            continue
        
        # Extract strategy
        if "topk" in spec_label.lower():
            strategy = "top-k"
        elif "randomk" in spec_label.lower():
            strategy = "random-k"
        else:
            continue
        
        # Extract k value
        k_val = 0
        for part in parts:
            if part.startswith("k"):
                try:
                    k_val = int(part[1:])
                except:
                    pass
        
        if k_val == 0:
            continue
        
        # Get metrics
        accuracy = metrics.get("accuracy", 0) * 100
        confidence = conf_results.get(spec_label, {}).get("mean_answer_logprob_per_token", -2.0)
        confidence = max(0, min(100, 100 * np.exp(confidence)))
        fluency = flu_results.get(spec_label, {}).get("mean_perplexity", 10.0)
        # Invert perplexity: lower ppl = higher score
        if fluency > 0:
            fluency = 10.0 / fluency * 100
        else:
            fluency = 0
        
        rows.append({
            "k": k_val,
            "strategy": strategy,
            "intervention_type": int_type,
            "accuracy": accuracy,
            "confidence": confidence,
            "fluency": fluency,
        })
    
    return pd.DataFrame(rows)


def plot_accuracy_comparison(
    metrics_df: pd.DataFrame,
    output_dir: str = "outputs/visualizations",
    layer: Optional[int] = None,
) -> None:
    """
    Line chart: k vs accuracy, with separate lines for each strategy and intervention type.
    Per-layer visualization.
    """
    ensure_dir(output_dir)
    
    if metrics_df.empty:
        logger.warning("No metrics data to plot accuracy comparison")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define line styles
    styles = {
        ("top-k", "ablation"): {"color": "#e74c3c", "linestyle": "-", "marker": "o", "label": "Top-K Ablation"}, # Red
        ("top-k", "amplification"): {"color": "#2ecc71", "linestyle": "-", "marker": "s", "label": "Top-K Amplification"}, # Green
        ("random-k", "ablation"): {"color": "#3498db", "linestyle": "-", "marker": "o", "label": "Random-K Ablation"}, # Blue
        ("random-k", "amplification"): {"color": "#9b59b6", "linestyle": "-", "marker": "s", "label": "Random-K Amplification"}, # Purple
    }
    
    # Plot lines for each strategy and intervention type
    for (strategy, int_type), style in styles.items():
        subset = metrics_df[
            (metrics_df["strategy"] == strategy) & 
            (metrics_df["intervention_type"] == int_type)
        ].sort_values("k")
        
        if len(subset) > 0:
            ax.plot(subset["k"], subset["accuracy"], linewidth=2.5, markersize=8, **style)
    
    ax.set_xlabel("Number of Patched Features (k)", fontsize=12)
    ax.set_ylabel("Answer Accuracy (%)", fontsize=12)
    
    if layer is not None:
        ax.set_title(f"Intervention Impact: Answer Accuracy – Layer {layer}", 
                     fontsize=14, fontweight="bold")
    else:
        ax.set_title("Intervention Impact: Answer Accuracy", fontsize=14, fontweight="bold")
    
    # DYNAMIC SCALING
    y_vals = metrics_df["accuracy"].values
    y_max = max(y_vals) if len(y_vals) > 0 else 100
    y_min = min(y_vals) if len(y_vals) > 0 else 0
    y_range = y_max - y_min
    y_padding = y_range * 0.1  # 10% padding
    ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
    
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.2)
    
    save_path = f"{output_dir}/03a_intervention_accuracy.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {save_path}")
    plt.close()


def plot_confidence_comparison(
    metrics_df: pd.DataFrame,
    output_dir: str = "outputs/visualizations",
    layer: Optional[int] = None,
) -> None:
    """
    Line chart: k vs confidence (log-probability), with separate lines for each strategy and intervention type.
    Per-layer visualization.
    """
    ensure_dir(output_dir)
    
    if metrics_df.empty:
        logger.warning("No metrics data to plot confidence comparison")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    styles = {
        ("top-k", "ablation"): {"color": "#e74c3c", "linestyle": "-", "marker": "o", "label": "Top-K Ablation"}, # Red
        ("top-k", "amplification"): {"color": "#2ecc71", "linestyle": "-", "marker": "s", "label": "Top-K Amplification"}, # Green
        ("random-k", "ablation"): {"color": "#3498db", "linestyle": "-", "marker": "o", "label": "Random-K Ablation"}, # Blue
        ("random-k", "amplification"): {"color": "#9b59b6", "linestyle": "-", "marker": "s", "label": "Random-K Amplification"}, # Purple
    }
    
    for (strategy, int_type), style in styles.items():
        subset = metrics_df[
            (metrics_df["strategy"] == strategy) & 
            (metrics_df["intervention_type"] == int_type)
        ].sort_values("k")
        
        if len(subset) > 0:
            ax.plot(subset["k"], subset["confidence"], linewidth=2.5, markersize=8, **style)
    
    ax.set_xlabel("Number of Patched Features (k)", fontsize=12)
    ax.set_ylabel("Answer Confidence (Log-Prob, %)", fontsize=12)
    
    if layer is not None:
        ax.set_title(f"Intervention Impact: Answer Confidence – Layer {layer}", 
                     fontsize=14, fontweight="bold")
    else:
        ax.set_title("Intervention Impact: Answer Confidence", fontsize=14, fontweight="bold")
    
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.2)
    
    # DYNAMIC SCALING
    y_vals = metrics_df["confidence"].values
    y_max = max(y_vals) if len(y_vals) > 0 else 100
    y_min = min(y_vals) if len(y_vals) > 0 else 0
    y_range = y_max - y_min
    y_padding = y_range * 0.1 if y_range > 0 else 5  
    ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
    
    save_path = f"{output_dir}/03b_intervention_confidence.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {save_path}")
    plt.close()


def plot_fluency_comparison(
    metrics_df: pd.DataFrame,
    output_dir: str = "outputs/visualizations",
    layer: Optional[int] = None,
) -> None:
    """
    Line chart: k vs fluency (inverted perplexity), with separate lines for each strategy and intervention type.
    Per-layer visualization.
    """
    ensure_dir(output_dir)
    
    if metrics_df.empty:
        logger.warning("No metrics data to plot fluency comparison")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    styles = {
        ("top-k", "ablation"): {"color": "#e74c3c", "linestyle": "-", "marker": "o", "label": "Top-K Ablation"}, # Red
        ("top-k", "amplification"): {"color": "#2ecc71", "linestyle": "-", "marker": "s", "label": "Top-K Amplification"}, # Green
        ("random-k", "ablation"): {"color": "#3498db", "linestyle": "-", "marker": "o", "label": "Random-K Ablation"}, # Blue
        ("random-k", "amplification"): {"color": "#9b59b6", "linestyle": "-", "marker": "s", "label": "Random-K Amplification"}, # Purple
    }
    
    for (strategy, int_type), style in styles.items():
        subset = metrics_df[
            (metrics_df["strategy"] == strategy) & 
            (metrics_df["intervention_type"] == int_type)
        ].sort_values("k")
        
        if len(subset) > 0:
            ax.plot(subset["k"], subset["fluency"], linewidth=2.5, markersize=8, **style)
    
    ax.set_xlabel("Number of Patched Features (k)", fontsize=12)
    ax.set_ylabel("Language Fluency (Inverted Perplexity)", fontsize=12)
    
    if layer is not None:
        ax.set_title(f"Intervention Impact: Language Fluency – Layer {layer}", 
                     fontsize=14, fontweight="bold")
    else:
        ax.set_title("Intervention Impact: Language Fluency", fontsize=14, fontweight="bold")
    
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.2)
    
    # DYNAMIC SCALING
    y_vals = metrics_df["fluency"].values
    y_max = max(y_vals) if len(y_vals) > 0 else 100
    y_min = min(y_vals) if len(y_vals) > 0 else 0
    y_range = y_max - y_min
    y_padding = y_range * 0.1 if y_range > 0 else 5
    ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
    
    save_path = f"{output_dir}/03c_intervention_fluency.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {save_path}")
    plt.close()


def plot_intervention_impact(
    metrics_df: pd.DataFrame,
    baseline_acc: float = 0.0,
    baseline_conf: float = 0.0,
    baseline_flu: float = 0.0,
    output_dir: str = "outputs/visualizations",
    layer: Optional[int] = None,
) -> None:
    """
    Grouped bar chart: Intervention types vs 3 metrics (Accuracy, Confidence, Fluency).
    Per-layer visualization with dynamic scaling.
    """
    ensure_dir(output_dir)
    
    if metrics_df.empty:
        logger.warning("No metrics data to plot intervention impact")
        return
    
    # Group by intervention type
    fig, ax = plt.subplots(figsize=(14, 7))
    
    int_types = sorted(metrics_df["intervention_type"].unique())
    x = np.arange(len(int_types))
    width = 0.25
    
    accuracy_vals = [metrics_df[metrics_df["intervention_type"] == t]["accuracy"].mean() 
                     for t in int_types]
    confidence_vals = [metrics_df[metrics_df["intervention_type"] == t]["confidence"].mean() 
                       for t in int_types]
    fluency_vals = [metrics_df[metrics_df["intervention_type"] == t]["fluency"].mean() 
                    for t in int_types]
    
    ax.bar(x - width, accuracy_vals, width, label='Answer Accuracy (%)', color='#3498db')
    ax.bar(x, confidence_vals, width, label='Answer Confidence (Log-Prob, %)', color='#e74c3c')
    ax.bar(x + width, fluency_vals, width, label='Language Fluency (inverted)', color='#2ecc71')

    # Add horizontal baseline reference lines
    if baseline_acc > 0:
        ax.axhline(baseline_acc, color='#3498db', linestyle='dashed', alpha=0.6, linewidth=1.5, label='Baseline Accuracy')
    if baseline_conf > 0:
        ax.axhline(baseline_conf, color='#e74c3c', linestyle='dashed', alpha=0.6, linewidth=1.5, label='Baseline Confidence')
    if baseline_flu > 0:
        ax.axhline(baseline_flu, color='#2ecc71', linestyle='dashed', alpha=0.6, linewidth=1.5, label='Baseline Fluency')
    
    ax.set_xlabel('Intervention Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    
    if layer is not None:
        ax.set_title(f'3-Axis Intervention Impact – Layer {layer}', 
                     fontsize=14, fontweight='bold')
    else:
        ax.set_title('3-Axis Intervention Impact', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(int_types)
    ax.legend(fontsize=10)
    
    # DYNAMIC SCALING: Set y-lim based on actual data
    all_vals = accuracy_vals + confidence_vals + fluency_vals
    max_val = max(all_vals) if all_vals else 100
    y_max = max_val * 1.15  # Add 15% padding
    ax.set_ylim(0, y_max)
    
    ax.grid(True, alpha=0.2, axis='y')
    
    save_path = f"{output_dir}/03_intervention_impact.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {save_path}")
    plt.close()

# ---------------------------------------------------------------------------
# 4. Cross-Layer Cognitive Evolution – Line Chart (CROSS-LAYER ONLY)
# ---------------------------------------------------------------------------

def plot_cross_layer_evolution(
    contrastive_results: Dict[int, ContrastiveResult],
    intervention_results: List[InterventionResult],
    output_dir: str = "outputs/visualizations",
) -> None:
    """
    Line chart: Layers vs accuracy drop when that layer's features are ablated.
    CROSS-LAYER visualization (goes in cross_layer subdirectory).
    """
    ensure_dir(output_dir)
    
    from src.eval_accuracy import evaluate_accuracy
    
    acc_results = evaluate_accuracy(intervention_results)
    
    # Group ablation results by layer
    layer_impacts = {}
    for spec_label, metrics in acc_results.items():
        if "ablation" not in spec_label:
            continue
        
        # Extract layer number
        layer_num = None
        for part in spec_label.split("_"):
            if part.startswith("L"):
                try:
                    layer_num = int(part[1:])
                except:
                    pass
        
        if layer_num is not None:
            if layer_num not in layer_impacts:
                layer_impacts[layer_num] = []
            layer_impacts[layer_num].append(metrics["accuracy"])
    
    if not layer_impacts:
        logger.warning("No ablation results found for cross-layer evolution")
        return
    
    # Compute mean accuracy drop per layer
    layers = sorted(layer_impacts.keys())
    mean_accuracies = [np.mean(layer_impacts[l]) * 100 for l in layers]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, mean_accuracies, marker='o', linewidth=2.5, markersize=8, color='#e74c3c')
    ax.fill_between(layers, mean_accuracies, alpha=0.2, color='#e74c3c')
    
    ax.set_xlabel('Transformer Layer', fontsize=12)
    ax.set_ylabel('Model Accuracy After Ablation (%)', fontsize=12)
    ax.set_title('Cross-Layer Cognitive Evolution', fontsize=14, fontweight='bold')
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    save_path = f"{output_dir}/04_cross_layer_evolution.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Qualitative "Lobotomy" Table (PER-LAYER)
# ---------------------------------------------------------------------------

def generate_lobotomy_table(
    baseline_results: List[Dict],
    intervention_results: List[InterventionResult],
    layer: Optional[int] = None,
    num_examples: int = 3,
    output_dir: str = "outputs/visualizations",
) -> None:
    """
    Side-by-side text table showing model outputs before/after ablation.
    Highlights where math turns into gibberish.
    Per-layer visualization (filtered by layer if specified).
    """
    ensure_dir(output_dir)
    
    # Filter ablation results by layer if specified
    ablation_results = [r for r in intervention_results if r.intervention_type == "ablation"]
    if layer is not None:
        ablation_results = [r for r in ablation_results if r.layer == layer]
    
    if not ablation_results:
        logger.warning(f"No ablation results found for layer {layer}. Skipping lobotomy table.")
        return
    
    # Match baseline to intervention results
    table_rows = []
    for i, ablation_res in enumerate(ablation_results[:num_examples]):
        # Find corresponding baseline
        baseline = next((b for b in baseline_results 
                        if b["sample_idx"] == ablation_res.sample_idx), None)
        if baseline is None:
            continue
        
        table_rows.append({
            "sample_idx": ablation_res.sample_idx,
            "layer": ablation_res.layer,
            "prompt": ablation_res.original_prompt[:100] + "...",
            "baseline_output": baseline["generated_text"][:200],
            "ablated_output": ablation_res.generated_text[:200],
            "ground_truth": ablation_res.ground_truth,
        })
    
    # Save as JSON (can be rendered as HTML table)
    table_path = f"{output_dir}/05_lobotomy_table.json"
    save_json(table_rows, table_path)
    
    # Also create a simple text version
    text_path = f"{output_dir}/05_lobotomy_table.txt"
    with open(text_path, "w") as f:
        f.write("=" * 100 + "\n")
        if layer is not None:
            f.write(f"QUALITATIVE ANALYSIS: MODEL BEHAVIOR BEFORE & AFTER REASONING FEATURE ABLATION – LAYER {layer}\n")
        else:
            f.write("QUALITATIVE ANALYSIS: MODEL BEHAVIOR BEFORE & AFTER REASONING FEATURE ABLATION\n")
        f.write("=" * 100 + "\n\n")
        
        for row in table_rows:
            f.write(f"\n--- Sample {row['sample_idx']} (Layer {row['layer']}) ---\n")
            f.write(f"Prompt (truncated): {row['prompt']}\n\n")
            f.write(f"BASELINE OUTPUT:\n{row['baseline_output']}\n\n")
            f.write(f"AFTER ABLATION:\n{row['ablated_output']}\n\n")
            f.write(f"Ground Truth: {row['ground_truth']}\n")
            f.write("-" * 100 + "\n")
    
    logger.info(f"Saved: {table_path}")
    logger.info(f"Saved: {text_path}")


# ---------------------------------------------------------------------------
# Main visualization pipeline (REFACTORED FOR LAYER-BASED ORGANIZATION)
# ---------------------------------------------------------------------------

def generate_all_visualizations(
    contrastive_results: Dict[int, ContrastiveResult],
    sae_features: Dict[str, Dict[int, torch.Tensor]],
    baseline_results: List[Dict],
    intervention_results: List[InterventionResult],
    model,
    confidence_baseline: Optional[Dict] = None,
    confidence_intervention: Optional[Dict] = None,
    output_dir: str = "outputs/visualizations",
    layers: Optional[List[int]] = None,
    # ADD THESE FLAGS:
    generate_accuracy_plots: bool = True,
    generate_fluency_plots: bool = True,
    generate_confidence_plots: bool = True,
) -> None:
    """
    Generate all publication-ready visualizations organized by layer.
    
    Layers are determined from config (passed via 'layers' parameter).
    If 'layers' is None, defaults to sorted keys from contrastive_results.
    
    Directory structure (example with layers=[2, 4, 6]):
    - output_dir/
      - layer_2/
        - 01_feature_isolation_scatter.png
        - 02_sparsity_distribution.png
        - 03_intervention_impact.png
        - 03a_intervention_accuracy.png
        - 03b_intervention_confidence.png
        - 03c_intervention_fluency.png
        - 05_lobotomy_table.json
        - 05_lobotomy_table.txt
      - layer_4/
        - (same files as layer_2)
      - layer_6/
        - (same files as layer_2)
      - cross_layer/
        - 04_cross_layer_evolution.png
    
    Args:
        contrastive_results: Dict mapping layer -> ContrastiveResult
        sae_features: Dict mapping condition -> layer -> features
        baseline_results: List of baseline generation results
        intervention_results: List of InterventionResult objects
        model: The model for evaluation
        confidence_baseline: Optional baseline confidence metrics
        confidence_intervention: Optional intervention confidence metrics
        output_dir: Root output directory for visualizations
        layers: Optional list of layer indices from config.layers.all_layers.
               If None, uses sorted keys from contrastive_results.
    """
    import torch
    
    ensure_dir(output_dir)
    logger.info("=" * 60)
    logger.info("Generating Publication Visualizations (Per-Layer Organization)")
    logger.info("=" * 60)
    
    if layers is None:
        layers = sorted(contrastive_results.keys())
    
    # Global evaluation results (used for all layers)
        # Global evaluation results (used for all layers)
    acc_results = None
    conf_results = None
    flu_results = None
    
    if generate_accuracy_plots:
        from src.eval_accuracy import evaluate_accuracy
        acc_results = evaluate_accuracy(intervention_results)
    
    if generate_confidence_plots:
        conf_results = evaluate_confidence(intervention_results, model)
    
    if generate_fluency_plots:
        from src.eval_fluency import evaluate_fluency
        flu_results = evaluate_fluency(intervention_results, model, None)

    
    # Generate visualizations for each layer
    for layer in layers:
        logger.info(f"\nProcessing Layer {layer}...")
        layer_dir = f"{output_dir}/layer_{layer}"
        ensure_dir(layer_dir)
        
        # 1. Feature Isolation Scatter (always run - doesn't depend on interventions)
        logger.info(f"  1. Feature isolation scatter plot...")
        plot_feature_isolation_scatter(contrastive_results, layer_dir, layer=layer)
        
        # 2. Sparsity Distribution (always run - it's foundational)
        logger.info(f"  2. Sparsity distribution...")
        layer_sae_features = filter_sae_features_by_layer(sae_features, layer)
        if layer_sae_features:
            sparsity_data = compute_sparsity_stats_per_layer(layer_sae_features)
            plot_sparsity_distribution(sparsity_data, layer_dir, layer=layer)
        
        # 3. Intervention Impact Comparison (per-layer) - CONDITIONAL
        if generate_confidence_plots or generate_accuracy_plots or generate_fluency_plots:
            logger.info(f"  3. Intervention comparison charts...")
            metrics_df = aggregate_metrics_by_strategy(
                intervention_results,
                acc_results or {},
                conf_results or {},
                flu_results or {},
                layer=layer,
            )
            
            if not metrics_df.empty:
                if generate_accuracy_plots:
                    plot_accuracy_comparison(metrics_df, layer_dir, layer=layer)
                if generate_confidence_plots:
                    plot_confidence_comparison(metrics_df, layer_dir, layer=layer)
                if generate_fluency_plots:
                    plot_fluency_comparison(metrics_df, layer_dir, layer=layer)
                
                # Only call plot_intervention_impact if we have all three metrics
                if generate_accuracy_plots and generate_confidence_plots and generate_fluency_plots:
                    base_a = evaluate_accuracy_baseline(baseline_results)['accuracy'] * 100 if baseline_results else 0
                    base_c = max(0, min(100, 100 * np.exp(confidence_baseline.get("mean_answer_logprob_per_token", -2.0)))) if confidence_baseline else 0 
                    plot_intervention_impact(metrics_df, base_a, base_c, 0.0, layer_dir, layer=layer)
            else:
                logger.info(f"  (No intervention data for layer {layer})")
        
        # 5. Lobotomy Table (per-layer) - only if interventions were run
        if generate_accuracy_plots or generate_fluency_plots:
            logger.info(f"  5. Qualitative lobotomy table...")
            generate_lobotomy_table(
                baseline_results, 
                intervention_results, 
                layer=layer,
                output_dir=layer_dir
            )
    
    # Generate cross-layer visualizations only if we have accuracy results
    if generate_accuracy_plots:
        logger.info("\nProcessing Cross-Layer Visualizations...")
        cross_layer_dir = f"{output_dir}/cross_layer"
        ensure_dir(cross_layer_dir)
        
        # 4. Cross-Layer Evolution (cross-layer only)
        logger.info("  4. Cross-layer cognitive evolution...")
        plot_cross_layer_evolution(contrastive_results, intervention_results, cross_layer_dir)

    logger.info("=" * 60)
    logger.info("All visualizations complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("Directory structure:")
    logger.info(f"  - {output_dir}/layer_0/")
    logger.info(f"  - {output_dir}/layer_2/")
    logger.info(f"  - ... (one per layer)")
    logger.info(f"  - {output_dir}/cross_layer/")
    logger.info("=" * 60)