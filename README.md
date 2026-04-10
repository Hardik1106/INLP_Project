# Mechanistic Faithfulness of Chain-of-Thought (CoT)

Investigating whether internal features active during Chain-of-Thought prompting
are causally necessary for logical reasoning via targeted ablation and amplification
interventions using Sparse Autoencoders (SAEs).

## Project Structure

```
ours/
├── configs/
│   └── default.yaml          # Central experiment configuration
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration dataclasses
│   ├── data_pipeline.py       # Module 1: Dataset loading & prompt formatting
│   ├── cache_activations.py   # Module 1: Activation extraction & caching
│   ├── sae_encoder.py         # Module 2: SAE encoding of cached activations
│   ├── contrastive_analysis.py# Module 2: Contrastive filtering & feature selection
│   ├── interventions.py       # Module 3: Ablation & amplification hooks
│   ├── eval_accuracy.py       # Module 4: Answer accuracy scoring
│   ├── eval_coherence.py      # Module 4: LLM-as-a-judge coherence scoring
│   ├── eval_fluency.py        # Module 4: Perplexity-based fluency evaluation
│   └── utils.py               # Shared utilities
├── scripts/
│   ├── run_experiments.sh     # Cross-model sweep launcher
│   └── run_pipeline.py        # Full pipeline orchestrator
├── data/                      # Cached datasets & activations
├── outputs/                   # Experiment results
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --config configs/default.yaml
```

## Experiments

1. **Contrastive Feature Extraction** — Intra-dataset (CoT vs No-CoT) and inter-dataset (GSM8K vs TriviaQA) contrasts
2. **Cross-Layer Analysis** — Feature extraction at shallow (Layer 2) and deep (Layers 12–16) transformer layers
3. **Targeted Causal Interventions** — Top-K and Random-K ablation/amplification of candidate reasoning features
4. **Three-Axis Evaluation** — Answer accuracy, reasoning coherence, language fluency
5. **Cross-Model Generalization** — Pythia, Llama, Gemma architectures


The code for training SAEs is adapted from `thecotFaithfulness` repo:
https://github.com/sekirodie1000/cotFaithfulness,  
based on the works of [Chen et al 2025](https://arxiv.org/abs/2507.22928).
