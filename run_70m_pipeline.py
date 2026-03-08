"""
Pipeline to reproduce Pythia-70M CoT faithfulness results.
Covers: activation collection, SAE training, and experiment execution.

Usage:
    python run_70m_pipeline.py --phase all       # Run everything
    python run_70m_pipeline.py --phase collect    # Phase 1: Collect activations
    python run_70m_pipeline.py --phase train      # Phase 2: Train SAEs
    python run_70m_pipeline.py --phase experiment  # Phase 3: Run experiments
    python run_70m_pipeline.py --phase verify     # Quick sanity checks
"""

import argparse
import os
import sys
import pickle

import torch
import numpy as np

# Add sparse_coding to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SC_DIR = os.path.join(SCRIPT_DIR, "sparse_coding")
sys.path.insert(0, SC_DIR)

from device_utils import get_device, get_device_for_training, get_model_dtype, empty_cache

# --- Configuration ---
MODEL_NAME = "EleutherAI/pythia-70m"
LAYER = 2
LAYER_LOC = "residual"
DATASET_NAME = "openai/gsm8k"
DICT_RATIOS = [4, 8]
L1_VALUES = [0, 1e-3, 3e-4, 1e-4]
BEST_RANK = 2  # index into L1_VALUES for 3e-4
N_CHUNKS = 10
CHUNK_SIZE_GB = 0.2  # small chunks for laptop
BATCH_SIZE = 128

# Directories
DATA_DIR = os.path.join(SC_DIR, "data")
NOCOT_ACTS_DIR = os.path.join(DATA_DIR, "nocot_acts")
COT_ACTS_DIR = os.path.join(DATA_DIR, "cot_acts")
OUTPUT_DIR = os.path.join(SC_DIR, "output")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def phase_collect():
    """Phase 1: Collect CoT and NoCoT activations from Pythia-70M."""
    from transformer_lens import HookedTransformer
    from activation_dataset import setup_data

    device = get_device()
    dtype = get_model_dtype()
    print(f"Loading {MODEL_NAME} on {device} with dtype={dtype}")

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        torch_dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
    )
    tokenizer = model.tokenizer

    for use_cot, acts_dir, label in [
        (False, NOCOT_ACTS_DIR, "NoCoT"),
        (True, COT_ACTS_DIR, "CoT"),
    ]:
        if os.path.exists(os.path.join(acts_dir, "0.pt")):
            print(f"  {label} activations already exist in {acts_dir}, skipping.")
            continue

        os.makedirs(acts_dir, exist_ok=True)
        print(f"  Collecting {label} activations → {acts_dir}")
        n = setup_data(
            tokenizer,
            model,
            dataset_name=DATASET_NAME,
            dataset_folder=acts_dir,
            layer=LAYER,
            layer_loc=LAYER_LOC,
            n_chunks=N_CHUNKS,
            chunk_size_gb=CHUNK_SIZE_GB,
            device=device,
            use_cot=use_cot,
            shuffle=False,
        )
        print(f"  {label}: collected {n} activation vectors.")

    empty_cache()
    print("Phase 1 complete: activations collected.")


def phase_train():
    """Phase 2: Train SAEs on collected activations (CPU-only, uses torch.vmap)."""
    import torchopt
    from autoencoders.sae_ensemble import FunctionalTiedSAE
    from autoencoders.ensemble import FunctionalEnsemble
    from big_sweep import ensemble_train_loop, unstacked_to_learned_dicts
    from config import TrainArgs

    train_device = str(get_device_for_training())
    print(f"Training SAEs on device: {train_device}")

    for ratio in DICT_RATIOS:
        for acts_dir, label in [
            (NOCOT_ACTS_DIR, "nocot"),
            (COT_ACTS_DIR, "cot"),
        ]:
            out_dir = os.path.join(OUTPUT_DIR, f"{label}_sae_r{ratio}")
            out_file = os.path.join(out_dir, "learned_dicts.pt")
            pkl_file = os.path.join(out_dir, "dict.pkl")

            if os.path.exists(pkl_file):
                print(f"  {label} r{ratio} SAE already trained ({pkl_file}), skipping.")
                continue

            os.makedirs(out_dir, exist_ok=True)
            print(f"  Training {label} SAE (ratio={ratio})...")

            # Load first chunk to get activation dim
            chunk0 = torch.load(os.path.join(acts_dir, "0.pt"))
            activation_dim = chunk0.shape[1]
            latent_dim = int(activation_dim * ratio)
            del chunk0

            # Initialize ensemble of SAEs (one per L1 value)
            models = [
                FunctionalTiedSAE.init(activation_dim, latent_dim, l1, device=train_device)
                for l1 in L1_VALUES
            ]
            ensemble = FunctionalEnsemble(
                models, FunctionalTiedSAE,
                torchopt.adam, {"lr": 1e-3},
                device=train_device,
            )
            args = {
                "batch_size": BATCH_SIZE,
                "device": train_device,
                "dict_size": latent_dim,
            }

            # Train on all chunks
            n_chunks = len([f for f in os.listdir(acts_dir) if f.endswith(".pt") and not f.startswith("input_ids")])
            chunk_order = np.random.permutation(n_chunks)

            for ci, chunk_id in enumerate(chunk_order):
                chunk_path = os.path.join(acts_dir, f"{chunk_id}.pt")
                if not os.path.exists(chunk_path):
                    continue
                dataset = torch.load(chunk_path).to(dtype=torch.float32)

                sampler = torch.utils.data.BatchSampler(
                    torch.utils.data.RandomSampler(range(dataset.shape[0])),
                    batch_size=BATCH_SIZE,
                    drop_last=False,
                )

                cfg = TrainArgs.__new__(TrainArgs)
                cfg.use_wandb = False

                print(f"    Chunk {ci+1}/{n_chunks} (id={chunk_id}, {dataset.shape[0]} samples)")

                class ProgressCounter:
                    def __init__(self): self.value = 0

                ensemble_train_loop(ensemble, cfg, args, f"{label}_r{ratio}", sampler, dataset, ProgressCounter())

            # Save
            learned_dicts = unstacked_to_learned_dicts(ensemble, args, ["dict_size"], ["l1_alpha"])
            torch.save(learned_dicts, out_file)

            # Convert to pickle format expected by experiment scripts
            # dict[layer][rank] = (TiedSAE, hyperparams)
            save_dict = {LAYER: learned_dicts}
            with open(pkl_file, "wb") as f:
                pickle.dump(save_dict, f)

            print(f"  Saved {label} r{ratio}: {out_file} and {pkl_file}")

    print("Phase 2 complete: SAEs trained.")


def phase_experiment():
    """Phase 3: Run activation patching, patch curves, and box plot experiments."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exp_dir = os.path.join(SCRIPT_DIR, "experiments")

    for ratio in DICT_RATIOS:
        nocot_pkl = os.path.join(OUTPUT_DIR, f"nocot_sae_r{ratio}", "dict.pkl")
        cot_pkl = os.path.join(OUTPUT_DIR, f"cot_sae_r{ratio}", "dict.pkl")

        if not os.path.exists(nocot_pkl) or not os.path.exists(cot_pkl):
            print(f"  SAE pickles for ratio {ratio} not found, skipping. Run --phase train first.")
            continue

        # Activation Patching
        out_patch = os.path.join(RESULTS_DIR, f"patching_hist_r{ratio}.png")
        if not os.path.exists(out_patch):
            print(f"  Running activation patching (ratio={ratio})...")
            _run_script(exp_dir, [
                sys.executable, os.path.join(exp_dir, "activated_patching.py"),
                "--model", MODEL_NAME,
                "--layer", str(LAYER), "--layer_loc", "resid",
                "--rank", str(BEST_RANK),
                "--dict_nocot", nocot_pkl,
                "--dict_cot", cot_pkl,
                "--acts_nocot_dir", NOCOT_ACTS_DIR,
                "--acts_cot_dir", COT_ACTS_DIR,
                "--topk", "20", "--max_samples", "500",
                "--out", out_patch,
            ])
        else:
            print(f"  {out_patch} exists, skipping.")

        # Patch Curve
        out_curve = os.path.join(RESULTS_DIR, f"patch_curve_r{ratio}.png")
        if not os.path.exists(out_curve):
            print(f"  Running patch curve (ratio={ratio})...")
            _run_script(exp_dir, [
                sys.executable, os.path.join(exp_dir, "patch_curve.py"),
                "--model", MODEL_NAME,
                "--layer", str(LAYER), "--layer_loc", "resid",
                "--rank", str(BEST_RANK),
                "--dict_nocot", nocot_pkl,
                "--dict_cot", cot_pkl,
                "--acts_nocot_dir", NOCOT_ACTS_DIR,
                "--acts_cot_dir", COT_ACTS_DIR,
                "--max_samples", "500",
                "--out", out_curve,
                "--save_stats", os.path.join(RESULTS_DIR, f"ttest_r{ratio}.txt"),
            ])
        else:
            print(f"  {out_curve} exists, skipping.")

    # Box plots (use ratio 4 SAE)
    for label, pkl_subdir in [("nocot", "nocot_sae_r4"), ("cot", "cot_sae_r4")]:
        pkl_path = os.path.join(OUTPUT_DIR, pkl_subdir, "dict.pkl")
        out_box = os.path.join(RESULTS_DIR, f"boxplot_{label}.png")
        if os.path.exists(pkl_path) and not os.path.exists(out_box):
            print(f"  Running box plot ({label})...")
            _run_script(exp_dir, [
                sys.executable, os.path.join(exp_dir, "activated_box_plot.py"),
                "--model", MODEL_NAME,
                "--sae_pkl", pkl_path,
                "--layer", str(LAYER), "--layer_loc", "resid",
                "--out", out_box,
            ])
        elif os.path.exists(out_box):
            print(f"  {out_box} exists, skipping.")

    print(f"Phase 3 complete: results in {RESULTS_DIR}/")


def _run_script(cwd, cmd):
    """Run a Python script as a subprocess."""
    import subprocess
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: script exited with code {result.returncode}")


def phase_verify():
    """Quick sanity checks across all phases."""
    print("=== Verification ===")

    # Phase 0: device_utils
    device = get_device()
    print(f"[OK] Device: {device}")

    import torchopt
    print(f"[OK] torchopt {torchopt.__version__}")

    from autoencoders.sae_ensemble import FunctionalTiedSAE
    p, b = FunctionalTiedSAE.init(512, 2048, 1e-3)
    assert "bias_decay" in b, "bias_decay buffer missing!"
    print("[OK] FunctionalTiedSAE.init has bias_decay buffer")

    # Phase 1: activations
    for label, d in [("NoCoT", NOCOT_ACTS_DIR), ("CoT", COT_ACTS_DIR)]:
        p0 = os.path.join(d, "0.pt")
        if os.path.exists(p0):
            t = torch.load(p0, weights_only=True)
            print(f"[OK] {label} acts: shape={t.shape}, dtype={t.dtype}")
        else:
            print(f"[--] {label} acts not found at {d}")

    # Phase 2: SAEs
    for ratio in DICT_RATIOS:
        for label in ["nocot", "cot"]:
            pkl = os.path.join(OUTPUT_DIR, f"{label}_sae_r{ratio}", "dict.pkl")
            if os.path.exists(pkl):
                with open(pkl, "rb") as f:
                    d = pickle.load(f)
                sae = d[LAYER][BEST_RANK][0]
                print(f"[OK] {label} r{ratio} SAE: {type(sae).__name__}")
            else:
                print(f"[--] {label} r{ratio} SAE not found")

    # Phase 3: results
    for f in os.listdir(RESULTS_DIR) if os.path.isdir(RESULTS_DIR) else []:
        print(f"[OK] Result: {f}")

    print("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Pythia-70M CoT faithfulness results")
    parser.add_argument("--phase", default="all",
                        choices=["all", "collect", "train", "experiment", "verify"],
                        help="Which phase to run")
    args = parser.parse_args()

    if args.phase == "verify":
        phase_verify()
    elif args.phase == "collect":
        phase_collect()
    elif args.phase == "train":
        phase_train()
    elif args.phase == "experiment":
        phase_experiment()
    elif args.phase == "all":
        phase_collect()
        phase_train()
        phase_experiment()
        phase_verify()
