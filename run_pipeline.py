"""
Reproduce Pythia-70M CoT faithfulness results (Figures 3/5/7/8/9 from the paper).
Skips automated interpretability. Runs on Windows/AMD via DirectML + CPU.

Usage:
    python run_pipeline.py --phase collect     # Phase 1: Collect activations
    python run_pipeline.py --phase train       # Phase 2: Train SAEs
    python run_pipeline.py --phase experiment  # Phase 3: Run experiments
    python run_pipeline.py --phase all         # Run everything
"""

import argparse
import os
import sys
import pickle
import subprocess

# Ensure subprocesses can print Unicode on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"

# Use the venv Python executable, not sys.executable (which may point to system Python)
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Scripts", "python.exe")
if not os.path.isfile(VENV_PYTHON):
    VENV_PYTHON = sys.executable  # fallback

import torch
import numpy as np

# Ensure sparse_coding is importable
SC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse_coding")
sys.path.insert(0, SC_DIR)

from device_utils import get_device, get_device_for_training, get_model_dtype, empty_cache

# ---------- constants matching the paper ----------
MODEL_NAME = "EleutherAI/pythia-70m"
DATASET_NAME = "openai/gsm8k"
LAYER = 2
LAYER_LOC = "residual"
DICT_RATIOS = [4, 8]
L1_VALUES = [0, 1e-3, 3e-4, 1e-4]
BATCH_SIZE = 256
N_CHUNKS = 10
CHUNK_SIZE_GB = 0.2  # small chunks for laptop

DATA_DIR = os.path.join(SC_DIR, "data")
OUTPUT_DIR = os.path.join(SC_DIR, "output")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# =====================================================================
# Phase 1 — Collect activations
# =====================================================================
def phase_collect():
    from transformer_lens import HookedTransformer
    from activation_dataset import setup_data

    device = get_device()
    dtype = get_model_dtype()
    print(f"[collect] Loading {MODEL_NAME} on {device} (dtype={dtype})")

    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device=str(device), torch_dtype=dtype,
        fold_ln=False, center_writing_weights=False,
    )
    tokenizer = model.tokenizer

    for use_cot, tag in [(False, "nocot"), (True, "cot")]:
        out_dir = os.path.join(DATA_DIR, f"{tag}_acts")
        if os.path.exists(os.path.join(out_dir, "0.pt")):
            print(f"[collect] {out_dir} already has data, skipping. Delete folder to re-collect.")
            continue
        os.makedirs(out_dir, exist_ok=True)
        print(f"[collect] Collecting {tag.upper()} activations -> {out_dir}")
        setup_data(
            tokenizer, model,
            dataset_name=DATASET_NAME,
            dataset_folder=out_dir,
            layer=LAYER,
            layer_loc=LAYER_LOC,
            n_chunks=N_CHUNKS,
            chunk_size_gb=CHUNK_SIZE_GB,
            device=torch.device(str(device)),
            use_cot=use_cot,
            shuffle=False,
        )
        print(f"[collect] Done: {tag.upper()}")

    empty_cache()
    print("[collect] Phase 1 complete.")


# =====================================================================
# Phase 2 — Train SAEs (on CPU via torch.vmap)
# =====================================================================
def phase_train():
    import torchopt
    from autoencoders.sae_ensemble import FunctionalTiedSAE
    from autoencoders.ensemble import FunctionalEnsemble
    from big_sweep import ensemble_train_loop, unstacked_to_learned_dicts

    train_device = str(get_device_for_training())
    print(f"[train] Using device: {train_device}")

    for tag in ["nocot", "cot"]:
        acts_dir = os.path.join(DATA_DIR, f"{tag}_acts")
        if not os.path.exists(os.path.join(acts_dir, "0.pt")):
            print(f"[train] ERROR: No activations in {acts_dir}. Run --phase collect first.")
            return

        for ratio in DICT_RATIOS:
            out_dir = os.path.join(OUTPUT_DIR, f"{tag}_sae_r{ratio}")
            final_path = os.path.join(out_dir, "learned_dicts.pt")
            if os.path.exists(final_path):
                print(f"[train] {final_path} exists, skipping. Delete to retrain.")
                continue

            os.makedirs(out_dir, exist_ok=True)

            # Peek at activation dim
            sample = torch.load(os.path.join(acts_dir, "0.pt"), map_location="cpu")
            activation_dim = sample.shape[1]
            latent_dim = int(activation_dim * ratio)
            del sample

            print(f"[train] Training {tag.upper()} ratio={ratio}  "
                  f"({activation_dim}→{latent_dim})  l1s={L1_VALUES}")

            models = [
                FunctionalTiedSAE.init(activation_dim, latent_dim, l1, device=train_device)
                for l1 in L1_VALUES
            ]
            ensemble = FunctionalEnsemble(
                models, FunctionalTiedSAE,
                torchopt.adam, {"lr": 1e-3},
                device=train_device,
            )
            args = {"batch_size": BATCH_SIZE, "device": train_device, "dict_size": latent_dim}

            n_chunks = len([f for f in os.listdir(acts_dir) if f.endswith(".pt") and not f.startswith("input_ids")])
            chunk_order = np.random.permutation(n_chunks)

            for ci, chunk_id in enumerate(chunk_order):
                chunk_path = os.path.join(acts_dir, f"{chunk_id}.pt")
                if not os.path.exists(chunk_path):
                    continue
                dataset = torch.load(chunk_path, map_location="cpu").to(dtype=torch.float32)
                sampler = torch.utils.data.BatchSampler(
                    torch.utils.data.RandomSampler(range(dataset.shape[0])),
                    batch_size=BATCH_SIZE, drop_last=False,
                )

                class ProgressCounter:
                    def __init__(self): self.value = 0

                class MinimalCfg:
                    use_wandb = False

                print(f"  chunk {ci+1}/{n_chunks} ({chunk_path}, {dataset.shape[0]} samples)")
                ensemble_train_loop(ensemble, MinimalCfg(), args, "sweep", sampler, dataset, ProgressCounter())

            learned_dicts = unstacked_to_learned_dicts(ensemble, args, ["dict_size"], ["l1_alpha"])
            torch.save(learned_dicts, final_path)

            # Also save as pickle format expected by experiment scripts:
            # dict[layer][rank] = [(sae, hyperparams), ...]
            pkl_path = os.path.join(out_dir, "dict.pkl")
            save_dict = {LAYER: list(enumerate([(ld, hp) for ld, hp in learned_dicts]))}
            # experiment scripts index as dict[layer][rank][0]
            # so we need: dict[layer] = [(sae, hp), (sae, hp), ...]
            save_dict = {LAYER: [(ld, hp) for ld, hp in learned_dicts]}
            with open(pkl_path, "wb") as f:
                pickle.dump(save_dict, f)

            print(f"[train] Saved {final_path} and {pkl_path}")

    print("[train] Phase 2 complete.")


# =====================================================================
# Phase 3 — Experiments
# =====================================================================
def phase_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")

    # Pick best rank (use rank=2 = l1=3e-4 as default, matching paper)
    rank = 2

    def run_exp(script_args):
        """Run experiment script using venv Python via subprocess."""
        cmd = [VENV_PYTHON] + script_args
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  [WARNING] Script exited with code {result.returncode}, continuing...")

    for ratio in DICT_RATIOS:
        nocot_pkl = os.path.join(OUTPUT_DIR, f"nocot_sae_r{ratio}", "dict.pkl")
        cot_pkl = os.path.join(OUTPUT_DIR, f"cot_sae_r{ratio}", "dict.pkl")
        nocot_acts = os.path.join(DATA_DIR, "nocot_acts")
        cot_acts = os.path.join(DATA_DIR, "cot_acts")

        for pkl in [nocot_pkl, cot_pkl]:
            if not os.path.exists(pkl):
                print(f"[experiment] Missing {pkl}. Run --phase train first.")
                return

        # 3a. Activation Patching histogram
        print(f"\n[experiment] Activation Patching (ratio={ratio})")
        run_exp([
            os.path.join(exp_dir, "activated_patching.py"),
            "--model", MODEL_NAME, "--layer", str(LAYER), "--layer_loc", "resid",
            "--rank", str(rank),
            "--dict_nocot", nocot_pkl, "--dict_cot", cot_pkl,
            "--acts_nocot_dir", nocot_acts, "--acts_cot_dir", cot_acts,
            "--topk", "20", "--max_samples", "1000",
            "--out", os.path.join(RESULTS_DIR, f"patching_hist_r{ratio}.png"),
        ])

        # 3b. Patch Curve
        print(f"\n[experiment] Patch Curve (ratio={ratio})")
        run_exp([
            os.path.join(exp_dir, "patch_curve.py"),
            "--model", MODEL_NAME, "--layer", str(LAYER), "--layer_loc", "resid",
            "--rank", str(rank),
            "--dict_nocot", nocot_pkl, "--dict_cot", cot_pkl,
            "--acts_nocot_dir", nocot_acts, "--acts_cot_dir", cot_acts,
            "--max_samples", "1000",
            "--out", os.path.join(RESULTS_DIR, f"patch_curve_r{ratio}.png"),
            "--save_stats", os.path.join(RESULTS_DIR, f"ttest_r{ratio}.txt"),
        ])

    # 3c. Box plots
    for tag in ["nocot", "cot"]:
        pkl = os.path.join(OUTPUT_DIR, f"{tag}_sae_r4", "dict.pkl")
        print(f"\n[experiment] Box Plot ({tag.upper()})")
        run_exp([
            os.path.join(exp_dir, "activated_box_plot.py"),
            "--model", MODEL_NAME, "--sae_pkl", pkl,
            "--layer", str(LAYER), "--layer_loc", "resid",
            "--out", os.path.join(RESULTS_DIR, f"boxplot_{tag}.png"),
        ])

    print(f"\n[experiment] Phase 3 complete. Results in {RESULTS_DIR}/")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pythia-70M CoT Faithfulness Reproduction")
    parser.add_argument("--phase", choices=["collect", "train", "experiment", "all"],
                        required=True, help="Which phase to run")
    args = parser.parse_args()

    if args.phase in ("collect", "all"):
        phase_collect()
    if args.phase in ("train", "all"):
        phase_train()
    if args.phase in ("experiment", "all"):
        phase_experiment()
