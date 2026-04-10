#!/usr/bin/env python3
"""
Convert trained SAEs from cotFaithfulness to INLP format.
Run this AFTER big_sweep_experiments.py completes training.
"""
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# SRC_PATH = PROJECT_ROOT / "src"
print({PROJECT_ROOT})
sys.path.insert(0, str(PROJECT_ROOT))

COT_FAITHFULNESS_ROOT = Path("/ssd_scratch/arnav.sharma/INLP/project/cotFaithfulness_ours")
sys.path.insert(0, str(COT_FAITHFULNESS_ROOT))
sys.path.insert(0, str(COT_FAITHFULNESS_ROOT / "sparse_coding"))

from src.sae_converter import batch_convert_from_mnt_ssd
from src.utils import setup_logger, logger

if __name__ == "__main__":
    setup_logger("sae_converter", "DEBUG")
    
    # Define your SAE sources (one per training run)
    sae_sources = [
        # {
        #     "mnt_path": "/ssd_scratch/arnav.sharma/INLP/project/cotFaithfulness_ours/sparse_coding/mnt/ssd-cluster/pythia70m_centered_gsm8k_NoCoT/",
        #     "output_dir": "./local_sae_checkpoints/pythia70m_nocot",
        #     "name": "Pythia-70M No-CoT",
        # },
        # {
        #     "mnt_path": "/ssd_scratch/arnav.sharma/INLP/project/cotFaithfulness_ours/sparse_coding/mnt/ssd-cluster/pythia70m_centered_gsm8k_CoT",
        #     "output_dir": "./local_sae_checkpoints/pythia70m_cot",
        #     "name": "Pythia-70M CoT",
        # },
        # {
        #     "mnt_path": "/ssd_scratch/arnav.sharma/INLP/project/cotFaithfulness_ours/sparse_coding/mnt/ssd-cluster/pythia2.8b_centered_gsm8k_NoCoT",
        #     "output_dir": "./local_sae_checkpoints/pythia2.8b_nocot",
        #     "name": "Pythia-2.8B No-CoT",
        # },
        {
            "mnt_path": "/ssd_scratch/arnav.sharma/INLP/project/cotFaithfulness_ours/sparse_coding/mnt/ssd-cluster/pythia2.8b_centered_gsm8k_CoT",
            "output_dir": "./local_sae_checkpoints/pythia2.8b_cot",
            "name": "Pythia-2.8B No-CoT",
        },
    ]
    
    all_converted = {}
    
    for source in sae_sources:
        logger.info(f"\n{'='*70}")
        logger.info(f"Converting {source['name']}")
        logger.info(f"{'='*70}")
        
        converted = batch_convert_from_mnt_ssd(
            mnt_base_path=source["mnt_path"],
            output_dir=source["output_dir"],
            use_final_checkpoint=True,
        )
        
        all_converted.update(converted)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"CONVERSION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total SAEs converted: {len(all_converted)}\n")
    
    for key, path in sorted(all_converted.items()):
        logger.info(f"✓ {key:<50} → {path}")