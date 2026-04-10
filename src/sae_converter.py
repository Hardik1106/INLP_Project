"""
Convert SAE checkpoints from cotFaithfulness format to INLP_Project format.
"""
import torch
import os
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger("sae_converter")


def convert_cotfaithfulness_checkpoint(
    input_checkpoint_path: str,
    output_path: str,
    sae_index: int = 0,
    activation_dim: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert a cotFaithfulness learned_dict checkpoint to INLP format.
    
    Args:
        input_checkpoint_path: Path to learned_dicts_epoch_*.pt from cotFaithfulness
        output_path: Where to save converted checkpoint
        sae_index: Which SAE in the ensemble to convert (0-indexed)
        activation_dim: Expected d_model (for validation)
    
    Returns:
        State dict with W_enc, b_enc, W_dec, b_dec, and metadata
    """
    logger.info(f"Loading cotFaithfulness checkpoint: {input_checkpoint_path}")
    learned_dicts = torch.load(input_checkpoint_path, map_location="cpu")
    
    if not isinstance(learned_dicts, list) or len(learned_dicts) == 0:
        raise ValueError(f"Expected list of (LearnedDict, hyperparams), got {type(learned_dicts)}")
    
    learned_dict, hyperparams = learned_dicts[sae_index]
    
    # Extract weights from TiedSAE
    encoder = learned_dict.encoder  # [d_sae, d_model]
    encoder_bias = learned_dict.encoder_bias  # [d_sae]
    
    # Get dimensions
    d_sae, d_model = encoder.shape
    
    if activation_dim is not None and d_model != activation_dim:
        logger.warning(
            f"Dimension mismatch: checkpoint has d_model={d_model}, "
            f"but expected {activation_dim}"
        )
    
    # For tied SAEs, decoder is normalized encoder
    # Store both raw encoder and normalized version
    norms = torch.norm(encoder, 2, dim=-1)
    encoder_normalized = encoder / torch.clamp(norms, 1e-8)[:, None]
    
    # Create state dict matching INLP's SparseAutoencoder format
    state = {
        "W_enc": encoder.T,  # ← TRANSPOSE from [d_sae, d_model] to [d_model, d_sae]
        "b_enc": encoder_bias,  # [d_sae]
        "W_dec": encoder_normalized.T,  # Use normalized encoder as decoder (tied)
        "b_dec": torch.zeros(d_model),  # Bias for reconstructions
        "_metadata": {
            "source": "cotfaithfulness",
            "checkpoint_path": input_checkpoint_path,
            "sae_index": sae_index,
            "d_model": d_model,
            "d_sae": d_sae,
            "ratio": d_sae / d_model,
            "hyperparams": hyperparams,  # l1_alpha, bias_decay, etc.
            "center_trans": learned_dict.center_trans if hasattr(learned_dict, "center_trans") else None,
            "center_rot": learned_dict.center_rot if hasattr(learned_dict, "center_rot") else None,
            "center_scale": learned_dict.center_scale if hasattr(learned_dict, "center_scale") else None,
        }
    }
    
    # Save converted checkpoint
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(state, output_path)
    logger.info(
        f"Converted SAE: d_model={d_model}, d_sae={d_sae}, ratio={d_sae/d_model:.2f}"
    )
    logger.info(f"Saved to: {output_path}")
    
    return state


def find_local_sae_checkpoint(
    base_path: str,
    model_name: str,
    layer: int,
    dict_ratio: Optional[float] = None,
) -> Optional[str]:
    """
    Discover local SAE checkpoint by model/layer/dict_ratio pattern.
    
    Args:
        base_path: Root directory containing converted SAE checkpoints
        model_name: Model identifier (e.g., "pythia-70m-deduped")
        layer: Layer number
        dict_ratio: Dict expansion ratio (optional, if None returns first match)
    
    Returns:
        Path to checkpoint if found, None otherwise
    """
    base_path = Path(base_path)
    
    # Expected naming pattern: {model_shortname}_{layer}_{ratio}.pt
    # Also handle: {model_shortname}_layer{layer}_ratio{ratio}.pt
    
    candidates = []
    for pattern in [
        f"*{model_name}*layer{layer}*",
        f"*{model_name}*l{layer}*",
        f"*{model_name.split('/')[-1]}*layer{layer}*",
        f"*{model_name.split('/')[-1]}*l{layer}*",
    ]:
        for checkpoint in base_path.glob(pattern):
            candidates.append(str(checkpoint))
    
    if not candidates:
        return None
    
    # If dict_ratio specified, filter candidates
    if dict_ratio is not None:
        for ckpt in candidates:
            # Try to extract ratio from filename
            filename = Path(ckpt).stem
            try:
                if f"ratio{dict_ratio}" in filename or f"r{dict_ratio}" in filename:
                    return ckpt
            except:
                pass
        # If no exact match, return first
        return candidates[0]
    
    return candidates[0]

def batch_convert_from_mnt_ssd(
    mnt_base_path: str,
    output_dir: str,
    use_final_checkpoint: bool = True,
) -> Dict[str, str]:
    """
    Auto-discover and convert SAE checkpoints from /mnt/ssd-cluster structure.
    
    Args:
        mnt_base_path: Path to /mnt/ssd-cluster or similar
        output_dir: Where to save INLP-format checkpoints
        use_final_checkpoint: If True, use largest checkpoint dir (e.g., _256/),
                             else use first checkpoint (_0/)
    
    Returns:
        Dict mapping "model_layer" → checkpoint_path
    """
    mnt_path = Path(mnt_base_path)
    converted = {}

    logger.debug(f"Scanning base path: {mnt_path}")
    # Look for both direct children and nested one level deep
    tied_dirs = sorted(set(mnt_path.glob("tied_*")) | set(mnt_path.glob("*/tied_*")))
    untied_dirs = sorted(set(mnt_path.glob("untied_*")) | set(mnt_path.glob("*/untied_*")))
    logger.debug(f"Found {len(tied_dirs)} tied_* dirs: {[str(d) for d in tied_dirs]}")
    logger.debug(f"Found {len(untied_dirs)} untied_* dirs: {[str(d) for d in untied_dirs]}")

    for experiment_dir in tied_dirs + untied_dirs:
        logger.debug(f"Checking experiment_dir: {experiment_dir}")
        if not experiment_dir.is_dir():
            logger.debug(f"Not a dir: {experiment_dir}")
            continue

        # Parse: tied_residual_l2_r8 or untied_mlp_l3_r4
        parts = experiment_dir.name.split("_")
        logger.debug(f"Split parts: {parts}")
        if len(parts) < 4:
            logger.warning(f"Not enough parts in {experiment_dir.name}")
            continue
        tied_str = parts[0]  # "tied" or "untied"
        layer_loc = parts[1]  # "residual", "mlp", etc.
        layer_str = parts[2]  # "l2"
        ratio_str = parts[3]  # "r8"

        try:
            layer = int(layer_str[1:])  # Extract "2" from "l2"
            ratio = int(ratio_str[1:])  # Extract "8" from "r8"
        except (ValueError, IndexError):
            logger.warning(f"Could not parse layer/ratio from {experiment_dir.name}")
            continue

        # Find checkpoint directories: _0, _8, _256, etc.
        checkpoint_dirs = sorted(
            [d for d in experiment_dir.glob("_*/") if d.is_dir()],
            key=lambda x: int(x.name[1:])  # Sort by step number
        )
        logger.debug(f"Found checkpoint_dirs in {experiment_dir}: {[str(d) for d in checkpoint_dirs]}")

        if not checkpoint_dirs:
            logger.warning(f"No checkpoints found in {experiment_dir}")
            continue

        # Choose checkpoint: final or first
        checkpoint_dir = checkpoint_dirs[-1] if use_final_checkpoint else checkpoint_dirs[0]
        logger.debug(f"Selected checkpoint_dir: {checkpoint_dir}")
        learned_dicts_path = checkpoint_dir / "learned_dicts.pt"
        logger.debug(f"Looking for learned_dicts.pt at: {learned_dicts_path}")

        if not learned_dicts_path.exists():
            logger.warning(f"learned_dicts.pt not found in {checkpoint_dir}")
            continue

        # Determine output structure
        model_name = experiment_dir.parent.name  # e.g. "pythia70m_centered_gsm8k_NoCoT"
        output_subdir = Path(output_dir) / model_name / f"layer_{layer}"
        output_path = output_subdir / "sae_weights.pt"

        try:
            # Load to check if tied or untied
            logger.debug(f"Loading learned_dicts.pt from {learned_dicts_path}")
            learned_dicts = torch.load(learned_dicts_path, map_location="cpu")
            learned_dict, hyperparams = learned_dicts[0]  # First SAE in ensemble

            # Convert based on type
            if hasattr(learned_dict, 'decoder') and learned_dict.decoder is not None:
                logger.info(f"Converting UNTIED SAE: {experiment_dir.name}")
                state = _convert_untied_sae(learned_dict, learned_dicts_path, layer)
            else:
                logger.info(f"Converting TIED SAE: {experiment_dir.name}")
                state = _convert_tied_sae(learned_dict, learned_dicts_path, layer)

            output_subdir.mkdir(parents=True, exist_ok=True)
            torch.save(state, output_path)

            converted[f"{model_name}_l{layer}_r{ratio}"] = str(output_path)
            logger.info(f"\u2713 Saved to {output_path}\n")

        except Exception as e:
            logger.error(f"\u2717 Failed to convert {experiment_dir.name}: {e}\n")

    logger.debug(f"Conversion summary: {converted}")
    return converted


def _convert_tied_sae(learned_dict, checkpoint_path, layer):
    """Convert TiedSAE (encoder only, decoder is transpose)."""
    encoder = learned_dict.encoder  # [d_sae, d_model]
    encoder_bias = learned_dict.encoder_bias  # [d_sae]
    d_sae, d_model = encoder.shape
    
    norms = torch.norm(encoder, 2, dim=-1)
    encoder_normalized = encoder / torch.clamp(norms, 1e-8)[:, None]
    
    return {
        "W_enc": encoder.T,  # [d_model, d_sae]
        "b_enc": encoder_bias,
        "W_dec": encoder_normalized.T,  # Tied decoder
        "b_dec": torch.zeros(d_model),
        "_metadata": {
            "type": "tied",
            "d_model": d_model,
            "d_sae": d_sae,
            "checkpoint_path": str(checkpoint_path),
        }
    }


def _convert_untied_sae(learned_dict, checkpoint_path, layer):
    """Convert UntiedSAE (separate encoder and decoder)."""
    encoder = learned_dict.encoder  # [d_sae, d_model]
    encoder_bias = learned_dict.encoder_bias
    decoder = learned_dict.decoder  # [d_sae, d_model]
    d_sae, d_model = encoder.shape
    
    return {
        "W_enc": encoder.T,  # [d_model, d_sae]
        "b_enc": encoder_bias,
        "W_dec": decoder.T,  # [d_model, d_sae]
        "b_dec": torch.zeros(d_model),
        "_metadata": {
            "type": "untied",
            "d_model": d_model,
            "d_sae": d_sae,
            "checkpoint_path": str(checkpoint_path),
        }
    }