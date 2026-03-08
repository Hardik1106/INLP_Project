"""
Module 2a – SAE Encoding
=========================
Loads pre-trained Sparse Autoencoders (via SAELens or custom weights)
and encodes cached dense activations into the sparse latent space.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src import config
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import SAEConfig
from src.utils import logger, save_tensor, load_tensor, ensure_dir


# ---------------------------------------------------------------------------
# SAE wrapper (compatible with SAELens or custom)
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """
    Minimal SAE wrapper that can either:
      1. Load from SAELens pre-trained checkpoints, or
      2. Be initialized with custom encoder/decoder weights.

    The SAE decomposes a d_model-dimensional activation vector into a
    d_sae-dimensional sparse feature vector via:
        z = ReLU(W_enc @ (x - b_dec) + b_enc)
        x_hat = W_dec @ z + b_dec

    For our pipeline we only need the encoder direction (activation -> features).
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.register_buffer("W_enc", W_enc)
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("W_dec", W_dec)
        self.register_buffer("b_dec", b_dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode dense activations to sparse feature activations.

        Args:
            x: [..., d_model]
        Returns:
            z: [..., d_sae]  (non-negative, sparse)
        """
        z = torch.relu(
            # (x - self.b_dec) @ self.W_enc.T + self.b_enc
            (x - self.b_dec) @ self.W_enc + self.b_enc
        )
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct dense activations from sparse features."""
        # return z @ self.W_dec.T + self.b_dec
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (sparse_features, reconstruction)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


def load_sae_from_saelens(
    release: str,
    sae_id: Optional[str],
    layer: int,
    device: str = "cpu",
) -> SparseAutoencoder:
    """
    Load a pre-trained SAE from the SAELens library.

    SAELens organizes SAEs by release (e.g., 'pythia-2.8b-res-jb') and
    layer-specific IDs.
    """
    try:
        print("==============================")
        print(release, sae_id, layer)
        # from sae_lens import SAE
        # if sae_id is None:
        #     sae_id = f"blocks.{layer}.hook_resid_post"
        # logger.info(f"Loading SAELens SAE: release={release}, id={sae_id}")
        # sae_lens_obj = SAE.from_pretrained(release, sae_id)

        # from sae_lens import SAE
        # sae_lens_obj, cfg_dict, sparsity = SAE.from_pretrained(
        #     release="pythia-70m-deduped-res-sm",
        #     sae_id=f"blocks.2.hook_resid_post",
        #     device=device,
        # )

        from sae_lens import SAE
        if sae_id is None:
            sae_id = f"blocks.{layer}.hook_resid_post"
        sae_lens_obj, cfg_dict, sparsity = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
        )

        # Extract the underlying weight tensors
        # SAELens stores these as attributes on the SAE object
        if hasattr(sae_lens_obj, "W_enc"):
            W_enc = sae_lens_obj.W_enc.data.clone()
            b_enc = sae_lens_obj.b_enc.data.clone()
            W_dec = sae_lens_obj.W_dec.data.clone()
            b_dec = sae_lens_obj.b_dec.data.clone()
        else:
            # Handle tuple return from from_pretrained
            sae_obj = sae_lens_obj[0] if isinstance(sae_lens_obj, tuple) else sae_lens_obj
            W_enc = sae_obj.W_enc.data.clone()
            b_enc = sae_obj.b_enc.data.clone()
            W_dec = sae_obj.W_dec.data.clone()
            b_dec = sae_obj.b_dec.data.clone()

        d_model = W_enc.shape[0]
        d_sae = W_enc.shape[1]

        sae = SparseAutoencoder(d_model, d_sae, W_enc, b_enc, W_dec, b_dec)
        sae = sae.to(device)
        logger.info(f"  SAE loaded: d_model={d_model}, d_sae={d_sae}")
        return sae

    except Exception as e:
        logger.error(f"Failed to load SAE from SAELens: {e}")
        raise


def load_sae_from_weights(
    weights_path: str,
    device: str = "cpu",
) -> SparseAutoencoder:
    """Load a custom SAE from a saved state dict."""
    logger.info(f"Loading custom SAE from {weights_path}")
    state = torch.load(weights_path, map_location=device, weights_only=True)
    sae = SparseAutoencoder(
        d_model=state["W_enc"].shape[0],
        d_sae=state["W_enc"].shape[1],
        W_enc=state["W_enc"],
        b_enc=state["b_enc"],
        W_dec=state["W_dec"],
        b_dec=state["b_dec"],
    )
    sae = sae.to(device)
    return sae


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_activations(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Encode a batch of dense activations through the SAE.

    Args:
        sae: The SparseAutoencoder.
        activations: [N, d_model] dense activations.
        batch_size: Mini-batch size for memory efficiency.

    Returns:
        features: [N, d_sae] sparse feature activations.
    """
    device = next(sae.parameters()).device if list(sae.parameters()) else "cpu"
    # Use buffers if no parameters
    if not list(sae.parameters()):
        device = sae.W_enc.device

    all_features = []
    for i in range(0, len(activations), batch_size):
        batch = activations[i : i + batch_size].to(device)
        features = sae.encode(batch)
        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


# def encode_and_cache_all(
#     sae_config: SAEConfig,
#     cached_activations: Dict[str, Dict[int, torch.Tensor]],
#     layers: List[int],
#     cache_dir: str,
#     device: str = "cpu",
# ) -> Dict[str, Dict[int, torch.Tensor]]:
#     """
#     For each dataset condition and layer, encode cached dense activations
#     through the layer-specific SAE and save the sparse features.

#     Returns:
#         Nested dict: dataset_label -> layer -> [N, d_sae] features
#     """
#     results = {}

#     for layer in layers:
#         logger.info(f"--- Encoding layer {layer} ---")
#         sae = load_sae_from_saelens(
#             release=sae_config.release,
#             sae_id=sae_config.sae_id,
#             layer=layer,
#             device=device,
#         )

#         for label, layer_acts in cached_activations.items():
#             if layer not in layer_acts:
#                 continue

#             acts = layer_acts[layer]  # [N, d_model]
#             features = encode_activations(sae, acts)

#             if label not in results:
#                 results[label] = {}
#             results[label][layer] = features

#             # Save
#             out_path = f"{cache_dir}/{label}/sae_features_layer_{layer}.pt"
#             save_tensor(features, out_path)
#             logger.info(
#                 f"  {label} layer {layer}: encoded {acts.shape} -> {features.shape}"
#             )

#         # Free SAE memory before loading the next layer's SAE
#         del sae
#         torch.cuda.empty_cache()

#     return results


def encode_and_cache_all(
    sae_config: SAEConfig,
    cached_activations: Dict[str, Dict[int, torch.Tensor]],
    layers: List[int],
    cache_dir: str,
    device: str = "cpu",
    batch_size: int = 512,  # increased from 64
) -> Dict[str, Dict[int, torch.Tensor]]:
    results = {}
    for layer in layers:
        logger.info(f"--- Encoding layer {layer} ---")
        sae = load_sae_from_saelens(
            release=sae_config.release,
            sae_id=sae_config.sae_id,
            layer=layer,
            device=device,
        )
        for label, layer_acts in cached_activations.items():
            if layer not in layer_acts:
                continue
            acts = layer_acts[layer].to(device)  # keep on GPU
            features = encode_activations(sae, acts, batch_size=batch_size)
            if label not in results:
                results[label] = {}
            results[label][layer] = features
            out_path = f"{cache_dir}/{label}/sae_features_layer_{layer}.pt"
            save_tensor(features, out_path)
            logger.info(f"  {label} layer {layer}: {acts.shape} -> {features.shape}")
        del sae
        torch.cuda.empty_cache()
    return results