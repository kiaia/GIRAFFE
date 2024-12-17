import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding


# Modified Qwen2RotaryEmbedding with mRoPE++ support
class Qwen2RotaryEmbeddingMRoPEPP(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=16384,
        base=10000,
        device=None,
        max_target_position=65536,
        mrope_section=[16, 24, 24],
        scale=1,
        ntk_factor=1,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Generate three types of inv_freq for different sections
        # 1. Base extrapolation (unchanged theta)
        inv_freq_base = 1.0 / (
            base ** (torch.arange(0, dim, 2).float().to(device) / dim)
        )

        # 2. Linear scaling
        inv_freq_linear = 1.0 / (
            scale * (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        )

        # 3. NTK scaling
        ntk_base = base * (
            (max_target_position - 1) / (max_position_embeddings - 1)
        ) ** (dim / (dim - 2))
        inv_freq_ntk = 1.0 / (
            ntk_base ** (torch.arange(0, dim, 2).float().to(device) / dim)
        )

        # Validate dimensions
        total_dims = sum(mrope_section)
        assert (
            dim // 2 == total_dims
        ), f"Dimension mismatch: dim//2({dim//2}) != sum(mrope_section)({total_dims})"

        # Combine inv_freq for three sections
        inv_freq_list = []
        current_pos = 0

        # Temporal section: use base version (pure extrapolation)
        inv_freq_list.append(inv_freq_base[: mrope_section[0]])
        current_pos += mrope_section[0]

        # Height section: mix of linear and ntk
        height_section = (
            inv_freq_linear[current_pos : current_pos + mrope_section[1]]
            * (1 - ntk_factor)
            + inv_freq_ntk[current_pos : current_pos + mrope_section[1]] * ntk_factor
        )
        inv_freq_list.append(height_section)
        current_pos += mrope_section[1]

        # Width section: use ntk version
        inv_freq_list.append(inv_freq_ntk[current_pos : current_pos + mrope_section[2]])

        # Merge all sections
        inv_freq = torch.cat(inv_freq_list)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


def enable_mrope_plus():
    """
    Monkey patch to enable mRoPE++ for Qwen2 models.
    Apply this patch before loading the model.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Store original class for potential restoration
    original_class = Qwen2RotaryEmbedding

    # Replace the original class with our modified version
    try:
        from transformers.models.qwen2 import modeling_qwen2

        modeling_qwen2.Qwen2RotaryEmbedding = Qwen2RotaryEmbeddingMRoPEPP
        logger.info("Successfully enabled mRoPE++ for Qwen2")
    except ImportError:
        logger.error("Failed to patch Qwen2: transformers library not found")
    except Exception as e:
        logger.error(f"Failed to patch Qwen2: {str(e)}")
        # Restore original class if patching fails
        modeling_qwen2.Qwen2RotaryEmbedding = original_class
