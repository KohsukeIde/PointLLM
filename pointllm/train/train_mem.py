# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# PyTorch 2.6 compatibility: Set safe globals BEFORE any other imports
import torch
import numpy as np

# Add comprehensive safe globals for PyTorch 2.6 compatibility
torch.serialization.add_safe_globals([
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.core.multiarray._reconstruct,
    np.random.RandomState,
    np.random.Generator,
    # Python built-in types
    list, tuple, dict, set, frozenset,
    int, float, str, bool, bytes, type(None),
    # Common numpy types
    np.int32, np.int64, np.float32, np.float64,
    np.uint8, np.uint16, np.uint32, np.uint64,
])

# Also set PyTorch's default load behavior for compatibility
import functools
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Need to call this before importing transformers.
from pointllm.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from pointllm.train.train import train

if __name__ == "__main__":
    train()
