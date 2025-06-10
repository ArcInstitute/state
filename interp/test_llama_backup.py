import sys
from pathlib import Path

import torch
from transformers import LlamaConfig
import transformers.models.llama.modeling_llama as llama_mod

# 1) Show what apply_rotary_pos_emb points to before and after patch
print("Before any model instantiation:")
print("  apply_rotary_pos_emb =", llama_mod.apply_rotary_pos_emb)

# allow imports from your project
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "vci_pretrain"))

from models.utils import LlamaBidirectionalModel

# 2) Load and patch
config = LlamaConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = LlamaBidirectionalModel.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", config=config
)

print("\nAfter LlamaBidirectionalModel init:")
print("  apply_rotary_pos_emb =", llama_mod.apply_rotary_pos_emb)

# 3) Sanity‐check that the patched function is identity
seq_len = 5
hidden_size = config.hidden_size

# grab the layer’s rotary helper to generate cos/sin
rotary_emb = model.layers[0].self_attn.rotary_emb
# dummy tensor just to get correctly-shaped cos & sin
dummy = torch.randn(1, seq_len, hidden_size)
cos, sin = rotary_emb(dummy, seq_len)

# dummy Q/K
q = torch.randn_like(dummy)
k = torch.randn_like(dummy)
q2, k2 = llama_mod.apply_rotary_pos_emb(q, k, cos, sin)

print(f"\nMax abs(q2 - q): {(q2 - q).abs().max().item():.3e}")
print(f"Max abs(k2 - k): {(k2 - k).abs().max().item():.3e}")
assert torch.allclose(q2, q) and torch.allclose(k2, k), "RoPE was *not* disabled!"

# 4) Finally, run a tiny forward and print one head’s attention
input_ids = torch.tensor([[50256, 314, 617, 198, 198]])  # “Hello”
outputs = model(input_ids, output_attentions=True)

attn0 = outputs.attentions[0][0, 0]  # layer 0, batch 0, head 0
print("\nLayer 0, Head 0 attention matrix:\n", attn0)
