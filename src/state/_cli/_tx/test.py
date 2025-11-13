import sys
from pathlib import Path

import torch
from transformers import LlamaConfig

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# NOTE: keep the import local so we don’t trigger heavy deps for other modules.
from src.state.tx.models.utils import LlamaBidirectionalModel

# Build a tiny LLaMA config locally so we don’t hit gated HuggingFace weights.
config = LlamaConfig(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=128,
    vocab_size=52000,
)

# Force eager attention so we can request attention weights from the forward pass.
config._attn_implementation = "eager"
config.attn_implementation = "eager"

model = LlamaBidirectionalModel(config)

# Sanity check: run a tiny forward pass and print one head’s attention.
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
outputs = model(input_ids, output_attentions=True, return_dict=True)
if outputs.attentions is None:
    raise RuntimeError(
        "Attention weights were not returned; ensure attention implementation supports `output_attentions`."
    )

# `outputs.attentions` is a tuple of length n_layers; each is [batch, heads, seq_len, seq_len].
attn0 = outputs.attentions[0][0, 0]  # layer 0, batch 0, head 0
print("Layer 0, Head 0 attention matrix:\n", attn0)
# You should see non‐zero entries in BOTH upper and lower triangles.