import sys
from pathlib import Path

import torch
from transformers import GPT2Config

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# NOTE: keep the import local so we don’t trigger heavy deps for other modules.
from src.state.tx.models.utils import GPT2BidirectionalModel

config = GPT2Config.from_pretrained("gpt2")  # or load from your own checkpoint
config._attn_implementation = "eager"
config.attn_implementation = "eager"

model = GPT2BidirectionalModel(config)

# (or, if you’re using HF’s `from_pretrained` pattern, you can do:)
model = GPT2BidirectionalModel.from_pretrained("gpt2", config=config)

# Sanity check: run a tiny forward pass and print one head’s attention.
input_ids = torch.tensor([[50256,  314,  617,  198,  198]])  # “Hello”
outputs = model(input_ids, output_attentions=True)  
# `outputs.attentions` is a tuple of length n_layers; each is [batch, heads, seq_len, seq_len].
attn0 = outputs.attentions[0][0, 0]  # layer 0, batch 0, head 0
print("Layer 0, Head 0 attention matrix:\n", attn0)