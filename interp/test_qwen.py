from transformers import Qwen3Config
import sys
import torch

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  # adjust if needed
sys.path.append(str(project_root))
sys.path.append(str(project_root / "vci_pretrain"))

from models.utils import Qwen3BidirectionalModel

config = Qwen3Config.from_pretrained("Qwen/Qwen3-1.7B")  # or load from your own checkpoint
model = Qwen3BidirectionalModel(config)

# (or, if you’re using HF’s `from_pretrained` pattern, you can do:)
model = Qwen3BidirectionalModel.from_pretrained("Qwen/Qwen3-1.7B", config=config)

# Sanity check: run a tiny forward pass and print one head’s attention.
input_ids = torch.tensor([[50256,  314,  617,  198,  198]])  # “Hello”
outputs = model(input_ids, output_attentions=True)  
# `outputs.attentions` is a tuple of length n_layers; each is [batch, heads, seq_len, seq_len].
attn0 = outputs.attentions[0][0, 0]  # layer 0, batch 0, head 0
print("Layer 0, Head 0 attention matrix:\n", attn0)
# You should see non‐zero entries in BOTH upper and lower triangles.
