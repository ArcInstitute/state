#!/usr/bin/env python3
"""
generate_configs.py

Reads a CSV/TSV file with columns:
    Backbone,Config FIle Name,Active Params,# Layers,Head Dim,# Heads,Hidden Dim,Intermediate Dim

and emits, for each row, a YAML config file named <Config FIle Name>.yaml
using either the “llama” template or the “gpt” template depending on Backbone.

Usage:
    1. Save your table to 'models.csv' (comma-separated) or 'models.tsv' (tab-separated).
    2. Run: python generate_configs.py models.csv
       (if you use a TSV, supply the “--tsv” flag).
    3. You’ll find files like 'tahoe_llama_31911552.yaml', 'tahoe_gpt_42537024.yaml', etc., in the same folder.
"""

import csv
import os
import sys
from typing import Dict

# ──────────────────────────────────────────────────────────────────────────────
# Helper templates
# ──────────────────────────────────────────────────────────────────────────────

LLAMA_TEMPLATE = """\
name: PertSets
checkpoint: null
device: cuda

kwargs:
  cell_set_len: 512
  blur: 0.05
  hidden_dim: {hidden_dim}      # hidden dimension going into the transformer backbone
  loss: energy
  confidence_head: False
  n_encoder_layers: 4
  n_decoder_layers: 4
  predict_residual: True
  softplus: True
  freeze_pert: False
  transformer_decoder: False
  finetune_vci_decoder: False
  residual_decoder: False
  batch_encoder: False
  nb_decoder: False
  mask_attn: False
  distributional_loss: energy
  init_from: null
  transformer_backbone_key: llama
  transformer_backbone_kwargs:
      max_position_embeddings: ${{model.kwargs.cell_set_len}}
      hidden_size: ${{model.kwargs.hidden_dim}}
      intermediate_size: {intermediate_dim}
      num_hidden_layers: {num_layers}
      num_attention_heads: {num_heads}
      num_key_value_heads: {num_heads}
      head_dim: {head_dim}
      use_cache: false
      attention_dropout: 0.0
      hidden_dropout: 0.0
      layer_norm_eps: 1e-6
      pad_token_id: 0
      bos_token_id: 1
      eos_token_id: 2
      tie_word_embeddings: false
      rotary_dim: 0
      use_rotary_embeddings: false
"""

GPT_TEMPLATE = """\
name: PertSets
checkpoint: null
device: cuda

kwargs:
  cell_set_len: 512   # how many cells to group together into a single set of cells
  extra_tokens: 1     # configurable buffer for confidence/special tokens
  blur: 0.05
  hidden_dim: {hidden_dim}   # hidden dimension going into the transformer backbone
  loss: energy
  confidence_token: False    # if true, model tries to predict its own confidence
  n_encoder_layers: 4        # number of MLP layers for pert, basal encoders
  n_decoder_layers: 4
  predict_residual: True      # if true, predicts the residual in embedding space to the basal cells
  freeze_pert_backbone: False # if true, the perturbation model is frozen
  finetune_vci_decoder: False # if true, the pretrained state decoder is used in finetuning
  batch_encoder: False        # if true, batch variables are used
  nb_decoder: False           # if true, use a negative binomial decoder
  mask_attn: False            # if true, mask the attention
  distributional_loss: energy
  init_from: null             # initial checkpoint to start the model
  transformer_backbone_key: GPT2
  transformer_backbone_kwargs:
      n_positions: ${{model.kwargs.cell_set_len}}
      n_embd: ${{model.kwargs.hidden_dim}}
      n_layer: {num_layers}
      n_head: {num_heads}
      resid_pdrop: 0.0
      embd_pdrop: 0.0
      attn_pdrop: 0.0
      use_cache: false
"""

# ──────────────────────────────────────────────────────────────────────────────
# Primary generation logic
# ──────────────────────────────────────────────────────────────────────────────

def generate_llama_config(params: Dict[str, str]) -> str:
    """
    Fill in the llama‐style YAML template using the fields from one row.
    Expects:
      params["# Layers"], params["Head Dim"], params["# Heads"],
      params["Hidden Dim"], params["Intermediate Dim"] all as strings parseable to int.
    """
    return LLAMA_TEMPLATE.format(
        hidden_dim=int(params["Hidden Dim"]),
        intermediate_dim=int(params["Intermediate Dim"]),
        num_layers=int(params["# Layers"]),
        num_heads=int(params["# Heads"]),
        head_dim=int(params["Head Dim"])
    )


def generate_gpt_config(params: Dict[str, str]) -> str:
    """
    Fill in the GPT‐style YAML template using the fields from one row.
    Expects the same numeric columns as for llama.
    """
    return GPT_TEMPLATE.format(
        hidden_dim=int(params["Hidden Dim"]),
        num_layers=int(params["# Layers"]),
        num_heads=int(params["# Heads"])
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_configs.py <models.csv> [--tsv]")
        sys.exit(1)

    input_path = sys.argv[1]
    use_tsv = "--tsv" in sys.argv
    delimiter = "\t" if use_tsv else ","

    if not os.path.isfile(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    # Read the table
    with open(input_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        required_fields = {
            "Backbone",
            "Config FIle Name",
            "Active Params",
            "# Layers",
            "Head Dim",
            "# Heads",
            "Hidden Dim",
            "Intermediate Dim"
        }
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            print(f"Error: missing columns in input file: {missing}")
            sys.exit(1)

        for row in reader:
            backbone = row["Backbone"].strip().lower()
            cfg_name = row["Config FIle Name"].strip()
            # Decide which template to use:
            if backbone == "llama":
                yaml_text = generate_llama_config(row)
            elif backbone == "gpt":
                yaml_text = generate_gpt_config(row)
            else:
                print(f"Warning: skipping unknown backbone '{row['Backbone']}' in row: {row}")
                continue

            # Write out to `<Config FIle Name>.yaml`
            out_filename = f"{cfg_name}.yaml"
            with open(out_filename, "w") as out_f:
                out_f.write(yaml_text)

            print(f"Generated {out_filename}")


if __name__ == "__main__":
    main()
