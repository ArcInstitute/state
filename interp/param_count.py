def count_transformer_params(hidden_dim, intermediate_dim, num_layers):
    d = hidden_dim
    inter = intermediate_dim

    # LayerNorm parameters (scale + bias) per LN: 2 * d
    ln1_params = 2 * d

    # Attention projection parameters:
    #   c_attn: project from d -> 3d (for Q, K, V)
    c_attn_params = d * (3 * d) + (3 * d)  # weight + bias

    #   c_proj_attn: project from d -> d
    c_proj_attn_params = (d) * d + d  # weight + bias

    # Second LayerNorm before MLP
    ln2_params = 2 * d

    # MLP projection parameters:
    #   c_fc: project from d -> intermediate_dim
    c_fc_params = d * inter + inter  # weight + bias

    #   c_proj_mlp: project from intermediate_dim -> d
    c_proj_mlp_params = inter * d + d  # weight + bias

    # Total parameters per transformer block
    block_params = (
        ln1_params
        + c_attn_params
        + c_proj_attn_params
        + ln2_params
        + c_fc_params
        + c_proj_mlp_params
    )

    # Total parameters across all blocks
    total_blocks = num_layers * block_params

    # Final LayerNorm after all blocks
    final_ln_params = 2 * d

    # Grand total (excluding embeddings)
    total_params = total_blocks + final_ln_params
    return total_params

def count_llama_params(
    hidden_dim: int,
    intermediate_dim: int,
    num_layers: int,
    num_attention_heads: int,
) -> int:
    d = hidden_dim
    m = intermediate_dim
    h = num_attention_heads
    kv = num_attention_heads

    # 1) Each LlamaDecoderLayer has:
    #
    #   a) input_layernorm (LlamaRMSNorm) → weight vector of length d
    ln_in = d
    #
    #   b) Self‐Attention (LlamaAttention):
    #       head_dim = d // h
    #       kv_out  = kv * head_dim   # output dim of k_proj / v_proj
    #
    head_dim = d // h
    kv_out = kv * head_dim
    #
    #     • q_proj: Linear(d → h * head_dim = d)   → (d*d) + d
    q_proj = d * d + d
    #
    #     • k_proj: Linear(d → kv_out)             → (d * kv_out) + kv_out
    k_proj = d * kv_out + kv_out
    #
    #     • v_proj: Linear(d → kv_out)             → (d * kv_out) + kv_out
    v_proj = d * kv_out + kv_out
    #
    #     • o_proj: Linear(h * head_dim = d → d)   → (d*d) + d
    o_proj = d * d + d
    #
    #     Total self_attn = q_proj + k_proj + v_proj + o_proj
    self_attn = q_proj + k_proj + v_proj + o_proj
    #
    #   c) post_attention_layernorm (LlamaRMSNorm) → weight vector of length d
    ln_post = d
    #
    #   d) MLP (LlamaMLP):
    #       • gate_proj: Linear(d → m) → (d*m) + m
    gate_proj = d * m + m
    #
    #       • up_proj:   Linear(d → m) → (d*m) + m
    up_proj = d * m + m
    #
    #       • down_proj: Linear(m → d) → (m*d) + d
    down_proj = m * d + d
    #
    #     Total MLP = gate_proj + up_proj + down_proj
    mlp = gate_proj + up_proj + down_proj

    # Per‐layer parameters in a single LlamaDecoderLayer:
    per_layer = ln_in + self_attn + ln_post + mlp

    # 2) Stack num_layers of those blocks, then add final RMSNorm(self.norm):
    total_blocks = num_layers * per_layer
    final_norm = d

    return total_blocks + final_norm

print("Total parameters for tahoe gpt hidden_dim=576, intermediate_dim=2304, num_layers=8 (excluding embeddings):",
      count_transformer_params(hidden_dim=576, intermediate_dim=2304, num_layers=8))
print("Total parameters for tahoe gpt hidden_dim=696, intermediate_dim=2784, num_layers=8 (excluding embeddings):",
      count_transformer_params(hidden_dim=696, intermediate_dim=2784, num_layers=8))
print("Total parameters for tahoe gpt hidden_dim=1104, intermediate_dim=4416, num_layers=4 (excluding embeddings):",
      count_transformer_params(hidden_dim=1104, intermediate_dim=4416, num_layers=4))
print("Total parameters for tahoe llama hidden_dim=576, intermediate_dim=2304, num_layers=8, num_attention_heads=8 (excluding embeddings):",
      count_llama_params(hidden_dim=576, intermediate_dim=2304, num_layers=8, num_attention_heads=8))
print("Total parameters for tahoe llama hidden_dim=696, intermediate_dim=2784, num_layers=8, num_attention_heads=12 (excluding embeddings):",
      count_llama_params(hidden_dim=696, intermediate_dim=2784, num_layers=8, num_attention_heads=12))
print("Total parameters for tahoe llama hidden_dim=1104, intermediate_dim=4416, num_layers=4, num_attention_heads=12 (excluding embeddings):",
      count_llama_params(hidden_dim=1104, intermediate_dim=4416, num_layers=4, num_attention_heads=12))

print("Total parameters for replogle gpt hidden_dim=328, intermediate_dim=1024, num_layers=8 (excluding embeddings):",
      count_transformer_params(hidden_dim=328, intermediate_dim=1024, num_layers=8))
print("Total parameters for replogle gpt hidden_dim=448, intermediate_dim=1792, num_layers=4 (excluding embeddings):",
      count_transformer_params(hidden_dim=448, intermediate_dim=1792, num_layers=4))
print("Total parameters for replogle gpt hidden_dim=492, intermediate_dim=1968, num_layers=8 (excluding embeddings):",
      count_transformer_params(hidden_dim=492, intermediate_dim=1968, num_layers=8))
print("Total parameters for replogle gpt hidden_dim=672, intermediate_dim=2688, num_layers=4 (excluding embeddings):",
      count_transformer_params(hidden_dim=672, intermediate_dim=2688, num_layers=4))
print("Total parameters for replogle gpt hidden_dim=328, intermediate_dim=1024, num_layers=4 (excluding embeddings):",
      count_transformer_params(hidden_dim=328, intermediate_dim=1024, num_layers=4))
print("Total parameters for replogle gpt hidden_dim=492, intermediate_dim=1968, num_layers=4 (excluding embeddings):",
      count_transformer_params(hidden_dim=492, intermediate_dim=1968, num_layers=4))

print("Total parameters for replogle llama hidden_dim=328, intermediate_dim=1024, num_layers=8, num_attention_heads=8 (excluding embeddings):",
      count_llama_params(hidden_dim=328, intermediate_dim=1024, num_layers=8, num_attention_heads=8))
print("Total parameters for replogle llama hidden_dim=448, intermediate_dim=1792, num_layers=4, num_attention_heads=8 (excluding embeddings):",
      count_llama_params(hidden_dim=448, intermediate_dim=1792, num_layers=4, num_attention_heads=8))
print("Total parameters for replogle llama hidden_dim=492, intermediate_dim=1968, num_layers=8, num_attention_heads=12 (excluding embeddings):",
      count_llama_params(hidden_dim=492, intermediate_dim=1968, num_layers=8, num_attention_heads=12))
print("Total parameters for replogle llama hidden_dim=672, intermediate_dim=2688, num_layers=4, num_attention_heads=12 (excluding embeddings):",
      count_llama_params(hidden_dim=672, intermediate_dim=2688, num_layers=4, num_attention_heads=12))
print("Total parameters for replogle llama hidden_dim=328, intermediate_dim=1024, num_layers=4, num_attention_heads=8 (excluding embeddings):",
      count_llama_params(hidden_dim=328, intermediate_dim=1024, num_layers=4, num_attention_heads=8))
print("Total parameters for replogle llama hidden_dim=492, intermediate_dim=1968, num_layers=4, num_attention_heads=12 (excluding embeddings):",
      count_llama_params(hidden_dim=492, intermediate_dim=1968, num_layers=4, num_attention_heads=12))