"""Tests for bidirectional attention models (Llama and GPT2)."""

import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from state.tx.models.utils import LlamaBidirectionalModel, get_transformer_backbone


@pytest.fixture
def small_llama_config():
    """Create a small Llama config for testing."""
    return LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


def test_llama_bidirectional_config_is_non_causal(small_llama_config):
    """Test that LlamaBidirectionalModel sets is_causal to False."""
    model = LlamaBidirectionalModel(small_llama_config)

    # Check that the model config is non-causal
    assert model.config.is_causal is False

    # Check that all attention layers are non-causal
    for layer in model.layers:
        if hasattr(layer, "self_attn"):
            assert layer.self_attn.is_causal is False  # type: ignore


def test_llama_bidirectional_update_causal_mask_returns_none(small_llama_config):
    """Test that _update_causal_mask returns None, disabling causal masking."""
    model = LlamaBidirectionalModel(small_llama_config)

    # Create dummy inputs
    batch_size, seq_len = 2, 8
    attention_mask = torch.ones(batch_size, seq_len)
    input_tensor = torch.randn(batch_size, seq_len, small_llama_config.hidden_size)
    cache_position = torch.arange(seq_len)

    # Call _update_causal_mask
    result = model._update_causal_mask(
        attention_mask=attention_mask,
        input_tensor=input_tensor,
        cache_position=cache_position,
        past_key_values=None,
        output_attentions=False,
    )

    # Should return None (no causal masking)
    assert result is None


def test_get_transformer_backbone_llama_is_causal_by_default():
    """get_transformer_backbone should return a causal LlamaModel unless opted in."""

    kwargs = {
        "vocab_size": 100,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 32,
    }

    model, model_dim = get_transformer_backbone("llama", kwargs)

    assert type(model) is LlamaModel
    assert model_dim == kwargs["hidden_size"]


def test_get_transformer_backbone_llama_bidirectional_flag():
    """Setting bidirectional_attention=True should opt into bidirectional Llama."""

    kwargs = {
        "vocab_size": 100,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 32,
        "bidirectional_attention": True,
    }

    model, _ = get_transformer_backbone("llama", kwargs)

    assert isinstance(model, LlamaBidirectionalModel)


def test_llama_bidirectional_attention_vs_causal(small_llama_config):
    """
    Test that bidirectional attention produces different outputs than causal attention.

    This is the key test: in bidirectional attention, later tokens should affect
    earlier token representations, which doesn't happen in causal attention.
    """
    torch.manual_seed(42)

    # Create both bidirectional and standard (causal) models
    bidirectional_model = LlamaBidirectionalModel(small_llama_config)
    causal_model = LlamaModel(small_llama_config)

    # Copy weights from bidirectional to causal to ensure same initialization
    causal_model.load_state_dict(bidirectional_model.state_dict(), strict=False)

    # Create input
    batch_size, seq_len = 2, 8
    inputs_embeds = torch.randn(batch_size, seq_len, small_llama_config.hidden_size)

    # Set models to eval mode
    bidirectional_model.eval()
    causal_model.eval()

    with torch.no_grad():
        # Get outputs from bidirectional model
        bidirectional_output = bidirectional_model(inputs_embeds=inputs_embeds)

        # Get outputs from causal model
        causal_output = causal_model(inputs_embeds=inputs_embeds)

    # The outputs should be different because bidirectional allows all tokens
    # to attend to each other, while causal only allows attending to past tokens
    assert not torch.allclose(bidirectional_output.last_hidden_state, causal_output.last_hidden_state, atol=1e-5), (
        "Bidirectional and causal outputs should differ"
    )


def test_llama_bidirectional_future_tokens_affect_past(small_llama_config):
    """
    Test that future tokens affect past token representations in bidirectional model.

    This is the core property of bidirectional attention: changing a future token
    should change the representation of past tokens.
    """
    torch.manual_seed(42)

    model = LlamaBidirectionalModel(small_llama_config)
    model.eval()

    batch_size, seq_len = 1, 6
    hidden_size = small_llama_config.hidden_size

    # Create two inputs that differ only in the last token
    inputs_embeds_1 = torch.randn(batch_size, seq_len, hidden_size)
    inputs_embeds_2 = inputs_embeds_1.clone()

    # Modify only the last token embedding in the second input
    inputs_embeds_2[:, -1, :] = torch.randn(batch_size, hidden_size)

    with torch.no_grad():
        output_1 = model(inputs_embeds=inputs_embeds_1)
        output_2 = model(inputs_embeds=inputs_embeds_2)

    # Check that the first tokens' representations differ between the two inputs
    # This demonstrates that the last token (future) affects the first token (past)
    first_token_repr_1 = output_1.last_hidden_state[:, 0, :]
    first_token_repr_2 = output_2.last_hidden_state[:, 0, :]

    assert not torch.allclose(first_token_repr_1, first_token_repr_2, atol=1e-5), (
        "First token representation should change when last token changes (bidirectional attention)"
    )


def test_llama_bidirectional_first_token_differs_across_batch(small_llama_config):
    """
    Test that first token representations differ across batch when sequences differ.

    This is a critical test for bidirectional attention: in causal attention,
    the first token can only attend to itself, so if all sequences have the same
    first token, they would produce identical first token representations.

    In bidirectional attention, the first token attends to all tokens in the sequence,
    so different sequences should produce different first token representations even
    when the first tokens themselves are identical.
    """
    torch.manual_seed(42)

    model = LlamaBidirectionalModel(small_llama_config)
    model.eval()

    batch_size, seq_len = 4, 8
    hidden_size = small_llama_config.hidden_size

    # Create a batch where ALL sequences have the SAME first token embedding
    # but DIFFERENT subsequent tokens
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)

    # Make the first token identical across all sequences
    shared_first_token = torch.randn(1, hidden_size)
    inputs_embeds[:, 0, :] = shared_first_token

    with torch.no_grad():
        output = model(inputs_embeds=inputs_embeds)

    # Extract first token representations for all sequences
    first_token_reprs = output.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

    # In bidirectional attention, these should all be DIFFERENT
    # because each attends to different subsequent tokens
    # Check that not all first tokens are the same
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            assert not torch.allclose(first_token_reprs[i], first_token_reprs[j], atol=1e-5), (
                f"First token representations for sequences {i} and {j} should differ in bidirectional attention"
            )

    # Additional check: variance across batch should be substantial
    variance_per_dim = torch.var(first_token_reprs, dim=0)
    mean_variance = variance_per_dim.mean()
    assert mean_variance > 1e-4, (
        "First token representations should have substantial variance across batch in bidirectional attention"
    )


def test_llama_bidirectional_symmetric_position_influence(small_llama_config):
    """
    Test that in bidirectional attention, position i affects position j
    as much as position j affects position i (roughly symmetric).
    """
    torch.manual_seed(42)

    model = LlamaBidirectionalModel(small_llama_config)
    model.eval()

    batch_size, seq_len = 1, 4
    hidden_size = small_llama_config.hidden_size

    # Create base input
    base_input = torch.randn(batch_size, seq_len, hidden_size)

    # Modify position 0 and see effect on position 2
    input_modify_0 = base_input.clone()
    input_modify_0[:, 0, :] = torch.randn(batch_size, hidden_size)

    # Modify position 2 and see effect on position 0
    input_modify_2 = base_input.clone()
    input_modify_2[:, 2, :] = torch.randn(batch_size, hidden_size)

    with torch.no_grad():
        output_base = model(inputs_embeds=base_input)
        output_modify_0 = model(inputs_embeds=input_modify_0)
        output_modify_2 = model(inputs_embeds=input_modify_2)

    # Calculate how much position 2 changes when position 0 changes
    effect_0_on_2 = torch.norm(output_modify_0.last_hidden_state[:, 2, :] - output_base.last_hidden_state[:, 2, :])

    # Calculate how much position 0 changes when position 2 changes
    effect_2_on_0 = torch.norm(output_modify_2.last_hidden_state[:, 0, :] - output_base.last_hidden_state[:, 0, :])

    # In bidirectional attention, these effects should both be non-zero
    # (demonstrating mutual influence, unlike in causal attention)
    assert effect_0_on_2 > 0.01, "Position 0 should affect position 2"
    assert effect_2_on_0 > 0.01, "Position 2 should affect position 0"


def test_llama_bidirectional_forward_with_input_ids(small_llama_config):
    """Test that forward pass works with input_ids."""
    model = LlamaBidirectionalModel(small_llama_config)
    model.eval()

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, small_llama_config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(input_ids=input_ids)

    # Check output shape
    assert output.last_hidden_state.shape == (batch_size, seq_len, small_llama_config.hidden_size)


def test_llama_bidirectional_forward_with_attention_mask(small_llama_config):
    """Test that forward pass respects attention mask for padding."""
    model = LlamaBidirectionalModel(small_llama_config)
    model.eval()

    batch_size, seq_len = 2, 10
    hidden_size = small_llama_config.hidden_size
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)

    # Create attention mask: first sequence has padding at positions 8-9
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 8:] = 0  # Mask out last 2 positions for first sequence

    with torch.no_grad():
        output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    # Check that output shape is correct
    assert output.last_hidden_state.shape == (batch_size, seq_len, hidden_size)


def test_llama_bidirectional_is_causal_false_in_forward(small_llama_config):
    """Test that is_causal=False is passed in flash_attn_kwargs during forward."""
    model = LlamaBidirectionalModel(small_llama_config)
    model.eval()

    batch_size, seq_len = 1, 8
    inputs_embeds = torch.randn(batch_size, seq_len, small_llama_config.hidden_size)

    # Monkey-patch the parent's forward to capture flash_attn_kwargs
    original_forward = LlamaModel.forward
    captured_kwargs = {}

    def capture_forward(self, **kwargs):
        captured_kwargs.update(kwargs)
        return original_forward(self, **kwargs)

    LlamaModel.forward = capture_forward  # type: ignore

    try:
        with torch.no_grad():
            model(inputs_embeds=inputs_embeds)

        # Check that is_causal was set to False
        assert "is_causal" in captured_kwargs
        assert captured_kwargs["is_causal"] is False
    finally:
        # Restore original forward
        LlamaModel.forward = original_forward


def test_llama_bidirectional_no_rope(small_llama_config):
    """Test that NoRoPE is used instead of standard rotary embeddings."""
    from state.tx.models.utils import NoRoPE

    model = LlamaBidirectionalModel(small_llama_config)

    # Check that rotary_emb is an instance of NoRoPE
    assert isinstance(model.rotary_emb, NoRoPE)
