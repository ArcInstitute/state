import torch
import torch.nn as nn

from transformers import GPT2Model, GPT2Config, LlamaModel, LlamaConfig, PreTrainedModel, Qwen3Model, Qwen3Config
from transformers.cache_utils import Cache
from typing import Union, Optional, Dict, Any
from lora.lora import prepare_model_for_lora


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    activation: nn.Module = nn.ReLU,  # default to nn.ReLU class
) -> nn.Sequential:
    """
    Build an MLP of `n_layers` from `in_dim` to `out_dim`.
    ...
    """
    layers = []
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    if n_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())  # instantiate the class
        layers.append(nn.Dropout(dropout))

        # Intermediate layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())  # instantiate again
            layers.append(nn.Dropout(dropout))

        # Final layer
        layers.append(nn.Linear(hidden_dim, out_dim))

    return nn.Sequential(*layers)


def get_activation_class(name: str) -> nn.Module:
    """
    Given a string activation name, return the corresponding nn.Module class.

    Supported activation functions (add any more here):
    - ReLU
    - LeakyReLU
    - ELU
    - SELU
    - GELU
    """
    name = name.lower()

    if name == "relu":
        return nn.ReLU
    elif name == "leakyrelu":
        return nn.LeakyReLU
    elif name == "elu":
        return nn.ELU
    elif name == "selu":
        return nn.SELU
    elif name == "gelu":
        return nn.GELU
    # Add more as needed...
    else:
        raise ValueError(f"Unsupported activation function: {name}")


def get_loss_fn(loss: Union[str, nn.Module]) -> nn.Module:
    """
    Given a string loss function name, return the corresponding nn.Module class.

    Supported loss functions (add any more here):
    - MSELoss
    - L1Loss
    - SmoothL1Loss
    """
    if isinstance(loss, nn.Module):
        return loss

    loss = loss.lower()

    if loss == "mse":
        return nn.MSELoss()
    # Add more as needed...
    else:
        raise ValueError(f"Unsupported loss function: {loss}")


def get_transformer_backbone(key, kwargs) -> PreTrainedModel:
    # Extract LoRA config if present
    lora_config = kwargs.pop("lora_config", None)
    
    if key == "GPT2":
        config = GPT2Config(**kwargs)
        model = GPT2Bidirectional(config)

        # Zero out position embeddings and freeze them
        with torch.no_grad():
            model.wpe.weight.zero_()
            model.wpe.weight.requires_grad = False
            model.wte.weight.zero_()
            model.wte.weight.requires_grad = False

        model_dim = config.n_embd
        
    elif key == "qwen3": # using this because of good performance for 0.5B model, key differences are RMSNorm instead of LayerNorm, and no bias in the attention layers
        config = Qwen3Config(**kwargs)
        # Use our bidirectional wrapper instead of the standard Qwen3Model
        model = Qwen3BidirectionalModel(config)
        model_dim = config.hidden_size

        with torch.no_grad():
            model.embed_tokens.weight.zero_()
            model.embed_tokens.weight.requires_grad = False
            
    elif key == "llama":
        config = LlamaConfig(**kwargs)
        model = LlamaBidirectionalModel(config)
        model_dim = config.hidden_size
        needs_bidirectional_mask = True  # Llama defaults to causal, needs explicit bidirectional mask

        with torch.no_grad():
            model.embed_tokens.weight.zero_()
            model.embed_tokens.weight.requires_grad = False
    else:
        raise ValueError(f"Unknown backbone key {key}")
    
    # Apply LoRA if config is provided
    if lora_config is not None:
        model = prepare_model_for_lora(model, key, lora_config)

    return model, model_dim



class LlamaBidirectionalModel(LlamaModel):
    """
    A drop-in replacement for LlamaModel with bidirectional attention.
    By overriding _update_causal_mask to return None, all tokens attend to each other.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        # By returning None, we disable any causal‐(look‐ahead) masking.
        # The only mask that remains is whatever “attention_mask” the user has passed
        # (e.g. padding‐mask), which will be handled by Flash/SDPA internally as non‐causal.
        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: Cache = None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        cache_position: torch.LongTensor = None,
        **flash_attn_kwargs,
    ):


        flash_attn_kwargs["is_causal"] = False

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )

class Qwen3BidirectionalModel(Qwen3Model):
    """
    A drop-in replacement for Qwen3Model with bidirectional attention.
    By overriding _update_causal_mask to return None, all tokens attend to each other.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        # By returning None, we disable any causal‐(look‐ahead) masking.
        # The only mask that remains is whatever “attention_mask” the user has passed
        # (e.g. padding‐mask), which will be handled by Flash/SDPA internally as non‐causal.
        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: Cache = None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        cache_position: torch.LongTensor = None,
        **flash_attn_kwargs,
    ):
        # Exactly the same signature as Qwen3Model.forward().
        # We do not need to change anything else—once _update_causal_mask returns None,
        # all layers will compute full, bidirectional attention.

        flash_attn_kwargs["is_causal"] = False

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )
    
class GPT2Bidirectional(GPT2Model):
    """
    A thin wrapper around GPT2Model that disables the causal (unidirectional) mask,
    allowing full bidirectional attention—and prints the internal bias mask each forward pass.
    """
    def __init__(self, config: GPT2Config):
        # Mark as not‐a‐decoder (for downstream utilities).
        config.is_decoder = False
        super().__init__(config)

        # Overwrite each attention's bias so no triangular masking occurs.
        for block in self.h:
            # block.attn.bias is a bool‐tensor of shape (1, 1, max_pos, max_pos).
            block.attn.bias.data.fill_(True)
            block.attn.is_causal = False
        
        def _no_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values,
            output_attentions: bool,
        ):
            return None

        self._update_causal_mask = _no_causal_mask.__get__(self, GPT2Model)


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Determine sequence length for printing the relevant slice of bias
        if input_ids is not None:
            seq_len = input_ids.size(1)
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.size(1)
        else:
            seq_len = None  # If neither is given, we can’t infer seq_len

        if seq_len is not None:
            # Print the (1, 1, seq_len, seq_len) slice of the bias for the first block
            bias_mask = self.h[0].attn.bias[0, 0, :seq_len, :seq_len]
            print("Bias mask (block 0) slice [0,0,:seq_len,:seq_len]:")
            print(bias_mask)
        else:
            print("Cannot infer sequence length to print bias mask.")

        # If a 2D attention_mask was provided, print its expanded 4D version:
        if attention_mask is not None:
            # Expand to (batch_size, 1, seq_len, seq_len)
            B, S = attention_mask.size()
            expanded = attention_mask.unsqueeze(1).unsqueeze(2).expand(B, 1, S, S)
            # Convert to float mask (1→0.0, 0→-inf) just like GPT2 does internally
            neg_inf = torch.finfo(self.dtype).min
            float_mask = (1.0 - expanded.to(self.dtype)) * neg_inf
            print(f"Expanded attention_mask (shape {expanded.shape}) → float mask:")
            print(float_mask)

        # Finally, call the parent forward method
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
