import torch
import torch.nn as nn

from transformers import GPT2Model, GPT2Config, LlamaModel, LlamaConfig, PreTrainedModel, Qwen3Model, Qwen3Config
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
        model = GPT2Model(config)

        # Zero out position embeddings and freeze them
        with torch.no_grad():
            model.wpe.weight.zero_()
            model.wpe.weight.requires_grad = False

        model_dim = config.n_embd
    elif key == "qwen3": # using this because of good performance for 0.5B model, key differences are RMSNorm instead of LayerNorm, and no bias in the attention layers
        config = Qwen3Config(**kwargs)
        model = Qwen3Model(config)
        model_dim = config.hidden_size
    elif key == "llama":
        config = LlamaConfig(**kwargs)
        model = LlamaModel(config)
        model_dim = config.hidden_size
    else:
        raise ValueError(f"Unknown backbone key {key}")
    
    # Apply LoRA if config is provided
    if lora_config is not None:
        model = prepare_model_for_lora(model, key, lora_config)

    return model, model_dim
