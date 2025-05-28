import torch
import torch.nn as nn
from peft import get_peft_model
from peft.tuners.lora import LoraConfig
from transformers import PreTrainedModel
from typing import Optional, Dict, Any

def get_model_target_modules(model_type: str) -> list:
    """
    Get the appropriate target modules for LoRA based on model type.
    
    Args:
        model_type: The type of transformer model ("GPT2", "qwen3", or "llama") for now
    
    Returns:
        List of module names to apply LoRA to
    """
    if model_type == "GPT2":
        return ["c_attn", "c_proj"]  # GPT2 specific attention modules
    elif model_type == "qwen3":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]  # Qwen3 attention modules
    elif model_type == "llama":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]  # LLaMA attention modules
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def prepare_model_for_lora(
    model: PreTrainedModel,
    model_type: str,
    lora_config: Optional[Dict[str, Any]] = None,
) -> PreTrainedModel:
    """
    Prepare a transformer model for LoRA fine-tuning.
    
    Args:
        model: The base transformer model
        model_type: The type of transformer model ("GPT2", "qwen3", or "llama") for now
        lora_config: Optional configuration for LoRA. If None, uses default config.
    
    Returns:
        Model configured for LoRA training
    """
    if lora_config is None:
        lora_config = {
            "r": 8,  # rank
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none",
            "target_modules": get_model_target_modules(model_type),
        }
    
    # Create LoRA config
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        target_modules=lora_config["target_modules"],
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def get_lora_config(
    model_type: str,
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    bias: str = "none",
    target_modules: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Get a default LoRA configuration.
    
    Args:
        model_type: The type of transformer model ("GPT2", "qwen3", or "llama") for now
        r: LoRA attention dimension
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout probability
        bias: Bias type for LoRA layers
        target_modules: Optional list of module names to apply LoRA to
    
    Returns:
        Dictionary containing LoRA configuration
    """
    if target_modules is None:
        target_modules = get_model_target_modules(model_type)
        
    return {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": bias,
        "target_modules": target_modules,
    } 