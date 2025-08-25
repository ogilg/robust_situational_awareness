"""Custom evaluations package - HuggingFace provider functionality with LoRA fine-tuning support."""

from .lora_config import LoRAConfig
from .lora_finetuner import LoRAFineTuner
from .loss_functions import CustomLossFunction, KLDivergenceLoss, TVDLoss
from .data_models import (
    Sample,
    HUGGINGFACE_MODEL_MAPPING,
    CHAT_MODELS,
    GetProbsRequest,
    GetProbsResponse,
    GetTextRequest,
    GetTextResponse,
    Message,
    Prompt,
)
from .huggingface_provider import (
    HuggingFaceProvider,
    DefaultHFProvider,
    GPTOSSProvider,
    get_provider_for_model,
    preload_model,
    validate_model_availability,
    clear_model_cache,
    clear_huggingface_disk_cache,
    clear_gpu_memory,
    extract_model_name,
)

__version__ = "0.1.0"
__all__ = [
    # HuggingFace provider
    "HuggingFaceProvider",
    "DefaultHFProvider",
    "GPTOSSProvider",
    "get_provider_for_model",
    "HUGGINGFACE_MODEL_MAPPING",
    "CHAT_MODELS",
    
    # Request templates
    "GetTextRequest",
    "GetTextResponse", 
    "GetProbsRequest",
    "GetProbsResponse",
    "Message",
    "Prompt",
    
    # Data models
    "Sample",
    
    # LoRA configuration
    "LoRAConfig",
    
    # LoRA fine-tuning
    "LoRAFineTuner",
    "CustomLossFunction",
    "TVDLoss",
    "KLDivergenceLoss",
    
    # Model management
    "preload_model",
    "validate_model_availability",
    "clear_model_cache", 
    "clear_huggingface_disk_cache",
    "clear_gpu_memory",
    "extract_model_name",
] 