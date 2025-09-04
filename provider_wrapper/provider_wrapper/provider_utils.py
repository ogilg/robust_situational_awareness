import gc
import os
import shutil
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer


from .data_models import HUGGINGFACE_MODEL_MAPPING
from .huggingface_provider import HuggingFaceProvider, DefaultHFProvider, extract_model_name
from .mock_provider import MockProvider
from .data_models import (
    GetProbsRequest,
    GetTextRequest,
    HUGGINGFACE_MODEL_MAPPING,
)

from openai_harmony import (
    ReasoningEffort
)

def provides_model(model_id: str) -> bool:
    """Return True if this provider supports the given model_id."""
    name = extract_model_name(model_id)
    return name in HUGGINGFACE_MODEL_MAPPING


def execute(model_id: str, request):
    """Handle both text generation and probability requests."""
    if not provides_model(model_id):
        raise NotImplementedError(f"Model {model_id} is not supported by HuggingFace provider")
    
    provider = get_provider_for_model(model_id, prefer_transformerlens=False)
    if isinstance(request, GetTextRequest):
        return provider.generate_text(request)
    elif isinstance(request, GetProbsRequest):
        return provider.get_probs(request)
    else:
        raise NotImplementedError(
            f"Request {type(request).__name__} for model {model_id} is not implemented"
        )


def get_provider_for_model(
    model_id: str,
    request: Any | None = None,
    *,
    lora_adapter_path: str | None = None,
    reasoning_effort: ReasoningEffort | str | None = None,
    feed_empty_analysis: bool | None = None,
    prefer_transformerlens: bool = False,
) -> HuggingFaceProvider:
    """Factory to return the appropriate provider instance for a model_id.

    Optional args allow configuring GPT-OSS-specific behavior from callers.
    """
    name = extract_model_name(model_id)

    # Provide a mock provider for tests
    if name in {"mock-llm", "mock"}:
        # Return a small shim exposing the same methods as HF providers
        return MockProvider(name)  # type: ignore
    # Prefer explicit args; fall back to request context for lora
    if lora_adapter_path is None and request is not None:
        lora_adapter_path = getattr(request, "lora_adapter_path", None)

    if name == "gpt-oss-20b":
        # Defer import to avoid circular dependency
        from .gptoss_provider import GPTOSSProvider  # type: ignore
        kwargs: Dict[str, Any] = {"lora_adapter_path": lora_adapter_path}
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if feed_empty_analysis is not None:
            kwargs["feed_empty_analysis"] = feed_empty_analysis
        return GPTOSSProvider(name, **kwargs)
    if prefer_transformerlens:
        from .transformer_lens_provider import TransformerLensProvider  # type: ignore
        return TransformerLensProvider(name)
    return DefaultHFProvider(name, lora_adapter_path=lora_adapter_path)


# Model Management Functions

def preload_model(model_id: str, verbose: bool = False) -> bool:
    """
    Preload a model into the HuggingFace cache.
    
    Args:
        model_id: Model identifier (e.g., "llama-3.1-8b")
        verbose: If True, print detailed loading messages
        
    Returns:
        True if successful, False otherwise
    """
    
    # Extract actual model name
    model_name = extract_model_name(model_id)
    
    if model_name not in HUGGINGFACE_MODEL_MAPPING:
        if verbose:
            print(f"ERROR: Model {model_id} not found in HUGGINGFACE_MODEL_MAPPING")
        return False
    
    hf_model_name = HUGGINGFACE_MODEL_MAPPING[model_name]
    if verbose:
        print(f"Preloading {model_id} ({hf_model_name})")
    
    try:
        # Use the factory to choose the right provider class and trigger load
        provider = get_provider_for_model(model_name, prefer_transformerlens=False)
        _ = provider.model  # ensure loaded
        
        if verbose:
            print(f"✓ {model_id} preloaded successfully")
        
        # DO NOT clear from memory - keep it cached for reuse
        # Only clear GPU memory to free up space for other operations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"✗ Failed to preload {model_id}: {e}")
        return False


def validate_model_availability(model_id: str, local_only: bool = True) -> bool:
    """
    Check if a model is available (downloaded) without loading it fully.
    
    Args:
        model_id: Model identifier
        local_only: If True, only check local cache (don't download)
        
    Returns:
        True if model is available, False otherwise
    """
    
    # Extract actual model name
    model_name = extract_model_name(model_id)
    
    if model_name not in HUGGINGFACE_MODEL_MAPPING:
        return False
    
    hf_model_name = HUGGINGFACE_MODEL_MAPPING[model_name]
    
    try:
        # Try to load tokenizer (lightweight check)
        AutoTokenizer.from_pretrained(hf_model_name, local_files_only=local_only)
        return True
    except Exception:
        return False


def clear_huggingface_disk_cache():
    """Clear HuggingFace disk cache to free space."""
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        try:
            # Only clear the hub cache, keep tokenizers cache for efficiency
            hub_cache = os.path.join(cache_dir, "hub")
            if os.path.exists(hub_cache):
                shutil.rmtree(hub_cache)
                print("HuggingFace hub cache cleared")
        except Exception as e:
            print(f"Warning: Could not clear HuggingFace cache: {e}")

