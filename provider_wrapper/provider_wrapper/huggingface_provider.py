"""
HuggingFace provider for evalugator.

Refactored to a class-based design so we can support multiple prompt formats
while sharing model/tokenizer loading, generation and logits logic.
"""

import os
import gc
from typing import Any, Dict, List, Tuple, Optional

import backoff
import torch
from torch import no_grad, softmax, topk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .data_models import (
    GetProbsRequest,
    GetProbsResponse,
    GetTextRequest,
    GetTextResponse,
    Prompt,
    HUGGINGFACE_MODEL_MAPPING,
    CHAT_MODELS,
)


from transformer_lens import HookedTransformer

# Ensure Transformers logs with INFO verbosity for clearer warning messages
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")

# Global cache for TransformerLens models to avoid repeated shard reloads
_tl_models: Dict[str, HookedTransformer] = {}

def on_backoff(details):
    print(f"Repeating failed request (attempt {details['tries']}). Reason: {details['exception']}")
    

# Singleton storage (cached across providers)
_models: Dict[str, Any] = {}
_tokenizers: Dict[str, Any] = {}

def extract_model_name(model_id: str) -> str:
    """Extract the model name from the model_id."""
    return model_id.split("/")[1] if model_id.startswith("huggingface/") else model_id



class HuggingFaceProvider:
    """Parent provider with shared functionality.

    Child classes should override ``format_prompt`` to return both tokens and text.
    """

    @classmethod
    def get_model_and_tokenizer(
        cls,
        model_id: str,
        lora_adapter_path: str | None = None,
    ):
        """Get or load model and tokenizer using singleton pattern.

        Caches by model_id + quantization + lora-adapter presence so callers can
        share models safely across providers.
        """

        # Cache key includes provider class to allow different loading strategies
        model_key = f"{model_id}_{cls.__name__}_lora" if lora_adapter_path else f"{model_id}_{cls.__name__}"

        if model_key not in _models:
            hf_model_name = HUGGINGFACE_MODEL_MAPPING[model_id]

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Build base kwargs and allow subclass to extend
            model_kwargs: Dict[str, Any] = {
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            # Allow subclasses to add provider-specific kwargs (e.g., quantization)
            extra_kwargs = cls._extra_model_kwargs()
            if extra_kwargs:
                model_kwargs.update(extra_kwargs)

            model = AutoModelForCausalLM.from_pretrained(hf_model_name, **model_kwargs)

            # Load LoRA adapter if provided
            if lora_adapter_path:
                try:
                    from peft import PeftModel

                    print(f"Loading LoRA adapter from {lora_adapter_path}")
                    model = PeftModel.from_pretrained(model, lora_adapter_path)
                    print("LoRA adapter loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load LoRA adapter from {lora_adapter_path}: {e}")
                    print("Continuing with base model")

            _models[model_key] = model
            _tokenizers[model_key] = tokenizer

        return _models[model_key], _tokenizers[model_key]


    def __init__(
        self,
        model_id: str,
        *,
        lora_adapter_path: str | None = None,
    ) -> None:
        self.model_id = extract_model_name(model_id)
        self.lora_adapter_path = lora_adapter_path
        self.model, self.tokenizer = self.get_model_and_tokenizer(
            self.model_id, lora_adapter_path
        )

    @classmethod
    def _extra_model_kwargs(cls) -> Dict[str, Any]:
        """Subclass hook: return extra kwargs for model.from_pretrained.

        Base: no extras.
        """
        return {}

    # ----- Prompt formatting API -----
    def format_prompt(self, prompt: Prompt) -> Tuple[List[int], str]:
        """Return (tokens, text) for the model input.

        Base implementation uses a simple newline-joined prompt.
        """
        text = "\n\n".join([msg.content for msg in prompt])
        tokens = self.tokenizer.encode(text)
        return tokens, text

    # ----- Public helpers -----
    def encode(self, data: str) -> List[int]:
        return self.tokenizer.encode(data)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def parse_completion(self, output_tokens: List[int]) -> Dict[str, Any]:
        """Default completion parser.

        Returns a dict with:
          - raw_messages: None (no structured messages for default models)
          - analysis: "" (no analysis channel)
          - final: decoded text from tokens
        Subclasses may override to return structured messages and channels.
        """
        text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return {"raw_messages": None, "analysis": "", "final": text}

    # ----- Core ops -----
    def generate_text(self, request: GetTextRequest) -> GetTextResponse:
        tokens, text = self.format_prompt(request.prompt)
        if os.environ.get("SA_DEBUG_PROMPT") == "1":
            try:
                print(f"[DEBUG PROMPT HF:{self.model_id}]\n{text}\n[END DEBUG PROMPT]\n")
            except Exception:
                pass

        # Build input tensor from tokens directly for full control
        input_ids = torch.tensor([tokens], dtype=torch.long)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)

        with no_grad():
            is_greedy = request.temperature is not None and request.temperature <= 0.0
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": request.max_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
                "attention_mask": attention_mask,
            }
            if is_greedy:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                # Only add temperature if we're sampling and temperature is valid
                if request.temperature is not None and request.temperature > 0.0:
                    gen_kwargs["temperature"] = request.temperature


            # For DefaultHFProvider, do not use Harmony stop tokens; rely on model defaults

            outputs = self.model.generate(input_ids=input_ids, **gen_kwargs)

        generated_tokens = outputs[0][input_ids.shape[-1]:]
        # Parse from tokens (base decodes with tokenizer; subclasses can override)
        generated_text = self.parse_completion(
            generated_tokens.tolist() if hasattr(generated_tokens, "tolist") else generated_tokens
        )

        return GetTextResponse(
            model_id=self.model_id,
            request=request,
            txt=generated_text["final"],
            raw_responses=[generated_text],
            context={
                "method": "generation",
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "hf_model_name": HUGGINGFACE_MODEL_MAPPING[self.model_id],
                "local_model": True,
                "prompt_text": text,
                "generated_tokens": (generated_tokens.tolist() if hasattr(generated_tokens, "tolist") else list(generated_tokens)),
                "prompt_tokens_len": len(tokens),
            },
        )

    def get_probs(self, request: GetProbsRequest) -> GetProbsResponse:
        tokens, text = self.format_prompt(request.prompt)
        probs, k = self._compute_next_token_topk(tokens, top_k=50)

        return GetProbsResponse(
            model_id=self.model_id,
            request=request,
            probs=probs,
            raw_responses=[probs],
            context={
                "method": "logits_based",
                "top_k": k,
                "hf_model_name": HUGGINGFACE_MODEL_MAPPING[self.model_id],
                "local_model": True,
                "prompt_text": text,
            },
        )

    def _compute_next_token_topk(self, tokens: List[int], top_k: int = 50) -> Tuple[Dict[str, float], int]:
        input_ids = torch.tensor([tokens], dtype=torch.long)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)

        with no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, -1, :]

        probs_tensor = softmax(logits, dim=-1)
        k = min(top_k, probs_tensor.shape[-1])
        top_probs, top_indices = topk(probs_tensor, k)

        probs: Dict[str, float] = {}
        for prob, idx in zip(top_probs.detach().cpu().tolist(), top_indices.detach().cpu().tolist()):
            token_str = self.tokenizer.decode([idx])
            probs[token_str] = prob
        return probs, k


class DefaultHFProvider(HuggingFaceProvider):
    """Default prompt formatting.

    - Chat models: use tokenizer chat template
    - Non-chat models: simple join
    """

    def format_prompt(self, prompt: Prompt) -> Tuple[List[int], str]:
        if self.model_id in CHAT_MODELS:
            messages = [{"role": msg.role, "content": msg.content} for msg in prompt]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = "\n\n".join([msg.content for msg in prompt])
        tokens = self.tokenizer.encode(text)
        return tokens, text

    @classmethod
    def _extra_model_kwargs(cls) -> Dict[str, Any]:
        # Apply 8-bit quantization by default for standard HF models, compute in bfloat16
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
            ),
            "torch_dtype": torch.bfloat16,
        }

def clear_model_cache(model_id: Optional[str] = None):
    """
    Clear models from memory cache.
    
    Args:
        model_id: Specific model to clear, or None to clear all
    """
    
    if model_id is None:
        # Clear all models
        _models.clear()
        _tokenizers.clear()
    else:
        # Clear specific model
        model_name = extract_model_name(model_id)

        # Remove from singleton dictionaries
        keys_to_remove = [key for key in list(_models.keys()) if key.startswith(model_name)]
        for key in keys_to_remove:
            if key in _models:
                del _models[key]
            if key in _tokenizers:
                del _tokenizers[key]
    
    # Clear GPU memory
    clear_gpu_memory()


def clear_gpu_memory():
    """Clear GPU memory safely."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()






# ----- Simple module-level helpers for compatibility -----
# def encode(model_id: str, data: str) -> List[int]:
#     provider = DefaultHFProvider(extract_model_name(model_id))
#     return provider.encode(data)


# def decode(model_id: str, tokens: List[int]) -> str:
#     provider = DefaultHFProvider(extract_model_name(model_id))
#     return provider.decode(tokens)


