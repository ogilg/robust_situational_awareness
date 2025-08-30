from typing import Any, Dict, Tuple
import os

import torch
from transformer_lens import HookedTransformer

from .data_models import GetTextRequest, GetTextResponse
from .huggingface_provider import DefaultHFProvider
from .huggingface_provider import clear_gpu_memory

# Global cache for TransformerLens models to avoid repeated shard reloads
_tl_models: Dict[str, HookedTransformer] = {}


class TransformerLensProvider(DefaultHFProvider):
    """HF provider extended with TransformerLens-specific generation paths.

    Exposes the TL residual capture and intervention utilities already implemented
    on DefaultHFProvider.
    """
    
    # ---------------- First-token residuals during generation (TransformerLens) -----------------
    def _get_tl_model(self, tl_name: str):
        # Prefer CUDA for TL regardless of HF model device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use module-level cache so repeated provider instances don't reload shards
        global _tl_models
        tl_model = _tl_models.get(tl_name)
        if tl_model is None:
            clear_gpu_memory()
            # Align TL behavior with HF chat formatting; avoid auto-BOS
            tl_model = HookedTransformer.from_pretrained(
                tl_name,
                device=str(device),
                dtype=torch.float16,
                center_writing_weights=False,
                default_prepend_bos=False,
            )
            _tl_models[tl_name] = tl_model
        return tl_model

    def generate_text_with_first_token_residuals(self, request: GetTextRequest):
        """
        Generate text and capture resid_pre for the FIRST generated token across all layers
        in a single TransformerLens generation pass.

        Returns: (GetTextResponse, {layer_idx: tensor[1, d_model] on CPU})
        """
        # Map provider id to TL model id (extend as needed)
        TL_COMPATIBLE: dict[str, str] = {
            # Use the Instruct checkpoint to match HF generation behavior
            # and chat templating, otherwise outputs will look off.
            "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
            "qwen-2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
        }
        name = self.model_id
        if name not in TL_COMPATIBLE:
            raise NotImplementedError(
                f"First-token residuals via TransformerLens not implemented for model '{name}'."
            )
        tl_name = TL_COMPATIBLE[name]

        # Format the prompt exactly like HF generate_text (chat template + generation prompt)
        # then let TransformerLens tokenize the TEXT (its own tokenizer/vocab).
        _, text = self.format_prompt(request.prompt)
        tl_model = self._get_tl_model(tl_name)
        if os.environ.get("SA_DEBUG_PROMPT") == "1":
            try:
                print(f"[DEBUG PROMPT TL:{self.model_id}]\n{text}\n[END DEBUG PROMPT]\n")
            except Exception:
                pass
        toks = tl_model.to_tokens(text, prepend_bos=False)
        prompt_len = toks.shape[1]

        cached: dict[int, torch.Tensor] = {}
        hook_variant = os.environ.get("SA_TL_HOOK", "pre").strip().lower()
        hook_attr = {
            "pre": "hook_resid_pre",
            "mid": "hook_resid_mid",
            "post": "hook_resid_post",
        }.get(hook_variant, "hook_resid_pre")

        def make_hook(layer_idx: int, p_len: int):
            def hook_fn(activation, hook):
 
                # Capture first generated token. With KV cache, seq len can be 1 on gen step
                if layer_idx not in cached:
                    pos_len = activation.shape[1]
                    if pos_len > p_len or pos_len == 1:
                        cached[layer_idx] = activation[:, -1, :].detach().cpu()

                return activation
            return hook_fn

        fwd_hooks = [
            (f"blocks.{i}.{hook_attr}", make_hook(i, prompt_len))
            for i in range(tl_model.cfg.n_layers)
        ]

        tl_model.reset_hooks()
        with tl_model.hooks(fwd_hooks=fwd_hooks):
            with torch.no_grad():
                out_toks = tl_model.generate(
                    toks,
                    max_new_tokens=int(getattr(request, "max_tokens", 16) or 16),
                    temperature=float(getattr(request, "temperature", 0.0) or 0.0),
                    verbose=False,
                )

        # Decode only generated tail
        gen_only = out_toks[:, prompt_len:]
        # Decode with TL tokenizer for the generated portion
        txt = tl_model.to_string(gen_only[0]) if gen_only.shape[1] > 0 else ""

        text_resp = GetTextResponse(
            model_id=self.model_id,
            request=request,
            txt=txt,
            raw_responses=[{"final": txt}],
            context={
                "method": "tl_generate_with_first_token_resid",
                "prompt_text": text,
                "prompt_len": prompt_len,
                "generated_tokens": gen_only[0].detach().cpu().tolist(),
            },
        )

        # Optional token debug: print prompt boundary token and first generated token
        if os.environ.get("SA_DEBUG_TOKENS") == "1":
            try:
                if prompt_len > 0:
                    prompt_last_id = int(toks[0, prompt_len - 1].item())
                    prompt_last_str = tl_model.to_string(toks[0, prompt_len - 1:prompt_len])
                else:
                    prompt_last_id = -1
                    prompt_last_str = ""
                if gen_only.shape[1] > 0:
                    first_gen_id = int(gen_only[0, 0].item())
                    first_gen_str = tl_model.to_string(gen_only[0, :1])
                else:
                    first_gen_id = -1
                    first_gen_str = ""
                print(f"[TL] boundary_token id={prompt_last_id} text={repr(prompt_last_str)} | first_generated_token id={first_gen_id} text={repr(first_gen_str)}")
            except Exception as e:
                try:
                    print(f"[TL] token_debug_failed: {e}")
                except Exception:
                    pass

        # Proactive cleanup to minimize peak memory
        try:
            del out_toks
            del gen_only
            del toks
        except Exception:
            pass
        try:
            tl_model.reset_hooks()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        return text_resp, cached

    # ---------------- Generation with activation interventions (TransformerLens) -----------------
    # --- Shared TL utilities ---
    def _tl_compat_name(self) -> str:
        TL_COMPATIBLE: dict[str, str] = {
            "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
            "qwen-2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
        }
        name = self.model_id
        if name not in TL_COMPATIBLE:
            raise NotImplementedError(
                f"TransformerLens path not implemented for model '{name}'."
            )
        return TL_COMPATIBLE[name]

    def _tl_tokens_for_request(self, request: GetTextRequest):
        _, text = self.format_prompt(request.prompt)
        tl_model = self._get_tl_model(self._tl_compat_name())
        toks = tl_model.to_tokens(text, prepend_bos=False)
        return tl_model, toks

    def _tl_hook_attr(self, hook_variant: str) -> str:
        return {
            "pre": "hook_resid_pre",
            "mid": "hook_resid_mid",
            "post": "hook_resid_post",
        }.get(hook_variant.strip().lower(), "hook_resid_pre")

    def _tl_prepare_vectors(self, tl_model: HookedTransformer, vectors: dict[int, Any], layers: list[int] | None) -> tuple[list[int], dict[int, torch.Tensor]]:
        target_layers = sorted(list(vectors.keys()) if layers is None else list(layers))
        device = tl_model.cfg.device
        d_model = tl_model.cfg.d_model
        torch_vecs: dict[int, torch.Tensor] = {}
        for li in target_layers:
            vec = vectors.get(li)
            if vec is None:
                continue
            try:
                t = torch.as_tensor(vec, device=device, dtype=torch.float32)
                if t.ndim != 1 or int(t.shape[0]) != int(d_model):
                    continue
                torch_vecs[li] = t
            except Exception:
                continue
        return target_layers, torch_vecs

    def _tl_generate_with_hooks(self, tl_model: HookedTransformer, toks, request: GetTextRequest, fwd_hooks, *, context_extra: dict) -> GetTextResponse:
        tl_model.reset_hooks()
        with tl_model.hooks(fwd_hooks=fwd_hooks):
            with torch.no_grad():
                out_toks = tl_model.generate(
                    toks,
                    max_new_tokens=int(getattr(request, "max_tokens", 16) or 16),
                    temperature=float(getattr(request, "temperature", 0.0) or 0.0),
                    verbose=False,
                )
        gen_only = out_toks[:, toks.shape[1]:]
        txt = tl_model.to_string(gen_only[0]) if gen_only.shape[1] > 0 else ""
        ctx = {
            "method": "tl_generate_with_interventions",
        }
        ctx.update(context_extra or {})
        return GetTextResponse(
            model_id=self.model_id,
            request=request,
            txt=txt,
            raw_responses=[{"final": txt}],
            context=ctx,
        )

    def generate_text_with_additions(
        self,
        request: GetTextRequest,
        *,
        vectors: dict[int, Any],
        coefficient: float = 1.0,
        layers: list[int] | None = None,
        hook_variant: str = "pre",
        positions: str = "all",
    ):
        tl_model, toks = self._tl_tokens_for_request(request)
        hook_attr = self._tl_hook_attr(hook_variant)
        target_layers, torch_vecs = self._tl_prepare_vectors(tl_model, vectors, layers)

        def make_add_hook(v: torch.Tensor):
            def hook_fn(activation, hook):
                if positions == "last":
                    activation[:, -1, :] = activation[:, -1, :] + float(coefficient) * v
                else:
                    activation = activation + float(coefficient) * v.view(1, 1, -1)
                return activation
            return hook_fn

        fwd_hooks = [
            (f"blocks.{i}.{hook_attr}", make_add_hook(torch_vecs[i]))
            for i in target_layers if i in torch_vecs
        ]
        return self._tl_generate_with_hooks(
            tl_model,
            toks,
            request,
            fwd_hooks,
            context_extra={
                "layers": target_layers,
                "mode": "add",
                "coefficient": coefficient,
                "hook_variant": hook_variant,
                "positions": positions,
            },
        )

    def generate_text_with_projection(
        self,
        request: GetTextRequest,
        *,
        vectors: dict[int, Any],
        layers: list[int] | None = None,
        hook_variant: str = "pre",
        positions: str = "all",
    ):
        tl_model, toks = self._tl_tokens_for_request(request)
        hook_attr = self._tl_hook_attr(hook_variant)
        target_layers, torch_vecs = self._tl_prepare_vectors(tl_model, vectors, layers)

        def make_project_hook(v: torch.Tensor):
            u = v / (v.norm() + 1e-12)
            def hook_fn(activation, hook):
                if positions == "last":
                    x = activation[:, -1, :]
                    dot = (x * u).sum(dim=-1, keepdim=True)
                    activation[:, -1, :] = x - dot * u.view(1, -1)
                else:
                    dot = torch.tensordot(activation, u, dims=([-1], [0]))
                    activation = activation - dot.unsqueeze(-1) * u.view(1, 1, -1)
                return activation
            return hook_fn

        fwd_hooks = [
            (f"blocks.{i}.{hook_attr}", make_project_hook(torch_vecs[i]))
            for i in target_layers if i in torch_vecs
        ]
        return self._tl_generate_with_hooks(
            tl_model,
            toks,
            request,
            fwd_hooks,
            context_extra={
                "layers": target_layers,
                "mode": "project",
                "hook_variant": hook_variant,
                "positions": positions,
            },
        )

