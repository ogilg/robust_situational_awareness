from typing import Any, Dict, List, Tuple

from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role as HarmonyRole,
    Message as HarmonyMessage,
    Conversation as HarmonyConversation,
    SystemContent,
    ReasoningEffort,
    StreamableParser,
)

from .data_models import (
    Prompt,
    GetTextRequest,
    GetTextResponse,
    GetProbsRequest,
    GetProbsResponse,
    HUGGINGFACE_MODEL_MAPPING,
)
from .huggingface_provider import HuggingFaceProvider


class GPTOSSProvider(HuggingFaceProvider):
    """GPT-OSS prompt formatting via openai-harmony library."""

    def __init__(
        self,
        model_id: str,
        *,
        lora_adapter_path: str | None = None,
        reasoning_effort: ReasoningEffort | str = ReasoningEffort.MEDIUM,
        feed_empty_analysis: bool = False,
    ) -> None:
        super().__init__(model_id, lora_adapter_path=lora_adapter_path)
        effort_map = {"low": ReasoningEffort.LOW, "medium": ReasoningEffort.MEDIUM, "high": ReasoningEffort.HIGH}
        self.reasoning_effort = effort_map[reasoning_effort] if isinstance(reasoning_effort, str) else reasoning_effort
        self.feed_empty_analysis = feed_empty_analysis

    def format_prompt(self, prompt: Prompt) -> Tuple[List[int], str]:
        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        msgs: List[HarmonyMessage] = []
        sys_content = (
            SystemContent.new()
            .with_required_channels(["final"])
            .with_reasoning_effort(self.reasoning_effort)
        )
        msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, sys_content))
        for msg in prompt:
            role = (msg.role or "user").lower()
            if role == "user":
                msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, msg.content))
            elif role == "assistant":
                msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, msg.content))
            elif role == "system":
                msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, msg.content))
            else:
                msgs.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, msg.content))

        if self.feed_empty_analysis:
            msgs.append(
                HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, "").with_channel("analysis")
            )

        convo = HarmonyConversation.from_messages(msgs)
        token_list: List[int] = enc.render_conversation_for_completion(convo, HarmonyRole.ASSISTANT)
        text = self.tokenizer.decode(token_list)
        return token_list, text

    def parse_completion(self, output_tokens: List[int]) -> Dict[str, Any]:
        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        messages = enc.parse_messages_from_completion_tokens(output_tokens, HarmonyRole.ASSISTANT)
        analysis_parts: List[str] = []
        final_parts: List[str] = []
        for m in messages:
            role_str = str(m.author.role.value) if hasattr(m.author.role, "value") else str(m.author.role)
            channel_norm = str(m.channel).lower()
            text_segments: List[str] = [c.text for c in m.content]
            combined = "".join(text_segments)
            role_norm = role_str.lower()
            if role_norm.endswith("assistant"):
                if channel_norm == "analysis":
                    analysis_parts.append(combined)
                elif channel_norm == "final":
                    final_parts.append(combined)

        analysis_text = "".join(analysis_parts).strip()
        final_text = "".join(final_parts).strip()
        return {"raw_messages": messages, "analysis": analysis_text, "final": final_text}

    def find_first_final_content_index(self, completion_tokens: List[int]) -> int | None:
        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        parser = StreamableParser(enc, HarmonyRole.ASSISTANT)
        for i, tok in enumerate(completion_tokens):
            parser.process(tok)
            if parser.current_channel == "final" and parser.current_role == HarmonyRole.ASSISTANT:
                delta = parser.last_content_delta
                if isinstance(delta, str) and len(delta) > 0:
                    return i
        return None

    def generate_text_and_probs(self, request: GetTextRequest) -> Tuple[GetTextResponse, GetProbsResponse]:
        # Format prompt once (shared)
        prompt_tokens, _ = self.format_prompt(request.prompt)

        # Generate text first (may include analysis)
        text_resp = self.generate_text(request)
        completion_tokens: List[int] = text_resp.context.get("generated_tokens", [])

        boundary = self.find_first_final_content_index(completion_tokens)
        tokens_for_logits: List[int] = list(prompt_tokens)
        if boundary is not None:
            tokens_for_logits.extend(completion_tokens[:boundary])
        else:
            tokens_for_logits.extend(completion_tokens)

        probs, k = self._compute_next_token_topk(tokens_for_logits)

        probs_resp = GetProbsResponse(
            model_id=self.model_id,
            request=GetProbsRequest(context=request.context, prompt=request.prompt, min_top_n=50, num_samples=None),
            probs=probs,
            raw_responses=[probs],
            context={
                "method": "logits_based_boundary",
                "top_k": k,
                "hf_model_name": HUGGINGFACE_MODEL_MAPPING[self.model_id],
                "local_model": True,
                "prompt_text": text_resp.context.get("prompt_text"),
                "final_boundary_index": boundary,
            },
        )

        return text_resp, probs_resp