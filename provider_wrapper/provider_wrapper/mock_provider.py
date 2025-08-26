import re
from typing import Any, Dict, List, Tuple

from .data_models import GetTextRequest, GetTextResponse, GetProbsRequest, GetProbsResponse, Prompt


class MockProvider:
    """
    Rule-based fake LLM provider for tests. Implements a subset of the provider API:
      - generate_text
      - get_probs
    """

    def __init__(self, model_id: str = "mock-llm") -> None:
        self.model_id = model_id

    # ----- Prompt formatting (no-op for mock) -----
    def _prompt_to_text(self, prompt: Prompt) -> str:
        return "\n\n".join([msg.content for msg in prompt])

    # ----- Tokenization helpers for tests -----
    def encode(self, data: str) -> List[int]:
        # Simple char-level encoding
        return [ord(ch) for ch in data]

    def decode(self, tokens: List[int]) -> str:
        try:
            return "".join(chr(t) for t in tokens)
        except Exception:
            # Fallback to safe join for unexpected ints
            return "".join(chr(t % 256) for t in tokens)

    # ----- Core ops -----
    def generate_text(self, request: GetTextRequest) -> GetTextResponse:
        text = self._prompt_to_text(request.prompt)

        # Stages/self_recognition style: enforce (A)/(B)
        if re.search(r"\(A\)\s*or\s*\(B\)", text, re.I):
            # Always choose (A) for determinism
            out = "(A)"
            return GetTextResponse(
                model_id=self.model_id,
                request=request,
                txt=out,
                raw_responses=[{"final": out}],
                context={"method": "mock", "prompt_text": text},
            )

        # ID leverage generic: detect target language instruction
        m = re.search(r"respond\s+strictly\s+in\s+(Spanish|French|German|Italian)", text, re.I)
        if m:
            lang = m.group(1).lower()
            samples = {
                "spanish": "Este es un breve ejemplo en español sobre un tema sencillo.",
                "french": "Ceci est un court exemple en français sur un sujet simple.",
                "german": "Dies ist ein kurzes Beispiel auf Deutsch zu einem einfachen Thema.",
                "italian": "Questo è un breve esempio in italiano su un argomento semplice.",
            }
            out = samples.get(lang, "This is a simple English sentence.")
            return GetTextResponse(
                model_id=self.model_id,
                request=request,
                txt=out,
                raw_responses=[{"final": out}],
                context={"method": "mock", "prompt_text": text},
            )

        # Default fallback
        # If this looks like a continuation task, produce a long enough string
        if any(k in text.lower() for k in ["continue", "continuation", "repeat the text", "extend it"]):
            goal = max(int(getattr(request, "max_tokens", 128) or 128), 128)
            chunk = "This is a long continuation sentence designed for testing. "
            out = (chunk * (goal // len(chunk) + 2)).strip()
        else:
            out = "(A)"
        return GetTextResponse(
            model_id=self.model_id,
            request=request,
            txt=out,
            raw_responses=[{"final": out}],
            context={"method": "mock", "prompt_text": text},
        )

    def get_probs(self, request: GetProbsRequest) -> GetProbsResponse:
        text = self._prompt_to_text(request.prompt)

        # Try to extract target percentages p and q
        # e.g., "have 60% of the answers be 'word1', and the remaining 40% should be 'word2'"
        p_match = re.search(r"have\s+(\d+)%\s+of\s+the\s+answers", text, re.I)
        q_match = re.search(r"remaining\s+(\d+)%", text, re.I)

        # Extract given words
        words = re.findall(r"'([^']+)'", text)
        w1 = words[0] if len(words) >= 1 else "alpha"
        w2 = words[1] if len(words) >= 2 else "beta"

        try:
            p = int(p_match.group(1)) if p_match else 50
            q = int(q_match.group(1)) if q_match else (100 - p)
        except Exception:
            p, q = 50, 50

        total = max(p + q, 1)
        p_f = p / total
        q_f = q / total

        probs = {w1: p_f, w2: q_f}
        return GetProbsResponse(
            model_id=self.model_id,
            request=request,
            probs=probs,
            raw_responses=[probs],
            context={"method": "mock", "prompt_text": text, "top_k": 2},
        )


