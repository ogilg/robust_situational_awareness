"""
Data models for custom evaluations package.

This module contains data structures used throughout the custom-evals package.
"""

from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


# Model mapping from evalugator model IDs to HuggingFace model names
HUGGINGFACE_MODEL_MAPPING = {
    # OpenAI GPT-OSS
    "gpt-oss-20b": "openai/gpt-oss-20b",
    # Meta Llama Instruct (extend/edit as needed)
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}

# Models that support chat format (instruct/chat models)
CHAT_MODELS = {
    # Treat instruct models as chat-capable
    "llama-3.1-8b-instruct": True,
}


@dataclass
class Message:
    role: str
    content: str


Prompt = list[Message]

@dataclass
class Sample:
    """A sample for training or evaluation with prompt and target distribution."""
    prompt: List[Prompt]
    target_distribution: Dict[str, float]
    word1: str
    word2: str
    tvd_tolerance: float
    is_given_words: bool
    seed: Optional[int] = None
    case_type: str = "given_words"  # Default case type 
    # Flags to describe prompt composition
    with_context: bool = False
    # Only meaningful for not_given_words when context includes ICL examples
    with_icl_examples: bool = False


@dataclass(kw_only=True)
class Request:
    #   Request context stores information that should be consider private/internal
    #   and providers should not access it. The structure might be different for different evals.
    context: Any
    prompt: Prompt

    def as_dict(self):
        return asdict(self)


@dataclass(kw_only=True)
class Response:
    model_id: str
    request: Request

    #   Raw responses from the API should go here. This is intended mostly for debuging.
    raw_responses: list[Any]

    #   Response context is meant mostly for low-level logging. 
    #   Provider is allowed to put anything (json-serializable) here, e.g.:
    #   *   exact prompt/messages structure sent to the API
    #   *   execution time
    #   *   additional response data (logprobs or whatever)
    #   *   provider config (if there's a config)
    #   It's also perfectly fine to set it to None.
    #
    #   NOTE: In some contexts (e.g. in solvers), evalugator translates some responses to different
    #         responses, so the response returned by the provider will not necessarely be the same
    #         as the response returned to the eval code.
    context: Any

    @abstractmethod
    def as_dict(self):
        raise NotImplementedError


@dataclass(kw_only=True)
class GetTextRequest(Request):
    temperature: float
    max_tokens: int


@dataclass(kw_only=True)
class GetTextResponse(Response):
    txt: str

    def as_dict(self):
        return {
            "model_id": self.model_id,
            "request": self.request.as_dict(),
            "txt": self.txt,
            "context": self.context,
        }


@dataclass(kw_only=True)
class GetProbsRequest(Request):
    #   If you need only top N most likely tokens, set this value.
    #   Provider is allowed to return more tokens than this.
    #   The only effect (right now) is that with this value set to <=5, 
    #   openai uses logprobs, and with a higher value raises NotImplementedError.
    min_top_n: int

    #   If this is not None, providers that don't support logprobs will
    #   do sampling with temperature=1
    num_samples: int | None


@dataclass(kw_only=True)
class GetProbsResponse(Response):
    probs: dict[str, float]

    def as_dict(self):
        return {
            "model_id": self.model_id,
            "request": self.request.as_dict(),
            "probs": self.probs,
            "context": self.context,
        }
