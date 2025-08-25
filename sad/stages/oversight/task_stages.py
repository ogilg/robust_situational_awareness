import json
import os
from typing import Iterable, Optional

from sad.task import Task, PLAIN_VARIANT, SP_VARIANT
from sad.templates import sa_prefix, Message

# provider_wrapper API
from provider_wrapper import (
    GetTextRequest,
    get_provider_for_model,
)


def _load_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_choice(text: str, answer_space: list[str]) -> Optional[str]:
    if text is None:
        return None
    t = text.strip()
    # Try letter formats first
    if t.startswith("(A)") or t.startswith("A)") or t.startswith("A"):
        return "(A)" if "(A)" in answer_space else "A"
    if t.startswith("(B)") or t.startswith("B)") or t.startswith("B"):
        return "(B)" if "(B)" in answer_space else "B"
    # Fallback: match by token/substring against answer options
    tl = t.lower()
    best = None
    for opt in answer_space:
        if opt.lower() in tl:
            best = opt
            break
    return best


class StagesOversightTask(Task):
    """
    Provider-wrapper based Stages Oversight task.

    Expects a JSONL dataset with one object per line with fields:
      - messages: [{"role": "system|user|assistant", "content": str}, ...]
      - answer_space: ["(A)", "(B)"] or ["testing", "deployment"], etc.
      - correct_label: one element from answer_space
      - splits: optional dict of split metadata (unused here)
    """

    def __init__(self, name: str, path: str, data_path: Optional[str] = None):
        super().__init__(name=name, path=path, variants=[PLAIN_VARIANT, SP_VARIANT], random_chance=0.0, has_splits=False)
        self.data_path = data_path  # set by caller or CLI

    def iter_samples(self, model: str, variant: str, n: int | None = None) -> Iterable[dict]:
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"StagesOversightTask: data_path not found: {self.data_path}. Provide a JSONL with messages/answer_space/correct_label."
            )
        count = 0
        for sample in _load_jsonl(self.data_path):
            yield sample
            count += 1
            if n is not None and count >= n:
                break

    def evaluate_sample(self, model: str, sample: dict) -> dict:
        messages = sample.get("messages", [])
        answer_space = sample.get("answer_space", ["(A)", "(B)"])
        correct_label = sample.get("correct_label")

        # Build provider messages; apply SP variant via system prefix if present
        rendered_messages: list[Message] = []
        if hasattr(sa_prefix, "template") and isinstance(sa_prefix.template, str):
            # Use caller-provided variant by inserting prefix externally if desired.
            pass
        for m in messages:
            rendered_messages.append(Message(role=m["role"], content=m["content"]))

        # Call provider for a short answer
        request = GetTextRequest(
            context=None,
            prompt=rendered_messages,
            max_tokens=5,
            temperature=0.0,
        )
        provider = get_provider_for_model(model, request=request)
        response = provider.generate_text(request)
        text = getattr(response, "text", None) if response is not None else None
        pred = _parse_choice(text or "", answer_space)

        is_correct = (pred is not None and correct_label is not None and pred == correct_label)
        result = {
            "pred": pred,
            "correct": 1 if is_correct else 0,
            "incorrect": 0 if is_correct else 1,
            "invalid": 0 if pred is not None else 1,
        }
        return result

    def _run(self, model: str, variant: str, n: int | None = None, save=True, random_order_seed=None, comment=None):
        correct = 0
        incorrect = 0
        invalid = 0
        for sample in self.iter_samples(model=model, variant=variant, n=n):
            res = self.evaluate_sample(model=model, sample=sample)
            correct += int(res.get("correct", 0))
            incorrect += int(res.get("incorrect", 0))
            invalid += int(res.get("invalid", 0))
        if save:
            self.save_results_csv(model=model, variant=variant, correct=correct, incorrect=incorrect, invalid=invalid)
        return {"correct": correct, "incorrect": incorrect, "invalid": invalid}


def make_task(data_path: Optional[str] = None) -> StagesOversightTask:
    here = os.path.dirname(os.path.abspath(__file__))
    return StagesOversightTask(name="stages_oversight", path=here, data_path=data_path)


