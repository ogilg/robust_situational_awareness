import os
from typing import Iterable

from sad.task import Task, PLAIN_VARIANT

from .utils import get_combined_samples, get_model_probabilities
from .parsers import tvd_parser
from provider_wrapper import get_provider_for_model, GetTextRequest


class OutputControlTask(Task):
    """
    Provider-wrapper based Output Control task.

    Reuses existing sample generators (given/random words, with/without context/ICL)
    and the TVD parser to score whether the model's top-token distribution matches
    the target distribution within tolerance.
    """

    def __init__(self, name: str = "output_control", path: str | None = None):
        here = path or os.path.dirname(os.path.abspath(__file__))
        super().__init__(name=name, path=here, variants=[PLAIN_VARIANT], random_chance=0.0, has_splits=False)

    def iter_samples(self, model: str, variant: str, n: int | None = None) -> Iterable[dict]:
        num_examples = n if n is not None and n > 0 else 10
        # get_combined_samples returns a list of provider_wrapper.Sample objects
        for sample in get_combined_samples(num_examples=num_examples, shuffle=False):
            yield sample

    def evaluate_sample(self, model: str, sample) -> dict:
        probs = get_model_probabilities(model_id=model, prompt=sample.prompt)
        result, _ = self._score_from_probs(probs, sample)
        return result

    def _run(self, model: str, variant: str, n: int | None = None, save=True, random_order_seed=None, comment=None):
        correct = 0
        incorrect = 0
        invalid = 0  # always 0, since we compare distributions directly

        for sample in self.iter_samples(model=model, variant=variant, n=n):
            res = self.evaluate_sample(model=model, sample=sample)
            if res.get("within_tolerance"):
                correct += 1
            else:
                incorrect += 1

        if save:
            self.save_results_csv(model=model, variant=variant, correct=correct, incorrect=incorrect, invalid=invalid)

        return {"correct": correct, "incorrect": incorrect, "invalid": invalid}

    def evaluate_and_capture_sample(self, model: str, sample, variant: str):
        provider = get_provider_for_model(model)
        # Generate a short audit output and capture first-token residuals
        req = GetTextRequest(context=None, prompt=sample.prompt, max_tokens=5, temperature=0.0)
        text_resp, residuals = provider.generate_text_with_first_token_residuals(req)
        txt = getattr(text_resp, "txt", None)

        # Score via probabilities (T=1)
        probs = get_model_probabilities(model_id=model, prompt=sample.prompt)
        result, extra = self._score_from_probs(probs, sample)
        aux_meta = {
            "txt": txt,
            "tvd": extra.get("tvd"),
            "top_tokens": extra.get("top_tokens"),
            "top_probs": extra.get("top_probs"),
            "target_distribution": sample.target_distribution,
            "tvd_tolerance": sample.tvd_tolerance,
        }
        return result, residuals, aux_meta

    # ----- Helpers to avoid duplication -----
    def _build_sample_data(self, sample) -> dict:
        prompt_content = ""
        try:
            for msg in reversed(sample.prompt):
                if getattr(msg, "role", "user") == "user":
                    prompt_content = getattr(msg, "content", "")
                    break
        except Exception:
            pass
        return {
            "prompt_content": prompt_content,
            "target_distribution": sample.target_distribution,
            "tvd_tolerance": sample.tvd_tolerance,
            "is_given_words": getattr(sample, "is_given_words", False),
            "word1": getattr(sample, "word1", ""),
            "word2": getattr(sample, "word2", ""),
            "seed": getattr(sample, "seed", None),
            "case_type": getattr(sample, "case_type", None),
            "with_context": getattr(sample, "with_context", False),
            "with_icl_examples": getattr(sample, "with_icl_examples", False),
        }

    def _score_from_probs(self, probs: dict, sample) -> tuple[dict, dict]:
        sample_data = self._build_sample_data(sample)
        parsed = tvd_parser(probs, sample_data)
        within = parsed.get("score", 0) == 1
        result = {
            "within_tolerance": within,
            "tvd": parsed.get("tvd_diff", float("inf")),
            "case_type": sample_data["case_type"],
            "with_context": sample_data["with_context"],
            "with_icl_examples": sample_data["with_icl_examples"],
            "correct": 1 if within else 0,
            "incorrect": 0 if within else 1,
            "invalid": 0,
        }
        extra = {
            "tvd": parsed.get("tvd_diff", float("inf")),
            "top_tokens": parsed.get("top_tokens", []),
            "top_probs": parsed.get("top_probs", []),
        }
        return result, extra


def make_task() -> OutputControlTask:
    return OutputControlTask()


