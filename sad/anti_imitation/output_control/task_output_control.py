import os
from typing import Iterable

from sad.task import Task, PLAIN_VARIANT

from .utils import get_combined_samples, get_model_probabilities
from .parsers import tvd_parser


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
        # Query next-token probabilities
        probs = get_model_probabilities(model_id=model, prompt=sample.prompt)

        # Prepare sample metadata for the TVD parser
        # Attempt to include a human-readable prompt excerpt for logging
        prompt_content = ""
        try:
            # last user message content if available
            for msg in reversed(sample.prompt):
                if getattr(msg, "role", "user") == "user":
                    prompt_content = getattr(msg, "content", "")
                    break
        except Exception:
            pass

        sample_data = {
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

        parsed = tvd_parser(probs, sample_data)
        within = parsed.get("score", 0) == 1

        return {
            "within_tolerance": within,
            "tvd": parsed.get("tvd_diff", float("inf")),
            "case_type": sample_data["case_type"],
            "with_context": sample_data["with_context"],
            "with_icl_examples": sample_data["with_icl_examples"],
        }

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


def make_task() -> OutputControlTask:
    return OutputControlTask()


