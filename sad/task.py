from abc import ABC
from dataclasses import dataclass
from datetime import datetime
import os
from typing import Callable, Iterable, Optional
import pandas as pd

from .utils import bootstrap_error, change_dir


# Minimal path cleaner to replace evalugator utils
def _clean_for_path(s: str) -> str:
    return (
        str(s)
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )


def get_id(task: str, variant: str, model: str) -> str:
    ts = datetime.now().strftime("%y-%m-%d %H%M%S").replace(" ", "T")
    return f"{_clean_for_path(task)}--{_clean_for_path(variant)}--{_clean_for_path(model)}--{ts}"


@dataclass
class Variant:
    name: str
    pretty_name: str
    in_sad_score: bool
    in_sp_sad_score: bool

    def __lt__(self, other):
        assert isinstance(other, Variant), f"Cannot compare Variant to {type(other)}"
        return self.name < other.name

    def __gt__(self, other):
        assert isinstance(other, Variant), f"Cannot compare Variant to {type(other)}"
        return self.name > other.name


PLAIN_VARIANT = Variant(
    name="plain", pretty_name="Plain", in_sad_score=True, in_sp_sad_score=False
)

SP_VARIANT = Variant(
    name="sp", pretty_name="SP", in_sad_score=False, in_sp_sad_score=True
)

LARGE_SP_VARIANT = Variant(
    name="sp_large", pretty_name="SP (large)", in_sad_score=False, in_sp_sad_score=False
)


class Task(ABC):
    """
    Lightweight task base (provider-agnostic).

    Subclasses should implement either `_run(...)` end-to-end or override
    `iter_samples(...)` and `evaluate_sample(...)` and then call `run_loop(...)`.
    """

    results_file_name = "results.csv"
    splits_file_name = "results_ALL_SPLITS.csv"

    def __init__(
        self,
        name: str,
        path,
        variants: list[Variant],
        random_chance: float = 0.0,
        has_splits: bool = False,
        after_run_hooks: list[Callable] = [],
    ):
        self.name = name
        self.path = path
        self.variants = variants
        self.random_chance = random_chance
        self.default_variant = variants[0]
        self.has_splits = has_splits
        self.after_run_hooks = after_run_hooks

    # Hooks for subclasses
    def _preprocess(self, models: list[str] | None = None):
        pass

    def iter_samples(self, model: str, variant: str, n: int | None = None) -> Iterable[dict]:
        raise NotImplementedError("iter_samples not implemented")

    def evaluate_sample(self, model: str, sample: dict) -> dict:
        raise NotImplementedError("evaluate_sample not implemented")

    def _run(self, model: str, variant: str, n: int | None = None, save=True, random_order_seed=None, comment=None):
        raise NotImplementedError("_run not implemented")

    # Public API
    def preprocess(self, models: list[str] | None = None):
        print(f"Preprocessing for task: {self.name:>30}")
        with change_dir(self.path):
            self._preprocess(models=models)

    def _validate_variant(self, variant: Variant):
        if variant not in self.variants:
            raise ValueError(
                f"Task {self.name} has no variant {variant}. Please specify one of: {self.variants}."
            )

    def run(
        self,
        model: str,
        variant: Optional[Variant] = None,
        n: int | None = None,
        random_order_seed=None,
        save=True,
        comment=None,
    ):
        if variant is None:
            variant = self.default_variant
        self._validate_variant(variant)
        variant_str = variant.name
        print(f"Running task {self.name} variant {variant_str} for {model}")
        with change_dir(self.path):
            result = self._run(
                model=model,
                variant=variant_str,
                n=n,
                save=save,
                random_order_seed=random_order_seed,
                comment=comment,
            )
            for hook_fn in self.after_run_hooks:
                hook_fn(self)
            return result

    # Helper loop for subclasses that want a simple per-sample evaluation
    def run_loop(self, model: str, variant: str, n: int | None = None) -> list[dict]:
        results: list[dict] = []
        for sample in self.iter_samples(model=model, variant=variant, n=n):
            res = self.evaluate_sample(model=model, sample=sample)
            results.append(res)
        return results

    # Results helpers (CSV aggregation). Subclasses may call these.
    def save_results_csv(
        self,
        model: str,
        variant: str,
        correct: int,
        incorrect: int,
        invalid: int,
        timestamp: Optional[str] = None,
        extra_cols: Optional[dict] = None,
    ) -> pd.DataFrame:
        ts = timestamp or datetime.now().isoformat(timespec="seconds")
        total_trials = correct + incorrect + invalid
        accuracy = (correct + self.random_chance * invalid) / max(total_trials, 1)
        std = bootstrap_error(accuracy, total_trials)
        data = {
            "model": [model],
            "task": [self.name],
            "variant": [variant],
            "score": [accuracy],
            "num_trials": [total_trials],
            "std": [std],
            "correct": [correct],
            "incorrect": [incorrect],
            "invalid": [invalid],
            "timestamp": [ts],
        }
        if extra_cols:
            for k, v in extra_cols.items():
                data[k] = [v]
        df = pd.DataFrame(data)
        df.to_csv(self.results_file_name, index=False)
        return df


@dataclass
class Category:
    name: str
    tasks: list[Task]
    category_weighting: float = 1.0

    @property
    def task_weightings(self):
        return {task: self.category_weighting / max(len(self.tasks), 1) for task in self.tasks}
