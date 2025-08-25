from typing import Optional, Literal

from pydantic import BaseModel


class OutputControlExperimentConfig(BaseModel):
    model: str
    num_examples: int = 10
    lora_adapter: Optional[str] = None
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    feed_empty_analysis: bool = False
    save_individual: bool = True


