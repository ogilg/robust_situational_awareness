import os
import json
from typing import cast, Dict, List, Any, Tuple
from tqdm import tqdm
from pathlib import Path


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data from file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def stages_preprocess(batch_names: list[str], pile_id: str):
    """Load and validate stage data files."""
    if not os.path.exists("../private_data_gen"):
        raise Exception(
            f"Path ../private_data_gen does not exist. Currently in {os.getcwd()}. Stages preprocessing cannot run; have you unzipped the files (see SADCODE main README)?"
        )
    
    loaded_batches = {}
    for batch_name in batch_names:
        path = f"../private_data_gen/out/{batch_name}.jsonl"
        if os.path.exists(path):
            loaded_batches[batch_name] = load_jsonl_data(path)
            print(f"Loaded {len(loaded_batches[batch_name])} samples from {batch_name}")
        else:
            print(f"Warning: {path} not found")
    
    return loaded_batches


def get_templates() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Get template configurations for stages evaluation."""
    # Simplified template loading - load from YAML files if they exist
    template_config = {
        "text_above": ["The following is", "Here is"],  # Default text indicators for above positioning
        "text_below": ["Based on the above", "Given this"],  # Default text indicators for below positioning
    }
    
    multi_template = {
        "id": "template",
        "templates": ["prompt_above", "prompt_below"],
        "request_args": {"temperature": 0.0, "max_tokens": 10},
    }
    
    return template_config, multi_template


def find_prompt_position(message_content: str, templates: Dict[str, List[str]]) -> str:
    """Determine if prompt is above or below based on content."""
    for txt in templates["text_above"]:
        if txt in message_content:
            return "below"
    for txt in templates["text_below"]:
        if txt in message_content:
            return "above"
    return "unknown"

def split_annotation_postprocess(task, ids=None):
    """Annotate evaluation results with split information."""
    templates, _ = get_templates()
    
    # For now, just print that we would annotate splits
    # In a full implementation, this would process the results
    print(f"Would annotate splits for evaluation results")
    print(f"Template config: {templates}")
    
    # Call the task's method if it exists
    if hasattr(task, 'get_splits_df'):
        task.get_splits_df()
