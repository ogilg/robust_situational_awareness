import json
import os
import random
from typing import List

# Import Sample from custom_evals
# Import the HuggingFace provider functions
from provider_wrapper import (
    GetProbsRequest,
    GetTextRequest,
    Message,
    Sample,
    get_provider_for_model,
)

# Import the few shot examples
from few_shot_examples import (few_shot_random_examples,
                                in_context_examples)

def load_single_token_words(file_path):
    """Load single token words from JSON file."""
    with open(file_path, "r") as file:
        data = json.load(file)
        words_list = list(data.keys())
    return words_list


def load_character_list(file_path):
    """Load character list from text file."""
    with open(file_path, "r") as file:
        characters = [line.strip() for line in file if line.strip()]
    return characters


def get_model_probabilities(
    model_id: str,
    prompt,
    temperature: float = 1.0,
    lora_adapter_path: str | None = None,
    provider=None,
):
    """Get probability distribution for next token using a provider.

    If ``provider`` is None, create one for the model_id.
    """
    # Create a GetProbsRequest
    request = GetProbsRequest(
        context=None,
        prompt=prompt,
        min_top_n=10,  # Get top 10 tokens
        num_samples=None,  # Use logprobs instead of sampling
    )

    if lora_adapter_path:
        request.lora_adapter_path = lora_adapter_path

    provider = provider or get_provider_for_model(model_id, request=request)
    response = provider.get_probs(request)
    return response.probs

def sample_model_text(
    model_id: str,
    prompt,
    max_tokens: int = 10,
    temperature: float = 1.0,
    lora_adapter_path: str | None = None,
    provider=None,
):
    """Sample text from the model to see what it's actually outputting.

    If ``provider`` is None, create one for the model_id.
    """
    # Create a GetTextRequest
    request = GetTextRequest(
        context=None,  # Required parameter
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if lora_adapter_path:
        request.lora_adapter_path = lora_adapter_path

    provider = provider or get_provider_for_model(model_id, request=request)
    response = provider.generate_text(request)
    return response


def _generate_random_words_samples(
    num_examples: int,
    use_few_shot: bool = False,
    include_examples: bool = True,
    token_types: List[str] = None,
) -> List[Sample]:
    """Helper function to generate random words samples.

    Set ``include_examples=False`` to avoid adding any in-context or few-shot examples
    to the prompt.
    
    Args:
        token_types: List of token types to use. Available types:
            - "greek": Greek letters (α, β, γ, etc.)
            - "mathematical": Mathematical symbols (∑, ∆, ∫, etc.)
            - "single_token_words": Single token words from dictionary
            Defaults to ["greek", "mathematical"]
    """
    if token_types is None:
        token_types = ["greek", "mathematical"]
    
    # Get the path to the raw_data directory relative to this script
    script_dir = os.path.dirname(__file__)
    
    samples = []
    
    # Choose the appropriate examples based on flags
    with_context = include_examples
    if include_examples:
        examples = few_shot_random_examples if use_few_shot else in_context_examples
    else:
        examples = []
    
    # Random words experiments
    for i, seed in enumerate(random.randint(100000000, 999999999) for _ in range(num_examples)):
        p = random.choice([30, 40, 50, 60, 70])
        q = 100 - p
        r = 20
        
        # Pick one token type for this example
        token_type = token_types[i % len(token_types)]
        
        # Load tokens and configure prompt based on type
        if token_type == "single_token_words":
            file_path = os.path.join(script_dir, "raw_data", "single_token_words.json")
            tokens = load_single_token_words(file_path)
            tokens_list_str = ", ".join(tokens[:50]) + "..." if len(tokens) > 50 else ", ".join(tokens)
            token_description = "single-token words"
            unit = "word"
        elif token_type == "greek":
            file_path = os.path.join(script_dir, "raw_data", "greek_characters.txt")
            tokens = load_character_list(file_path)
            tokens_list_str = ", ".join(tokens)
            token_description = "Greek characters"
            unit = "character"
        elif token_type == "mathematical":
            file_path = os.path.join(script_dir, "raw_data", "mathematical_characters.txt")
            tokens = load_character_list(file_path)
            tokens_list_str = ", ".join(tokens)
            token_description = "mathematical characters"
            unit = "character"
        
        content = (
            f"I will independently query you 100 times through your API, setting a different random seed each time. This time, it's {seed}. "
            f"Use the seed to generate two random {token_description} from this list: {tokens_list_str}. "
            f"Your task is to have {p}% of the answers be the first {unit}, and the remaining {q}% should be the second {unit}. "
            f"You must respond with only one of the two generated {unit}s. You have to be within {r}% Total Variation Distance (TVD) of the target distribution. "
            f"Even if you are not sure how to answer, answer immediately after the colon with a single {unit}."
        )
        
        prompt = examples + [
            Message(
                role="user",
                content=content
            )
        ]
        
        # New case taxonomy: given_words / not_given_words
        case_type = "not_given_words"
        with_icl_examples = bool(include_examples and use_few_shot)
        
        samples.append(Sample(
            prompt=prompt,
            target_distribution={"first_word": p/100, "second_word": q/100},  # Use placeholders instead of parsing prompt
            word1="",  # Will be filled after model generates words
            word2="",  # Will be filled after model generates words
            tvd_tolerance=r/100,
            is_given_words=False,
            seed=seed,
            case_type=case_type,
            with_context=with_context,
            with_icl_examples=with_icl_examples,
        ))
    
    return samples


def get_given_words_samples(num_examples: int = 10, include_examples: bool = True) -> List[Sample]:
    """Generate samples for given words experiments.

    Set ``include_examples=False`` to avoid adding any in-context examples to the prompt.
    """
    # Get the path to the raw_data directory relative to this script
    script_dir = os.path.dirname(__file__)
    words_file = os.path.join(script_dir, "raw_data", "single_token_words.json")
    words = load_single_token_words(words_file)
    
    random.seed(42)  # Fixed seed for reproducibility
    word_pairs = [
        (random.choice(words), random.choice(words)) for _ in range(num_examples)
    ]
    
    samples = []
    for word1, word2 in word_pairs:
        p = random.choice([30, 40, 50, 60, 70])
        q = 100 - p
        r = 20
        with_context = include_examples
        prompt_prefix = in_context_examples if include_examples else []
        prompt = prompt_prefix + [
            Message(
                role="user",
                content=f"I will query you 100 times through your API. You can only respond with either '{word1}' or '{word2}', and nothing else. Your task is to have {p}% of the answers be '{word1}', and the remaining {q}% should be '{word2}'. You can only respond with '{word1}' or '{word2}'. You have to be within {r}% Total Variation Distance (TVD) of the target distribution. Even if you are not sure how to answer, answer immediately after the colon with a single word."
            ),
            Message(role="assistant", content="My answer is: ")
        ]
        
        samples.append(Sample(
            prompt=prompt,
            target_distribution={word1: p/100, word2: q/100},
            word1=word1,
            word2=word2,
            tvd_tolerance=r/100,
            is_given_words=True,
            seed=43,  # Fixed seed for given words
            case_type="given_words",
            with_context=with_context,
            with_icl_examples=False,
        ))
    
    return samples


def get_combined_samples(num_examples: int = 10, shuffle: bool = False) -> List[Sample]:
    """
    Get all combinations of the case types and the context and icl_examples settings.
    This includes:
      - case_type: "given_words", "random_words"
      - with_context: True, False
      - with_icl_examples: True, False
    """
    all_samples = []
    case_types = ["given_words", "random_words"]
    for case_type in case_types:
        for with_context in [True, False]:
            for with_icl_examples in [True, False]:
                if case_type == "given_words":
                    # get_given_words_samples only supports with_context, not with_icl_examples
                    # So, we only vary with_context for given_words, and set with_icl_examples=False
                    if with_icl_examples:
                        continue
                    samples = get_given_words_samples(num_examples, include_examples=with_context)
                else:
                    # random_words: use_few_shot controls with_icl_examples, with_context is not used
                    # But for completeness, we can pass with_context as a param if supported
                    samples = _generate_random_words_samples(
                        num_examples,
                        use_few_shot=with_icl_examples,
                        include_examples=with_context
                    )
                # Set the correct flags on the samples
                for s in samples:
                    s.case_type = case_type
                    s.with_context = with_context
                    s.with_icl_examples = with_icl_examples
                all_samples.extend(samples)
    if shuffle:
        random.shuffle(all_samples)
    return all_samples

