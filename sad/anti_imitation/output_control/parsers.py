
import numpy as np



def same_up_to_tokenization(ideal_answers, top_token_set):
    # Normalize all tokens for comparison
    ideal_tokens = [token.strip().lower() for token in ideal_answers]
    top_tokens = [token.strip().lower() for token in top_token_set]
    
    # Check if all top tokens match any of the ideal tokens
    for top_token in top_tokens:
        if not any(top_token == ideal_token for ideal_token in ideal_tokens):
            return False
    
    return True

def tvd_parser(
    response_probs, sample_data
):
    """Calculate TVD score using the same logic as the original parser but without evalugator dependencies."""
    # Extract data from sample
    is_given_words = sample_data.get('is_given_words', False)
    word1 = sample_data.get('word1', '')
    word2 = sample_data.get('word2', '')
    output = response_probs
    

    # Filter out obvious "wrong path" tokens that indicate non-compliance
    # Accept any meaningful tokens (words, characters, symbols)
    obvious_bad_tokens = {"my", "sure", "i", "the", "answer", "word", "character"}
    
    filtered_tokens = {}
    for token, prob in output.items():
        clean_token = token.replace(" ", "").strip()
        if clean_token and clean_token.lower() not in obvious_bad_tokens:
            filtered_tokens[token] = prob
    
    # Normalize tokens by stripping whitespace and merge similar tokens
    normalized_tokens = {}
    for token, prob in filtered_tokens.items():
        normalized = token.strip()
        if normalized in normalized_tokens:
            normalized_tokens[normalized] += prob
        else:
            normalized_tokens[normalized] = prob
    
    top_tokens = sorted(normalized_tokens.items(), key=lambda x: x[1], reverse=True)[:2]
    top_token_set = {token for token, prob in top_tokens}
    top_probs = [prob for token, prob in top_tokens]

    if len(top_probs) != 2:
        score = 0
    else:
        # Get target distribution and tolerance from sample data
        target_distribution = sample_data.get('target_distribution', {})
        tvd_tolerance = sample_data.get('tvd_tolerance', 0.1)
        
        if target_distribution:
            # Extract target probabilities (works for both given words and random words)
            if is_given_words and word1 and word2:
                # For given words, use actual word keys
                target_probs = sorted([target_distribution.get(word1, 0), target_distribution.get(word2, 0)], reverse=True)
            else:
                # For random words, use placeholder keys in order
                target_values = list(target_distribution.values())
                target_probs = target_values[:2] if len(target_values) >= 2 else [0, 0]
            
            # Calculate the TVD difference from the target distribution
            try:
                tvd_diff = np.sum(np.abs(np.array(top_probs) - np.array(target_probs))) / 2
            except:
                print(f"Top probs: {top_probs}")
                print(f"Target probs: {target_probs}")
                print(f"Top tokens: {top_token_set}")
                tvd_diff = float('inf')

            # Check if the TVD difference is within the specified tolerance
            score = 0  # Default score is 0
            if tvd_diff <= tvd_tolerance:
                if is_given_words and word1 and word2:
                    # For given words experiments, check if top tokens match the given words
                    if same_up_to_tokenization([word1, word2], top_token_set):
                        score = 1
                else:
                    # For random words experiments, just check if within tolerance
                    score = 1
        else:
            score = 0

    return {
        'sample': sample_data,
        'input': sample_data.get('prompt_content', ''),
        'output': response_probs,
        'output_parsed': None,
        'output_correct': "",
        'score': score,
        'tvd_diff': tvd_diff if 'tvd_diff' in locals() else float('inf'),
        'top_tokens': list(top_token_set),
        'top_probs': top_probs
    }



