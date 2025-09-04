#!/usr/bin/env python3
"""
Script to combine task vectors with language steering vectors.

Adds language difference-of-means vector (english_mean - non_english_mean) to task vectors.
Creates two output files for each variant:
1. Task vectors + Language difference (with coefficient)
2. Task vectors + Language difference (without coefficient)

Task vectors are divided by 100 due to different normalization.
"""

import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Combine task vectors with language vectors for steering")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--vector-variant", choices=["plain", "sp"], required=True, help="Task vector variant to use")
    parser.add_argument("--lang-variant", choices=["plain", "sp"], default="plain", help="Language vector variant to use")
    parser.add_argument("--multipliers", type=float, nargs="+", default=[1.0, 2.0, 5.0, 0.0], help="Projection coefficient multipliers (default: 1.0 2.0 5.0 0.0)")
    args = parser.parse_args()
    
    vectors_dir = os.path.join(os.path.dirname(__file__), "..", "vectors")
    
    # Load language vectors for specified variant
    lang_file = os.path.join(vectors_dir, f"language_vectors_{args.model}__{args.lang_variant}.npz")
    if not os.path.exists(lang_file):
        print(f"Error: {lang_file} not found")
        return
    lang_data = np.load(lang_file)
    
    # Load task vectors from weighted file
    weighted_file = os.path.join(vectors_dir, f"weighted_vectors_{args.model}__both.npz")
    if not os.path.exists(weighted_file):
        print(f"Error: {weighted_file} not found")
        return
    
    task_data = np.load(weighted_file)
    
    languages = ["spanish", "french", "german", "italian"]
    
    # Find all layers from language vectors
    layers = set()
    for key in lang_data.keys():
        if "__layer_" in key:
            try:
                layer_idx = int(key.split("__layer_")[-1])
                layers.add(layer_idx)
            except:
                pass
    layers = sorted(layers)
    
    # Create multiple output combinations with different coefficients
    for multiplier in args.multipliers:
        output_payload = {}
            
        # Process each layer
        for layer_idx in layers:
            # Calculate language difference: english_mean - non_english_mean
            english_key = f"english__layer_{layer_idx}"
            english_vec = lang_data[english_key] if english_key in lang_data else None
            
            # Calculate mean of non-English languages
            non_english_sum = None
            non_english_count = 0
            for lang in languages:
                lang_key = f"{lang}__layer_{layer_idx}"
                if lang_key in lang_data:
                    vec = lang_data[lang_key]
                    if non_english_sum is None:
                        non_english_sum = vec.copy()
                    else:
                        non_english_sum += vec
                    non_english_count += 1
            
            # Compute difference of means: english - mean(non_english)
            language_diff = None
            if english_vec is not None and non_english_sum is not None and non_english_count > 0:
                non_english_mean = non_english_sum / non_english_count
                language_diff = english_vec - non_english_mean
            
            # Add id_leverage task vectors from weighted file (divided by 100)
            for task_key in task_data.keys():
                if f"__layer_{layer_idx}" in task_key and "__weighted__" in task_key and "id_leverage_generic" in task_key and f"__{args.vector_variant}__" in task_key:
                    task_vec = task_data[task_key] / 100.0  # Divide by 100 for normalization
                    
                    # Compute projection-equivalent coefficient per layer
                    if language_diff is not None:
                        task_norm = np.linalg.norm(task_vec)
                        lang_norm = np.linalg.norm(language_diff)
                        cos_sim = np.dot(task_vec, language_diff) / (task_norm * lang_norm)
                        proj_coeff = -cos_sim * task_norm / lang_norm
                        
                        # Apply multiplier to projection coefficient
                        final_coeff = multiplier * proj_coeff
                        
                        if multiplier == 1.0:
                            print(f"Layer {layer_idx}: cos_sim={cos_sim:.4f}, proj_coeff={proj_coeff:.4f}")
                        
                        # Combine task vector with language difference
                        lang_component = final_coeff * language_diff
                        combined_key = f"{task_key.replace('__weighted__', '__lang_diff__')}"
                        output_payload[combined_key] = task_vec + lang_component
            
        # Save files with different multipliers
        if multiplier == 0.0:
            coeff_suffix = "_no_proj"
        else:
            coeff_suffix = f"_proj_{multiplier}x"
        
        if output_payload:
            output_file = os.path.join(vectors_dir, f"steering_vectors_{args.model}__{args.vector_variant}_diff{coeff_suffix}.npz")
            np.savez(output_file, **output_payload)
            print(f"Wrote language difference steering vectors: {output_file} ({len(output_payload)} vectors)")


if __name__ == "__main__":
    main()