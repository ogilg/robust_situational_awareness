#!/usr/bin/env python3
"""
Script to combine task vectors from weighted_ files with language vectors for steering.

Fetches task vectors from weighted_ files and precomputes them layer by layer for both variants.
Creates four output files:
1. Non-English subtraction + Task vectors (with coefficient)
2. Non-English subtraction + Task vectors (without coefficient) 
3. English addition + Task vectors (with coefficient)
4. English addition + Task vectors (without coefficient)

Task vectors are divided by 100 due to different normalization.
"""

import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Combine task vectors with language vectors for steering")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--coeff", type=float, default=0.5, help="Coefficient for language steering (default: 0.5)")
    args = parser.parse_args()
    
    vectors_dir = os.path.join(os.path.dirname(__file__), "..", "vectors")
    
    # Load language vectors
    language_files = {}
    for variant in ["plain", "sp"]:
        lang_file = os.path.join(vectors_dir, f"language_vectors_{args.model}__{variant}.npz")
        if os.path.exists(lang_file):
            language_files[variant] = np.load(lang_file)
        else:
            print(f"Warning: {lang_file} not found")
    
    if not language_files:
        print("No language vector files found - creating task-only steering vectors")
    
    # Load task vectors from weighted file
    weighted_file = os.path.join(vectors_dir, f"weighted_vectors_{args.model}__both.npz")
    if not os.path.exists(weighted_file):
        print(f"Error: {weighted_file} not found")
        return
    
    task_data = np.load(weighted_file)
    
    languages = ["spanish", "french", "german", "italian"]
    
    # Find all layers from language vectors
    layers = set()
    for variant_data in language_files.values():
        for key in variant_data.keys():
            if "__layer_" in key:
                try:
                    layer_idx = int(key.split("__layer_")[-1])
                    layers.add(layer_idx)
                except:
                    pass
    layers = sorted(layers)
    
    # Create four output combinations
    for variant in ["plain", "sp"]:
        if variant not in language_files:
            continue
            
        lang_data = language_files[variant]
        
        for use_coeff in [True, False]:
            # Create output payloads for both language combinations
            subtract_payload = {}
            add_payload = {}
            
            # Process each layer
            for layer_idx in layers:
                # Non-English subtraction combination
                non_english_sum = None
                for lang in languages:
                    lang_key = f"{lang}__layer_{layer_idx}"
                    if lang_key in lang_data:
                        vec = lang_data[lang_key]
                        if non_english_sum is None:
                            non_english_sum = vec.copy()
                        else:
                            non_english_sum += vec
                
                # English addition combination
                english_key = f"english__layer_{layer_idx}"
                english_vec = lang_data[english_key] if english_key in lang_data else None
                
                # Add task vectors from weighted file (divided by 100)
                for task_key in task_data.keys():
                    if f"__layer_{layer_idx}" in task_key and "__weighted__" in task_key:
                        task_vec = task_data[task_key] / 100.0  # Divide by 100 for normalization
                        
                        # Non-English subtraction + task vector
                        if non_english_sum is not None:
                            lang_component = -args.coeff * non_english_sum if use_coeff else -non_english_sum
                            combined_key = f"{task_key.replace('__weighted__', '__subtract_non_english__')}"
                            subtract_payload[combined_key] = lang_component + task_vec
                        
                        # English addition + task vector  
                        if english_vec is not None:
                            lang_component = args.coeff * english_vec if use_coeff else english_vec
                            combined_key = f"{task_key.replace('__weighted__', '__add_english__')}"
                            add_payload[combined_key] = lang_component + task_vec
            
            # Save files
            coeff_suffix = f"_coeff_{args.coeff}" if use_coeff else "_no_coeff"
            
            if subtract_payload:
                subtract_file = os.path.join(vectors_dir, f"steering_vectors_{args.model}__{variant}__subtract_non_english{coeff_suffix}.npz")
                np.savez(subtract_file, **subtract_payload)
                print(f"Wrote subtract non-English vectors: {subtract_file} ({len(subtract_payload)} vectors)")
            
            if add_payload:
                add_file = os.path.join(vectors_dir, f"steering_vectors_{args.model}__{variant}__add_english{coeff_suffix}.npz")
                np.savez(add_file, **add_payload)
                print(f"Wrote add English vectors: {add_file} ({len(add_payload)} vectors)")


if __name__ == "__main__":
    main()