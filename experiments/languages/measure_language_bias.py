#!/usr/bin/env python3
"""
Measure language bias in task vectors using extracted language difference vectors.
Computes cosine similarity between task vectors and language difference vectors.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Measure language bias in task vectors")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--variant", choices=["plain", "sp"], required=True, help="Variant")
    args = parser.parse_args()
    
    vectors_dir = os.path.join(os.path.dirname(__file__), "..", "vectors")
    
    # Load task vectors
    task_file = os.path.join(vectors_dir, f"weighted_vectors_{args.model}__both.npz")
    if not os.path.exists(task_file):
        print(f"Error: {task_file} not found")
        return
    task_data = np.load(task_file)
    
    # Load language vectors
    lang_file = os.path.join(vectors_dir, f"language_vectors_{args.model}__plain.npz")
    if not os.path.exists(lang_file):
        print(f"Error: {lang_file} not found")
        return
    lang_data = np.load(lang_file)
    
    # Calculate language difference vector per layer
    languages = ["spanish", "french", "german", "italian"]
    
    # Find all layers and id_leverage task
    layers = []
    bias_values = []
    
    for key in task_data.keys():
        if "__weighted__layer_" in key and f"__{args.variant}__" in key and "id_leverage_generic" in key:
            layer_idx = int(key.split("__layer_")[-1])
            task_vec = task_data[key]
            
            # Calculate language difference for this layer
            english_key = f"english__layer_{layer_idx}"
            english_vec = lang_data.get(english_key)
            
            non_english_sum = None
            count = 0
            for lang in languages:
                lang_key = f"{lang}__layer_{layer_idx}"
                if lang_key in lang_data:
                    vec = lang_data[lang_key]
                    if non_english_sum is None:
                        non_english_sum = vec.copy()
                    else:
                        non_english_sum += vec
                    count += 1
            
            if english_vec is not None and non_english_sum is not None and count > 0:
                non_english_mean = non_english_sum / count
                lang_diff = english_vec - non_english_mean
                
                # Compute cosine similarity
                cos_sim = np.dot(task_vec, lang_diff) / (np.linalg.norm(task_vec) * np.linalg.norm(lang_diff))
                
                layers.append(layer_idx)
                bias_values.append(cos_sim)
                print(f"Layer {layer_idx:<6} Bias: {cos_sim:<12.4f}")
    
    # Create plot
    if layers and bias_values:
        plt.figure(figsize=(10, 6))
        plt.plot(layers, bias_values, 'o-')
        plt.xlabel('Layer')
        plt.ylabel('Language Bias (Cosine Similarity)')
        plt.title(f'Language Bias in ID Leverage Task - {args.model} ({args.variant})')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Neutral')
        plt.legend()
        
        plot_file = os.path.join(vectors_dir, f"language_bias_{args.model}_{args.variant}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot_file}")
        plt.show()


if __name__ == "__main__":
    main()