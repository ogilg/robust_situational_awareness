#!/usr/bin/env python3
"""
Simple comparison of steered vs unsteered performance from a single CSV file.
Uses coefficient 0.0 as baseline and compares against specified coefficient.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
from typing import Dict

TASK_ORDER = [
    "stages_oversight",
    "output_control", 
    "id_leverage_generic",
]

TASK_COLORS: Dict[str, str] = {
    "stages_oversight": "tab:blue",
    "output_control": "tab:orange", 
    "id_leverage_generic": "tab:green",
}

def parse_single_layer(layers_value):
    """Parse layer from string representation like '[25]' or '25'"""
    if pd.isna(layers_value):
        return None
    if isinstance(layers_value, (int, float)):
        return int(layers_value)
    
    layers_str = str(layers_value).strip()
    try:
        # Try to parse as literal (e.g., "[25]" -> [25])
        val = ast.literal_eval(layers_str)
        if isinstance(val, list) and len(val) == 1:
            return int(val[0])
        elif isinstance(val, (int, float)):
            return int(val)
    except:
        # Try to parse as plain number
        try:
            return int(layers_str)
        except:
            return None
    return None

def main():
    parser = argparse.ArgumentParser(description="Simple steered vs unsteered comparison")
    parser.add_argument("--csv-file", required=True, help="Path to CSV file with results")
    parser.add_argument("--steered-coeff", type=float, required=True, help="Steering coefficient to compare")
    parser.add_argument("--layer", type=int, required=True, help="Layer to filter by")
    parser.add_argument("--vector-variant", choices=["plain", "sp"], required=True, help="Vector variant to filter by")
    parser.add_argument("--out-dir", default=".", help="Output directory for plot")
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.csv_file)
    
    # Parse layers and convert accuracy to numeric
    df['layer_parsed'] = df['layers'].apply(parse_single_layer)
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['coefficient'] = pd.to_numeric(df['coefficient'], errors='coerce')
    
    # Filter by layer and vector_variant
    df_filtered = df[(df['layer_parsed'] == args.layer) & (df['vector_variant'] == args.vector_variant)].copy()
    
    if df_filtered.empty:
        print(f"No data found for layer {args.layer} and vector_variant {args.vector_variant}")
        return
    
    # Get baseline (coefficient 0.0) and steered data
    baseline = df_filtered[df_filtered['coefficient'] == 0.0]
    steered = df_filtered[df_filtered['coefficient'] == args.steered_coeff]
    
    if baseline.empty:
        print("No baseline data (coefficient 0.0) found")
        return
    if steered.empty:
        print(f"No steered data for coefficient {args.steered_coeff} found")
        return
    
    # Get tasks and accuracies with proper ordering
    task_data = []
    
    # Collect data for all available tasks
    for task in baseline['task'].unique():
        baseline_task = baseline[baseline['task'] == task]['accuracy'].mean()
        steered_task_data = steered[steered['task'] == task]
        
        if not steered_task_data.empty:
            steered_task = steered_task_data['accuracy'].mean()
            task_data.append({
                'task': task,
                'baseline_acc': baseline_task,
                'steered_acc': steered_task
            })
    
    if not task_data:
        print("No matching tasks between baseline and steered data")
        return
    
    # Sort tasks according to TASK_ORDER
    task_df = pd.DataFrame(task_data)
    task_df['task'] = pd.Categorical(task_df['task'], categories=TASK_ORDER, ordered=True)
    task_df = task_df.sort_values('task')
    
    tasks = task_df['task'].astype(str).tolist()
    baseline_accs = task_df['baseline_acc'].tolist()
    steered_accs = task_df['steered_acc'].tolist()
    
    # Create plot with task-specific colors
    plt.figure(figsize=(10, 6))
    x = range(len(tasks))
    width = 0.35
    
    # Get colors for each task
    colors = [TASK_COLORS.get(task, "tab:gray") for task in tasks]
    
    plt.bar([i - width/2 for i in x], baseline_accs, width, color=colors, alpha=0.6, label=f'Baseline (coeff=0.0)')
    plt.bar([i + width/2 for i in x], steered_accs, width, color=colors, alpha=0.9, label=f'Steered (coeff={args.steered_coeff})', hatch='//')
    
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.title(f'Steered vs Baseline Performance (Layer {args.layer}, {args.vector_variant} vector variant)')
    plt.xticks(x, tasks, rotation=20)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(args.out_dir, exist_ok=True)
    plot_path = os.path.join(args.out_dir, f'compare_coeff_{args.steered_coeff}_layer_{args.layer}_{args.vector_variant}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\nSummary for layer {args.layer} ({args.vector_variant} vector variant):")
    print("Task                 Baseline   Steered    Delta")
    print("-" * 50)
    for i, task in enumerate(tasks):
        delta = steered_accs[i] - baseline_accs[i]
        print(f"{task:<20} {baseline_accs[i]:.3f}     {steered_accs[i]:.3f}     {delta:+.3f}")
    
    print(f"\nPlot saved to: {plot_path}")

if __name__ == "__main__":
    main()