#!/usr/bin/env python3
"""
Calculate and visualize cosine similarity between weighted vectors from different tasks.

Usage:
  python experiments/cosine_similarity.py --model llama-3.1-8b-instruct --layer 15
  python experiments/cosine_similarity.py --model llama-3.1-8b-instruct --all-layers
"""

import os
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional


def load_weighted_vectors(model: str, vectors_dir: str) -> Dict[str, np.ndarray]:
    """Load weighted vectors from the npz file."""
    weighted_path = os.path.join(vectors_dir, f"weighted_vectors_{model}.npz")
    if not os.path.exists(weighted_path):
        raise FileNotFoundError(f"Weighted vectors file not found: {weighted_path}")
    
    data = np.load(weighted_path)
    return {key: data[key] for key in data.files}


def parse_vector_key(key: str) -> Tuple[str, int]:
    """Parse vector key to extract task name and layer index.
    
    Example: 'stages__weighted__layer_15' -> ('stages', 15)
    """
    parts = key.split("__")
    if len(parts) >= 3 and parts[-2] == "weighted" and parts[-1].startswith("layer_"):
        task = "__".join(parts[:-2])
        layer_idx = int(parts[-1].split("_")[-1])
        return task, layer_idx
    raise ValueError(f"Cannot parse vector key: {key}")


def group_vectors_by_layer(vectors: Dict[str, np.ndarray]) -> Dict[int, Dict[str, np.ndarray]]:
    """Group vectors by layer index."""
    by_layer = {}
    
    for key, vector in vectors.items():
        try:
            task, layer_idx = parse_vector_key(key)
            if layer_idx not in by_layer:
                by_layer[layer_idx] = {}
            by_layer[layer_idx][task] = vector
        except ValueError:
            continue
    
    return by_layer


def calculate_cosine_similarity_matrix(task_vectors: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Calculate cosine similarity matrix between tasks."""
    tasks = list(task_vectors.keys())
    vectors = [task_vectors[task] for task in tasks]
    
    # Stack vectors into matrix (n_tasks, n_features)
    vector_matrix = np.stack(vectors)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(vector_matrix)
    
    return similarity_matrix, tasks


def filter_tasks(by_layer: Dict[int, Dict[str, np.ndarray]], tasks: Optional[List[str]]) -> Dict[int, Dict[str, np.ndarray]]:
    if not tasks:
        return by_layer
    keep = set(tasks)
    filtered: Dict[int, Dict[str, np.ndarray]] = {}
    for layer_idx, tv in by_layer.items():
        sub = {t: v for t, v in tv.items() if t in keep}
        if len(sub) >= 2:
            filtered[layer_idx] = sub
    return filtered


def compute_pairwise_similarities_by_layer(by_layer: Dict[int, Dict[str, np.ndarray]]) -> Dict[Tuple[str, str], List[Tuple[int, float]]]:
    """For each task pair, collect (layer_idx, cosine) across all layers where both exist."""
    # Collect all task names
    task_names = sorted({t for tv in by_layer.values() for t in tv.keys()})
    pair_sims: Dict[Tuple[str, str], List[Tuple[int, float]]] = {}
    for t1, t2 in itertools.combinations(task_names, 2):
        pair_key = (t1, t2)
        sims: List[Tuple[int, float]] = []
        for layer_idx, tv in by_layer.items():
            if t1 in tv and t2 in tv:
                v1 = tv[t1]
                v2 = tv[t2]
                # Cosine for two vectors
                sim = float(cosine_similarity([v1], [v2])[0, 0])
                sims.append((layer_idx, sim))
        if sims:
            pair_sims[pair_key] = sorted(sims, key=lambda x: x[0])
    return pair_sims


def compute_overall_average_similarity(pair_sims: Dict[Tuple[str, str], List[Tuple[int, float]]]) -> List[Tuple[str, str, float, int]]:
    """Average similarity across layers for each pair; returns list of (t1, t2, mean_sim, n_layers)."""
    rows: List[Tuple[str, str, float, int]] = []
    for (t1, t2), sims in sorted(pair_sims.items()):
        vals = [s for _, s in sims]
        if vals:
            rows.append((t1, t2, float(np.mean(vals)), len(vals)))
    return rows


def plot_similarity_heatmap(similarity_matrix: np.ndarray, task_names: List[str], 
                          layer_idx: int, model: str, save_path: str = None):
    """Plot cosine similarity as a heatmap."""
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        xticklabels=task_names,
        yticklabels=task_names,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    plt.title(f'Cosine Similarity Between Tasks\nModel: {model}, Layer: {layer_idx}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_similarity_across_layers(by_layer: Dict[int, Dict[str, np.ndarray]], 
                                 task_pairs: List[Tuple[str, str]], model: str, 
                                 save_path: str = None):
    """Plot cosine similarity for specific task pairs across all layers."""
    layers = sorted(by_layer.keys())
    
    plt.figure(figsize=(12, 8))
    
    for task1, task2 in task_pairs:
        similarities = []
        valid_layers = []
        
        for layer_idx in layers:
            task_vectors = by_layer[layer_idx]
            if task1 in task_vectors and task2 in task_vectors:
                sim = cosine_similarity([task_vectors[task1]], [task_vectors[task2]])[0, 0]
                similarities.append(sim)
                valid_layers.append(layer_idx)
        
        if similarities:
            plt.plot(valid_layers, similarities, marker='o', label=f'{task1} vs {task2}')
    
    plt.xlabel('Layer Index')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Cosine Similarity Across Layers\nModel: {model}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved layer plot: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_pc1_explained_variance_by_layer(by_layer: Dict[int, Dict[str, np.ndarray]], *, normalize: bool = True) -> List[Tuple[int, float, int]]:
    """Return list of (layer_idx, pc1_variance_ratio, n_tasks) for layers with >=2 tasks.

    If normalize=True, L2-normalize each task vector before PCA.
    """
    out: List[Tuple[int, float, int]] = []
    for layer_idx in sorted(by_layer.keys()):
        tv = by_layer[layer_idx]
        if len(tv) < 2:
            continue
        X = np.stack([v for _, v in sorted(tv.items())])
        if normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
        try:
            pca = PCA(n_components=min(5, X.shape[0], X.shape[1]))
            pca.fit(X)
            pc1 = float(pca.explained_variance_ratio_[0]) if len(pca.explained_variance_ratio_) > 0 else 0.0
        except Exception:
            pc1 = 0.0
        out.append((layer_idx, pc1, X.shape[0]))
    return out


def plot_pc1_variance(pc1_by_layer: List[Tuple[int, float, int]], *, model: str, save_path: Optional[str]) -> None:
    if not pc1_by_layer:
        return
    layers = [x[0] for x in pc1_by_layer]
    vals = [x[1] for x in pc1_by_layer]
    ns = [x[2] for x in pc1_by_layer]
    plt.figure(figsize=(10, 5))
    plt.plot(layers, vals, marker='o')
    plt.xlabel('Layer Index')
    plt.ylabel('PC1 explained variance ratio')
    plt.title(f'Explained Variance of PC1 for Task Vectors per Layer\nModel: {model}')
    # annotate n_tasks per point lightly
    for x, y, n in zip(layers, vals, ns):
        plt.text(x, y, str(n), fontsize=8, ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PC1 variance plot: {save_path}")
    else:
        plt.show()
    plt.close()


def save_overall_similarity_csv(rows: List[Tuple[str, str, float, int]], *, out_csv: str) -> None:
    import csv
    if not rows:
        return
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["task_1", "task_2", "mean_cosine_across_layers", "n_layers"])
        for t1, t2, m, n in rows:
            w.writerow([t1, t2, f"{m:.6f}", n])
    print(f"Saved overall pairwise similarities to {out_csv}")


def compute_pc1_and_cosines_by_layer(by_layer: Dict[int, Dict[str, np.ndarray]], *, normalize: bool = True) -> Tuple[List[Tuple[int, float]], Dict[int, Dict[str, float]]]:
    """
    For each layer, compute PC1 (from PCA over task vectors) and return:
    - list of (layer_idx, pc1_variance_ratio)
    - mapping layer_idx -> {task_name: cosine(task_vector, pc1_vector)}
    """
    pc1_var_list: List[Tuple[int, float]] = []
    cos_to_pc1: Dict[int, Dict[str, float]] = {}
    for layer_idx in sorted(by_layer.keys()):
        tv = by_layer[layer_idx]
        if len(tv) < 2:
            continue
        # Stable task order
        items = sorted(tv.items())
        tasks = [t for t, _ in items]
        X = np.stack([v for _, v in items])
        if normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
        try:
            pca = PCA(n_components=1)
            pca.fit(X)
            pc1_vec = pca.components_[0]
            pc1_var = float(pca.explained_variance_ratio_[0]) if len(pca.explained_variance_ratio_) else 0.0
        except Exception:
            pc1_vec = np.zeros((X.shape[1],), dtype=X.dtype)
            pc1_var = 0.0
        # Cosines to PC1
        pc1_norm = np.linalg.norm(pc1_vec)
        if pc1_norm == 0:
            cosines = {t: 0.0 for t in tasks}
        else:
            cosines = {t: float(np.dot(x, pc1_vec) / max(np.linalg.norm(x) * pc1_norm, 1e-12)) for t, x in zip(tasks, X)}
        pc1_var_list.append((layer_idx, pc1_var))
        cos_to_pc1[layer_idx] = cosines
    return pc1_var_list, cos_to_pc1


def plot_mean_cosine_to_pc1(cos_to_pc1: Dict[int, Dict[str, float]], *, model: str, save_path: Optional[str]) -> None:
    if not cos_to_pc1:
        return
    layers = sorted(cos_to_pc1.keys())
    means = []
    stds = []
    for l in layers:
        vals = list(cos_to_pc1[l].values())
        if not vals:
            means.append(0.0)
            stds.append(0.0)
        else:
            means.append(float(np.mean(np.abs(vals))))  # alignment magnitude
            stds.append(float(np.std(np.abs(vals))))
    plt.figure(figsize=(10, 5))
    means = np.asarray(means)
    stds = np.asarray(stds)
    plt.plot(layers, means, marker='o', label='mean |cos(task, PC1)|')
    plt.fill_between(layers, means - stds, means + stds, color='C0', alpha=0.2, label='±1 std')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean |cosine(task vector, layer PC1)|')
    plt.title(f'Alignment of Task Vectors to Each Layer\'s PC1 (Mean |cos|)\nModel: {model}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved mean cosine-to-PC1 plot: {save_path}")
    else:
        plt.show()
    plt.close()


def save_cosine_to_pc1_csv(cos_to_pc1: Dict[int, Dict[str, float]], *, out_csv: str) -> None:
    import csv
    if not cos_to_pc1:
        return
    layers = sorted(cos_to_pc1.keys())
    tasks = sorted({t for l in layers for t in cos_to_pc1[l].keys()})
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["layer"] + tasks)
        for l in layers:
            row = [l] + [f"{float(cos_to_pc1[l].get(t, 0.0)):.6f}" for t in tasks]
            w.writerow(row)
    print(f"Saved cosine-to-PC1 table to {out_csv}")


# -------- Per-task stability across layers (adjacent-layer cosine) --------
def compute_adjacent_layer_cosines(by_layer: Dict[int, Dict[str, np.ndarray]], *, normalize: bool = True) -> Dict[str, List[Tuple[int, int, float]]]:
    """
    For each task, compute cosine(v_layer_i, v_layer_{i+1}) for consecutive layers where the task exists.
    Returns: {task: [(layer_i, layer_j, cosine), ...]} with layer_j = next layer after layer_i.
    """
    # Collect per-task vectors by layer
    task_to_layers: Dict[str, Dict[int, np.ndarray]] = {}
    for layer_idx, tv in by_layer.items():
        for task, vec in tv.items():
            task_to_layers.setdefault(task, {})[layer_idx] = vec

    task_adj: Dict[str, List[Tuple[int, int, float]]] = {}
    for task, layer_map in task_to_layers.items():
        layers_sorted = sorted(layer_map.keys())
        pairs: List[Tuple[int, int, float]] = []
        for a, b in zip(layers_sorted, layers_sorted[1:]):
            v1 = layer_map[a]
            v2 = layer_map[b]
            if normalize:
                n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
                if n1 == 0 or n2 == 0:
                    cos = 0.0
                else:
                    cos = float(np.dot(v1, v2) / (n1 * n2))
            else:
                # fallback to sklearn for consistency
                cos = float(cosine_similarity([v1], [v2])[0, 0])
            pairs.append((a, b, cos))
        if pairs:
            task_adj[task] = pairs
    return task_adj


def plot_task_stability_adjacent(adj_cosines: Dict[str, List[Tuple[int, int, float]]], *, model: str, save_path: Optional[str]) -> None:
    """Plot per-task cosine between consecutive layers.
    X-axis: starting layer index (layer_i).
    Y-axis: cosine(v_i, v_{i+1}). One line per task.
    """
    if not adj_cosines:
        return
    plt.figure(figsize=(12, 7))
    for task, pairs in sorted(adj_cosines.items()):
        xs = [i for (i, j, c) in pairs]
        ys = [c for (i, j, c) in pairs]
        if xs and ys:
            plt.plot(xs, ys, marker='o', label=task)
    plt.xlabel('Starting Layer Index (i)')
    plt.ylabel('Cosine between consecutive layers cos(v_i, v_{i+1})')
    plt.title(f'Per-Task Representation Stability Across Layers (Adjacent Cosine)\nModel: {model}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved task stability plot: {save_path}")
    else:
        plt.show()
    plt.close()


def save_task_stability_csv(adj_cosines: Dict[str, List[Tuple[int, int, float]]], *, out_csv: str) -> None:
    import csv
    if not adj_cosines:
        return
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["task", "layer_i", "layer_j", "cosine_adjacent"])
        for task, pairs in sorted(adj_cosines.items()):
            for i, j, c in pairs:
                w.writerow([task, i, j, f"{c:.6f}"])
    print(f"Saved task stability table to {out_csv}")

def print_similarity_summary(similarity_matrix: np.ndarray, task_names: List[str], layer_idx: int):
    """Print a summary of similarity values."""
    print(f"\nLayer {layer_idx} Similarity Summary:")
    print("-" * 40)
    
    n_tasks = len(task_names)
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            sim = similarity_matrix[i, j]
            print(f"{task_names[i]} vs {task_names[j]}: {sim:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Calculate and visualize cosine similarity between task vectors")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--vectors-dir", default=None, help="Directory containing vectors (defaults to experiments/vectors)")
    parser.add_argument("--layer", type=int, default=None, help="Specific layer to analyze")
    parser.add_argument("--all-layers", action="store_true", help="Analyze all layers and plot trends")
    parser.add_argument("--output-dir", default=None, help="Directory to save plots (defaults to experiments/results)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively instead of saving")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional subset of tasks to include (space-separated)")
    parser.add_argument("--save-overall-csv", action="store_true", help="Save overall average pairwise similarity CSV")
    parser.add_argument("--pca", action="store_true", help="Run PCA per layer and plot PC1 explained variance and cosine-to-PC1")
    parser.add_argument("--save-cos-to-pc1-csv", action="store_true", help="Save per-layer cosine(task, PC1) table")
    parser.add_argument("--save-pc1-vectors", action="store_true", help="Save PC1 vectors per layer to NPZ (experiments/vectors)")
    
    args = parser.parse_args()
    
    # Set default paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if args.vectors_dir is None:
        args.vectors_dir = os.path.join(root_dir, "experiments", "vectors")
    if args.output_dir is None:
        args.output_dir = os.path.join(root_dir, "experiments", "results")
    
    # Load vectors
    print(f"Loading weighted vectors for model: {args.model}")
    vectors = load_weighted_vectors(args.model, args.vectors_dir)
    print(f"Loaded {len(vectors)} vectors")
    
    # Group by layer
    by_layer = filter_tasks(group_vectors_by_layer(vectors), args.tasks)
    layers = sorted(by_layer.keys())
    print(f"Found vectors for layers: {layers}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.layer is not None:
        # Analyze specific layer
        if args.layer not in by_layer:
            print(f"Layer {args.layer} not found in vectors")
            return
        
        task_vectors = by_layer[args.layer]
        similarity_matrix, task_names = calculate_cosine_similarity_matrix(task_vectors)
        
        print_similarity_summary(similarity_matrix, task_names, args.layer)
        
        # Plot heatmap
        save_path = None if args.show else os.path.join(args.output_dir, f"cosine_similarity_{args.model}_layer_{args.layer}.png")
        plot_similarity_heatmap(similarity_matrix, task_names, args.layer, args.model, save_path)
        
    elif args.all_layers:
        # Analyze all layers
        # First, create heatmaps for a few key layers
        key_layers = [layers[0], layers[len(layers)//2], layers[-1]] if len(layers) >= 3 else layers
        
        for layer_idx in key_layers:
            task_vectors = by_layer[layer_idx]
            similarity_matrix, task_names = calculate_cosine_similarity_matrix(task_vectors)
            
            save_path = None if args.show else os.path.join(args.output_dir, f"cosine_similarity_{args.model}_layer_{layer_idx}.png")
            plot_similarity_heatmap(similarity_matrix, task_names, layer_idx, args.model, save_path)
        
        # Plot similarity trends across layers for all task pairs
        all_tasks = sorted({t for tv in by_layer.values() for t in tv.keys()})
        task_pairs = [(all_tasks[i], all_tasks[j]) for i in range(len(all_tasks)) 
                     for j in range(i + 1, len(all_tasks))]
        
        save_path = None if args.show else os.path.join(args.output_dir, f"cosine_similarity_across_layers_{args.model}.png")
        plot_similarity_across_layers(by_layer, task_pairs, args.model, save_path)
        
        # Overall average similarity across layers for each pair
        pair_sims = compute_pairwise_similarities_by_layer(by_layer)
        overall_rows = compute_overall_average_similarity(pair_sims)
        if overall_rows:
            print("\nOverall average cosine similarity across layers (pairwise):")
            for t1, t2, m, n in overall_rows:
                print(f"  {t1:>20} vs {t2:<20} : mean={m:.3f} over {n} layers")
            if args.save_overall_csv:
                out_csv = os.path.join(args.output_dir, f"overall_pairwise_similarity_{args.model}.csv")
                save_overall_similarity_csv(overall_rows, out_csv=out_csv)

        if args.pca:
            # PCA PC1 variance per layer and cosine-to-PC1
            pc1_by_layer = compute_pc1_explained_variance_by_layer(by_layer, normalize=True)
            pc1_path = None if args.show else os.path.join(args.output_dir, f"pc1_explained_variance_{args.model}.png")
            plot_pc1_variance(pc1_by_layer, model=args.model, save_path=pc1_path)

            # Also get PC1 vectors to optionally save
            _, cos_to_pc1 = compute_pc1_and_cosines_by_layer(by_layer, normalize=True)
            if args.save_pc1_vectors:
                # Re-run PCA to extract actual PC1 vectors (not just cosines)
                pc1_vecs = {}
                for layer_idx in sorted(by_layer.keys()):
                    tv = by_layer[layer_idx]
                    if len(tv) < 2:
                        continue
                    X = np.stack([v for _, v in sorted(tv.items())])
                    norms = np.linalg.norm(X, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    Xn = X / norms
                    try:
                        pca = PCA(n_components=1)
                        pca.fit(Xn)
                        pc1_vec = pca.components_[0]
                    except Exception:
                        pc1_vec = np.zeros((Xn.shape[1],), dtype=Xn.dtype)
                    pc1_vecs[f"pc1__layer_{layer_idx}"] = pc1_vec.astype(np.float32)
                vectors_dir = os.path.join(os.path.dirname(args.output_dir), "vectors")
                os.makedirs(vectors_dir, exist_ok=True)
                out_npz = os.path.join(vectors_dir, f"pc1_vectors_{args.model}.npz")
                np.savez(out_npz, **pc1_vecs)
                print(f"Saved PC1 vectors to {out_npz}")
            cos_pc1_path = None if args.show else os.path.join(args.output_dir, f"cosine_to_pc1_{args.model}.png")
            plot_mean_cosine_to_pc1(cos_to_pc1, model=args.model, save_path=cos_pc1_path)
            if args.save_cos_to_pc1_csv:
                out_csv = os.path.join(args.output_dir, f"cosine_to_pc1_{args.model}.csv")
                save_cosine_to_pc1_csv(cos_to_pc1, out_csv=out_csv)

        # Per-task adjacent-layer stability
        adj = compute_adjacent_layer_cosines(by_layer, normalize=True)
        stab_path = None if args.show else os.path.join(args.output_dir, f"task_stability_adjacent_{args.model}.png")
        plot_task_stability_adjacent(adj, model=args.model, save_path=stab_path)
        stab_csv = os.path.join(args.output_dir, f"task_stability_adjacent_{args.model}.csv")
        save_task_stability_csv(adj, out_csv=stab_csv)
        
        # Print summary for last layer
        if layers:
            last_layer = layers[-1]
            task_vectors = by_layer[last_layer]
            similarity_matrix, task_names = calculate_cosine_similarity_matrix(task_vectors)
            print_similarity_summary(similarity_matrix, task_names, last_layer)
            
    else:
        print("Please specify either --layer <N> or --all-layers")


if __name__ == "__main__":
    main()