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


def load_weighted_vectors(model: str, vectors_dir: str, *, variant: str = "plain") -> Dict[str, np.ndarray]:
    """Load weighted vectors for a specific variant from the npz file.

    Expects files named weighted_vectors_{model}__{variant}.npz
    """
    weighted_path = os.path.join(vectors_dir, f"weighted_vectors_{model}__{variant}.npz")
    if os.path.exists(weighted_path):
        data = np.load(weighted_path)
        return {key: data[key] for key in data.files}

    # Fallback: if separate variant file is missing but a combined 'both' file exists,
    # filter keys by variant from that file (keys include task__{variant}__weighted__layer_k)
    both_path = os.path.join(vectors_dir, f"weighted_vectors_{model}__both.npz")
    if os.path.exists(both_path) and variant in {"plain", "sp"}:
        data = np.load(both_path)
        out: Dict[str, np.ndarray] = {}
        for key in data.files:
            try:
                task_with_variant, _ = parse_vector_key(key)
            except ValueError:
                continue
            _, v = strip_variant_suffix(task_with_variant)
            if v == variant:
                out[key] = data[key]
        if out:
            return out
    raise FileNotFoundError(f"Weighted vectors file not found: {weighted_path}")


def parse_vector_key(key: str) -> Tuple[str, int]:
    """Parse vector key to extract task name and layer index.

    Robust to an optional variant segment in the task part, e.g.:
      'stages__plain__weighted__layer_15' or 'stages__weighted__layer_15'
    Returns task including variant portion; caller may strip variant.
    """
    if "__weighted__layer_" not in key:
        raise ValueError(f"Cannot parse vector key: {key}")
    task_part, layer_part = key.split("__weighted__layer_")
    try:
        layer_idx = int(layer_part)
    except Exception:
        raise ValueError(f"Cannot parse vector key: {key}")
    return task_part, layer_idx


def strip_variant_suffix(task_with_variant: str) -> Tuple[str, str]:
    """Return (task_name, variant) where variant is '' if absent.

    Examples:
      'stages__plain' -> ('stages', 'plain')
      'output_control__sp' -> ('output_control', 'sp')
      'id_leverage_generic' -> ('id_leverage_generic', '')
    """
    parts = task_with_variant.split("__")
    if len(parts) >= 2 and parts[-1] in {"plain", "sp"}:
        return "__".join(parts[:-1]), parts[-1]
    return task_with_variant, ""


def group_vectors_by_layer(vectors: Dict[str, np.ndarray], *, expect_variant: str | None = None) -> Dict[int, Dict[str, np.ndarray]]:
    """Group vectors by layer index, optionally filtering to a specific variant.

    Task keys in the returned mapping are variant-stripped (just the task name).
    """
    by_layer: Dict[int, Dict[str, np.ndarray]] = {}

    for key, vector in vectors.items():
        try:
            task_with_variant, layer_idx = parse_vector_key(key)
            task_name, variant = strip_variant_suffix(task_with_variant)
            if expect_variant is not None and variant != expect_variant:
                # Skip keys from other variants
                continue
            if layer_idx not in by_layer:
                by_layer[layer_idx] = {}
            by_layer[layer_idx][task_name] = vector
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


def plot_plain_vs_sp_final_layer(vectors_plain: Dict[str, np.ndarray], vectors_sp: Dict[str, np.ndarray], *, model: str, output_dir: str, show: bool = False) -> None:
    """Compute cosine similarity between plain and sp vectors at the final layer per task and plot a bar chart.

    Skips tasks missing in either variant.
    """
    # Build per-layer grouped maps (variant-stripped)
    by_layer_plain = group_vectors_by_layer(vectors_plain)
    by_layer_sp = group_vectors_by_layer(vectors_sp)

    if not by_layer_plain or not by_layer_sp:
        print("Not enough vectors to compare plain vs sp.")
        return

    final_layer_plain = max(by_layer_plain.keys())
    final_layer_sp = max(by_layer_sp.keys())
    final_layer = min(final_layer_plain, final_layer_sp)

    tp = by_layer_plain.get(final_layer, {})
    ts = by_layer_sp.get(final_layer, {})

    tasks = sorted(set(tp.keys()) & set(ts.keys()))
    if not tasks:
        print("No overlapping tasks at final layer for plain vs sp comparison.")
        return

    vals = []
    for t in tasks:
        v1 = tp.get(t)
        v2 = ts.get(t)
        if v1 is None or v2 is None:
            continue
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            cos = 0.0
        else:
            cos = float(np.dot(v1, v2) / (max(n1 * n2, 1e-12)))
        vals.append((t, cos))

    if not vals:
        print("No task pairs available for plain vs sp comparison at final layer.")
        return

    tasks_plot = [t for t, _ in vals]
    cosines = [c for _, c in vals]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(tasks_plot)), cosines, color="purple")
    plt.xticks(range(len(tasks_plot)), tasks_plot, rotation=30, ha='right')
    plt.ylim(-1, 1)
    plt.ylabel('cosine(plain, sp)')
    plt.title(f'Plain vs SP Cosine at Final Layer (L={final_layer})\nModel: {model}')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"plain_vs_sp_final_layer_{model}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plain vs sp comparison: {out_path}")
    if show:
        plt.show()
    plt.close()


def plot_plain_vs_sp_avg_across_layers(vectors_plain: Dict[str, np.ndarray], vectors_sp: Dict[str, np.ndarray], *, model: str, output_dir: str, show: bool = False) -> None:
    """Compute average cosine similarity between plain and sp across all overlapping layers per task.

    Produces a bar chart and prints a small table. Skips tasks missing in either variant.
    """
    by_layer_plain = group_vectors_by_layer(vectors_plain)
    by_layer_sp = group_vectors_by_layer(vectors_sp)

    # Collect all layers present in both
    common_layers = sorted(set(by_layer_plain.keys()) & set(by_layer_sp.keys()))
    if not common_layers:
        print("No overlapping layers across variants; skipping avg plain vs sp plot.")
        return

    # Collect per-task cosines across layers
    task_to_cosines: Dict[str, list] = {}
    for layer_idx in common_layers:
        tp = by_layer_plain.get(layer_idx, {})
        ts = by_layer_sp.get(layer_idx, {})
        for t in sorted(set(tp.keys()) & set(ts.keys())):
            v1 = tp[t]; v2 = ts[t]
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                cos = 0.0
            else:
                cos = float(np.dot(v1, v2) / max(n1 * n2, 1e-12))
            task_to_cosines.setdefault(t, []).append(cos)

    if not task_to_cosines:
        print("No overlapping tasks across layers; skipping avg plain vs sp plot.")
        return

    tasks = sorted(task_to_cosines.keys())
    means = [float(np.mean(task_to_cosines[t])) for t in tasks]

    # Print summary
    print("\nAverage cosine(plain, sp) across layers by task:")
    for t, m in zip(tasks, means):
        print(f"  {t:>20} : {m:.3f}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(tasks)), means, color="teal")
    plt.xticks(range(len(tasks)), tasks, rotation=30, ha='right')
    plt.ylim(-1, 1)
    plt.ylabel('mean cosine across layers')
    plt.title(f'Plain vs SP Mean Cosine Across Layers\nModel: {model}')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"plain_vs_sp_avg_across_layers_{model}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plain vs sp mean-cosine plot: {out_path}")
    if show:
        plt.show()
    plt.close()


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


def compute_pc1_pc2_and_cosines_by_layer(by_layer: Dict[int, Dict[str, np.ndarray]], *, normalize: bool = True) -> Tuple[List[Tuple[int, float, float]], Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """
    For each layer, compute PC1 and PC2 (from PCA over task vectors) and return:
    - list of (layer_idx, pc1_variance_ratio, pc2_variance_ratio)
    - mapping layer_idx -> {task_name: cosine(task_vector, pc1_vector)}
    - mapping layer_idx -> {task_name: cosine(task_vector, pc2_vector)}
    """
    pc_var_list: List[Tuple[int, float, float]] = []
    cos_to_pc1: Dict[int, Dict[str, float]] = {}
    cos_to_pc2: Dict[int, Dict[str, float]] = {}
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
            n_components = min(2, X.shape[0], X.shape[1])
            pca = PCA(n_components=n_components)
            pca.fit(X)
            pc1_vec = pca.components_[0] if len(pca.components_) > 0 else np.zeros((X.shape[1],), dtype=X.dtype)
            pc2_vec = pca.components_[1] if len(pca.components_) > 1 else np.zeros((X.shape[1],), dtype=X.dtype)
            pc1_var = float(pca.explained_variance_ratio_[0]) if len(pca.explained_variance_ratio_) > 0 else 0.0
            pc2_var = float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else 0.0
        except Exception:
            pc1_vec = np.zeros((X.shape[1],), dtype=X.dtype)
            pc2_vec = np.zeros((X.shape[1],), dtype=X.dtype)
            pc1_var = 0.0
            pc2_var = 0.0
        
        # Cosines to PC1
        pc1_norm = np.linalg.norm(pc1_vec)
        if pc1_norm == 0:
            cosines1 = {t: 0.0 for t in tasks}
        else:
            cosines1 = {t: float(np.dot(x, pc1_vec) / max(np.linalg.norm(x) * pc1_norm, 1e-12)) for t, x in zip(tasks, X)}
        
        # Cosines to PC2
        pc2_norm = np.linalg.norm(pc2_vec)
        if pc2_norm == 0:
            cosines2 = {t: 0.0 for t in tasks}
        else:
            cosines2 = {t: float(np.dot(x, pc2_vec) / max(np.linalg.norm(x) * pc2_norm, 1e-12)) for t, x in zip(tasks, X)}
        
        pc_var_list.append((layer_idx, pc1_var, pc2_var))
        cos_to_pc1[layer_idx] = cosines1
        cos_to_pc2[layer_idx] = cosines2
    return pc_var_list, cos_to_pc1, cos_to_pc2


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


def plot_mean_cosine_to_pc2(cos_to_pc2: Dict[int, Dict[str, float]], *, model: str, save_path: Optional[str]) -> None:
    if not cos_to_pc2:
        return
    layers = sorted(cos_to_pc2.keys())
    means = []
    stds = []
    for l in layers:
        vals = list(cos_to_pc2[l].values())
        if not vals:
            means.append(0.0)
            stds.append(0.0)
        else:
            means.append(float(np.mean(np.abs(vals))))  # alignment magnitude
            stds.append(float(np.std(np.abs(vals))))
    plt.figure(figsize=(10, 5))
    means = np.asarray(means)
    stds = np.asarray(stds)
    plt.plot(layers, means, marker='o', label='mean |cos(task, PC2)|', color='C1')
    plt.fill_between(layers, means - stds, means + stds, color='C1', alpha=0.2, label='±1 std')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean |cosine(task vector, layer PC2)|')
    plt.title(f'Alignment of Task Vectors to Each Layer\'s PC2 (Mean |cos|)\nModel: {model}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved mean cosine-to-PC2 plot: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_individual_cosines_to_pc(cos_to_pc: Dict[int, Dict[str, float]], pc_name: str, *, model: str, save_path: Optional[str]) -> None:
    """Plot individual task cosines to PC across layers."""
    if not cos_to_pc:
        return
    layers = sorted(cos_to_pc.keys())
    tasks = sorted({t for layer_data in cos_to_pc.values() for t in layer_data.keys()})
    
    plt.figure(figsize=(12, 8))
    for task in tasks:
        task_cosines = []
        task_layers = []
        for layer in layers:
            if task in cos_to_pc[layer]:
                task_cosines.append(cos_to_pc[layer][task])
                task_layers.append(layer)
        if task_cosines:
            plt.plot(task_layers, task_cosines, marker='o', label=task)
    
    plt.xlabel('Layer Index')
    plt.ylabel(f'cosine(task vector, {pc_name})')
    plt.title(f'Individual Task Alignment to {pc_name} Across Layers\nModel: {model}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved individual {pc_name} cosines plot: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_pc1_pc2_variance(pc_var_list: List[Tuple[int, float, float]], *, model: str, save_path: Optional[str]) -> None:
    if not pc_var_list:
        return
    layers = [x[0] for x in pc_var_list]
    pc1_vals = [x[1] for x in pc_var_list]
    pc2_vals = [x[2] for x in pc_var_list]
    
    plt.figure(figsize=(10, 5))
    plt.plot(layers, pc1_vals, marker='o', label='PC1 explained variance')
    plt.plot(layers, pc2_vals, marker='s', label='PC2 explained variance')
    plt.xlabel('Layer Index')
    plt.ylabel('Explained variance ratio')
    plt.title(f'Explained Variance of PC1 and PC2 for Task Vectors per Layer\nModel: {model}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PC1-PC2 variance plot: {save_path}")
    else:
        plt.show()
    plt.close()


def save_cosine_to_pc2_csv(cos_to_pc2: Dict[int, Dict[str, float]], *, out_csv: str) -> None:
    import csv
    if not cos_to_pc2:
        return
    layers = sorted(cos_to_pc2.keys())
    tasks = sorted({t for l in layers for t in cos_to_pc2[l].keys()})
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["layer"] + tasks)
        for l in layers:
            row = [l] + [f"{float(cos_to_pc2[l].get(t, 0.0)):.6f}" for t in tasks]
            w.writerow(row)
    print(f"Saved cosine-to-PC2 table to {out_csv}")


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
    parser.add_argument("--variant", choices=["plain", "sp", "both"], default="plain", help="Which variant vectors to analyze for the main analysis")
    parser.add_argument("--compare-variants", action="store_true", help="Also plot cosine between plain and sp at final layer (if both exist)")
    
    args = parser.parse_args()
    
    # Set default paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if args.vectors_dir is None:
        args.vectors_dir = os.path.join(root_dir, "experiments", "vectors")
    if args.output_dir is None:
        args.output_dir = os.path.join(root_dir, "experiments", "results")
    
    # Load vectors (main analysis on selected variant; default plain)
    main_variant = "plain" if args.variant in (None, "both") else args.variant
    print(args.variant)
    print(f"Loading weighted vectors for model: {args.model}, variant: {main_variant}")
    vectors = load_weighted_vectors(args.model, args.vectors_dir, variant=main_variant)
    print(f"Loaded {len(vectors)} vectors for {main_variant}")
    
    # Group by layer
    by_layer = filter_tasks(group_vectors_by_layer(vectors, expect_variant=main_variant), args.tasks)
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
            # PCA PC1 and PC2 analysis
            pc_var_list, cos_to_pc1, cos_to_pc2 = compute_pc1_pc2_and_cosines_by_layer(by_layer, normalize=True)
            
            # Plot PC1 and PC2 explained variance
            pc_var_path = None if args.show else os.path.join(args.output_dir, f"pc1_pc2_explained_variance_{args.model}.png")
            plot_pc1_pc2_variance(pc_var_list, model=args.model, save_path=pc_var_path)
            
            # Plot mean cosine alignments
            cos_pc1_path = None if args.show else os.path.join(args.output_dir, f"cosine_to_pc1_{args.model}.png")
            plot_mean_cosine_to_pc1(cos_to_pc1, model=args.model, save_path=cos_pc1_path)
            
            cos_pc2_path = None if args.show else os.path.join(args.output_dir, f"cosine_to_pc2_{args.model}.png")
            plot_mean_cosine_to_pc2(cos_to_pc2, model=args.model, save_path=cos_pc2_path)
            
            # Plot individual task alignments to PC1 and PC2
            individual_pc1_path = None if args.show else os.path.join(args.output_dir, f"individual_cosines_to_pc1_{args.model}.png")
            plot_individual_cosines_to_pc(cos_to_pc1, "PC1", model=args.model, save_path=individual_pc1_path)
            
            individual_pc2_path = None if args.show else os.path.join(args.output_dir, f"individual_cosines_to_pc2_{args.model}.png")
            plot_individual_cosines_to_pc(cos_to_pc2, "PC2", model=args.model, save_path=individual_pc2_path)
            
            # Save CSV files
            if args.save_cos_to_pc1_csv:
                out_csv_pc1 = os.path.join(args.output_dir, f"cosine_to_pc1_{args.model}.csv")
                save_cosine_to_pc1_csv(cos_to_pc1, out_csv=out_csv_pc1)
                
                out_csv_pc2 = os.path.join(args.output_dir, f"cosine_to_pc2_{args.model}.csv")
                save_cosine_to_pc2_csv(cos_to_pc2, out_csv=out_csv_pc2)
            
            # Save PC vectors
            if args.save_pc1_vectors:
                pc1_vecs = {}
                pc2_vecs = {}
                for layer_idx in sorted(by_layer.keys()):
                    tv = by_layer[layer_idx]
                    if len(tv) < 2:
                        continue
                    X = np.stack([v for _, v in sorted(tv.items())])
                    norms = np.linalg.norm(X, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    Xn = X / norms
                    try:
                        n_components = min(2, X.shape[0], X.shape[1])
                        pca = PCA(n_components=n_components)
                        pca.fit(Xn)
                        pc1_vec = pca.components_[0] if len(pca.components_) > 0 else np.zeros((Xn.shape[1],), dtype=Xn.dtype)
                        pc2_vec = pca.components_[1] if len(pca.components_) > 1 else np.zeros((Xn.shape[1],), dtype=Xn.dtype)
                    except Exception:
                        pc1_vec = np.zeros((Xn.shape[1],), dtype=Xn.dtype)
                        pc2_vec = np.zeros((Xn.shape[1],), dtype=Xn.dtype)
                    pc1_vecs[f"pc1__layer_{layer_idx}"] = pc1_vec.astype(np.float32)
                    pc2_vecs[f"pc2__layer_{layer_idx}"] = pc2_vec.astype(np.float32)
                
                vectors_dir = os.path.join(os.path.dirname(args.output_dir), "vectors")
                os.makedirs(vectors_dir, exist_ok=True)
                
                out_npz_pc1 = os.path.join(vectors_dir, f"pc1_vectors_{args.model}__{main_variant}.npz")
                np.savez(out_npz_pc1, **pc1_vecs)
                print(f"Saved PC1 vectors to {out_npz_pc1}")
                
                out_npz_pc2 = os.path.join(vectors_dir, f"pc2_vectors_{args.model}__{main_variant}.npz")
                np.savez(out_npz_pc2, **pc2_vecs)
                print(f"Saved PC2 vectors to {out_npz_pc2}")

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

    # Optional: compare plain vs sp at final layer
    if args.compare_variants:
        try:
            vec_plain = load_weighted_vectors(args.model, args.vectors_dir, variant="plain")
            vec_sp = load_weighted_vectors(args.model, args.vectors_dir, variant="sp")
            plot_plain_vs_sp_final_layer(vec_plain, vec_sp, model=args.model, output_dir=args.output_dir, show=args.show)
            plot_plain_vs_sp_avg_across_layers(vec_plain, vec_sp, model=args.model, output_dir=args.output_dir, show=args.show)
        except FileNotFoundError:
            print("Plain or SP vectors not found; skipping variant comparison plot.")


if __name__ == "__main__":
    main()