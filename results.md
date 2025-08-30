# Preliminary results (analysis_1)

This note summarizes outputs produced under `experiments/results/analysis_1/` for model `llama-3.1-8b-instruct`. The figures report relationships between task vectors extracted per layer after aggregating first-token residuals into per-class sums and reducing to weighted vectors.

Artifacts
- cosine_similarity_llama-3.1-8b-instruct_layer_{0,16,31}.png: Per-layer heatmaps of cosine similarity between tasks.
- cosine_similarity_across_layers_llama-3.1-8b-instruct.png: Cosine similarity vs. layer for all task pairs.
- overall_pairwise_similarity_llama-3.1-8b-instruct.csv: Mean cosine across layers per task pair.
- pc1_explained_variance_llama-3.1-8b-instruct.png: Explained variance of PC1 for task vectors per layer.
- cosine_to_pc1_llama-3.1-8b-instruct.{png,csv}: Mean |cos(task vector, layer PC1)| vs. layer and per-layer table.

Observations (descriptive)
- Per-layer heatmaps (layers 0, mid, last):
  - Show a consistent block of positive similarity for several task pairs across the displayed layers.
  - Later layers typically exhibit stronger clustering structure than the earliest layer, where similarities are lower and more diffuse.
- Similarity vs. layer curves:
  - For most pairs, cosine similarity tends to increase from early to mid layers, with saturation or modest variation in later layers.
  - Some task pairs remain moderately similar across the depth, with limited crossings between pairs.
- Overall pairwise similarity (CSV):
  - Mean cosine values reflect the trend from the curves; pairs with higher late-layer similarity score higher overall.
- PC1 explained variance per layer:
  - PC1 captures a nontrivial fraction of variance at many layers, with a tendency to increase toward mid/late layers (exact magnitudes visible in the figure).
- Alignment to layer PC1 (mean |cos|):
  - Average absolute cosine of task vectors to each layer’s PC1 rises through the stack in tandem with PC1 variance, indicating a stronger common direction at certain depths.

Notes
- These are descriptive summaries; we intentionally avoid causal interpretations here.
- The analysis used weighted per-layer task vectors (sum_correct − sum_incorrect), computed from aggregated first-token residuals.
- Task set includes Stages Oversight and others integrated in the vector benchmark at the time of analysis. Refer to the analysis_{run}/overall_pairwise_similarity CSV for exact task names included.
