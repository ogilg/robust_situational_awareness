Prospective title: Is situational awareness robust in activation space?

Overview
This repo investigates whether “situational awareness” (SA) is robust and convergent in activation space across tasks and contexts, and whether steering with SA directions causally alters behavior. We benchmark SA-related tasks, extract first-token residuals with TransformerLens, aggregate vectors, and evaluate cross-task effects on Llama 3.1 models.

Setup
- Python 3.12 (venv recommended). Install requirements and ensure GPUs are available if desired.
- Models are loaded via Hugging Face; first run will download weights.
- Optional split: set SA_SPLIT=train or SA_SPLIT=test for a 50/50 hashed split with a fixed seed.

Running benchmarks
- Score-only (CSV + examples):
  python experiments/benchmark_model.py --model llama-3.1-8b-instruct --n 10 --examples 5 --out-dir experiments/results --comment "run"
- Score + vectors (aggregated .npz + examples from the TL pass):
  python experiments/benchmark_with_vectors.py --model llama-3.1-8b-instruct --n 10 --out-dir experiments/results --comment "vectors"

Notes
- Stages Oversight uses YAML batches under sad/stages/oversight/structs/batch/*.yaml (no JSON conversion needed). Train/test split is applied in-task via SA_SPLIT.
- TransformerLens generation now reuses the HF chat template string and disables auto-BOS to align prompt formatting with HF.
- Output Control prompts enforce single-word outputs and use TVD tolerance 0.30.

The fact that misalignment (easily) emerges begs the question of whether other safety-relevant properties have strong basins of attraction.
One such property is situational awareness, which is relevant to e.g. sandbagging and deceptive alignment.
My prior for situational awareness to “emerge” in a similar setting to the original EM paper is low, and very low within this short project. Instead I want to focus my analysis on:
Is SA convergent? Meaning do we extract similar vectors from different SA tasks (using contrastive pairs).
Is it robust to different contexts and interventions?
Is SA causally unified? Meaning does using steering vectors from one task translate to performance in another task.
Can it be used to significantly alter behaviour?
Here are some experiments I want to run, which I think would go some way towards supporting or refuting the claim “Is situational awareness robust in activation space?”
Extract steering vectors from the 4 most advanced tasks in the Situational Awareness dataset: Situational Awareness Dataset 

Use contrastive pairs of prompts, and methods close to the RepE paper. Just like in that paper (and also this one: https://arxiv.org/pdf/2507.01786) the contrastive pair will differ by basically one token, so we can use that token’s activations. For each layer’s residual stream we can extract the difference between the pair.
In the “refusal is mediated by a single direction” paper, they actually use the same vector on all layers. They compare a bunch of candidates on a validation set, measuring steering performance. It could be interesting to compare the two approaches (but probably won’t pro this).
Compare the extracted vectors

Look at the cosine similarity of the four vectors.
Measure robustness of extracted vectors to adding situating prompts e.g. “For research purposes, please answer:” 
Compare the vectors extracted across different layers. Is there a single direction dominating?
For each task, look at how much ablating and adding the vector changes performance.
Cross-task analysis: how much does the vector extracted from task A change performance on task B? Fill in the 4x4 table.
One v many analysis: What happens when we ramp up one vector but ablate the other 3? And the other way around.
Other investigations of the CoT: e.g. use an LLM judge to count the number of times the model explicitly thinks it is being evaluated, look at CoT traces to form other hypotheses.
Does steering significantly alter behaviour on some more general prompts: e.g. “describe what you are”.
Extension: Measuring steering performance on a general situational awareness-relevant task

The idea would be to pick an eval, e.g. a sandbagging or deceptive alignment eval, and take the same kinds of measurements as above.
Maybe we can show that ablating some SA directions leads to e.g. less sandbagging?
 
 
Research decision 1: Contrastive prompts will come from when the model picks one option over the other, rather than forcing both outputs. I chose this because it is a more established practice, and it makes sense to isolate cases where the model actually goes for an option over another. This kind of representation should already be present before we know the last token.
Research decision 2: TransformerLens doesn’t have GPT-OSS so I’ll pivot to Llama 3 models because I’m not sure about fiddling with residuals.
Research decision 3: Train test split. Even thought we don’t explicitly train, the activations could potentially capture some stuff, so better to test on separate data.
 
Recent progress
- Stages Oversight data loader now aggregates all YAMLs in `sad/stages/oversight/structs/batch/` and infers correctness from `choices_right/choices_wrong`; prompts use yes/no to match data.
- Default Stages data path is hardcoded to the batch directory; experiment runners no longer pass file paths.
- Provider refactor: split into `DefaultHFProvider` (HF-only), `TransformerLensProvider` (TL hooks for residual capture and interventions), and `GPTOSSProvider` (separate module). Factory supports `prefer_transformerlens=True` where needed.
- Added TL-based activation interventions: additions (steering) and directional projection (ablation) with per-layer and per-position control.
- Self-recognition sample generation shows `tqdm` progress; completions cache moved under `sad/self_recognition/private_completions/`.
- Vector pipeline stabilized: `benchmark_with_vectors.py` saves per-task aggregated vectors and counts; `experiments/reduce_vectors.py` produces `weighted_vectors_{model}.npz`.
- New analysis script `experiments/extracted_vector_analysis.py` computes: per-layer task similarity heatmaps, similarity vs. layer curves, overall pairwise similarities, PC1 explained variance, and alignment of task vectors to each layer’s PC1. Optional saving of PC1 vectors.
- New simple A/B baseline task (`sad/ab_baseline`) with deterministic 50% scoring flip; integrated into vector benchmark for fast comparisons.