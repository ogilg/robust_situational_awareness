Link to project write-up (situational awareness focus): https://docs.google.com/document/d/1-sZtqzyDRNVt5FkRHGkbPrv4q8l1OVr1dtbCD5gS6Jw/edit?pli=1&tab=t.0#heading=h.mjtns0hcjgju

Overview
This repository studies whether “situational awareness” (SA) has a robust, convergent representation in the activation space of modern LLMs and whether steering along SA directions causally alters behavior. We:
- Benchmark SA-related tasks across datasets
- Extract per-layer residual directions (first-token focus) using TransformerLens
- Aggregate and reduce task vectors, then analyze cross-task similarity and alignment
- Run causal interventions (add/projection) to test steering and ablation effects

Primary models: Llama 3.1 variants (historical runs) and Qwen2.5 Instruct variants for current sweeps. See citations and motivation in the project write-up above.

Quick start
1) Clone and unzip datasets
2) Create a Python 3.12 virtual environment
3) Install the package in editable mode
4) Configure Hugging Face access
5) Run benchmarks or steering sweeps

Setup
- Python 3.12 required. GPUs recommended for vector extraction and steering.
- Models load from Hugging Face; first use downloads weights.
- Optional split: set `SA_SPLIT=train` or `SA_SPLIT=test` for a fixed 50/50 hashed split.

Option A: one-shot setup script
  bash setup.sh

Notes for setup.sh:
- It unzips datasets via `sad/unzip.sh --all-dirs`
- It creates a `.venv` with `uv` and installs this repo with `uv pip install -e .`
- Replace any personal `git config` or tokens in your environment rather than adopting placeholders from the script

Option B: manual setup
  bash sad/unzip.sh --all-dirs
  pip install uv
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install -e .
  # Authenticate with Hugging Face (choose one)
  # 1) CLI login (recommended):
  huggingface-cli login --token <YOUR_HF_TOKEN>
  # 2) Or export a token in your shell (non-interactive):
  export HF_TOKEN=<YOUR_HF_TOKEN>

Running benchmarks
- Score-only (CSV + sampled examples):
  python experiments/benchmark_model.py \
    --model llama-3.1-8b-instruct \
    --n 10 --examples 5 \
    --out-dir experiments/results \
    --comment "run"

- Score + vectors (saves aggregated .npz and examples from the TransformerLens pass):
  python experiments/benchmark_with_vectors.py \
    --model llama-3.1-8b-instruct \
    --n 10 \
    --out-dir experiments/results \
    --comment "vectors"

Vector aggregation and analysis
- Reduce/weight vectors across tasks into a single file:
  python experiments/reduce_vectors.py --model llama-3.1-8b-instruct --in-dir experiments/results --out experiments/results/weighted_vectors_llama-3.1-8b-instruct.npz

- Analyze extracted vectors (similarity heatmaps, layer curves, PC1 alignment):
  python experiments/extracted_vector_analysis.py --vectors experiments/results/weighted_vectors_llama-3.1-8b-instruct.npz --out-dir experiments/analysis

Steering experiments
- Single-machine, multi-GPU parallel sweep helper exists under `scripts/steering_sweep_parallel.sh`.
- Default sweep targets Qwen2.5 Instruct and varies vector variant, coefficient, and layers.

To launch an example parallel sweep (edit variables inside the script as needed):
  bash scripts/steering_sweep_parallel.sh

The script internally runs a module like:
  SA_SPLIT=test python -m experiments.benchmark_with_steering \
    --model qwen-2.5-14b-instruct \
    --n 100 --examples 0 \
    --steering-mode add --vector-source id_leverage \
    --variant plain --vector-variant sp \
    --coefficient 0.01 --layers 25 \
    --out-dir experiments/results/steering_sweep \
    --comment "vv=sp coef=0.01 layers=25"

Data and tasks
- Stages Oversight YAML batches live under `sad/stages/oversight/structs/batch/*.yaml`. Train/test split is applied via `SA_SPLIT` during task execution.
- Self-recognition samples and caches are under `sad/self_recognition/private_completions/`.
- A simple A/B baseline task exists in `sad/ab_baseline` for quick checks.

Implementation notes
- HF chat templates are reused inside TransformerLens runs; auto-BOS is disabled to keep prompts aligned with HF.
- Output Control prompts enforce single-token outputs with a TVD tolerance of ~0.30.
- Providers are split into `DefaultHFProvider`, `TransformerLensProvider` (residual hooks and interventions), and `GPTOSSProvider`. A factory flag enables preferring TransformerLens where needed.
- Interventions supported: additions (steering) and directional projection (ablation) with per-layer/per-position control.

Results layout (default)
- `experiments/results/` stores CSVs, vector artifacts (`.npz`), and sweep outputs
- `experiments/logs/` stores logs for sweeps and long runs

Citation and further context
- Project motivation, hypotheses, and design choices are summarized in the write-up: `https://docs.google.com/document/d/1-sZtqzyDRNVt5FkRHGkbPrv4q8l1OVr1dtbCD5gS6Jw/edit?pli=1&tab=t.0#heading=h.mjtns0hcjgju`.

Troubleshooting
- Missing weights: ensure you are logged into Hugging Face or `HF_TOKEN` is set
- CUDA errors: verify GPU visibility (`nvidia-smi`) and CUDA/PyTorch compatibility
- Empty results: check `SA_SPLIT` and dataset unzip step, then rerun with `--examples 5` for spot checks