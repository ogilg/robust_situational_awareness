# Robust Situational Awareness

This repository studies whether "situational awareness" (SA) has a robust, convergent representation in the activation space of Qwen-2.5-14b-instruct and whether steering along SA directions causally alters behavior.

For a comprehensive research report, see the [project write-up](https://docs.google.com/document/d/1-sZtqzyDRNVt5FkRHGkbPrv4q8l1OVr1dtbCD5gS6Jw/edit?pli=1&tab=t.0#heading=h.mjtns0hcjgju).

## Overview

We investigate situational awareness in LLMs through:
- Benchmarking SA-related tasks across datasets
- Extracting per-layer residual directions using TransformerLens
- Analyzing cross-task similarity and vector alignment
- Running causal interventions (steering/ablation) to test behavioral changes

**Models**: Llama 3.1 variants (historical) and Qwen2.5 Instruct variants (current)

## Setup

**Requirements**: Python 3.12, GPUs recommended

### Quick Setup
```bash
bash setup.sh
```

### Manual Setup
```bash
# Unzip datasets
bash sad/unzip.sh --all-dirs

# Create virtual environment
pip install uv
uv venv --python 3.12
source .venv/bin/activate

# Install package
uv pip install -e .

# Authenticate with Hugging Face
huggingface-cli login --token <YOUR_HF_TOKEN>
# OR
export HF_TOKEN=<YOUR_HF_TOKEN>
```

**Optional**: Set `SA_SPLIT=train` or `SA_SPLIT=test` for fixed 50/50 data split

## Usage

### Run Benchmarks

**Score-only benchmarks:**
```bash
python experiments/benchmark_model.py \
  --model llama-3.1-8b-instruct \
  --n 10 --examples 5 \
  --out-dir experiments/results \
  --comment "run"
```

**Extract vectors + benchmarks:**
```bash
python experiments/benchmark_with_vectors.py \
  --model llama-3.1-8b-instruct \
  --n 10 \
  --out-dir experiments/results \
  --comment "vectors"
```

### Vector Analysis

**Aggregate vectors:**
```bash
python experiments/reduce_vectors.py \
  --model llama-3.1-8b-instruct \
  --in-dir experiments/results \
  --out experiments/results/weighted_vectors_llama-3.1-8b-instruct.npz
```

**Analyze vectors:**
```bash
python experiments/extracted_vector_analysis.py \
  --vectors experiments/results/weighted_vectors_llama-3.1-8b-instruct.npz \
  --out-dir experiments/analysis
```

### Steering Experiments

**Parallel sweep:**
```bash
bash scripts/steering_sweep_parallel.sh
```

**Manual steering:**
```bash
SA_SPLIT=test python -m experiments.benchmark_with_steering \
  --model qwen-2.5-14b-instruct \
  --n 100 --examples 0 \
  --steering-mode add --vector-source id_leverage \
  --variant plain --vector-variant sp \
  --coefficient 0.01 --layers 25 \
  --out-dir experiments/results/steering_sweep \
  --comment "vv=sp coef=0.01 layers=25"
```

## Data Structure

- **Stages Oversight**: YAML batches in `sad/stages/oversight/structs/batch/*.yaml`
- **Self-recognition**: Samples in `sad/self_recognition/private_completions/`
- **A/B Baseline**: Quick checks in `sad/ab_baseline`

## Results

- **Output**: `experiments/results/` (CSVs, .npz vectors, sweep outputs)
- **Logs**: `experiments/logs/` (detailed run logs)

