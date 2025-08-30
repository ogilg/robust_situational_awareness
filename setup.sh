#!/bin/bash


bash sad/unzip.sh --all-dirs

pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .

git config --global user.name "Oscar Gilg"
git config --global user.email "oscargilg18@gmail.com"

export HF_TOKEN="ADD_TOKEN"

# Remember to update provider_wrapper package path in pyproject.toml to something like: /workspace/robust_situational_awareness/provider_wrapper


