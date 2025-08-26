#!/bin/bash

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .

git config --global user.name "Oscar Gilg"
git config --global user.email "oscargilg18@gmail.com"

export HF_TOKEN="ADD_TOKEN"



