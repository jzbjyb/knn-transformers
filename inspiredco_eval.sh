#!/usr/bin/env bash

source openai_keys.sh

input=$1

eval "$(conda shell.bash hook)"
conda deactivate
conda activate knn

INSPIREDCO_API_KEY=${inspiredco_key} python inspiredco_eval.py --input ${input}
