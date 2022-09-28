#!/usr/bin/env bash
#SBATCH --job-name=gen
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=3:00:00
#SBATCH --partition=learnlab
#SBATCH --mem=256GB
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

# env
source env.sh

task=normal

model=google/t5-xl-lm-adapt
data_file=data/eli5/val_astarget_selfanswer_evidence.json
out_file=test.tsv

if [[ ${task} == "normal" ]]; then
    sp="Definition: Given a question, generate a descriptive answer. Question: "
    ss=" Answer:"
    e=no
    ep=""
    es=""

elif [[ ${task} == "prefix" ]]; then
    sp="Definition: Given a question, generate a descriptive answer. Question: "
    ss=""
    e=decoder_prefix
    ep="Evidence: "
    es=" Answer:"
else
    exit
fi

python generate.py \
    --model ${model} \
    --data_file ${data_file} \
    --out_file ${out_file} \
    --batch_size 32 \
    --source_prefix "${sp}" \
    --source_suffix "${ss}" \
    --use_evidence ${e} \
    --evidence_prefix "${ep}" \
    --evidence_suffix "${es}"
