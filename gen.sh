#!/usr/bin/env bash
#SBATCH --job-name=gen
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=3:00:00
#SBATCH --partition=learnlab
#SBATCH --mem=256GB
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

# env
source env.sh

task=retrieve

model=google/t5-xl-lm-adapt
#data_file=data/eli5/val_astarget_selfanswer_evidence.json
#out_file=checkpoints/eli5/t53b/val_astarget_selfanswer/prompt1/t53b_evidence_evidencelen64.tsv
data_file=data/eli5/val_astarget_selfanswer_qa.json

out_root=checkpoints/eli5/t53b/val_astarget_answer/memtrans_reproduce_prefix_layerall
#out_file=${out_root}/gen_topk64_byids_skip1_nopad_afterfirst_nospace.filter100_asc.tsv
#out_file=${out_root}/gen_topk4.l23.tsv
#track_file=${out_root}/track_topk4.l23.txt
out_file=${out_root}/gen_topk4.lall_h9.tsv
#track_file=${out_root}/track_topk4.lall_h9.txt

batch_size=32
evi_len=0
gen_len=256
retrieval_topk=0
retrieval_layers="[0]"
filter_topk=0
filter_order=original

if [[ ${task} == "normal" ]]; then
    sp="Definition: Given a question, generate a descriptive answer. Question: "
    ss=" Answer:"
    e=no
    ep=""
    es=""
elif [[ ${task} == "prefix" ]]; then
    evi_len=64
    gen_len=$( expr ${evi_len} + 256 )
    sp="Definition: Given a question, generate a descriptive answer. Question: "
    ss=""
    e=decoder_prefix
    ep="Evidence: "
    es=" Answer:"
elif [[ ${task} == "retrieve" ]]; then
    retrieval_topk=4
    retrieval_layers="list(range(24))"
    filter_topk=0
    filter_order=original
    sp="Definition: Given a question, generate a descriptive answer. Question: "
    ss=""
    e=fixed
    ep="Answer:"
    es=""
else
    exit
fi

srun python generate.py \
    --model ${model} \
    --data_file ${data_file} \
    --out_file ${out_file} \
    --batch_size ${batch_size} \
    --source_prefix "${sp}" \
    --source_suffix "${ss}" \
    --use_evidence ${e} \
    --evidence_prefix "${ep}" \
    --evidence_suffix "${es}" \
    --max_evidence_len ${evi_len} \
    --max_gen_len ${gen_len} \
    --retrieval_topk ${retrieval_topk} \
    --retrieval_layers ${retrieval_layers} \
    --filter_topk ${filter_topk} \
    --filter_order ${filter_order}
