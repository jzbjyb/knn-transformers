#!/usr/bin/env bash
#SBATCH --job-name=gen
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=512GB

# env
source env.sh

task=retrieve+targetprefix

model=google/t5-xl-lm-adapt

if [[ ${task} == "normal" || ${task} == "evidence" || ${task} == "targetprefix" || ${task} == "evidence+targetprefix" ]]; then
    # === w/o memtrans exp ===
    data_file=data/eli5/val_astarget_selfanswer_evidence.json
    out_root=checkpoints/eli5/prefix_exp/val_astarget_selfanswer/prompt1
    #out_file=${out_root}/t53b_targetprefix16.tsv
    out_file=${out_root}/t53b_evidencelen64_targetprefix16.tsv
elif [[ ${task} == "retrieve" || ${task} == "retrieve+targetprefix" ]]; then
    # === w/ memtrans exp ===
    data_file=data/eli5/val_astarget_selfanswer_qa.json
    out_root=checkpoints/eli5/prefix_exp/val_astarget_answer/memtrans_reproduce_prefix_layerall
    #out_file=${out_root}/gen_topk64_byids_skip1_nopad_afterfirst_nospace.filter100_asc.tsv
    #out_file=${out_root}/gen_topk4.l23.tsv
    #track_file=${out_root}/track_topk4.l23.txt
    #out_file=${out_root}/gen_topk4.lall_h9.tsv
    #track_file=${out_root}/track_topk4.lall_h9.txt
    #out_file=${out_root}/gen_topk64_byids_skip1_nopad_afterfirst_nospace.cache.tsv
    out_file=${out_root}/gen_topk64_byids_skip8_accum8_targetprefix16.tsv
else
    echo "${task} is not defined"
    exit
fi

batch_size=32
evi_len=0
gen_len=32
targetprefix_len=0
retrieval_topk=0
retrieval_layers="[]"
skip_retrieval_steps=0
accum_retrieval_steps=0
filter_topk=0
filter_order=original

if [[ ${task} == "normal" ]]; then
    src_pre="Definition: Given a question, generate a descriptive answer. Question: "
    src_suf=" Answer:"
    evi=no
    evi_pre=""
    evi_suf=""
elif [[ ${task} == "evidence" ]]; then
    evi_len=64
    gen_len=$( expr ${evi_len} + ${gen_len} )
    src_pre="Definition: Given a question, generate a descriptive answer. Question: "
    src_suf=""
    evi=decoder_prefix
    evi_pre="Evidence: "
    evi_suf=" Answer:"
elif [[ ${task} == "targetprefix" ]]; then
    targetprefix_len=16
    src_pre="Definition: Given a question, generate a descriptive answer. Question: "
    src_suf=""
    evi=fixed
    evi_pre=""
    evi_suf="Answer:"
elif [[ ${task} == "evidence+targetprefix" ]]; then
    evi_len=64
    gen_len=$( expr ${evi_len} + ${gen_len} )
    targetprefix_len=16
    src_pre="Definition: Given a question, generate a descriptive answer. Question: "
    src_suf=""
    evi=decoder_prefix
    evi_pre="Evidence: "
    evi_suf=" Answer:"
elif [[ ${task} == "retrieve" ]]; then
    retrieval_topk=64
    retrieval_layers="list(range(24))"
    filter_topk=0
    filter_order=original
    src_pre="Definition: Given a question, generate a descriptive answer. Question: "
    src_suf=""
    evi=fixed
    evi_pre=""
    evi_suf="Answer:"
elif [[ ${task} == "retrieve+targetprefix" ]]; then
    targetprefix_len=16
    retrieval_topk=64
    retrieval_layers="[0]"
    skip_retrieval_steps=8
    accum_retrieval_steps=8
    filter_topk=0
    filter_order=original
    src_pre="Definition: Given a question, generate a descriptive answer. Question: "
    src_suf=""
    evi=fixed
    evi_pre=""
    evi_suf="Answer:"
else
    exit
fi

python generate.py \
    --model ${model} \
    --data_file ${data_file} \
    --out_file ${out_file} \
    --batch_size ${batch_size} \
    --source_prefix "${src_pre}" \
    --source_suffix "${src_suf}" \
    --use_evidence ${evi} \
    --evidence_prefix "${evi_pre}" \
    --evidence_suffix "${evi_suf}" \
    --max_evidence_len ${evi_len} \
    --max_gen_len ${gen_len} \
    --target_as_prefix_len ${targetprefix_len} \
    --retrieval_topk ${retrieval_topk} \
    --retrieval_layers ${retrieval_layers} \
    --skip_retrieval_steps ${skip_retrieval_steps} \
    --accum_retrieval_steps ${accum_retrieval_steps} \
    --filter_topk ${filter_topk} \
    --filter_order ${filter_order}
