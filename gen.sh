#!/usr/bin/env bash
#SBATCH --job-name=gen
#SBATCH --time=3:00:00
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=512GB

# env
source env.sh

dataset=wow
task=retrieve+targetprefix

#model=google/t5-xl-lm-adapt
model=checkpoints/models/t53b_wow_alpha4_hard_layer12_head4_ctx32_bm25_sepcrossattn

dstore_dir=checkpoints/test
dstore_size=0

if [[ ${task} == "normal" || ${task} == "evidence" || ${task} == "targetprefix" || ${task} == "evidence+targetprefix" ]]; then
    # === w/o memtrans exp ===
    if [[ ${dataset} == 'eli5' ]]; then
        #data_file=data/eli5/val_astarget_selfprov_evidence.json
        data_file=data/eli5/val_astarget_selfanswer_evidence.json
        #out_root=checkpoints/eli5/prefix_exp/val_astarget_selfprov/prompt1
        out_root=checkpoints/eli5/prefix_exp/val_astarget_selfanswer/prompt1
        #out_file=${out_root}/t53b_targetprefix16.tsv
        out_file=${out_root}/t53b_evidencelen64_targetprefix16.tsv
    elif [[ ${dataset} == 'wow' ]]; then
        data_file=data/wow/val_astarget_selfprov_evidence.json
        out_root=checkpoints/wow/prefix_exp/val_astarget_selfprov/prompt1
        out_file=${out_root}/t53b_wow_alpha4_hard_layer12_head4_ctx32.tsv
    elif [[ ${dataset} == 'test' ]]; then
        data_file=data/test/test.json
        out_root=checkpoints/test
        out_file=${out_root}/test.tsv
    else
        echo "${dataset} is not supported"
        exit
    fi
elif [[ ${task} == "retrieve" || ${task} == "retrieve+targetprefix" ]]; then
    # === w/ memtrans exp ===
    if [[ ${dataset} == 'eli5' ]]; then
        dstore_size=206896
        data_file=data/eli5/val_astarget_selfanswer_qa.json
        dstore_dir=checkpoints/eli5/prefix_exp/val_astarget_answer/memtrans_reproduce_prefix_layerall
        #out_file=${dstore_dir}/gen_topk64_byids_skip1_nopad_afterfirst_nospace.filter100_asc.tsv
        #out_file=${dstore_dir}/gen_topk4.l23.tsv
        #track_file=${dstore_dir}/track_topk4.l23.txt
        #out_file=${dstore_dir}/gen_topk4.lall_h9.tsv
        #track_file=${dstore_dir}/track_topk4.lall_h9.txt
        #out_file=${dstore_dir}/gen_topk64_byids_skip1_nopad_afterfirst_nospace.cache.tsv
        #out_file=${dstore_dir}/gen_topk64_byids_skip8_accum8_targetprefix16.tsv
        out_file=${dstore_dir}/gen_evi64_tgt16_skip1_every8_max1_head30.tsv
    elif [[ ${dataset} == 'wow' ]]; then
        dstore_size=38131
        data_file=data/wow/val_astarget_selfprov_qa.json
        #dstore_dir=checkpoints/wow/prefix_exp/val_astarget_selfprov/memtrans_reproduce_prefix_layerall
        #out_file=${dstore_dir}/gen_evi32_tgt16_byids_skip4.tsv
        #out_file=${dstore_dir}/gen_evi32_tgt16_skip1_every12_max1_head9.tsv
        #dstore_dir=checkpoints/wow/prefix_exp/val_astarget_selfprov/t53b_wow_alpha4_hard_neg20
        #out_file=${dstore_dir}/gen_t53b_wow_alpha4_hard_neg20_evi32_tgt16_skip1_every8_max1_head9_fornext.reindex.tsv
        #dstore_dir=checkpoints/wow/prefix_exp/val_astarget_selfprov/t53b_wow_alpha4_hard_layer12_head4
        #out_file=${dstore_dir}/gen_t53b_wow_alpha4_hard_layer12_head4_evi32_tgt16_skip1_every8_max1_head9_fornext.tsv
        #dstore_dir=checkpoints/wow/prefix_exp/val_astarget_selfprov/t53b_wow_alpha4_hard_layer12_head4_ctx32
        #out_file=${dstore_dir}/gen_t53b_wow_alpha4_hard_layer12_head4_ctx32_evi32_tgt16_skip1_every8_max1_head9_fornext.tsv
        dstore_dir=checkpoints/wow/prefix_exp/val_astarget_selfprov/t53b_wow_alpha4_hard_layer12_head4_ctx32_bm25_sepcrossattn
        #dstore_dir=checkpoints/wow/prefix_exp/val_astarget_selfprov/t53b
        out_file=${dstore_dir}/test.tsv
    elif [[ ${dataset} == 'wow_train_5k' ]]; then
        dstore_size=176143
        data_file=data/wow/train_astarget_selfprov_qa.5000.json
        #dstore_dir=checkpoints/wow/prefix_exp/train5k_astarget_selfprov/t53b_wow_alpha4_hard_neg20
        #out_file=${dstore_dir}/gen_t53b_wow_alpha4_hard_neg20_evi32_tgt16_skip1_every8_max1_head9.reindex.tsv
        dstore_dir=checkpoints/wow/prefix_exp/train5k_astarget_selfprov/t53b_wow_alpha4_hard_layer12_head4
        #out_file=${dstore_dir}/gen_t53b_wow_alpha4_hard_layer12_head4_evi32_tgt16_skip1_every8_max1_head9_fornext.tsv
        out_file=${dstore_dir}/test.tsv
    else
        echo "${dataset} is not supported"
        exit
    fi
elif [[ ${task} == "save" ]]; then
    # === save datastore for memtrans exp ===
    if [[ ${dataset} == 'eli5' ]]; then
        data_file=data/eli5/val_astarget_answer_qa.json
        dstore_dir=checkpoints/eli5/prefix_exp/val_astarget_answer/memtrans_reproduce_prefix_layerall
        out_file=""
    elif [[ ${dataset} == 'wow' ]]; then
        data_file=data/wow/val_astarget_selfprov_evidence.json
        dstore_dir=checkpoints/wow/prefix_exp/val_astarget_selfprov/t53b_wow_alpha4_hard_layer12_head4_ctx32
        out_file=""
    elif [[ ${dataset} == 'wow_train_5k' ]]; then
        data_file=data/wow/train_astarget_selfprov_evidence.5000.json
        dstore_dir=checkpoints/wow/prefix_exp/train5k_astarget_selfprov/t53b_wow_alpha4_hard_layer12_head4
        out_file=""
    else
        echo "${dataset} is not supported"
        exit
    fi
elif [[ ${task} == "save_same_encoder_input" ]]; then
    # === save datastore (no duplication) for memtrans exp ===
    if [[ ${dataset} == 'wow' ]]; then
        data_file=data/wow/val_astarget_selfprov_evidence.dedup.json
        #dstore_dir=checkpoints/wow/prefix_exp/val_astarget_selfprov/t53b_wow_alpha4_hard_layer12_head4_ctx32_bm25_sepcrossattn
        dstore_dir=checkpoints/wow/prefix_exp/val_astarget_selfprov/t53b
        out_file=""
    else
        echo "${dataset} is not supported"
        exit
    fi
else
    echo "${task} is not defined"
    exit
fi

batch_size=32
evi_len=0
gen_len=256
targetprefix_len=0
stage='retrieve'
retrieval_topk=0
retrieval_layers="[]"
skip_retrieval_steps=0
accum_retrieval_steps=0
retrieval_for_next_step_at_layer=-1
retrieval_every_steps=1
max_retrieval_times=100000
filter_topk=0
filter_order=original
only_use_head_idx=-1
num_ctxs=1
ctx_order=parallel
evidence_encoder_input=""

if [[ ${task} == "normal" ]]; then
    src_pre="Definition: Given a question, generate a descriptive answer. Question: "
    src_suf=""
    evi=fixed
    evi_pre=""
    evi_suf="Answer:"
elif [[ ${task} == "evidence" ]]; then
    evi_len=32
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
    evi_len=32
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
elif [[ ${task} == "save" ]]; then
    stage="save"
    retrieval_layers="list(range(24))"
    src_pre="Definition: Given a question, generate a descriptive answer. Question: "
    src_suf=""
    evi=fixed
    evi_pre=""
    evi_suf="Evidence:"
elif [[ ${task} == "save_same_encoder_input" ]]; then
    stage="save"
    retrieval_layers="list(range(24))"
    src_pre=""
    src_suf=""
    evi=fixed
    evi_pre=""
    evi_suf="Evidence:"
    evidence_encoder_input="Definition: Given a question, generate a descriptive answer."
elif [[ ${task} == "retrieve+targetprefix" ]]; then
    targetprefix_len=16
    retrieval_topk=32
    retrieval_layers="list(range(24))"
    skip_retrieval_steps=1
    accum_retrieval_steps=0
    retrieval_for_next_step_at_layer=12
    retrieval_every_steps=8
    max_retrieval_times=1
    only_use_head_idx=4
    filter_topk=0
    filter_order=original
    num_ctxs=1
    ctx_order=parallel
    src_pre="Definition: Given a question, generate a descriptive answer. Question: "
    src_suf=""
    evi=fixed
    evi_pre=""
    evi_suf="Answer:"
else
    exit
fi

srun python generate.py \
    --model ${model} \
    --stage ${stage} \
    --dstore_dir ${dstore_dir} \
    --dstore_size ${dstore_size} \
    --data_file ${data_file} \
    --out_file "${out_file}" \
    --batch_size ${batch_size} \
    --source_prefix "${src_pre}" \
    --source_suffix "${src_suf}" \
    --use_evidence ${evi} \
    --evidence_prefix "${evi_pre}" \
    --evidence_suffix "${evi_suf}" \
    --evidence_encoder_input "${evidence_encoder_input}" \
    --max_evidence_len ${evi_len} \
    --max_gen_len ${gen_len} \
    --target_as_prefix_len ${targetprefix_len} \
    --retrieval_topk ${retrieval_topk} \
    --retrieval_layers ${retrieval_layers} \
    --skip_retrieval_steps ${skip_retrieval_steps} \
    --accum_retrieval_steps ${accum_retrieval_steps} \
    --retrieval_for_next_step_at_layer ${retrieval_for_next_step_at_layer} \
    --retrieval_every_steps ${retrieval_every_steps} \
    --max_retrieval_times ${max_retrieval_times} \
    --filter_topk ${filter_topk} \
    --filter_order ${filter_order} \
    --only_use_head_idx ${only_use_head_idx} \
    --num_ctxs ${num_ctxs} \
    --ctx_order ${ctx_order}
