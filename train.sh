#!/usr/bin/env bash
#SBATCH --job-name=fustiont5
#SBATCH --time=8:00:00
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=512GB

module purge
module load anaconda3
. /usr/share/modules/init/sh
eval "$(conda shell.bash hook)"
conda activate knn

export WANDB_PROJECT=unifiedrlm
export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a
export WANDB_WATCH=all

debug=false

# random docs
#train_file=data/wow/train_neg10_dpr.json
#val_file=data/wow/val_neg10_dpr.json

# bm25 docs
train_file=data/wow/train_astarget_selfprov_evidence.json.beir_dedup_ans.fid/dev.json
val_file=data/wow/val_astarget_selfprov_evidence.json.beir_dedup_ans.fid/dev.json

output_dir=checkpoints/models/t53b_wow_alpha4_hard_layer12_head4_ctx32_bm25_sepcrossattn_singlebos_aggreproduce
#output_dir=checkpoints/models/t53b_wow_ctx32_bm25_sepcrossattn_singlebos_onlyblock8

init_model=google/t5-xl-lm-adapt
depth=10
max_context_len=32
use_context=true
context_bos=true
answer_bos=true
bos_attention=single
ctx_attention_loss="block:8_layer2heads:12,[4]_loss:hard_alpha:4"

eval_steps=100
max_eval_samples=1000

if [[ ${debug} == "small" ]]; then
    init_model=google/t5-small-lm-adapt
    ctx_attention_loss="block:8_layer2heads:0,list(range(4))|2,list(range(4))_layerheadagg:softmax-mean_layerheadtau:0.001_loss:hard_alpha:4"
    output_dir=checkpoints/models/test
    rm -r ${output_dir}
    eval_steps=5
    max_eval_samples=16
    extra="--report_to none"
elif [[ ${debug} == "large" ]]; then
    output_dir=checkpoints/models/test
    rm -r ${output_dir}
    eval_steps=5
    max_eval_samples=16
    extra="--report_to none"
elif [[ ${debug} == "false" ]]; then
    extra="--report_to wandb"
else
    exit
fi

run_name="$(basename $output_dir)"
cat ./train.sh &> ${output_dir}.sh

deepspeed train.py \
    --deepspeed deepspeed/lr-decay-zero1.json \
    --model_name_or_path ${init_model} \
    --train_file ${train_file} \
    --validation_file ${val_file} \
    --output_dir ${output_dir} \
    --remove_unused_columns false \
    --depth ${depth} \
    --max_question_len 128 \
    --max_context_len ${max_context_len} \
    --max_answer_len 128 \
    --use_context ${use_context} \
    --context_bos ${context_bos} \
    --answer_bos ${answer_bos} \
    --bos_attention ${bos_attention} \
    --ctx_attention_loss ${ctx_attention_loss} \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --learning_rate 5e-5 \
    --max_steps 1000 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --eval_steps ${eval_steps} \
    --evaluation_strategy steps \
    --max_eval_samples ${max_eval_samples} \
    --predict_with_generate \
    --save_steps 500 \
    --dataloader_num_workers 0 \
    --run_name ${run_name} \
    ${extra}
