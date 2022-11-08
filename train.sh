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

# random docs
#train_file=data/wow/train_neg10_dpr.json
#val_file=data/wow/val_neg10_dpr.json

# bm25 docs
train_file=data/wow/train_astarget_selfprov_evidence.json.beir_dedup_ans.fid/dev.json
val_file=data/wow/val_astarget_selfprov_evidence.json.beir_dedup_ans.fid/dev.json

output_dir=checkpoints/models/t53b_wow_alpha4_hard_layer12_head4_ctx32_bm25_sepcrossattn
#output_dir=checkpoints/models/test
run_name="$(basename $output_dir)"
depth=10
use_context=true
ctx_attention_loss="block:8-layer:12-head:4-loss:hard-alpha:4"
max_context_len=32

deepspeed train.py \
    --deepspeed deepspeed/lr-decay-zero1.json \
    --model_name_or_path google/t5-xl-lm-adapt \
    --train_file ${train_file} \
    --validation_file ${val_file} \
    --output_dir ${output_dir} \
    --remove_unused_columns false \
    --depth ${depth} \
    --max_question_len 128 \
    --max_context_len ${max_context_len} \
    --max_answer_len 128 \
    --use_context ${use_context} \
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
    --eval_steps 100 \
    --evaluation_strategy steps \
    --max_eval_samples 1000 \
    --predict_with_generate \
    --save_steps 500 \
    --dataloader_num_workers 0 \
    --run_name ${run_name} \
    --report_to 'wandb'
