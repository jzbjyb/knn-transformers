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

train_file=data/wow/train_neg10_dpr.json
#val_file=data/wow/train_neg10_dpr.json
val_file=data/wow/val_neg10_dpr.json
output_dir=test
#model=google/t5-xl-lm-adapt
model=checkpoints/models/t53b_wow_alpha4_hard_layer12_head4_ctx32

depth=10
max_context_len=32
use_context=true

ctx_attention_loss="block:8-layer:12-head:4-loss:hard-alpha:4"
#ctx_attention_loss="block:8-layer:0-head:0-loss:hard-alpha:4"

deepspeed train.py \
    --model_name_or_path ${model} \
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
    --do_eval \
    --do_eval_rerank \
    --per_device_eval_batch_size 1 \
    --max_eval_samples 1000 \
    --predict_with_generate \
    --dataloader_num_workers 0
