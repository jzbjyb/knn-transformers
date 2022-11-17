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

debug=false

setting=generate
data=bm25
model=$1  # model to test
need_model_args=$2  # specify model args or not

# -- original models --
# google/t5-xl-lm-adapt
# google/t5-small-lm-adapt
# -- finetuned models --
# checkpoints/models/t53b_wow_alpha4_hard_layer12_head4_ctx32_bm25_sepcrossattn_singlebos

# ------------- data -------------
train_file=data/wow/train_neg100_dpr.json
if [[ ${data} == "bm25" ]]; then
    # random docs
    #val_file=data/wow/train_neg100_dpr.json
    #val_file=data/wow/val_neg100_dpr.json
    # bm25 docs
    #val_file=data/wow/train_astarget_selfprov_evidence.5000.json.beir_ans.fid/dev.json
    #val_file=data/wow/val_astarget_selfprov_evidence.json.beir_ans.fid/dev.json
    # bm25 docs (dedup)
    val_file=data/wow/val_astarget_selfprov_evidence.json.beir_dedup_ans.fid/dev.json
    depth=100
elif [[ ${data} == "random" ]]; then
    # all docs (full ranking)
    val_file=data/wow/val_all_dpr.json
    depth=500
else
    exit
fi


# ------------- hyperparameters -------------
max_question_len=128
max_context_len=32
generation_prefix_len=0
use_context=true
context_bos=true
answer_bos=true
max_eval_samples=1000

if [[ ${setting} == "rerank" ]]; then
    max_answer_len=12
    setting_extra="--do_eval_rerank rerank"
elif [[ ${setting} == "generate_rerank" ]]; then
    generation_prefix_len=4
    max_answer_len=12
    setting_extra="--do_eval_rerank generate_rerank"
elif [[ ${setting} == "generate" ]]; then
    generation_prefix_len=8
    max_answer_len=128
    ctx_topk=1
    setting_extra="--ctx_topk ${ctx_topk}"
else
    exit
fi

if [[ ${need_model_args} == "true" ]]; then  # use additional model args for public pretrained models
    bos_attention=single
    ctx_attention_loss="block:8_layer:12_head:4_loss:hard_alpha:4"
    model_args="--bos_attention ${bos_attention} --ctx_attention_loss ${ctx_attention_loss}"
elif [[ ${need_model_args} == "false" ]]; then
    model_args=""
else
    exit
fi

if [[ ${debug} == "small" ]]; then
    model=google/t5-small-lm-adapt
    bos_attention=single
    ctx_attention_loss="block:8_layer:0_head:0_loss:hard_alpha:4"
    model_args="--bos_attention ${bos_attention} --ctx_attention_loss ${ctx_attention_loss}"
    max_eval_samples=4
fi

deepspeed train.py \
    --model_name_or_path ${model} \
    --train_file ${train_file} \
    --validation_file ${val_file} \
    --output_dir test \
    --remove_unused_columns false \
    --depth ${depth} \
    --max_question_len ${max_question_len} \
    --max_context_len ${max_context_len} \
    --max_answer_len ${max_answer_len} \
    --generation_prefix_len ${generation_prefix_len} \
    --use_context ${use_context} \
    --context_bos ${context_bos} \
    --answer_bos ${answer_bos} \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --max_eval_samples ${max_eval_samples} \
    --predict_with_generate \
    --dataloader_num_workers 4 \
    --report_to none \
    ${model_args} ${setting_extra}