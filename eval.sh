#!/usr/bin/env bash
#SBATCH --time=8:00:00
#SBATCH --partition=learnlab
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%x.%j.out
#SBATCH -e slurm/%x.%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=512GB
#SBATCH --job-name=fustiont5

eval "$(conda shell.bash hook)"
conda activate knn

export WANDB_PROJECT=unifiedrlm
export WANDB_API_KEY=9caada2c257feff1b6e6a519ad378be3994bc06a

debug=false

model=$1  # model to test
setting=generate
data=wow
output_file=test.jsonl

# ------------- hyperparameters -------------
use_context=true
batch_size=32
depth=1

# ------------- data -------------
if [[ ${data} == "wow" ]]; then
    question_prefix=$'Given the context, generate the next response.\nContext:\n'
    encoder_input_for_context="Given the context, generate the next response."
    context_prefix="Evidence: "
    answer_prefix="Response: "
    train_file=data/wow/train_neg100_dpr.json
    val_file=data/wow/val_astarget_selfprov_evidence.json.beir_dedup_ans.fid/dev.json
    beir_dir=data/wow/val_astarget_selfprov_evidence.json.beir_dedup_ans
    dstore_dir=checkpoints/wow/val_astarget_selfprov_evidence.json.beir_dedup_ans/flant5xl/knn
    dstore_size=38089
    max_eval_samples=1000
else
    exit
fi

# ------------- hyperparameters -------------
max_question_len=128
max_context_len=128
max_answer_len=128
generation_prefix_len=0
context_bos=true
answer_bos=true
bos_attention=single

if [[ ${setting} == "perplexity" ]]; then
    max_eval_samples=100000
    setting_extra="--do_eval_special perplexity"
elif [[ ${setting} == "generate_perplexity" ]]; then
    generation_prefix_len=128
    setting_extra="--do_eval_special generate_perplexity --predict_with_generat"
elif [[ ${setting} == "generate" ]]; then
    generation_prefix_len=2  # for 'Answer:'
    setting_extra="--do_eval_special generate --predict_with_generate"
else
    exit
fi

if [[ ${debug} == "small" ]]; then
    model=google/t5-small-lm-adapt
    max_eval_samples=8
    batch_size=2
fi

deepspeed train.py \
    --model_name_or_path ${model} \
    --train_file ${train_file} \
    --validation_file ${val_file} \
    --beir_dir ${beir_dir} \
    --question_prefix "${question_prefix}" \
    --context_prefix "${context_prefix}" \
    --answer_prefix "${answer_prefix}" \
    --encoder_input_for_context "${encoder_input_for_context}" \
    --output_dir test \
    --output_file ${output_file} \
    --remove_unused_columns false \
    --depth ${depth} \
    --max_question_len ${max_question_len} \
    --max_context_len ${max_context_len} \
    --max_answer_len ${max_answer_len} \
    --generation_prefix_len ${generation_prefix_len} \
    --use_context ${use_context} \
    --context_bos ${context_bos} \
    --answer_bos ${answer_bos} \
    --bos_attention ${bos_attention} \
    --do_eval \
    --per_device_eval_batch_size ${batch_size} \
    --max_eval_samples ${max_eval_samples} \
    --dataloader_num_workers 4 \
    --report_to none \
    --dstore_size ${dstore_size} \
    --dstore_dir ${dstore_dir} \
    ${setting_extra}
