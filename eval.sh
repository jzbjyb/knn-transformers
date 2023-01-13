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

setting=generate_perplexity
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
    #val_file=data/bioasq_test.json
    depth=100
elif [[ ${data} == "random" ]]; then
    # all docs (full ranking)
    val_file=data/wow/val_all_dpr.json
    depth=500
elif [[ ${data} == "bm25_split" ]]; then
    val_file=data/wow/val_astarget_selfprov_evidence.json.beir_dedup_ans.fid/dev.split.json
    depth=1
else
    exit
fi

# ------------- hyperparameters -------------
max_question_len=128
max_context_len=128
generation_prefix_len=0
use_context=true
context_bos=true
answer_bos=true
max_eval_samples=1000
batch_size=32

if [[ ${setting} == "rerank" ]]; then
    max_answer_len=12
    setting_extra="--do_eval_special rerank"
elif [[ ${setting} == "generate_rerank" ]]; then
    generation_prefix_len=4
    max_answer_len=12
    setting_extra="--do_eval_special generate_rerank --predict_with_generate"
elif [[ ${setting} == "perplexity" ]]; then
    max_answer_len=128
    batch_size=50
    max_eval_samples=100000
    setting_extra="--do_eval_special perplexity"
elif [[ ${setting} == "generate_perplexity" ]]; then
    generation_prefix_len=128
    max_answer_len=128
    depth=1
    setting_extra="--do_eval_special generate_perplexity --predict_with_generat"
elif [[ ${setting} == "gradient" ]]; then
    max_answer_len=128
    batch_size=8
    max_eval_samples=10000
    setting_extra="--do_eval_special gradient"
elif [[ ${setting} == "gradient-batch" ]]; then
    max_answer_len=128
    setting_extra="--do_eval_special gradient-batch"
elif [[ ${setting} == "generate" ]]; then
    generation_prefix_len=128
    max_answer_len=128
    ctx_topk=0
    setting_extra="--ctx_topk ${ctx_topk} --predict_with_generate"
else
    exit
fi

if [[ ${need_model_args} == "true" ]]; then  # use additional model args for public pretrained models
    bos_attention=single
    # - misc
    # ctx_attention_loss="block:8_layer2heads:12.[4,5]_layerheadagg:normalize-softmax-mean_layerheadtau:0.001_tokenagg:premean_loss:hard_alpha:4"
    # - test reranking performance of all heads
    # ctx_attention_loss="block:8_layer2heads:0.list(range(24))|3.list(range(24))|6.list(range(24))|9.list(range(24))|12.list(range(24))|15.list(range(24))|18.list(range(24))|21.list(range(24))|23.list(range(24))_layerheadagg:none_loss:hard_alpha:4"
    # - test perplexity of different contexts
    ctx_attention_loss="block:8_layer2heads:_loss:hard_alpha:4"
    #model_args="--bos_attention ${bos_attention} --ctx_attention_loss ${ctx_attention_loss}"
    model_args="--bos_attention ${bos_attention}"
elif [[ ${need_model_args} == "false" ]]; then
    model_args=""
else
    exit
fi

if [[ ${debug} == "small" ]]; then
    model=google/t5-small-lm-adapt
    bos_attention=single
    ctx_attention_loss="block:8_layer2heads:_loss:hard_alpha:4"
    #model_args="--bos_attention ${bos_attention} --ctx_attention_loss ${ctx_attention_loss}"
    model_args="--bos_attention ${bos_attention}"
    max_eval_samples=8
    batch_size=2
    use_context=false
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
    --per_device_eval_batch_size ${batch_size} \
    --max_eval_samples ${max_eval_samples} \
    --dataloader_num_workers 4 \
    --report_to none \
    ${model_args} ${setting_extra}
