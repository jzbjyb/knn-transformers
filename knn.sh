#!/usr/bin/env bash
#SBATCH --job-name=knn
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --partition=learnlab
#SBATCH --mem=256GB
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

# env
source env.sh

: '
model=t5-small
output=checkpoints-translation/wmt16_enro_train_knn
dataset=wmt16
dataset_config=ro-en
source_lang=en
target_lang=ro
split=train
num_samples=1000000000
prefix="translate English to Romanian: "
dstore_size=26565876
'

: '
model=K024/mt5-zh-ja-en-trimmed
output=checkpoints-translation/wmt19_enzh_val_knn
dataset=wmt19
dataset_config=zh-en
source_lang=en
target_lang=zh
split=validation
num_samples=1000000000
prefix="en2zh: "
dstore_size=116168
'

: '
model=K024/mt5-zh-ja-en-trimmed
output=checkpoints-translation/wmt19_enzh_train200k_knn
dataset=wmt19
dataset_config=zh-en
source_lang=en
target_lang=zh
split=train
num_samples=200000
prefix="en2zh: "
dstore_size=5980103
'

model=K024/mt5-zh-ja-en-trimmed
output=checkpoints-translation/wmt19_enzh_train100k_knn
dataset=wmt19
dataset_config=zh-en
source_lang=en
target_lang=zh
split=train
num_samples=100000
prefix="en2zh: "
dstore_size=2974871

python -u run_translation.py  \
  --model_name_or_path ${model} \
  --dataset_name ${dataset} --dataset_config_name ${dataset_config} \
  --source_lang ${source_lang} --target_lang ${target_lang} \
  --output_dir ${output} \
  --dstore_dir ${output} \
  --per_device_train_batch_size 4 --per_device_eval_batch_size=4 \
  --do_eval --eval_subset ${split} --max_eval_samples ${num_samples} \
  --source_prefix "${prefix}" \
  --save_knnlm_dstore

python -u run_translation.py  \
  --model_name_or_path ${model} \
  --dataset_name ${dataset} --dataset_config_name ${dataset_config} \
  --source_lang ${source_lang} --target_lang ${target_lang} \
  --output_dir ${output} \
  --dstore_dir ${output} \
  --per_device_train_batch_size 4 --per_device_eval_batch_size=4 \
  --dstore_size ${dstore_size} \
  --build_index

python -u run_translation.py  \
  --model_name_or_path ${model} \
  --dataset_name ${dataset} --dataset_config_name ${dataset_config} \
  --source_lang ${source_lang} --target_lang ${target_lang} \
  --output_dir ${output} \
  --dstore_dir ${output} \
  --per_device_train_batch_size 4 --per_device_eval_batch_size=4 \
  --do_predict --eval_subset validation --predict_with_generate --max_predict_samples 500 \
  --source_prefix "${prefix}" \
  --dstore_size ${dstore_size} \
  --knn_temp 50 --k 1 --lmbda 0.5 --retomaton
