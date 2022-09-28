#!/usr/bin/env bash
#SBATCH --job-name=knn
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --partition=learnlab
#SBATCH --mem=128GB
#SBATCH --constraint=volta32gb
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

# env
source env.sh

batch_size=4
max_target_length=256
generation_file=generated_predictions.txt

: '
model=allenai/tk-instruct-base-def-pos
output=checkpoints/eli5/val_astarget_answer/memtrans
train_file=data/eli5/val_astarget_answer_evidence.json
validation_file=data/eli5/val_astarget_answer_qa.json
source_lang=en
target_lang=zh
split=train
num_samples=1000000000
prefix="Definition: Given a question, generate a relevant answer to the question. Input: "
suffix=" Output:"
dstore_size=1328738
'

: '
model=google/t5-xl-lm-adapt
output=checkpoints/eli5/t53b/val_astarget_prov/memtrans
train_file=data/eli5/val_astarget_prov_evidence.json
validation_file=data/eli5/val_astarget_prov_qa.json
source_lang=en
target_lang=zh
split=train
num_samples=1000000000
prefix="Definition: Given a question, generate a descriptive answer. Question: "
#suffix=" Evidence:"
suffix=" Answer:"
use_approx_index=false
dstore_size=306645
'

model=google/t5-xl-lm-adapt
output=checkpoints/eli5/t53b/val_astarget_answer/memtrans
train_file=data/eli5/val_astarget_answer_qa.json
validation_file=data/eli5/val_astarget_answer_qa.json
source_lang=en
target_lang=zh
split=train
num_samples=1000000000
prefix="Definition: Given a question, generate a descriptive answer. Question: "
#suffix=" Evidence:"
suffix=" Answer:"
use_approx_index=false
dstore_size=204498

python -u run_translation.py \
  --model_name_or_path ${model} \
  --train_file ${train_file} --validation_file ${validation_file} \
  --source_lang ${source_lang} --target_lang ${target_lang} \
  --output_dir ${output} \
  --dstore_dir ${output} \
  --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size ${batch_size} \
  --do_eval --eval_subset ${split} --max_eval_samples ${num_samples} --max_target_length ${max_target_length} \
  --source_prefix "${prefix}" \
  --source_suffix "${suffix}" \
  --save_knnlm_dstore --build_index --memtrans

python -u run_translation.py \
  --model_name_or_path ${model} \
  --train_file ${train_file} --validation_file ${validation_file} \
  --source_lang ${source_lang} --target_lang ${target_lang} \
  --output_dir ${output} \
  --dstore_dir ${output} \
  --generation_file ${generation_file} \
  --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size ${batch_size} \
  --do_predict --eval_subset validation --predict_with_generate --max_target_length ${max_target_length} \
  --source_prefix "${prefix}" \
  --source_suffix "${suffix}" \
  --dstore_size ${dstore_size} \
  --target_prefix "Answer: "
  #--memtrans --k 1 --max_predict_samples 500
