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

eval "$(conda shell.bash hook)"
conda activate knn

output=checkpoints/wow/val_astarget_selfprov_evidence.json.beir_dedup_ans/flant5xl/knn
train_file=data/wow/val_astarget_selfprov_evidence.json.beir_dedup_ans/corpus.jsonl
validation_file=${train_file}
source_prefix="Given the context, generate the next response."
target_prefix="Evidence: "

batch_size=32
max_evi_length=128
max_gen_length=128

model=google/flan-t5-xl
use_approx_index=false
generation_file=${output}/generation.txt

python -u run_translation.py \
  --model_name_or_path ${model} \
  --train_file ${train_file} --validation_file ${validation_file} \
  --source_lang xxx --target_lang xxx \
  --output_dir ${output} \
  --dstore_dir ${output} \
  --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size ${batch_size} \
  --do_eval --eval_subset train --max_target_length ${max_evi_length} \
  --source_prefix "${source_prefix}" \
  --target_prefix "${target_prefix}" \
  --save_knnlm_dstore --build_index --use_approx_index ${use_approx_index}
exit

python -u run_translation.py \
  --model_name_or_path ${model} \
  --train_file ${train_file} --validation_file ${validation_file} \
  --source_lang xxx --target_lang xxx \
  --output_dir ${output} \
  --dstore_dir ${output} \
  --generation_file ${generation_file} \
  --per_device_train_batch_size ${batch_size} --per_device_eval_batch_size ${batch_size} \
  --do_eval --eval_subset validation --predict_with_generate --max_eval_samples 100 --max_target_length ${max_gen_length} \
  --source_prefix "Definition: Given a question, generate a descriptive answer. Question: " \
  --target_prefix "Answer: " \
  --dstore_size ${dstore_size} \
  --knn_temp 50 --k 32 --lmbda 0.25 --knn
