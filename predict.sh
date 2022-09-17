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

model=t5-small
output=checkpoints-translation/t5-small-fixbug

python -u run_translation.py  \
  --model_name_or_path ${model} \
  --dataset_name wmt16 --dataset_config_name ro-en \
  --source_lang en --target_lang ro \
  --output_dir ${output} \
  --dstore_dir ${output} \
  --per_device_train_batch_size 4 --per_device_eval_batch_size=4 \
  --do_eval --predict_with_generate \
  --source_prefix "translate English to Romanian: " \
  --dstore_size 26565876 \
  --knn_temp 50 --k 32 --lmbda 0.25 --retomaton
