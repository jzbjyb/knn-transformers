#!/usr/bin/env bash
#SBATCH --job-name=build_es
#SBATCH --partition=RM-shared

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --mem=100GB

#SBATCH -o slurm/%x.%j.out
#SBATCH -e slurm/%x.%j.err

eval "$(conda shell.bash hook)"
conda activate knn

pushd ../elasticsearch-7.17.9
nohup bin/elasticsearch &
popd
sleep 20
python prep.py --task build_elasticsearch --inp data/dpr/beir/corpus.jsonl wikipedia_dpr
