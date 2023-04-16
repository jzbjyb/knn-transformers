#!/usr/env/bin bash

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
conda install -c pytorch faiss-gpu=1.7.2 cudatoolkit=11.3
conda install -c anaconda mongodb
conda install -c conda-forge mongo-tools
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install wandb
pip install accelerate
# need to set CUDA_HOME
# has bugs with --global-option="build_ext" --global-option="-j8"
DS_BUILD_OPS=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_SPARSE_ATTN=0 DS_BUILD_AIO=0 pip install deepspeed
pip install sentencepiece==0.1.83
pip install filelock
pip install spacy==3.5.0
python -m spacy download en_core_web_sm
conda install -c conda-forge gsutil

# mongo for KILT
#mongod --dbpath /path/to/data --logpath /path/to/mongod.log --fork

# elasticsearch disable disk 95%
#curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_cluster/settings -d '{ "transient": { "cluster.routing.allocation.disk.threshold_enabled": false } }'
