#!/usr/env/bin bash

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
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

# mongo for KILT
#mongod --dbpath /path/to/data --logpath /path/to/mongod.log --fork
