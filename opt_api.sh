#!/usr/bin/env bash
set -e

debug=false

num_shards=1
output=$1
max_generation_len=256
batch_size=8
max_num_examples=250
model=opt-iml-max-175b
prompt_type=sa_ctx

# build index
python -m models.openai_api \
    --input /projects/metis1/users/zhengbaj/exp/knn-transformers/data/strategyqa/train_cot_beir \
    --build_index

# query api
if [[ ${debug} == "true" ]]; then
    OPENAI_API_KEY=test OPENAI_API_BASE="http://localhost:6010" python -m models.openai_api \
        --model ${model} \
        --prompt_type ${prompt_type} \
        --input /projects/metis1/users/zhengbaj/exp/knn-transformers/data/strategyqa/train_cot_beir \
        --max_num_examples 32 \
        --max_generation_len ${max_generation_len} \
        --batch_size ${batch_size} \
        --output test.jsonl \
        --num_shards 1 \
        --shard_id 0
    exit
fi

OPENAI_API_KEY=test OPENAI_API_BASE="http://localhost:6010" python -m models.openai_api \
    --model ${model} \
    --prompt_type ${prompt_type} \
    --input /projects/metis1/users/zhengbaj/exp/knn-transformers/data/strategyqa/train_cot_beir \
    --max_num_examples ${max_num_examples} \
    --max_generation_len ${max_generation_len} \
    --batch_size ${batch_size} \
    --output ${output}
