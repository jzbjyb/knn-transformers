#!/usr/bin/env bash
set -e

debug=false

source openai_keys.sh
num_shards=${#keys[@]}

output=$1
max_generation_len=256

if [[ ${debug} == "true" ]]; then
    okey="${keys[0]}"
    OPENAI_API_KEY=${okey} python -m models.openai_api \
        --input data/strategyqa/train_cot_beir \
        --max_num_examples 250 \
        --max_generation_len ${max_generation_len} \
        --batch_size 8 \
        --output test.jsonl \
        --num_shards 1 \
        --shard_id 0 \
        --max_num_examples 32
    exit
fi

for (( i=0; i<${num_shards}; i++ )); do
    okey="${keys[$i]}"
    OPENAI_API_KEY=${okey} python -m models.openai_api \
        --input data/strategyqa/train_cot_beir \
        --max_num_examples 250 \
        --max_generation_len ${max_generation_len} \
        --batch_size 8 \
        --output ${output}.${i} \
        --num_shards ${num_shards} \
        --shard_id ${i} &
done
wait
cat ${output}.* > ${output}
rm ${output}.*
