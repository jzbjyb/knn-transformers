#!/usr/bin/env bash
set -e

debug=false

source openai_keys.sh
num_shards=${#keys[@]}

output=$1
max_generation_len=256
batch_size=8
max_num_examples=250
model=code-davinci-002
fewshot=6
index_name=wikipedia_dpr

if [[ ${model} != code-* ]]; then
    num_shards=1
fi

if [[ ${index_name} == "test" ]]; then  # build index
    OPENAI_API_KEY=${keys[0]} python -m models.openai_api \
        --input data/strategyqa/train_cot_beir \
        --index_name ${index_name} \
        --build_index
fi

# query api
if [[ ${debug} == "true" ]]; then
    okey="${keys[0]}"
    OPENAI_API_KEY=${okey} BING_SEARCH_V7_SUBSCRIPTION_KEY=${bing_key} python -m models.openai_api \
        --model ${model} \
        --input data/strategyqa/train_cot_beir \
        --index_name ${index_name} \
        --max_num_examples 32 \
        --max_generation_len ${max_generation_len} \
        --fewshot ${fewshot} \
        --batch_size ${batch_size} \
        --output test.jsonl \
        --num_shards 1 \
        --shard_id 0
    exit
fi

for (( i=0; i<${num_shards}; i++ )); do
    okey="${keys[$i]}"
    OPENAI_API_KEY=${okey} BING_SEARCH_V7_SUBSCRIPTION_KEY=${bing_key} python -m models.openai_api \
        --model ${model} \
        --input data/strategyqa/train_cot_beir \
        --index_name ${index_name} \
        --max_num_examples ${max_num_examples} \
        --max_generation_len ${max_generation_len} \
        --fewshot ${fewshot} \
        --batch_size ${batch_size} \
        --output ${output}.${i} \
        --num_shards ${num_shards} \
        --shard_id ${i} &
done
wait
cat ${output}.* > ${output}
rm ${output}.*
