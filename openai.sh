#!/usr/bin/env bash
set -e

debug=false

source openai_keys.sh
num_shards=${#keys[@]}

output=$1
dataset=2wikihop
max_generation_len=256
batch_size=8
model=code-davinci-002
index_name=wikipedia_dpr
consistency=1

if [[ ${dataset} == 'hotpotqa' ]]; then
    input=""
    fewshot=4
    max_num_examples=250
elif [[ ${dataset} == 'strategyqa' ]]; then
    input="--input data/strategyqa/train_cot_beir"
    fewshot=6
    max_num_examples=250
elif [[ ${dataset} == '2wikihop' ]]; then
    input="--input data/2wikimultihopqa/dev_beir"
    fewshot=4
    max_num_examples=250
else
    exit
fi

if [[ ${model} != code-* ]]; then
    num_shards=1
fi

if [[ ${index_name} == "test" && ${input} != "none" ]]; then  # build index
    okey="${keys[0]}"
    OPENAI_API_KEY=${okey} BING_SEARCH_V7_SUBSCRIPTION_KEY=${bing_key} python -m models.openai_api \
        --dataset ${dataset} ${input} \
        --fewshot ${fewshot} \
        --index_name ${index_name} \
        --build_index
    echo '========= index built ========='
fi

# query api
if [[ ${debug} == "true" ]]; then
    okey="${keys[0]}"
    OPENAI_API_KEY=${okey} BING_SEARCH_V7_SUBSCRIPTION_KEY=${bing_key} python -m models.openai_api \
        --model ${model} \
        --dataset ${dataset} ${input} \
        --fewshot ${fewshot} \
        --index_name ${index_name} \
        --max_num_examples 100 \
        --max_generation_len ${max_generation_len} \
        --batch_size 1 \
        --output test.jsonl \
        --num_shards 1 \
        --shard_id 0
    exit
fi

for (( run=0; run<${consistency}; run++ )); do
    if [[ ${consistency} == 1 ]]; then
        temperature=0
        oneoutput=${output}
    else
        temperature=0.7
        oneoutput=${output}.run${run}
    fi
    file_lock=$(mktemp)
    for (( i=0; i<${num_shards}; i++ )); do
        okey="${keys[$i]}"
        OPENAI_API_KEY=${okey} BING_SEARCH_V7_SUBSCRIPTION_KEY=${bing_key} python -m models.openai_api \
            --model ${model} \
            --dataset ${dataset} ${input} \
            --fewshot ${fewshot} \
            --index_name ${index_name} \
            --max_num_examples ${max_num_examples} \
            --max_generation_len ${max_generation_len} \
            --temperature ${temperature} \
            --batch_size ${batch_size} \
            --output ${oneoutput}.${i} \
            --num_shards ${num_shards} \
            --shard_id ${i} \
            --file_lock ${file_lock} &
    done
    wait
    rm ${file_lock}
    cat ${oneoutput}.* > ${oneoutput}
    rm ${oneoutput}.*
done
