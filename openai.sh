#!/usr/bin/env bash
set -e

debug=false

declare -a keys=(
    "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"
    "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"
    "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"
    "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"
    "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"
)
num_shards=${#keys[@]}

output=$1

if [[ ${debug} == "true" ]]; then
    okey="${keys[0]}"
    OPENAI_API_KEY=${okey} python -m models.openai_api \
        --input data/strategyqa/dev_beir \
        --batch_size 8 \
        --output test.jsonl \
        --num_shards 1 \
        --shard_id 0 \
        --max_num_examples 32
    exit
fi

for (( i=0; i<${num_shards}; i++ )); do
    okey="${keys[$i]}"
    echo OPENAI_API_KEY=${okey} python -m models.openai_api \
        --input data/strategyqa/dev_beir \
        --batch_size 8 \
        --output ${output}.${i} \
        --num_shards ${num_shards} \
        --shard_id ${i} &
done
wait
cat ${output}.* > ${output}
