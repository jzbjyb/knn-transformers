#!/usr/bin/env bash
set -e

debug=true

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
        --max_num_examples 2
    exit
fi

for (( i=0; i<${num_shards}; i++ )); do
    okey="${keys[$i]}"
    OPENAI_API_KEY=${okey} python -m models.openai_api \
        --input data/strategyqa/dev_beir \
        --batch_size 8 \
        --output ${output}.${i} \
        --num_shards ${num_shards} \
        --shard_id ${i} &
done
wait
cat ${output}.* > ${output}
