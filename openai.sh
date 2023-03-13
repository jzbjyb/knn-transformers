#!/usr/bin/env bash
set -e

debug=false

source openai_keys.sh
num_keys=${#keys[@]}

output=$1
dataset=wow
batch_size=8
model=code-davinci-002
index_name=wikipedia_dpr  # wikipedia_dpr, wikisum_all_beir
consistency=1

if [[ ${dataset} == 'hotpotqa' ]]; then
    input=""
    fewshot=12
    max_num_examples=1000
    max_generation_len=256
elif [[ ${dataset} == 'strategyqa_dev' ]]; then
    input="--input data/strategyqa/dev_beir"
    fewshot=6
    max_num_examples=229
    max_generation_len=256
elif [[ ${dataset} == '2wikihop' ]]; then
    input="--input data/2wikimultihopqa/dev_beir"
    fewshot=15
    max_num_examples=1000
    max_generation_len=256
elif [[ ${dataset} == 'eli5' ]]; then
    input=""  # "--input data/eli5/val_astarget_selfprov_evidence.json.beir_dedup"
    fewshot=8
    max_num_examples=1000000
    max_generation_len=256
elif [[ ${dataset} == 'wow' ]]; then
    input=""  # "--input data/wow/val_astarget_selfprov_evidence.json.beir_dedup"
    fewshot=8
    max_num_examples=1000
    max_generation_len=256
elif [[ ${dataset} == 'wikisum_all_beir' ]]; then
    input="--input data/wikisum/wikisum_all_beir"
    fewshot=8
    max_num_examples=1000
    max_generation_len=256
else
    exit
fi

if [[ ${model} != code-* ]]; then
    num_keys=1
fi

if [[ ${index_name} == "test" && ${input} != "none" ]]; then  # build index
    BING_SEARCH_V7_SUBSCRIPTION_KEY=${bing_key} python -m models.openai_api \
        --model ${model} \
        --dataset ${dataset} ${input} \
        --fewshot ${fewshot} \
        --index_name ${index_name} \
        --openai_keys ${keys[0]} \
        --build_index
    echo '========= index built ========='
fi

# query api
if [[ ${debug} == "true" ]]; then
    BING_SEARCH_V7_SUBSCRIPTION_KEY=${bing_key} python -m models.openai_api \
        --model ${model} \
        --dataset ${dataset} ${input} \
        --fewshot ${fewshot} \
        --index_name ${index_name} \
        --max_num_examples 100 \
        --max_generation_len ${max_generation_len} \
        --batch_size ${batch_size} \
        --output test.jsonl \
        --num_shards 1 \
        --shard_id 0 \
        --openai_keys "${keys[0]}" \
        --debug
    exit
fi

function join_by {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

joined_keys=$(join_by " " "${keys[@]:0:${num_keys}}")

for (( run=0; run<${consistency}; run++ )); do
    if [[ ${consistency} == 1 ]]; then
        temperature=0
        oneoutput=${output}
    else
        temperature=0.7
        oneoutput=${output}.run${run}
    fi
    file_lock=$(mktemp)
    #for (( i=0; i<${num_shards}; i++ )); do
    #okey="${keys[$i]}"
    BING_SEARCH_V7_SUBSCRIPTION_KEY=${bing_key} python -m models.openai_api \
        --model ${model} \
        --dataset ${dataset} ${input} \
        --fewshot ${fewshot} \
        --index_name ${index_name} \
        --max_num_examples ${max_num_examples} \
        --max_generation_len ${max_generation_len} \
        --temperature ${temperature} \
        --batch_size ${batch_size} \
        --output ${oneoutput} \
        --num_shards 1 \
        --shard_id 0 \
        --openai_keys ${joined_keys} \
        --file_lock ${file_lock}
    #done
    #wait
    rm ${file_lock}
    #cat ${oneoutput}.* > ${oneoutput}
    #rm ${oneoutput}.*
done

exit
# raw
retrieval_kwargs = {
    'retriever': retriever,
    'topk': 1,
    'use_ctx': False,
    'frequency': 0,
    'boundary': [],
    #'boundary': ['Intermediate answer:'],
    #'boundary': ['")]'],
    #'boundary': ['. '],
    'use_gold': False,
    'use_gold_iterative': False,
    'max_query_length': 16,
    'use_full_input_as_query': False,
    'retrieval_at_beginning': False,
    'look_ahead_steps': 0,
    'look_ahead_truncate_at_boundary': None,
    'look_ahead_filter_prob': 0.0,
    'look_ahead_mask_prob': 0.0,
    'look_ahead_boundary': [],
    'only_use_look_ahead': False,
    'retrieval_trigers': [],
    #'retrieval_trigers': [('Follow up:', 'Intermediate answer:')],
    #'retrieval_trigers': [('\[Search\("', '")]')],
    #'retrieval_trigers': [(None, '. ')],
    'force_generate': None,
    'forbid_generate_step': 0,
    'truncate_at_prob': 0.0,
    'truncate_at_boundary': None,
    'append_retrieval': False,
    'use_ctx_for_examplars': False,
    'use_retrieval_instruction': False,
    'format_reference_method': 'default',
    'ctx_position': 'before_case',
    'prompt_type': 'cot_interleave',
    'ctx_increase': 'replace',
    'add_ref_suffix': None,
    'add_ref_prefix': None,
    'debug': args.debug,
}

# only one ret
retrieval_kwargs = {
    'retriever': retriever,
    'topk': 1,
    'use_ctx': True,
    'frequency': 1,
    'boundary': [],
    #'boundary': ['Intermediate answer:'],
    #'boundary': ['")]'],
    #'boundary': ['. '],
    'use_gold': False,
    'use_gold_iterative': False,
    'max_query_length': 16,
    'use_full_input_as_query': True,
    'retrieval_at_beginning': True,
    'look_ahead_steps': 0,
    'look_ahead_truncate_at_boundary': None,
    'look_ahead_filter_prob': 0.0,
    'look_ahead_mask_prob': 0.0,
    'look_ahead_boundary': [],
    'only_use_look_ahead': False,
    'retrieval_trigers': [],
    #'retrieval_trigers': [('Follow up:', 'Intermediate answer:')],
    #'retrieval_trigers': [('\[Search\("', '")]')],
    #'retrieval_trigers': [(None, '. ')],
    'force_generate': None,
    'forbid_generate_step': 0,
    'truncate_at_prob': 0.0,
    'truncate_at_boundary': None,
    'append_retrieval': False,
    'use_ctx_for_examplars': 'gold',
    'use_retrieval_instruction': False,
    'format_reference_method': 'default',
    'ctx_position': 'before_case',
    'prompt_type': 'cot_interleave',
    'ctx_increase': 'replace',
    'add_ref_suffix': None,
    'add_ref_prefix': None,
    'debug': args.debug,
}

# config of lookahead with mask
retrieval_kwargs = {
    'retriever': retriever,
    'topk': 1,
    'use_ctx': True,
    'frequency': 64,
    'boundary': [],
    #'boundary': ['Intermediate answer:'],
    #'boundary': ['")]'],
    #'boundary': ['. '],
    'use_gold': False,
    'use_gold_iterative': False,
    'max_query_length': 64,
    'use_full_input_as_query': True,
    'retrieval_at_beginning': False,
    'look_ahead_steps': 64,
    'look_ahead_truncate_at_boundary': 'sentence',
    'look_ahead_filter_prob': 0.0,
    'look_ahead_mask_prob': 0.2,
    'look_ahead_boundary': [],
    'only_use_look_ahead': True,
    'retrieval_trigers': [],
    #'retrieval_trigers': [('Follow up:', 'Intermediate answer:')],
    #'retrieval_trigers': [('\[Search\("', '")]')],
    #'retrieval_trigers': [(None, '. ')],
    'force_generate': None,
    'forbid_generate_step': None,
    'truncate_at_prob': 0.0,
    'truncate_at_boundary': 'sentence',
    'append_retrieval': False,
    'use_ctx_for_examplars': 'gold',
    'use_retrieval_instruction': False,
    'format_reference_method': 'default',
    'ctx_position': 'before_case',
    'prompt_type': 'cot_interleave',
    'ctx_increase': 'replace',
    'add_ref_suffix': None,
    'add_ref_prefix': None,
    'debug': args.debug,
}

# config of interleave
retrieval_kwargs = {
    'retriever': retriever,
    'topk': 1,
    'use_ctx': True,
    'frequency': 64,
    'boundary': [],
    #'boundary': ['Intermediate answer:'],
    #'boundary': ['")]'],
    #'boundary': ['. '],
    'use_gold': False,
    'use_gold_iterative': False,
    'max_query_length': 64,
    'use_full_input_as_query': True,
    'retrieval_at_beginning': False,
    'look_ahead_steps': 0,
    'look_ahead_truncate_at_boundary': None,
    'look_ahead_filter_prob': 0.0,
    'look_ahead_mask_prob': 0.0,
    'look_ahead_boundary': [],
    'only_use_look_ahead': False,
    'retrieval_trigers': [],
    #'retrieval_trigers': [('Follow up:', 'Intermediate answer:')],
    #'retrieval_trigers': [('\[Search\("', '")]')],
    #'retrieval_trigers': [(None, '. ')],
    'force_generate': None,
    'forbid_generate_step': None,
    'truncate_at_prob': 0.0,
    'truncate_at_boundary': 'sentence',
    'append_retrieval': False,
    'use_ctx_for_examplars': 'gold',
    'use_retrieval_instruction': False,
    'format_reference_method': 'default',
    'ctx_position': 'before_case',
    'prompt_type': 'cot_interleave',
    'ctx_increase': 'replace',
    'add_ref_suffix': None,
    'add_ref_prefix': None,
    'debug': args.debug,
}

# config of prob
retrieval_kwargs = {
    'retriever': retriever,
    'topk': 1,
    'use_ctx': True,
    'frequency': 64,
    'boundary': [],
    #'boundary': ['Intermediate answer:'],
    #'boundary': ['")]'],
    #'boundary': ['. '],
    'use_gold': False,
    'use_gold_iterative': False,
    'max_query_length': 16,
    'use_full_input_as_query': True,
    'retrieval_at_beginning': False,
    'look_ahead_steps': 0,
    'look_ahead_truncate_at_boundary': None,
    'look_ahead_filter_prob': 0.0,
    'look_ahead_mask_prob': 0.0,
    'look_ahead_boundary': [],
    'only_use_look_ahead': False,
    'retrieval_trigers': [],
    #'retrieval_trigers': [('Follow up:', 'Intermediate answer:')],
    #'retrieval_trigers': [('\[Search\("', '")]')],
    #'retrieval_trigers': [(None, '. ')],
    'force_generate': None,
    'forbid_generate_step': None,
    'truncate_at_prob': 0.2,
    'truncate_at_boundary': None,
    'append_retrieval': False,
    'use_ctx_for_examplars': 'gold',
    'use_retrieval_instruction': False,
    'format_reference_method': 'default',
    'ctx_position': 'before_case',
    'prompt_type': 'cot_interleave',
    'ctx_increase': 'replace',
    'add_ref_suffix': None,
    'add_ref_prefix': None,
    'debug': args.debug,
}

# config of freq
retrieval_kwargs = {
    'retriever': retriever,
    'topk': 1,
    'use_ctx': True,
    'frequency': 16,
    'boundary': [],
    #'boundary': ['Intermediate answer:'],
    #'boundary': ['")]'],
    #'boundary': ['. '],
    'use_gold': False,
    'use_gold_iterative': False,
    'max_query_length': 16,
    'use_full_input_as_query': True,
    'retrieval_at_beginning': False,
    'look_ahead_steps': 0,
    'look_ahead_truncate_at_boundary': None,
    'look_ahead_filter_prob': 0.0,
    'look_ahead_mask_prob': 0.0,
    'look_ahead_boundary': [],
    'only_use_look_ahead': False,
    'retrieval_trigers': [],
    #'retrieval_trigers': [('Follow up:', 'Intermediate answer:')],
    #'retrieval_trigers': [('\[Search\("', '")]')],
    #'retrieval_trigers': [(None, '. ')],
    'force_generate': None,
    'forbid_generate_step': None,
    'truncate_at_prob': 0.0,
    'truncate_at_boundary': None,
    'append_retrieval': False,
    'use_ctx_for_examplars': 'gold',
    'use_retrieval_instruction': False,
    'format_reference_method': 'default',
    'ctx_position': 'before_case',
    'prompt_type': 'cot_interleave',
    'ctx_increase': 'replace',
    'add_ref_suffix': None,
    'add_ref_prefix': None,
    'debug': args.debug,
}





# ---- ret instruction ---

# config of raw
retrieval_kwargs = {
    'retriever': retriever,
    'topk': 1,
    'use_ctx': False,
    'frequency': 0,
    'boundary': [],
    #'boundary': ['Intermediate answer:'],
    #'boundary': ['")]'],
    #'boundary': ['. '],
    'use_gold': False,
    'use_gold_iterative': False,
    'max_query_length': 16,
    'use_full_input_as_query': False,
    'retrieval_at_beginning': False,
    'look_ahead_steps': 0,
    'look_ahead_truncate_at_boundary': None,
    'look_ahead_filter_prob': 0.0,
    'look_ahead_mask_prob': 0.0,
    'look_ahead_boundary': [],
    'only_use_look_ahead': False,
    'retrieval_trigers': [],
    #'retrieval_trigers': [('Follow up:', 'Intermediate answer:')],
    #'retrieval_trigers': [('\[Search\("', '")]')],
    #'retrieval_trigers': [(None, '. ')],
    'force_generate': None,
    'forbid_generate_step': 0,
    'truncate_at_prob': 0.0,
    'truncate_at_boundary': None,
    'append_retrieval': False,
    'use_ctx_for_examplars': False,
    'use_retrieval_instruction': 'cot',
    'format_reference_method': 'default',
    'ctx_position': 'before_case',
    'prompt_type': 'cot_interleave_ret',
    'ctx_increase': 'replace',
    'add_ref_suffix': None,
    'add_ref_prefix': None,
    'debug': args.debug,
}

# config of only one ret
retrieval_kwargs = {
    'retriever': retriever,
    'topk': 1,
    'use_ctx': True,
    'frequency': 1,
    'boundary': [],
    #'boundary': ['Intermediate answer:'],
    #'boundary': ['")]'],
    #'boundary': ['. '],
    'use_gold': False,
    'use_gold_iterative': False,
    'max_query_length': 16,
    'use_full_input_as_query': True,
    'retrieval_at_beginning': True,
    'look_ahead_steps': 0,
    'look_ahead_truncate_at_boundary': None,
    'look_ahead_filter_prob': 0.0,
    'look_ahead_mask_prob': 0.0,
    'look_ahead_boundary': [],
    'only_use_look_ahead': False,
    'retrieval_trigers': [],
    #'retrieval_trigers': [('Follow up:', 'Intermediate answer:')],
    #'retrieval_trigers': [('\[Search\("', '")]')],
    #'retrieval_trigers': [(None, '. ')],
    'force_generate': None,
    'forbid_generate_step': 0,
    'truncate_at_prob': 0.0,
    'truncate_at_boundary': None,
    'append_retrieval': False,
    'use_ctx_for_examplars': 'gold',
    'use_retrieval_instruction': 'cot',
    'format_reference_method': 'default',
    'ctx_position': 'before_case',
    'prompt_type': 'cot_interleave_ret',
    'ctx_increase': 'replace',
    'add_ref_suffix': None,
    'add_ref_prefix': None,
    'debug': args.debug,
}

# config of search term
retrieval_kwargs = {
    'retriever': retriever,
    'topk': 1,
    'use_ctx': True,
    'frequency': 0,
    #'boundary': [],
    #'boundary': ['Intermediate answer:'],
    'boundary': ['")]'],
    #'boundary': ['. '],
    'use_gold': False,
    'use_gold_iterative': False,
    'max_query_length': 64,
    'use_full_input_as_query': True,
    'retrieval_at_beginning': False,
    'look_ahead_steps': 0,
    'look_ahead_truncate_at_boundary': None,
    'look_ahead_filter_prob': 0.0,
    'look_ahead_mask_prob': 0.0,
    'look_ahead_boundary': [],
    'only_use_look_ahead': False,
    #'retrieval_trigers': [],
    #'retrieval_trigers': [('Follow up:', 'Intermediate answer:')],
    'retrieval_trigers': [('\[Search\("', '")]')],
    #'retrieval_trigers': [(None, '. ')],
    'force_generate': (685, 2.0),
    'forbid_generate_step': 5,
    'truncate_at_prob': 0.0,
    'truncate_at_boundary': None,
    'append_retrieval': False,
    'use_ctx_for_examplars': 'gold',
    'use_retrieval_instruction': 'cot',
    'format_reference_method': 'default',
    'ctx_position': 'before_case',
    'prompt_type': 'cot_interleave_ret',
    'ctx_increase': 'replace',
    'add_ref_suffix': None,
    'add_ref_prefix': None,
    'debug': args.debug,
}
