#!/usr/bin/env bash
set -e

debug=false

source openai_keys.sh
num_keys=${#keys[@]}

output=$1
dataset=strategyqa
debug_batch_size=1
batch_size=8
model=text-davinci-003  # code-davinci-002, gpt-3.5-turbo-0301, text-davinci-003, text-curie-001
consistency=1

function is_expensive() {
    if [[ ${model} == *turbo* || ${model} == *003 || ${model} == *001  ]]; then
        echo true
    else
        echo false
    fi
}
expensive=$(is_expensive)

if [[ ${dataset} == 'hotpotqa' ]]; then
    input=""
    index_name=wikipedia_dpr
    fewshot=12
    max_num_examples=1000
    max_generation_len=256
elif [[ ${dataset} == 'strategyqa' ]]; then
    input="--input data/strategyqa/dev_beir"
    index_name=wikipedia_dpr
    fewshot=6
    max_num_examples=229
    max_generation_len=256
elif [[ ${dataset} == '2wikihop' ]]; then
    input="--input data/2wikimultihopqa/dev_beir"
    index_name=wikipedia_dpr
    fewshot=15
    if [[ ${expensive} == true ]]; then
        fewshot=4
    fi
    max_num_examples=1000
    max_generation_len=256
elif [[ ${dataset} == 'eli5' ]]; then
    input=""  # "--input data/eli5/val_astarget_selfprov_evidence.json.beir_dedup"
    index_name=wikipedia_dpr
    fewshot=8
    if [[ ${expensive} == true ]]; then
        fewshot=4
    fi
    max_num_examples=1000000
    max_generation_len=256
elif [[ ${dataset} == 'asqa' ]]; then
    input="--input data/asqa/ASQA.json"
    index_name=wikipedia_dpr
    fewshot=8
    if [[ ${expensive} == true ]]; then
        fewshot=8
    fi
    max_num_examples=1000000
    max_generation_len=256
elif [[ ${dataset} == 'asqa_annotation' ]]; then
    input="--input data/asqa/ASQA.json"
    index_name=wikipedia_dpr
    fewshot=13
    max_num_examples=1000000
    max_generation_len=256
elif [[ ${dataset} == 'wow' ]]; then
    input=""  # "--input data/wow/val_astarget_selfprov_evidence.json.beir_dedup"
    index_name=wikipedia_dpr
    fewshot=8
    max_num_examples=1000
    max_generation_len=256
elif [[ ${dataset} == 'wow_train_1k' ]]; then
    input="--input data/wow/train_with_ref.1008.jsonl"
    index_name=wikipedia_dpr
    fewshot=8
    if [[ ${expensive} == true ]]; then
        fewshot=4
    fi
    max_num_examples=1000
    max_generation_len=256
elif [[ ${dataset} == 'wikisum_all_beir' ]]; then
    input="--input data/wikisum/wikisum_all_beir"
    index_name=wikisum_all_beir
    fewshot=8
    if [[ ${expensive} == true ]]; then
        fewshot=4
    fi
    max_num_examples=1000
    max_generation_len=512
elif [[ ${dataset} == 'wikiasp' ]]; then
    input="--input \"data/wikiasp/matched_with_bing_test.500.annotated\""
    index_name=wikiasp
    fewshot=8
    if [[ ${expensive} == true ]]; then
        fewshot=4
    fi
    max_num_examples=1000
    max_generation_len=512
elif [[ ${dataset} == 'arxiv' ]]; then
    input="--input data/pile/sliding_window_512/ArXiv_test.window.jsonl"
    index_name=arxiv00
    fewshot=0
    max_num_examples=1000
    max_generation_len=512
    dataset=lmdata
elif [[ ${dataset} == 'mmlu' ]]; then
    input=""
    index_name=wikipedia_dpr
    fewshot=4
    max_num_examples=1000
    max_generation_len=256
    if [[ ${expensive} == true ]]; then
        fewshot=4
        max_generation_len=512
    fi
else
    exit
fi

if [[ ${expensive} == true && ${dataset} != *_annotation ]]; then
    max_num_examples=$(( max_num_examples < 300 ? max_num_examples : 300 ))
fi

if [[ ${index_name} == "test" && ${input} != "none" ]]; then  # build index
    BING_SEARCH_V7_SUBSCRIPTION_KEY=${bing_key} python -m models.openai_api \
        --model ${model} \
        --dataset ${dataset} ${input} \
        --fewshot ${fewshot} \
        --index_name ${index_name} \
        --openai_keys ${test_key} \
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
        --batch_size ${debug_batch_size} \
        --output test.jsonl \
        --num_shards 1 \
        --shard_id 0 \
        --openai_keys ${test_key} \
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
    #file_lock=$(mktemp)
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
        #--file_lock ${file_lock}
    #rm ${file_lock}
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
    'use_instruction': False,
    'format_reference_method': 'searchresultsrank',
    'ctx_position': 'before_case',
    'prompt_type': 'cot',
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
    'use_instruction': False,
    'format_reference_method': 'searchresultsrank',
    'ctx_position': 'before_case',
    'prompt_type': 'cot',
    'ctx_increase': 'replace',
    'add_ref_suffix': None,
    'add_ref_prefix': None,
    'debug': args.debug,
}

# config of lookahead with mask
retrieval_kwargs = {
    'retriever': retriever,
    'topk': 3,
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
    'look_ahead_filter_prob': 0.4,
    'look_ahead_mask_prob': 0.4,
    'look_ahead_mask_method': 'wholeterm-askquestion',
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
    'reinit_ctx': True,
    'use_ctx_for_examplars': 'ret',
    'use_retrieval_instruction': False,
    'use_instruction': False,
    'format_reference_method': 'searchresultsrank',
    'ctx_position': 'before_case',
    'prompt_type': 'cot',
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
    'use_instruction': False,
    'format_reference_method': 'searchresultsrank',
    'ctx_position': 'before_case',
    'prompt_type': 'cot',
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
    'use_instruction': False,
    'format_reference_method': 'searchresultsrank',
    'ctx_position': 'before_case',
    'prompt_type': 'cot',
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
    'use_instruction': False,
    'format_reference_method': 'searchresultsrank',
    'ctx_position': 'before_case',
    'prompt_type': 'cot',
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
    'use_instruction': False,
    'format_reference_method': 'searchresultsrank',
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
    'use_instruction': False,
    'format_reference_method': 'searchresultsrank',
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
    'topk': 3,
    'use_ctx': False,
    'frequency': 0,
    #'boundary': [],
    #'boundary': ['Intermediate answer:'],
    'boundary': [')]'],
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
    'retrieval_trigers': [('\[Search\(', ')]')],
    #'retrieval_trigers': [(None, '. ')],
    'force_generate': (685, 2.0),
    'forbid_generate_step': 5,
    'truncate_at_prob': 0.0,
    'truncate_at_boundary': None,
    'append_retrieval': False,
    'use_ctx_for_examplars': False,
    'use_retrieval_instruction': 'cot',
    'use_instruction': False,
    'format_reference_method': 'searchresultsrank',
    'ctx_position': 'before_case',
    'prompt_type': 'cot_ret',
    'ctx_increase': 'replace',
    'add_ref_suffix': None,
    'add_ref_prefix': None,
    'debug': args.debug,
}

# LM ppl
retrieval_kwargs = {
    'retriever': retriever,
    'prefix_method': 'freq:32',
    'topk': 10,
    'use_ctx': True,
    'frequency': 64,
    'boundary': [],
    #'boundary': ['Intermediate answer:'],
    #'boundary': ['")]'],
    #'boundary': ['. '],
    'use_gold': False,
    'use_gold_iterative': False,
    'max_query_length': 32,
    'use_full_input_as_query': False,
    'retrieval_at_beginning': False,
    'look_ahead_steps': 16,
    'look_ahead_pre_retrieval': False,
    'look_ahead_truncate_at_boundary': None,
    'look_ahead_filter_prob': None,
    'look_ahead_mask_prob': None,
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
    'use_instruction': False,
    'format_reference_method': 'searchresultsrank',
    'ctx_position': 'before_case',
    'prompt_type': 'none',
    'ctx_increase': 'replace',
    'add_ref_suffix': None,
    'add_ref_prefix': None,
    'debug': args.debug,
}

# self ask
retrieval_kwargs = {
    'retriever': retriever,
    'topk': 3,
    'use_ctx': False,
    'frequency': 0,
    #'boundary': [],
    'boundary': ['Intermediate answer:'],
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
    #'retrieval_trigers': [],
    'retrieval_trigers': [('Follow up:', 'Intermediate answer:')],
    #'retrieval_trigers': [('\[Search\("', '")]')],
    #'retrieval_trigers': [(None, '. ')],
    'force_generate': None,
    'forbid_generate_step': None,
    'truncate_at_prob': 0.0,
    'truncate_at_boundary': None,
    'append_retrieval': False,
    'use_ctx_for_examplars': False,
    'use_retrieval_instruction': False,
    'use_instruction': False,
    'format_reference_method': 'searchresultsrank',
    'ctx_position': 'before_case',
    'prompt_type': 'sa',
    'ctx_increase': 'replace',
    'add_ref_suffix': None,
    'add_ref_prefix': None,
    'debug': args.debug,
}