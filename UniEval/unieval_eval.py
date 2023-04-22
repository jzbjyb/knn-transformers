from typing import List, Dict
import json
import argparse
from utils import convert_to_json
from metric.evaluator import get_evaluator, evaluate, multi_gpu


def extract_source_target_ref(input_file: str):
    dataset: List[Dict] = []
    with open(input_file, 'r') as fin:
        for l in fin:
            example = json.loads(l)
            source = example['question']
            target = example['output']
            refs = example['answers'] if 'answers' in example else [example['gold_output']]
            summarization_references = None
            if 'references' in example:
                summarization_references = example['references']
            elif 'raw_references' in example:
                summarization_references = example['raw_references']
            dataset.append({
                'source': source,
                'target': target,
                'references': refs,
                'summarization_references': summarization_references
            })
        return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='fact')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--devices', type=int, default=None, nargs='+')
    parser.add_argument('--gold_field', type=str, default='references')
    args = parser.parse_args()

    print('load data ...')
    dataset = extract_source_target_ref(args.input)
    src_list = ['\n'.join(e[args.gold_field]) for e in dataset]
    output_list = [e['target'] for e in dataset]
    data = convert_to_json(output_list=output_list, src_list=src_list)

    print('eval ...')
    evaluator = get_evaluator(args.task)
    multi_evaluate = multi_gpu(evaluate, cuda_devices=args.devices)
    eval_scores = multi_evaluate(evaluator, data, print_result=True)
