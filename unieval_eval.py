from typing import List, Dict
import json
import argparse
from utils import convert_to_json
from metric.evaluator import get_evaluator


def extract_source_target_ref(input_file: str):
    dataset: List[Dict] = []
    with open(input_file, 'r') as fin:
        for l in fin:
            example = json.loads(l)
            source = example['question']
            target = example['output']
            refs = example['answers'] if 'answers' in example else [example['gold_output']]
            dataset.append({
                'source': source,
                'target': target,
                'references': refs,
            })
        return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='fact')
    parser.add_argument('--input', type=str, default=None)
    args = parser.parse_args()

    print('load data ...')
    dataset = extract_source_target_ref(args.input)
    src_list = ['\n'.join(e['references']) for e in dataset]
    output_list = [e['target'] for e in dataset]
    data = convert_to_json(output_list=output_list, src_list=src_list)

    print('eval ...')
    evaluator = get_evaluator(args.task)
    eval_scores = evaluator.evaluate(data, print_result=True)
