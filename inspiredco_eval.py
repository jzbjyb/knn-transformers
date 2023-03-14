from typing import List, Dict
import os
import argparse
import json
from inspiredco import critique


metric = "rouge"
config = {
    "variety": "rouge_1"
}

metric = "uni_eval"
config = {
    "task": "summarization",
    "evaluation_aspect": "relevance",
}

metric = "bert_score"
config = {
    "variety": "f_measure",
    "model": "bert-base-uncased",
    "language": "eng"
}

metric = "bart_score"
config = {
    "variety": "reference_target_bidirectional",
    "model": "facebook/bart-large-cnn",
    "language": "eng"
}

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
    parser.add_argument('--input', type=str, default=None)
    args = parser.parse_args()

    print('load data ...')
    dataset = extract_source_target_ref(args.input)

    print('eval ...')
    client = critique.Critique(api_key=os.environ['INSPIREDCO_API_KEY'])
    results = client.evaluate(metric=metric, config=config, dataset=dataset)

    print(results['overall'])
