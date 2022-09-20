from typing import List, Tuple
import argparse
import random
import evaluate

def load_pred_file(pred_file: str) -> List[Tuple[str, str, str]]:  # input, gold, pred
    examples: List[Tuple[str, str, str]] = []
    with open(pred_file, 'r') as fin:
        for l in fin:
            inp, gold, pred = l.rstrip('\n').split('\t')
            examples.append((inp, gold, pred))
    return examples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True, help='prediction file')
    parser.add_argument('--pred_file2', type=str, required=False, default=None, help='another prediction file')
    parser.add_argument('--metric', type=str, default='sacrebleu', help='metrics to report')
    args = parser.parse_args()

    # set random seed to make sure the same examples are sampled across multiple runs
    random.seed(2022)

    metric = evaluate.load(args.metric)
    examples = load_pred_file(args.pred_file)
    examples2 = load_pred_file(args.pred_file2 or args.pred_file)

    # overall performance
    perf = metric.compute(
        predictions=[e[-1] for e in examples], 
        references=[[e[-2]] for e in examples], 
        tokenize='zh')
    print(perf)
