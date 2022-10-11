from typing import List, Tuple, Dict, Any
import argparse
import random
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
import evaluate

metric_key_map: Dict[str, Dict[str, str]] = {
    'rouge': None,  # no mapping
    'sacrebleu': {
        'score': 'sacrebleu',
        'sys_len': 'sys_len',
        'ref_len': 'ref_len'
    },
}

def load_pred_file(
    pred_file: str, 
    dedup: bool = False,  # keep the first instance among those with the same source
    remove_prediction_prefix: str = 'Answer:',
    remove_repetition: bool = True,
    debug: bool = False,
    ) -> List[Tuple[str, str, str]]:  # source, target, pred
    examples: List[Tuple[str, str, str]] = []
    with open(pred_file, 'r') as fin:
        prev_source = None
        for l in fin:
            items = l.rstrip('\n').split('\t')
            source, target, pred = items[:3]
            if dedup and prev_source == source:  # TODO: use target if evidence is included in source
                continue
            prev_source = source
            prefix = (items[3] if len(items) >= 4 else '').strip()
            if not pred.startswith(prefix):
                print('a case where prediction does not start with prefix')
                #print(f'prediction "{pred}" should start with the prefix "{prefix}"')
                continue
            pred = pred[len(prefix):].strip()
            if remove_prediction_prefix and pred.startswith(remove_prediction_prefix):
                pred = pred[len(remove_prediction_prefix):].strip()
            if remove_prediction_prefix and remove_repetition:
                pred = pred.split(remove_prediction_prefix, 1)[0].strip()
            examples.append((source, target, pred))
            if debug:
                print('SOURCE\t', source)
                print('TARGET\t', target)
                print('PRED\t', pred)
                input()
    return examples

class EvalWrapper(object):
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
    
    def evaluate(
        self, 
        sources: List[str], 
        targets: List[str], 
        predictions: List[str]):
        source2index: Dict[str, int] = defaultdict(lambda: len(source2index))
        metric2scores: Dict[str, List[float]] = defaultdict(list)
        for metric in self.metrics:
            metric_func = evaluate.load(metric)
            for s, t, p in tqdm(zip(sources, targets, predictions)):
                metric2scores['index'] = source2index[s]
                score = metric_func.compute(predictions=[p], references=[t])
                if metric_key_map[metric]:
                    score = {new_key: score[old_key] for old_key, new_key in metric_key_map[metric].items()}
                for k, v in score.items():
                    metric2scores[k].append(v)
        metric2score: Dict[str, float] = {m: np.mean(vs) for m, vs in metric2scores.items()}
        return metric2score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True, help='prediction file')
    parser.add_argument('--pred_file2', type=str, required=False, default=None, help='another prediction file')
    parser.add_argument('--tokenizer', type=str, default=None, choices=[None, 'zh'], help='tokenizer used in metrics')
    parser.add_argument('--metrics', type=str, default=['rouge', 'sacrebleu'], choices=['rouge', 'sacrebleu'], nargs='+', help='metrics to report')
    parser.add_argument('--aggregate', type=str, default=None, choices=[None, 'mean', 'max'], help='how to aggregate mutiple examples of the same source')
    parser.add_argument('--compare', action='store_true', help='compare the predictions from pred_file and pred_file2')
    args = parser.parse_args()

    # set random seed to make sure the same examples are sampled across multiple runs
    random.seed(2022)

    dedup = args.aggregate is None  # only keep one example with the same source
    examples = load_pred_file(args.pred_file, dedup=dedup)
    examples2 = load_pred_file(args.pred_file2 or args.pred_file, dedup=dedup)
    sources, targets, predictions = list(zip(*examples))
    targets = [e[1] for e in examples2][:len(sources)]

    '''
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/t5-xl-lm-adapt')
    inds = [i for i, t in enumerate(targets) if len(tokenizer.tokenize(t)) >= 70]
    sources = [sources[i] for i in inds]
    targets = [targets[i] for i in inds]
    predictions = [predictions[i] for i in inds]
    '''
    
    print(f'#total examples {len(predictions)}')
    '''
    ew = EvalWrapper(args.metrics)
    metric2score = ew.evaluate(sources, targets, predictions)
    print('\t'.join(metric2score.keys()))
    print('\t'.join(map(str, metric2score.values())))
    '''
    metric_keys: List[str] = []
    metric_vals: List[str] = []
    for metric in args.metrics:
        metric_func = evaluate.load(metric)
        if args.aggregate is None:
            print(f'#real examples {len(predictions)}')
            perf = metric_func.compute(predictions=predictions, references=targets)
        else:  # use custom aggregation function
            perf = metric_func.compute(predictions=predictions, references=targets, use_aggregator=False)
            source2index: Dict[str, int] = defaultdict(lambda: len(source2index))
            perf['_index'] = [source2index[s] for s in sources]  # TODO: use target if evidence is included in source
            perf = getattr(pd.DataFrame(perf).groupby('_index'), args.aggregate)()
            print(f'#real examples {len(perf)}')
            perf = perf.mean().to_dict()
        if metric_key_map[metric]:
            perf = {new_key: perf[old_key] for old_key, new_key in metric_key_map[metric].items()}
        metric_keys.extend(perf.keys())
        metric_vals.extend(perf.values())
    print('\t'.join(metric_keys))
    print('\t'.join(map(str, metric_vals)))

    if args.compare:
        for (s1, t1, p1), (s2, t2, p2) in zip(examples, examples2):
            assert t1 == t2
            print('Q:', s1)
            print('A:', t1)
            print('->', p1)
            print('->', p2)
            input()
