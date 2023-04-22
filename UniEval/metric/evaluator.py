from typing import List
import sys
import torch
import numpy as np
from nltk import sent_tokenize
from multiprocessing.pool import Pool
from metric.scorer import UniEvaluator
sys.path.append("..")
from utils import add_question, print_scores


def multi_gpu(
    evaluate_func,
    cuda_devices: List[int] = None
):
    def wrapper(*args, **kwargs):
        devices = [torch.device(f'cuda:{i}') for i in cuda_devices] if cuda_devices else [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        num_shards = len(devices)
        self = args[0]
        data = args[1]
        shard_size = round(len(data) / num_shards)
        all_result = []
        with Pool(num_shards) as pool:
            shards = [data[i * shard_size:i * shard_size + shard_size] for i in range(num_shards)]
            # only support *args
            for result in pool.map(evaluate_func, [(self, shard, devices[i]) + tuple(args[2:]) for i, shard in enumerate(shards)]):
                all_result.extend(result)
        print_scores(all_result)
        return all_result
    return wrapper


class SumEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for text summarization """
        self.max_length = max_length
        self.device = device
        self.cache_dir = cache_dir
        self.task = 'summarization'
        self.dimensions = ['coherence', 'consistency', 'fluency', 'relevance']

    def evaluate(self, data, device, dims=None, overall=True, print_result=False):
        """
            Get the scores of all the given dimensions

            dims: A list of dimensions to be evaluated. If dims is None, SumEvaluator will evaluate
                  four dimensions: coherence, consistency, fluency, relevance.

            overall: indicates whether the overall score is to be calculated.
                     Overall score can be customized to a combination of scores based on different
                     dimensions. The default here is the average score of all the given dimensions.

            print_result: whether to print the average score of each dimension on the screen
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        if dims == None:
            eval_dims = self.dimensions
        else:
            assert isinstance(dims, list)
            eval_dims = dims

        scorer = UniEvaluator(
            model_name_or_path='MingZhong/unieval-sum',
            max_length=self.max_length,
            device=device,
            cache_dir=self.cache_dir)

        for dim in eval_dims:
            print('Evaluating {} of {} samples !!!'.format(dim, n_data))

            # Calculate average sentence-level scores for 'consistency' and 'fluency'
            if dim == 'consistency' or dim == 'fluency':
                src_list, output_list = [], []
                n_sents = [] # the number of sentences in each generated summary
                for i in range(n_data):
                    source = data[i]['source']
                    system_outputs = sent_tokenize(data[i]['system_output'])
                    n_sents.append(len(system_outputs))
                    for j in range(len(system_outputs)):
                        src_list.append(source)
                        output_list.append(system_outputs[j])
                input_list = add_question(dimension=dim, output=output_list,
                                          src=src_list, task=self.task)
                sent_score = scorer.score(input_list)

                # Get average score for each sample
                start_idx = 0
                score = []
                for cur_n_sent in n_sents:
                    score.append(sum(sent_score[start_idx: start_idx + cur_n_sent]) / cur_n_sent)
                    start_idx += cur_n_sent

            # Calculate summary-level score for 'coherence' and 'relevance'
            elif dim == 'coherence' or dim == 'relevance':
                src_list, output_list, ref_list = [], [], []
                for i in range(n_data):
                    src_list.append(data[i]['source'])
                    output_list.append(data[i]['system_output'])
                    if dim == 'relevance':
                        ref_list.append(data[i]['reference'])
                input_list = add_question(dimension=dim, output=output_list,
                                          src=src_list, ref=ref_list, task=self.task)
                score = scorer.score(input_list)

            # Please customize other dimensions here for summarization
            else:
                raise NotImplementedError('The input format for this dimension is still undefined. \
                                           Please customize it first.')

            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        # Customize your overall score here.
        if overall == True:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores


class DialogEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for dialogues """
        self.max_length = max_length
        self.device = device
        self.cache_dir = cache_dir
        self.task = 'dialogue'
        self.dimensions = ['naturalness', 'coherence', 'engagingness',
                           'groundedness', 'understandability']

    def evaluate(self, data, device, dims=None, overall=True, print_result=False):
        """
            Get the scores of all the given dimensions

            dims: A list of dimensions to be evaluated. If dims is None, DialogEvaluator will evaluate
                  five dimensions: naturalness, coherence, engagingness, groundedness and understandability.

            overall: indicates whether the overall score is to be calculated.
                     Overall score can be customized to a combination of scores based on different
                     dimensions. The default here is the average score of all the given dimensions.

            print_result: whether to print the average score of each dimension on the screen
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        if dims == None:
            eval_dims = self.dimensions
        else:
            assert isinstance(dims, list)
            eval_dims = dims

        scorer = UniEvaluator(
            model_name_or_path='MingZhong/unieval-dialog',
            max_length=self.max_length,
            device=device,
            cache_dir=self.cache_dir)

        for dim in eval_dims:
            print('Evaluating {} of {} samples !!!'.format(dim, n_data))

            # Calculate summation score for 'engagingness'
            if dim == 'engagingness':
                src_list, output_list, context_list = [], [], []
                n_sents = [] # the number of sentences in each generated response
                for i in range(n_data):
                    source = data[i]['source']
                    context = data[i]['context']
                    system_outputs = sent_tokenize(data[i]['system_output'])
                    n_sents.append(len(system_outputs))
                    for j in range(len(system_outputs)):
                        src_list.append(source)
                        context_list.append(context)
                        output_list.append(system_outputs[j])
                input_list = add_question(dimension=dim, output=output_list,
                                          src=src_list, context=context_list, task=self.task)
                sent_score = scorer.score(input_list)

                # Get the summation score for each sample
                start_idx = 0
                score = []
                for cur_n_sent in n_sents:
                    score.append(sum(sent_score[start_idx: start_idx + cur_n_sent]))
                    start_idx += cur_n_sent

            # Calculate turn-level score for other dimensions
            elif dim in ['naturalness', 'coherence', 'groundedness', 'understandability']:
                src_list, output_list, context_list = [], [], []
                for i in range(n_data):
                    src_list.append(data[i]['source'])
                    output_list.append(data[i]['system_output'])
                    context_list.append(data[i]['context'])
                input_list = add_question(dimension=dim, output=output_list,
                                          src=src_list, context=context_list, task=self.task)
                score = scorer.score(input_list)

            # Please customize other dimensions here for summarization
            else:
                raise NotImplementedError('The input format for this dimension is still undefined. \
                                           Please customize it first.')

            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        # Customize your overall score here.
        if overall == True:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores


class D2tEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for data-to-text """
        self.max_length = max_length
        self.device = device
        self.cache_dir = cache_dir
        self.task = 'data2text'
        self.dimensions = ['naturalness', 'informativeness']

    def evaluate(self, data, device, dims=None, overall=True, print_result=False):
        """
            Get the scores of all the given dimensions

            dims: A list of dimensions to be evaluated. If dims is None, D2tEvaluator will evaluate
                  two dimensions: naturalness and informativeness.

            overall: indicates whether the overall score is to be calculated.
                     Overall score can be customized to a combination of scores based on different
                     dimensions. The default here is the average score of all the given dimensions.

            print_result: whether to print the average score of each dimension on the screen
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        if dims == None:
            eval_dims = self.dimensions
        else:
            assert isinstance(dims, list)
            eval_dims = dims

        scorer = UniEvaluator(
            model_name_or_path='MingZhong/unieval-sum',
            max_length=self.max_length,
            device=device,
            cache_dir=self.cache_dir)

        for dim in eval_dims:
            print('Evaluating {} of {} samples !!!'.format(dim, n_data))

            output_list, ref_list = [], []
            for i in range(n_data):
                output_list.append(data[i]['system_output'])
                ref_list.append(data[i]['reference'])

            input_list = add_question(dimension=dim, output=output_list,
                                      ref=ref_list, task=self.task)
            score = scorer.score(input_list)

            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        # Customize your overall score here.
        if overall == True:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores


class FactEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for factual consistency detection """
        self.max_length = max_length
        self.device = device
        self.cache_dir = cache_dir
        self.task = 'fact'
        self.dim = 'consistency'

    def evaluate(self, data, device, print_result=False):
        """
            Get the factual consistency score (only 1 dimension for this task)

            print_result: whether to print the average factual score on the screen
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        print('Evaluating {} of {} samples !!!'.format(self.dim, n_data))

        # Calculate average sentence-level scores for facutal consistency
        src_list, output_list = [], []
        n_sents = [] # the number of sentences in the claim
        for i in range(n_data):
            source = data[i]['source']
            system_outputs = sent_tokenize(data[i]['system_output'])
            n_sents.append(len(system_outputs))
            for j in range(len(system_outputs)):
                src_list.append(source)
                output_list.append(system_outputs[j])
        input_list = add_question(dimension=self.dim, output=output_list,
                                    src=src_list, task=self.task)

        scorer = UniEvaluator(
            model_name_or_path='MingZhong/unieval-fact',
            max_length=self.max_length,
            device=device,
            cache_dir=self.cache_dir)
        sent_score = scorer.score(input_list)

        # Get average score for each sample
        start_idx = 0
        score = []
        for cur_n_sent in n_sents:
            score.append(sum(sent_score[start_idx: start_idx + cur_n_sent]) / cur_n_sent)
            start_idx += cur_n_sent

        for i in range(n_data):
            eval_scores[i][self.dim] = score[i]

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores


def evaluate(all_arg):
    self = all_arg[0]
    return getattr(self, 'evaluate')(*all_arg[1:])


def get_evaluator(task, max_length=1024, device='cuda:0', cache_dir=None):
    assert task in ['summarization', 'dialogue', 'data2text', 'fact']
    if task == 'summarization':
        return SumEvaluator(max_length=max_length,
                            device=device,
                            cache_dir=cache_dir)
    elif task == 'dialogue':
        return DialogEvaluator(max_length=max_length,
                               device=device,
                               cache_dir=cache_dir)
    elif task == 'data2text':
        return D2tEvaluator(max_length=max_length,
                            device=device,
                            cache_dir=cache_dir)
    elif task == 'fact':
        return FactEvaluator(max_length=max_length,
                             device=device,
                             cache_dir=cache_dir)
    else:
        raise NotImplementedError('Other tasks are not implemented, \
                                   please customize specific tasks here.')
