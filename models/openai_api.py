from typing import List, Dict, Any, Tuple
from operator import itemgetter
import argparse
import random
from collections import namedtuple
import numpy as np
import logging
from tqdm import tqdm
import os
import time
import json
import copy
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search
import openai
from .retriever import BM25

logging.basicConfig(level=logging.INFO)

class Prompt:
    def __init__(self, demo: List[str] = [], ctx: str = '', case: str = ''):
        self.demo = demo
        self.ctx = ctx
        self.case = case

    @classmethod
    def from_dict(cls, adict):
        return cls(**{k: adict[k] for k in ['demo', 'ctx', 'case'] if k in adict})

    def __str__(self):
        return '\n\n'.join(self.demo) + '\n\n' + 'Evidence: ' + self.ctx + '\n' + self.case

    def format(self, use_ctx: bool = False):
        if use_ctx:
            return '\n\n'.join(self.demo) + '\n\n' + 'Evidence: ' + self.ctx + '\n' + self.case
        return '\n\n'.join(self.demo) + '\n\n' + self.case

class QueryAgent:
    def __init__(
        self,
        model: str = 'code-davinci-002',
        retrieval_kwargs: Dict[str, Any] = {},
    ):
        self.model = model

        # generation args
        self.final_stop_sym = '\n'
        self.max_generation_len = 128
        self.temperature = 0
        assert self.temperature == 0, f'do not support sampling'
        self.top_p = 0
        self.boundary_len = 1
        assert len(self.final_stop_sym) == self.boundary_len, f'do not support boundary with {self.boundary_len} chars'

        # retrieval args
        self.retriever = retrieval_kwargs.get('retriever', None)
        self.ret_frequency = retrieval_kwargs.get('frequency', 0)
        self.ret_boundary = retrieval_kwargs.get('boundary', [])
        self.use_gold = retrieval_kwargs.get('use_gold', False)
        if self.ret_boundary:  # otherwise cannot decide when to finally stop
            assert self.final_stop_sym not in self.ret_boundary
            for rb in self.ret_boundary:
                assert len(rb) == self.boundary_len, f'do not support boundary with {self.boundary_len} chars'

        self.look_ahead_steps = retrieval_kwargs.get('look_ahead_steps', 0)
        self.look_ahead_boundary = retrieval_kwargs.get('look_ahead_boundary', 0)
        self.max_query_length = retrieval_kwargs.get('max_query_length', None)
        self.only_use_look_ahead = retrieval_kwargs.get('only_use_look_ahead', False)

        self.ret_topk = retrieval_kwargs.get('topk', 1)

        self.retrieval_at_beginning = retrieval_kwargs.get('retrieval_at_beginning', False)
        if self.retrieval_at_beginning:
            self.ret_frequency = self.max_generation_len
            self.ret_boundary = []

    @property
    def use_retrieval(self):
        return self.ret_frequency > 0 or self.ret_boundary or self.use_gold

    def complete(
        self,
        queries: List[str],
        params: Dict[str, Any],
    ) -> List[Tuple[str, str]]:
        tosleep = 3
        expbf = 1.5
        while True:
            try:
                responses = openai.Completion.create(
                    model=self.model,
                    prompt=queries,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    **params)
                generations = [(r['text'], r['finish_reason']) for r in responses['choices']]
                break
            except openai.error.RateLimitError:  # TODO: make it exponential?
                logging.info(f'sleep {tosleep}')
                time.sleep(tosleep)
                tosleep = tosleep ** expbf
        return generations

    def prompt(
        self,
        queries: List[Prompt],
    ):
        if self.use_retrieval:
            if self.use_gold:  # directly generate all with gold context
                outputs = list(map(itemgetter(0), self.complete(
                    [q.format(use_ctx=True) for q in queries],
                    params={'max_tokens': self.max_generation_len, 'stop': self.final_stop_sym})))
                return outputs, None
            else:
                return self.ret_prompt(queries)
        else:  # directly generate all without gold context
            outputs = list(map(itemgetter(0), self.complete(
                [q.format(use_ctx=False) for q in queries],
                params={'max_tokens': self.max_generation_len, 'stop': self.final_stop_sym})))
            return outputs, None

    def ret_prompt(
        self,
        queries: List[Prompt],
    ):
        batch_size = len(queries)
        final_retrievals: List[List[List[str]]] = [[] for _ in range(len(queries))]  # (bs, n_ret_steps, ret_topk)
        final_outputs: List[str] = [''] * len(queries)
        queries: List[Tuple[int, Prompt]] = [(i, q) for i, q in enumerate(queries)]  # to query
        max_gen_len = 0

        while len(queries) and max_gen_len < self.max_generation_len:
            # retrieve
            look_aheads: List[str] = [''] * len(queries)
            if self.look_ahead_steps:  # generate a fixed number tokens for retrieval
                continues = self.complete(
                    [q.format(use_ctx=True) for i, q in queries],
                    params={'max_tokens': self.look_ahead, 'stop': self.final_stop_sym})
                look_aheads = [cont for cont, _ in continues]
            elif self.look_ahead_boundary:  # generate tokens until boundary for retrieval
                continues = self.complete(
                    [q.format(use_ctx=True) for i, q in queries],
                    params={'max_tokens': self.max_generation_len, 'stop': self.look_ahead_boundary})
                look_aheads = [cont for cont, _ in continues]
            assert len(look_aheads) == len(queries)

            # (bs, ret_topk) * 2
            ctx_ids, ctx_texts = self.retriever.retrieve_and_prepare(
                decoder_texts=[
                    (q.case + lh) if self.only_use_look_ahead else q.case
                    for (i, q), lh in zip(queries, look_aheads)],
                topk=self.ret_topk,
                max_query_length=self.max_query_length)
            for _i, (i, q) in enumerate(queries):
                final_retrievals[i].append(ctx_ids[_i].tolist())
                assert self.ret_topk == 1
                q.ctx = ctx_texts[_i][0]

            # complete
            if self.ret_frequency:
                continues = self.complete(
                    [q.format(use_ctx=True) for i, q in queries],
                    params={'max_tokens': self.ret_frequency, 'stop': self.final_stop_sym})
                max_gen_len += self.ret_frequency
            elif self.ret_boundary:
                continues = self.complete(
                    [q.format(use_ctx=True) for i, q in queries],
                    params={'max_tokens': self.max_generation_len - max_gen_len, 'stop': self.ret_boundary})
                # used to collect the generation with ret_boundary
                min_cont_len = 100000
                for i, (cont, reason) in enumerate(continues):
                    if reason == 'stop' and self.final_stop_sym not in cont:  # fake stop
                        assert len(self.ret_boundary) == 1
                        cont += self.ret_boundary[0]
                        reason = 'boundary'
                        assert len(cont) > 0, 'empty generation will cause dead lock'
                    if self.final_stop_sym in cont:
                        cont = cont.split(self.final_stop_sym, 1)[0]
                        reason = 'stop'
                    if len(cont) <= 0 and self.max_generation_len - max_gen_len > 0:  # TODO: why empty string are generated?
                        cont += ' '
                    continues[i] = (cont, reason)
                    min_cont_len = min(min_cont_len, len(cont.split()))  # TODO: split is not the same tokenization
                max_gen_len += min_cont_len
            else:
                raise NotImplementedError

            # decide whether to continue
            new_queries = []
            for (i, query), (cont, reason) in zip(queries, continues):
                final_outputs[i] += cont
                if reason == 'stop':
                    pass
                elif reason in {'length', 'boundary'}:
                    query.case += cont
                    new_queries.append((i, query))
                else:
                    raise ValueError
            queries = new_queries
        return final_outputs, final_retrievals

class BaseDataset:
    def format(
        self,
        fewshot: int = 0,
    ):
        def _format(
            example: Dict,
            use_answer: bool = False,
        ):
            q = example['question']
            a = example['answer']

            query = f'Q: {q}'
            if use_answer:
                query = f'{query}\nA: {a}'
            else:
                query = f'{query}\nA:'
            return query

        # demo
        demo = [_format(self.examplers[i], use_answer=True) for i in range(fewshot)] if fewshot else []

        def _format_for_dataset(example):
            # case
            case = _format(example, use_answer=False)
            # ctx
            ctx = ' '.join(example['references']) if 'references' in example else None
            example['demo'] = demo
            example['ctx'] = ctx
            example['case'] = case
            return example
        self.dataset = self.dataset.map(_format_for_dataset)

class StrategyQA(BaseDataset):
    examplers: List[Dict] = [
        {
            'question': 'Do hamsters provide food for any animals?',
            'answer': 'Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is yes.',
        },
        {
            'question': 'Could Brooke Shields succeed at University of Pennsylvania?',
            'answer': 'Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is yes.',
        },
        {
            'question': "Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?",
            'answer': "Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5. So the answer is no.",
        },
        {
            'question': "Yes or no: Is it common to see frost during some college commencements?",
            'answer': "College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is yes.",
        },
        {
            'question': "Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?",
            'answer': "The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is no.",
        },
        {
            'question': "Yes or no: Would a pear sink in water?",
            'answer': "The density of a pear is about 0.6g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is no.",
        }
    ]

    def __init__(self, beir_dir: str):
        self.dataset = self.load_data(beir_dir)

    def load_data(self, beir_dir: str):
        query_file = os.path.join(beir_dir, 'queries.jsonl')
        corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split='dev')
        dataset = []
        with open(query_file, 'r') as fin:
            for l in fin:
                example = json.loads(l)
                qid = example['_id']
                rel_dids = [did for did, rel in qrels[qid].items() if rel]
                rel_docs = [corpus[did]['text'] for did in rel_dids]
                dataset.append({
                    'qid': qid,
                    'question': example['text'],
                    'answer': example['metadata']['answer'],
                    'references': rel_docs,
                })
        return Dataset.from_list(dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='strategyqa', choices=['strategyqa'])
    parser.add_argument('--model', type=str, default='code-davinci-002', choices=['code-davinci-002', 'text-davinci-002'])
    parser.add_argument('--input', type=str, default='.')
    parser.add_argument('--output', type=str, default='.')
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_num_examples', type=int, default=None)
    parser.add_argument('--fewshot', type=int, default=6)

    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load data
    if args.dataset == 'strategyqa':
        data = StrategyQA(args.input)
        data.format(fewshot=args.fewshot)
    else:
        raise NotImplementedError
    data = data.dataset

    # downsample
    if args.max_num_examples and args.max_num_examples < len(data):
        data = data.shuffle()
        data = data.select(range(args.max_num_examples))
    if args.num_shards > 1:
        shard_size = int(np.ceil(len(data) / args.num_shards))
        data_from = args.shard_id * shard_size
        data_to = min((args.shard_id + 1) * shard_size, len(data))
        data = data.select(range(data_from, data_to))
    logging.info(f'#examples {len(data)}, shard {args.shard_id} / {args.num_shards}')
    logging.info(f'first example: {data[0]}')

    # load retrieval corpus and index
    index_name = 'test'
    corpus, queries, qrels = GenericDataLoader(data_folder=args.input).load(split='dev')
    BM25Search(index_name=index_name, hostname='localhost', initialize=True, number_of_shards=1).index(corpus)
    time.sleep(5)

    # retrieval kwargs
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
    retriever = BM25(
        tokenizer=tokenizer,
        dataset=(corpus, queries, qrels),
        index_name=index_name,
        use_decoder_input_ids=True)
    retrieval_kwargs = {
        'retriever': retriever,
        'topk': 1,
        'frequency': 0,
        'boundary': ['.'],
        'use_gold': False,
        'max_query_length': 16,
        'retrieval_at_beginning': False,
        'look_ahead_steps': 0,
        'look_ahead_boundary': ['.', '\n'],
        'only_use_look_ahead': True,
    }
    qagent = QueryAgent(model=args.model, retrieval_kwargs=retrieval_kwargs)

    # query
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with tqdm(total=len(data)) as pbar, open(args.output, 'w') as fout:
        for b in range(0, len(data), args.batch_size):
            batch = data.select(range(b, min(b + args.batch_size, len(data))))
            prompts = [Prompt.from_dict(example) for example in batch]
            generations, retrievals = qagent.prompt(prompts)
            retrievals = retrievals or [None] * len(generations)
            for example, generation, retrieval in zip(batch, generations, retrievals):
                example['output'] = generation
                example['retrieval'] = retrieval
                fout.write(json.dumps(example) + '\n')
            pbar.update(len(batch))
