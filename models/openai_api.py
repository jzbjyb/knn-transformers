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

class CtxPrompt:
    ctx_position = 'before_case'

    def __init__(self, demo: List["CtxPrompt"] = [], ctx: str = None, case: str = None):
        self.demo = demo
        self.ctx = ctx
        self.case = case

    @classmethod
    def from_dict(cls, adict):
        adict = dict(adict)
        if 'demo' in adict:
            adict['demo'] = [cls.from_dict(d) for d in adict['demo']]
        return cls(**{k: adict[k] for k in ['demo', 'ctx', 'case'] if k in adict})

    def format(self, use_ctx: bool = False):
        demo_formatted: List[str] = [d.format(use_ctx=use_ctx) for d in self.demo]
        use_ctx = use_ctx and self.ctx
        if use_ctx:
            if self.ctx_position == 'begin':
                if len(demo_formatted):
                    return 'Evidence: ' + self.ctx + '\n\n' + '\n\n'.join(demo_formatted) + '\n\n' + self.case
                else:
                    return 'Evidence: ' + self.ctx + '\n' + self.case
            elif self.ctx_position == 'before_case':
                if len(demo_formatted):
                    return '\n\n'.join(demo_formatted) + '\n\n' + 'Evidence: ' + self.ctx + '\n' + self.case
                else:
                    return 'Evidence: ' + self.ctx + '\n' + self.case
            else:
                raise NotImplementedError
        if len(demo_formatted):
            return '\n\n'.join(demo_formatted) + '\n\n' + self.case
        else:
            return self.case

class QueryAgent:
    def __init__(
        self,
        model: str = 'code-davinci-002',
        max_generation_len: int = 128,
        retrieval_kwargs: Dict[str, Any] = {},
    ):
        self.model = model

        # generation args
        self.final_stop_sym = 'Question:'
        self.max_generation_len = max_generation_len
        self.temperature = 0
        assert self.temperature == 0, f'do not support sampling'
        self.top_p = 1

        # retrieval args
        self.retriever = retrieval_kwargs.get('retriever', None)
        self.ret_frequency = retrieval_kwargs.get('frequency', 0)
        self.ret_boundary = retrieval_kwargs.get('boundary', [])
        self.use_gold = retrieval_kwargs.get('use_gold', False)
        if self.ret_boundary:  # otherwise cannot decide when to finally stop
            assert self.final_stop_sym not in self.ret_boundary

        self.look_ahead_steps = retrieval_kwargs.get('look_ahead_steps', 0)
        self.look_ahead_boundary = retrieval_kwargs.get('look_ahead_boundary', 0)
        self.max_query_length = retrieval_kwargs.get('max_query_length', None)
        self.only_use_look_ahead = retrieval_kwargs.get('only_use_look_ahead', False)
        self.retrieval_trigers = retrieval_kwargs.get('retrieval_trigers', [])
        for rts, rte in self.retrieval_trigers:
            assert rte in self.ret_boundary, 'end of retrieval trigers must be used as boundary'

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
        if 'max_tokens' in params:  # TODO: opt doesn't have this bug
            params['max_tokens'] = max(2, params['max_tokens'])  # openai returns nothing if set to 1
        tosleep = 3
        expbf = 1.5
        while True:
            try:
                responses = openai.Completion.create(
                    model=self.model,
                    prompt=queries,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    logprobs=0,
                    **params)
                if 'openai' in openai.api_base:
                    # GPT3 api
                    generations = [(r['text'], r['finish_reason']) for r in responses['choices']]
                else:
                    # OPT API
                    generations = [(r['text'], r['logprobs']['finish_reason']) for r in responses['choices']]
                break
            except openai.error.RateLimitError:
                logging.info(f'sleep {tosleep}')
                time.sleep(tosleep)
                tosleep = tosleep * expbf
        return generations

    def prompt(
        self,
        queries: List[CtxPrompt],
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
        queries: List[CtxPrompt],
    ):
        batch_size = len(queries)
        final_retrievals: List[List[List[str]]] = [[] for _ in range(len(queries))]  # (bs, n_ret_steps, ret_topk)
        final_outputs: List[str] = [''] * len(queries)
        queries: List[Tuple[int, CtxPrompt]] = [(i, q) for i, q in enumerate(queries)]  # to query
        max_gen_len = 0

        generate_queries: List[str] = []
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

            # send queries to index
            if generate_queries:  # some queries might be None which means no queries are generated
                queries_to_issue = [gq for gq in generate_queries if gq]
            else:
                queries_to_issue = [(q.case + lh) if self.only_use_look_ahead else q.case
                    for (i, q), lh in zip(queries, look_aheads)]
            if queries_to_issue:
                # (bs, ret_topk) * 2
                ctx_ids, ctx_texts = self.retriever.retrieve_and_prepare(
                    decoder_texts=queries_to_issue,
                    topk=self.ret_topk,
                    max_query_length=self.max_query_length)
                idx = -1
                for _i, (i, q) in enumerate(queries):
                    assert self.ret_topk == 1
                    if generate_queries:
                        if generate_queries[_i]:
                            idx += 1
                            ret_id, ret_text = ctx_ids[idx].tolist(), ctx_texts[idx][0]
                            final_retrievals[i].append(ret_id)
                            q.ctx = ret_text
                    else:
                        ret_id, ret_text = ctx_ids[_i].tolist(), ctx_texts[_i][0]
                        final_retrievals[i].append(ret_id)
                        q.ctx = ret_text
            generate_queries = []

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
                    if reason == 'stop' and self.final_stop_sym not in cont:  # stop at ret_boundary
                        assert len(self.ret_boundary) == 1
                        cont += self.ret_boundary[0]
                        reason = 'boundary'
                        assert len(cont) > 0, 'empty generation will cause dead lock'
                        if self.retrieval_trigers:  # extract queries from generation
                            assert len(self.retrieval_trigers) == 1
                            # TODO: check if it stops at retrieval trigers
                            ret_tri_start = self.retrieval_trigers[0][0]
                            ret_start = cont.find(ret_tri_start)
                            if ret_start != -1:
                                generate_queries.append(cont[ret_start + len(ret_tri_start):].strip())
                            else:
                                generate_queries.append(None)
                    else:
                        if self.retrieval_trigers:
                            generate_queries.append(None)
                    if self.final_stop_sym in cont:
                        cont = cont.split(self.final_stop_sym, 1)[0]
                        reason = 'stop'
                    continues[i] = (cont, reason)
                    min_cont_len = min(min_cont_len, len(cont.split()))  # TODO: split is not the same tokenization
                max_gen_len += min_cont_len
            else:
                raise NotImplementedError

            # decide whether to continue
            new_queries = []
            new_generate_queries = []
            assert len(queries) == len(continues)
            if self.retrieval_trigers:
                assert len(queries) == len(generate_queries)
            for _i, ((i, query), (cont, reason)) in enumerate(zip(queries, continues)):
                final_outputs[i] += cont
                if reason == 'stop':
                    pass
                elif reason in {'length', 'boundary'}:
                    query.case += cont
                    new_queries.append((i, query))
                    new_generate_queries.append(generate_queries[_i])
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
            cot = example['cot']
            a = example['answer']

            query = self.input_template(q)
            if use_answer:
                query += self.output_template(cot, a)
            return query

        # demo
        demo = [{
            'case': _format(self.examplers[i], use_answer=True),
            'ctx': ' '.join(self.examplers[i]['ctx']) if 'ctx' in self.examplers[i] else None,
        } for i in range(fewshot)] if fewshot else []

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
    cot_examplers: List[Dict] = [
        {
            'question': 'Do hamsters provide food for any animals?',
            'cot': ('Hamsters are prey animals. '
                'Prey are food for predators. '
                'Thus, hamsters provide food for some animals.'),
            'answer': 'yes',
        },
        {
            'question': 'Could Brooke Shields succeed at University of Pennsylvania?',
            'cot': ('Brooke Shields went to Princeton University. '
                'Princeton University is about as academically rigorous as the University of Pennsylvania. '
                'Thus, Brooke Shields could also succeed at the University of Pennsylvania.'),
            'answer': 'yes',
        },
        {
            'question': "Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?",
            'cot': ("Hydrogen has an atomic number of 1. "
                "1 squared is 1. "
                "There are 5 Spice Girls. "
                "Thus, Hydrogen's atomic number squared is less than 5."),
            'answer': 'no',
        },
        {
            'question': "Yes or no: Is it common to see frost during some college commencements?",
            'cot': ("College commencement ceremonies can happen in December, May, and June. "
                "December is in the winter, so there can be frost. "
                "Thus, there could be frost at some commencements."),
            'answer': 'yes',
        },
        {
            'question': "Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?",
            'cot': ("The War in Vietnam was 6 months. "
                "The gestation period for a llama is 11 months, which is more than 6 months. "
                "Thus, a llama could not give birth twice during the War in Vietnam."),
            'answer': 'no',
        },
        {
            'question': "Yes or no: Would a pear sink in water?",
            'cot': ("The density of a pear is about 0.6g/cm^3, which is less than water. "
                "Objects less dense than water float. "
                "Thus, a pear would float."),
            'answer': 'no',
        }
    ]
    cot_input_template = lambda self, ques: f'Q: {ques}\nA:'
    cot_output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    sa_examplers: List[Dict] = [
        {
            'question': 'Do hamsters provide food for any animals?',
            'cot': ('Follow up: What types of animal are hamsters?\n'
                'Intermediate answer: Hamsters are prey animals.\n'
                'Follow up: Do prey provide food for any other animals?\n'
                'Intermediate answer: Prey are food for predators.'),
            'answer': 'yes',
        },
        {
            'question': 'Could Brooke Shields succeed at University of Pennsylvania?',
            'cot': ('Follow up: What college did Brooke Shields go to?\n'
                'Intermediate answer: Brooke Shields went to Princeton University.\n'
                'Follow up: Out of all colleges in the US, how is Princeton University ranked?\n'
                'Intermediate answer: Princeton is ranked as the number 1 national college by US news.\n'
                'Follow up: Out of all colleges in the US, how is University of Pennsylvania ranked?\n'
                'Intermediate answer: University of Pennsylvania is ranked as number 6 national college by US news.\n'
                'Follow up: Is the ranking of University of Pennsylvania similar to Princeton University?\n'
                'Intermediate answer: Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.'),
            'answer': 'yes',
        },
        {
            'question': "Hydrogen's atomic number squared exceeds number of Spice Girls?",
            'cot': ('Follow up: What is the atomic number of hydrogen?\n'
                'Intermediate answer: Hydrogen has an atomic number of 1.\n'
                'Follow up: How many people are in the Spice Girls band?\n'
                'Intermediate answer: There are 5 Spice Girls.\n'
                'Follow up: Is the square of 1 greater than 5?\n'
                "Intermediate answer: 1 squared is 1. Thus, Hydrogen's atomic number squared is less than 5."),
            'answer': 'no',
        },
        {
            'question': "Is it common to see frost during some college commencements?",
            'cot': ('Follow up: What seasons can you expect see frost?\n'
                'Intermediate answer: Frost usually can be seen in the winter.\n'
                'Follow up: What months do college commencements occur?\n'
                'Intermediate answer: College commencement ceremonies can happen in December, May, and June.\n'
                'Follow up: Do any of December, May, and June occur during winter?\n'
                'Intermediate answer: December is in the winter, so there can be frost. Thus, there could be frost at some commencements.'),
            'answer': 'yes',
        },
        {
            'question': "Could a llama birth twice during War in Vietnam (1945-46)?",
            'cot': ('Follow up: How long did the Vietnam war last?\n'
                'Intermediate answer: The War in Vietnam was 6 months.\n'
                'Follow up: How long is llama gestational period?\n'
                'Intermediate answer: The gestation period for a llama is 11 months.\n'
                'Follow up: What is 2 times 11 months?\n'
                'Intermediate answer: 2 times 11 months is 22 months.\n'
                'Follow up: Is 6 months longer than 22 months?\n'
                'Intermediate answer: 6 months is not longer than 22 months.'),
            'answer': 'no',
        },
        {
            'question': "Would a pear sink in water?",
            'cot': ('Follow up: What is the density of a pear?\n'
                'Intermediate answer: The density of a pear is about 0.59 g/cm^3.\n'
                'Follow up: What is the density of water?\n'
                'Intermediate answer: The density of water is about 1 g/cm^3.\n'
                'Follow up: Is 0.59 g/cm^3 greater than 1 g/cm^3?\n'
                'Intermediate answer: 0.59 g/cm^3 is not greater than 1 g/cm^3? Thus, a pear would float.'),
            'answer': 'no',
        }
    ]
    sa_ctx_examplers: List[Dict] = [
        {
            'question': 'Do hamsters provide food for any animals?',
            'ctx': ["Hamsters are prey animals.",
                "Prey animals provide food for predators."],
            'cot': ('Follow up: What types of animal are hamsters?\n'
                'Hamsters are prey animals.\n'
                'Follow up: Do prey provide food for any other animals?\n'
                'Prey are food for predators.'),
            'answer': 'yes',
        },
        {
            'question': 'Could Brooke Shields succeed at University of Pennsylvania?',
            'ctx': ["Brooke Shields graduated from Princeton University.",
                "Princeton is ranked as the number 1 national college by US news.",
                "University of Pennsylvania is ranked as number 6 national college by US news.",
                "Princeton only admits around 6 percent of applicants as of 2018.",
                "University of Pennsylvania accepts around 9% of applicants as of 2018."],
            'cot': ('Follow up: What college did Brooke Shields go to?\n'
                'Brooke Shields went to Princeton University.\n'
                'Follow up: How is Princeton University ranked?\n'
                'Princeton is ranked as the number 1 national college by US news.\n'
                'Follow up: How is University of Pennsylvania ranked?\n'
                'University of Pennsylvania is ranked as number 6 national college by US news.\n'
                'Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.'),
            'answer': 'yes',
        },
        {
            'question': "Hydrogen's atomic number squared exceeds number of Spice Girls?",
            'ctx': ["Hydrogen is the first element and has an atomic number of one.",
                "To square a number, you multiply it by itself.",
                "The Spice Girls has five members."],
            'cot': ('Follow up: What is the atomic number of hydrogen?\n'
                'Hydrogen has an atomic number of 1.\n'
                'Follow up: How many people are in the Spice Girls band?\n'
                'There are 5 Spice Girls.\n'
                "1 squared is 1. Thus, Hydrogen's atomic number squared is less than 5."),
            'answer': 'no',
        },
        {
            'question': "Is it common to see frost during some college commencements?",
            'ctx': ["College commencement ceremonies often happen during the months of December, May, and sometimes June.",
                "Frost isn't uncommon to see during the month of December, as it is the winter."],
            'cot': ('Follow up: What seasons can you expect see frost?\n'
                'Frost usually can be seen in the winter.\n'
                'Follow up: What months do college commencements occur?\n'
                'College commencement ceremonies can happen in December, May, and June.\n'
                'December is in the winter, so there can be frost. Thus, there could be frost at some commencements.'),
            'answer': 'yes',
        },
        {
            'question': "Could a llama birth twice during War in Vietnam (1945-46)?",
            'ctx': ["The War in Vietnam (1945-46) lasted around 6 months.",
                "The gestation period for a llama is 11 months."],
            'cot': ('Follow up: How long did the Vietnam war last?\n'
                'The War in Vietnam was 6 months.\n'
                'Follow up: How long is llama gestational period?\n'
                'The gestation period for a llama is 11 months.\n'
                '2 times 11 months is 22 months. 6 months is not longer than 22 months.'),
            'answer': 'no',
        },
        {
            'question': "Would a pear sink in water?",
            'ctx': ["The density of a raw pear is about 0.59 g/cm^3.",
                "The density of water is about 1 g/cm^3.",
                "Objects only sink if they are denser than the surrounding fluid."],
            'cot': ('Follow up: What is the density of a pear?\n'
                'The density of a pear is about 0.59 g/cm^3.\n'
                'Follow up: What is the density of water?\n'
                'The density of water is about 1 g/cm^3.\n'
                '0.59 g/cm^3 is not greater than 1 g/cm^3? Thus, a pear would float.'),
            'answer': 'no',
        }
    ]
    sa_input_template = lambda self, ques: f'Question: {ques}\n'
    sa_output_template = lambda self, cot, ans: f'{cot}\nSo the final answer is: {ans}.'
    sa_ctx_input_template = sa_input_template
    sa_ctx_output_template = sa_output_template

    sa_ctx_nofollow_examplers: List[Dict] = [
        {
            'question': 'Do hamsters provide food for any animals?',
            'ctx': ["Hamsters are prey animals.",
                "Prey animals provide food for predators."],
            'cot': (
                'Hamsters are prey animals.\n'
                'Prey are food for predators.'),
            'answer': 'yes',
        },
        {
            'question': 'Could Brooke Shields succeed at University of Pennsylvania?',
            'ctx': ["Brooke Shields graduated from Princeton University.",
                "Princeton is ranked as the number 1 national college by US news.",
                "University of Pennsylvania is ranked as number 6 national college by US news.",
                "Princeton only admits around 6 percent of applicants as of 2018.",
                "University of Pennsylvania accepts around 9% of applicants as of 2018."],
            'cot': (
                'Brooke Shields went to Princeton University.\n'
                'Princeton is ranked as the number 1 national college by US news.\n'
                'University of Pennsylvania is ranked as number 6 national college by US news.\n'
                'Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.'),
            'answer': 'yes',
        },
        {
            'question': "Hydrogen's atomic number squared exceeds number of Spice Girls?",
            'ctx': ["Hydrogen is the first element and has an atomic number of one.",
                "To square a number, you multiply it by itself.",
                "The Spice Girls has five members."],
            'cot': (
                'Hydrogen has an atomic number of 1.\n'
                'There are 5 Spice Girls.\n'
                "1 squared is 1. Thus, Hydrogen's atomic number squared is less than 5."),
            'answer': 'no',
        },
        {
            'question': "Is it common to see frost during some college commencements?",
            'ctx': ["College commencement ceremonies often happen during the months of December, May, and sometimes June.",
                "Frost isn't uncommon to see during the month of December, as it is the winter."],
            'cot': (
                'Frost usually can be seen in the winter.\n'
                'College commencement ceremonies can happen in December, May, and June.\n'
                'December is in the winter, so there can be frost. Thus, there could be frost at some commencements.'),
            'answer': 'yes',
        },
        {
            'question': "Could a llama birth twice during War in Vietnam (1945-46)?",
            'ctx': ["The War in Vietnam (1945-46) lasted around 6 months.",
                "The gestation period for a llama is 11 months."],
            'cot': (
                'The War in Vietnam was 6 months.\n'
                'The gestation period for a llama is 11 months.\n'
                '2 times 11 months is 22 months. 6 months is not longer than 22 months.'),
            'answer': 'no',
        },
        {
            'question': "Would a pear sink in water?",
            'ctx': ["The density of a raw pear is about 0.59 g/cm^3.",
                "The density of water is about 1 g/cm^3.",
                "Objects only sink if they are denser than the surrounding fluid."],
            'cot': (
                'The density of a pear is about 0.59 g/cm^3.\n'
                'The density of water is about 1 g/cm^3.\n'
                '0.59 g/cm^3 is not greater than 1 g/cm^3? Thus, a pear would float.'),
            'answer': 'no',
        }
    ]

    sa_ctx_nofollow_input_template = sa_input_template
    sa_ctx_nofollow_output_template = sa_output_template


    def __init__(self, beir_dir: str, prompt_type: str = 'cot'):
        assert prompt_type in {'cot', 'sa', 'sa_ctx', 'sa_ctx_nofollow'}
        self.input_template = getattr(self, f'{prompt_type}_input_template')
        self.output_template = getattr(self, f'{prompt_type}_output_template')
        self.examplers = getattr(self, f'{prompt_type}_examplers')
        self.dataset = self.load_data(beir_dir)

    def load_data(self, beir_dir: str):
        query_file = os.path.join(beir_dir, 'queries.jsonl')
        corpus, queries, qrels = GenericDataLoader(data_folder=beir_dir).load(split='dev')
        dataset = []
        with open(query_file, 'r') as fin:
            for l in fin:
                example = json.loads(l)
                qid = example['_id']
                question = example['text']
                cot = example['metadata']['cot']
                ans = example['metadata']['answer']
                rel_dids = [did for did, rel in qrels[qid].items() if rel]
                rel_docs = [corpus[did]['text'] for did in rel_dids]
                output = self.output_template(cot, ans)
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'cot': cot,
                    'answer': ans,
                    'gold_output': output,
                    'references': rel_docs,
                })
        return Dataset.from_list(dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='strategyqa', choices=['strategyqa'])
    parser.add_argument('--model', type=str, default='code-davinci-002', choices=['code-davinci-002', 'text-davinci-002', 'text-davinci-003'])
    parser.add_argument('--input', type=str, default='.')
    parser.add_argument('--output', type=str, default='.')
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_num_examples', type=int, default=None)
    parser.add_argument('--fewshot', type=int, default=6)
    parser.add_argument('--max_generation_len', type=int, default=128)

    parser.add_argument('--build_index', action='store_true')
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load data
    if args.dataset == 'strategyqa':
        data = StrategyQA(args.input, prompt_type='sa')
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
    if args.build_index:
        BM25Search(index_name=index_name, hostname='localhost', initialize=True, number_of_shards=1).index(corpus)
        time.sleep(5)
        exit()

    # retrieval kwargs
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
    retriever = BM25(
        tokenizer=tokenizer,
        dataset=(corpus, queries, qrels),
        index_name=index_name,
        use_decoder_input_ids=True)

    # no ret: freq 0, boundary [], use gold false
    # gold ret: freq 0, boundary [], use gold true
    # ret once: freq 0, boundary [], use gold false, retrieval_at_beginning': True
    # ret every 16 tokens: freq 16, boundary [], use gold false

    retrieval_kwargs = {
        'retriever': retriever,
        'topk': 1,
        'frequency': 0,
        'boundary': [],
        'use_gold': False,
        'max_query_length': 16,
        'retrieval_at_beginning': False,
        'look_ahead_steps': 0,
        'look_ahead_boundary': [],
        'only_use_look_ahead': False,
        'retrieval_trigers': [('Follow up:', '?')],
    }
    qagent = QueryAgent(
        model=args.model,
        max_generation_len=args.max_generation_len,
        retrieval_kwargs=retrieval_kwargs)

    # query
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with tqdm(total=len(data)) as pbar, open(args.output, 'w') as fout:
        for b in range(0, len(data), args.batch_size):
            batch = data.select(range(b, min(b + args.batch_size, len(data))))
            prompts = [CtxPrompt.from_dict(example) for example in batch]
            generations, retrievals = qagent.prompt(prompts)
            retrievals = retrievals or [None] * len(generations)
            for example, generation, retrieval in zip(batch, generations, retrievals):
                example['output'] = generation
                example['retrieval'] = retrieval
                fout.write(json.dumps(example) + '\n')
            pbar.update(len(batch))
