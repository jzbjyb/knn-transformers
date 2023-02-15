from typing import List, Dict, Any, Tuple
import argparse
import random
import numpy as np
import logging
from tqdm import tqdm
import os
import re
import time
import json
from filelock import FileLock
from transformers import AutoTokenizer, GPT2TokenizerFast
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search
import openai
from .retriever import BM25
from .templates import CtxPrompt, RetrievalInstruction
from .datasets import StrategyQA, HotpotQA

logging.basicConfig(level=logging.INFO)


class ApiReturn:
    EOS = '<|endoftext|>'

    def __init__(
        self,
        prompt: str,
        text: str,
        tokens: List[str] = [],
        finish_reason: str = 'stop',
    ):
        self.prompt = prompt
        self.text = text
        self.tokens = tokens
        self.finish_reason = finish_reason

    @property
    def has_endoftext(self):
        return self.EOS in self.tokens


class QueryAgent:
    def __init__(
        self,
        model: str = 'code-davinci-002',
        max_generation_len: int = 128,
        retrieval_kwargs: Dict[str, Any] = {},
        tokenizer: AutoTokenizer = None,
        temperature: float = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer

        # generation args
        self.final_stop_sym = '\n\n'
        self.max_generation_len = max_generation_len
        self.temperature = temperature
        self.top_p = 1.0

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
        self.use_gold_iterative = retrieval_kwargs.get('use_gold_iterative', False)
        self.append_retrieval = retrieval_kwargs.get('append_retrieval', False)

        self.ret_topk = retrieval_kwargs.get('topk', 1)

        self.retrieval_at_beginning = retrieval_kwargs.get('retrieval_at_beginning', False)
        if self.retrieval_at_beginning:
            self.ret_frequency = self.max_generation_len
            self.ret_boundary = []

    @property
    def use_retrieval(self):
        return self.ret_frequency > 0 or self.ret_boundary or self.use_gold

    @staticmethod
    def clean_retrieval(texts: List[str]):
        return ' '.join(texts).replace('\n', ' ')

    def retrieve(self, queries: List[str]):
        ctx_ids, ctx_texts = self.retriever.retrieve_and_prepare(
            decoder_texts=queries,
            topk=self.ret_topk,
            max_query_length=self.max_query_length)
        return ctx_ids, ctx_texts

    def complete(
        self,
        queries: List[str],
        params: Dict[str, Any],
        max_num_req_per_min: int = 10,
        debug: bool = False,
    ) -> List[ApiReturn]:
        if 'max_tokens' in params:  # TODO: opt doesn't have this bug
            params['max_tokens'] = max(2, params['max_tokens'])  # openai returns nothing if set to 1
        min_sleep = 60 / max_num_req_per_min
        add_sleep = 3
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
                generations = [ApiReturn(
                    prompt=q,
                    text=r['text'],
                    tokens=r['logprobs']['tokens'],
                    finish_reason=r['finish_reason']) for r, q in zip(responses['choices'], queries)]
                if debug:
                    print(queries[0])
                    print('-->', generations[0].text)
                    input()
                break
            except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.Timeout):
                logging.info(f'sleep {add_sleep + min_sleep}')
                time.sleep(add_sleep + min_sleep)
                add_sleep = add_sleep * expbf
        time.sleep(min_sleep)
        return generations

    def prompt(
        self,
        queries: List[CtxPrompt],
    ):
        if self.use_retrieval:
            if self.use_gold:  # directly generate all with gold context
                ars = self.complete(
                    [q.format(use_ctx=True) for q in queries],
                    params={'max_tokens': self.max_generation_len, 'stop': self.final_stop_sym})
                outputs = [ar.text for ar in ars]
                traces = [[(ar.prompt, ar.text)] for ar in ars]
                return outputs, None, traces
            else:
                return self.ret_prompt(queries)
        else:  # directly generate all without gold context
            ars = self.complete(
                [q.format(use_ctx=False) for q in queries],
                params={'max_tokens': self.max_generation_len, 'stop': self.final_stop_sym})
            outputs = [ar.text for ar in ars]
            traces = [[(ar.prompt, ar.text)] for ar in ars]
            return outputs, None, traces

    def ret_prompt(
        self,
        queries: List[CtxPrompt],
    ):
        batch_size = len(queries)
        final_retrievals: List[List[List[str]]] = [[] for _ in range(len(queries))]  # (bs, n_ret_steps, ret_topk)
        final_outputs: List[str] = [''] * len(queries)
        traces: List[List[Tuple[str, str]]] = [[] for _ in range(len(queries))]
        queries: List[Tuple[int, CtxPrompt]] = [(i, q) for i, q in enumerate(queries)]  # to query
        max_gen_len = 0

        generate_queries: List[str] = []
        while len(queries) and max_gen_len < self.max_generation_len:
            # retrieve
            look_aheads: List[str] = [''] * len(queries)
            if self.look_ahead_steps:  # generate a fixed number tokens for retrieval
                apireturns = self.complete(
                    [q.format(use_ctx=True) for i, q in queries],
                    params={'max_tokens': self.look_ahead, 'stop': self.final_stop_sym})
                look_aheads = [ar.text for ar in apireturns]
            elif self.look_ahead_boundary:  # generate tokens until boundary for retrieval
                apireturns = self.complete(
                    [q.format(use_ctx=True) for i, q in queries],
                    params={'max_tokens': self.max_generation_len, 'stop': self.look_ahead_boundary})
                look_aheads = [ar.text for ar in apireturns]
            assert len(look_aheads) == len(queries)

            # send queries to index
            if generate_queries:  # some queries might be None which means no queries are generated
                assert len(generate_queries) == len(queries)
                queries_to_issue = [gq for gq in generate_queries if gq]
            else:
                # TODO: only use question
                queries_to_issue = [lh if self.only_use_look_ahead else (q.case.split('\n')[0].split(':', 1)[1].strip() + lh)
                    for (i, q), lh in zip(queries, look_aheads)]
            if queries_to_issue:
                # (bs, ret_topk) * 2
                ctx_ids, ctx_texts = self.retriever.retrieve_and_prepare(
                    decoder_texts=queries_to_issue,
                    topk=self.ret_topk,
                    max_query_length=self.max_query_length)
                idx = -1
                for _i, (i, q) in enumerate(queries):
                    if generate_queries:
                        if generate_queries[_i]:
                            idx += 1
                            if self.use_gold_iterative:
                                ret_id, ret_text = q.change_ctx()
                                ret_id = [ret_id]
                            else:
                                ret_id, ret_text = ctx_ids[idx].tolist(), self.clean_retrieval(ctx_texts[idx])
                            final_retrievals[i].append(ret_id)
                            if self.append_retrieval:
                                q.ctx = None
                                q.append_retrieval(ret_text, add_index=False)
                            else:
                                q.update_retrieval(ret_text, method='replace')
                    else:
                        ret_id, ret_text = ctx_ids[_i].tolist(), self.clean_retrieval(ctx_texts[_i])
                        if self.append_retrieval:
                            final_retrievals[i].append(ret_id)
                            q.ctx = None
                            q.append_retrieval(ret_text, add_index=False)
                        else:
                            final_retrievals[i].append(ret_id)
                            q.update_retrieval(ret_text, method='replace')
            generate_queries = []

            # complete
            if self.ret_frequency:
                apireturns = self.complete(
                    [q.format(use_ctx=True) for i, q in queries],
                    params={'max_tokens': self.ret_frequency, 'stop': self.final_stop_sym})
                max_gen_len += self.ret_frequency
            elif self.ret_boundary:
                apireturns = self.complete(
                    [q.format(use_ctx=True) for i, q in queries],
                    params={'max_tokens': self.max_generation_len - max_gen_len, 'stop': self.ret_boundary})
                # used to collect the generation with ret_boundary
                min_cont_len = 100000
                for i, ar in enumerate(apireturns):
                    cont, reason = ar.text, ar.finish_reason
                    if ar.has_endoftext:  # 003 stops proactively by returning endoftext
                        if self.retrieval_trigers:
                            generate_queries.append(None)
                    elif reason == 'stop' and self.final_stop_sym not in cont:  # stop at ret_boundary
                        if self.retrieval_trigers:  # extract queries from generation
                            assert len(self.retrieval_trigers) == 1
                            # TODO: check if it stops at retrieval trigers
                            ret_tri_start = self.retrieval_trigers[0][0]
                            found = re.search(ret_tri_start, cont)
                            if found:
                                generate_queries.append(cont[found.span()[1]:].strip())
                            else:
                                generate_queries.append(None)
                        assert len(self.ret_boundary) == 1
                        cont += self.ret_boundary[0]
                        reason = 'boundary'
                        assert len(cont) > 0, 'empty generation will cause dead lock'
                    else:
                        if self.retrieval_trigers:
                            generate_queries.append(None)
                    if self.final_stop_sym in cont:
                        cont = cont.split(self.final_stop_sym, 1)[0]
                        reason = 'stop'
                    apireturns[i].text = cont
                    apireturns[i].finish_reason = reason
                    min_cont_len = min(min_cont_len, len(self.tokenizer.tokenize(cont)))
                max_gen_len += min_cont_len
            else:
                raise NotImplementedError

            # decide whether to continue
            new_queries = []
            new_generate_queries = []
            assert len(queries) == len(apireturns)
            if self.retrieval_trigers:
                assert len(queries) == len(generate_queries), f'{len(queries)} {len(generate_queries)}'
            for _i, ((i, query), ar) in enumerate(zip(queries, apireturns)):
                cont, reason = ar.text, ar.finish_reason
                final_outputs[i] += cont
                traces[i].append((ar.prompt, cont))
                if reason == 'stop':
                    pass
                elif reason in {'length', 'boundary'}:
                    query.case += cont
                    new_queries.append((i, query))
                    if self.retrieval_trigers:
                        new_generate_queries.append(generate_queries[_i])
                else:
                    raise ValueError
            queries = new_queries
            generate_queries = new_generate_queries
        return final_outputs, final_retrievals, traces


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='strategyqa', choices=['strategyqa', 'hotpotqa'])
    parser.add_argument('--model', type=str, default='code-davinci-002', choices=['code-davinci-002', 'text-davinci-002', 'text-davinci-003'])
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--index_name', type=str, default='test')
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--file_lock', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_num_examples', type=int, default=None)
    parser.add_argument('--fewshot', type=int, default=0)
    parser.add_argument('--max_generation_len', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)

    parser.add_argument('--build_index', action='store_true')
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load retrieval corpus and index
    corpus, queries, qrels = GenericDataLoader(data_folder=args.input).load(split='dev') if args.input else (None, None, None)
    if args.build_index:
        if args.input:
            BM25Search(index_name=args.index_name, hostname='localhost', initialize=True, number_of_shards=1).index(corpus)
            time.sleep(5)
        exit()

    # init agent
    ret_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')
    prompt_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    retriever = BM25(
        tokenizer=ret_tokenizer,
        dataset=(corpus, queries, qrels),
        index_name=args.index_name,
        use_decoder_input_ids=True,
        engine='elasticsearch',
        file_lock=FileLock(args.file_lock) if args.file_lock else None)
    retrieval_kwargs = {
        'retriever': retriever,
        'topk': 1,
        'frequency': 0,
        'boundary': ['")]'],
        'use_gold': False,
        'use_gold_iterative': False,
        'max_query_length': 16,
        'retrieval_at_beginning': False,
        'look_ahead_steps': 0,
        'look_ahead_boundary': [],
        'only_use_look_ahead': False,
        'retrieval_trigers': [('\[Search\("', '")]')],
        'append_retrieval': False,
        'use_retrieval_instruction': False
    }
    qagent = QueryAgent(
        model=args.model,
        tokenizer=prompt_tokenizer,
        max_generation_len=args.max_generation_len,
        retrieval_kwargs=retrieval_kwargs,
        temperature=args.temperature)
    if retrieval_kwargs['use_retrieval_instruction']:
        CtxPrompt.ret_instruction = RetrievalInstruction()

    # load data
    if args.dataset == 'strategyqa':
        data = StrategyQA(args.input, prompt_type='cot')
        if qagent.append_retrieval:
            data.retrieval_augment_examplars(qagent, retrieval_at_beginning=retrieval_kwargs['retrieval_at_beginning'])
        data.format(fewshot=args.fewshot)
    elif args.dataset == 'hotpotqa':
        data = HotpotQA('validation', prompt_type='tool')
        if qagent.append_retrieval:
            data.retrieval_augment_examplars(qagent, retrieval_at_beginning=retrieval_kwargs['retrieval_at_beginning'])
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

    # query
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with tqdm(total=len(data)) as pbar, open(args.output, 'w') as fout:
        for b in range(0, len(data), args.batch_size):
            batch = data.select(range(b, min(b + args.batch_size, len(data))))
            prompts = [CtxPrompt.from_dict(example) for example in batch]
            generations, retrievals, traces = qagent.prompt(prompts)
            retrievals = retrievals or [None] * len(generations)
            traces = traces or [None] * len(generations)
            for example, generation, retrieval, trace in zip(batch, generations, retrievals, traces):
                example['output'] = generation
                example['retrieval'] = retrieval
                example['trace'] = trace
                fout.write(json.dumps(example) + '\n')
            pbar.update(len(batch))
