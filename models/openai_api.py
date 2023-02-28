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
import spacy
from .retriever import BM25
from .templates import CtxPrompt, RetrievalInstruction
from .datasets import StrategyQA, HotpotQA, WikiMultiHopQA

logging.basicConfig(level=logging.INFO)


class ApiReturn:
    EOS = '<|endoftext|>'
    nlp = spacy.load('en_core_web_sm')

    def __init__(
        self,
        prompt: str,
        text: str,
        tokens: List[str] = [],
        probs: List[float] = [],
        offsets: List[int] = [],
        finish_reason: str = 'stop',
    ):
        self.prompt = prompt
        self.text = text
        assert len(tokens) == len(probs) == len(offsets)
        self.tokens = tokens
        self.probs = probs
        self.offsets = offsets
        self.finish_reason = finish_reason
        if self.finish_reason is None:
            self.finish_reason = 'stop'  # TODO: a bug from openai?

    @property
    def num_tokens(self):
        return len(self.tokens)

    @property
    def has_endoftext(self):
        return self.EOS in self.tokens

    @classmethod
    def get_sent(cls, text: str, position: str = 'begin'):
        doc = cls.nlp(text)
        if position == 'begin':
            break_at = len(text)
            for sent in doc.sents:
                if sent.end_char > 0:
                    break_at = sent.end_char
                    break
            return text[:break_at], break_at
        if position == 'end':
            sents = list(doc.sents)
            break_at = 0
            for i in range(len(sents)):
                sent = sents[len(sents) - i - 1]
                if len(text) - sent.start_char >= 5:  # TODO: argument
                    break_at = sent.start_char
                    break
            return text[break_at:], break_at
        raise NotImplementedError

    def truncate_at_prob(self, low: float):
        if self.num_tokens <= 1:
            return self

        break_point = self.num_tokens
        for i in range(self.num_tokens):
            t, p, o = self.tokens[i], self.probs[i], self.offsets[i]
            if p <= low:
                break_point = i
                break
        if break_point == 0 and self.num_tokens > 0:  # avoid deadlock
            break_point = 1

        while break_point < self.num_tokens:  # truncation
            assert break_point > 0
            keep = self.offsets[break_point] - len(self.prompt)
            if keep <= 0:
                break_point += 1
                continue

            self.text = self.text[:keep]
            self.tokens = self.tokens[:break_point]
            self.probs = self.probs[:break_point]
            self.offsets = self.offsets[:break_point]
            self.finish_reason = 'boundary'
            break

        return self

    def truncate_at_boundary(self, unit: str = 'sentence'):
        if self.num_tokens <= 1:
            return self

        if unit == 'sentence':
            doc = self.nlp(self.text)
            break_at = len(self.text)
            for sent in doc.sents:
                if sent.end_char > 0:
                    break_at = sent.end_char
                    break

            if break_at > 0 and break_at < len(self.text):  # truncation
                i = 0
                for i in range(self.num_tokens):
                    if self.offsets[i] - len(self.prompt) >= break_at:
                        break_at = self.offsets[i] - len(self.prompt)
                        break
                assert i > 0 and break_at > 0
                self.text = self.text[:break_at]
                self.tokens = self.tokens[:i]
                self.probs = self.probs[:i]
                self.offsets = self.offsets[:i]
                self.finish_reason = 'boundary'
        else:
            raise NotImplementedError
        return self

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
        self.use_ctx = retrieval_kwargs.get('use_ctx', False)
        self.ret_frequency = retrieval_kwargs.get('frequency', 0)
        self.truncate_at_prob = retrieval_kwargs.get('truncate_at_prob', 0)
        self.truncate_at_boundary = retrieval_kwargs.get('truncate_at_boundary', None)
        self.ret_boundary = retrieval_kwargs.get('boundary', [])
        self.use_gold = retrieval_kwargs.get('use_gold', False)
        if self.ret_boundary:  # otherwise cannot decide when to finally stop
            assert self.final_stop_sym not in self.ret_boundary
        self.use_ctx_for_examplars = retrieval_kwargs.get('use_ctx_for_examplars', False)
        self.ctx_increase = retrieval_kwargs.get('ctx_increase', 'replace')

        self.look_ahead_steps = retrieval_kwargs.get('look_ahead_steps', 0)
        self.look_ahead_boundary = retrieval_kwargs.get('look_ahead_boundary', 0)
        self.look_ahead_truncate_at_boundary = retrieval_kwargs.get('look_ahead_truncate_at_boundary', None)
        self.max_query_length = retrieval_kwargs.get('max_query_length', None)
        self.use_full_input_as_query = retrieval_kwargs.get('use_full_input_as_query', False)
        self.only_use_look_ahead = retrieval_kwargs.get('only_use_look_ahead', False)
        self.retrieval_trigers = retrieval_kwargs.get('retrieval_trigers', [])
        for rts, rte in self.retrieval_trigers:
            assert rte in self.ret_boundary, 'end of retrieval trigers must be used as boundary'
        self.force_generate = retrieval_kwargs.get('force_generate', None)
        self.forbid_generate_step = retrieval_kwargs.get('forbid_generate_step', None)
        self.use_gold_iterative = retrieval_kwargs.get('use_gold_iterative', False)
        self.append_retrieval = retrieval_kwargs.get('append_retrieval', False)

        self.ret_topk = retrieval_kwargs.get('topk', 1)
        self.debug = retrieval_kwargs.get('debug', False)

        self.retrieval_at_beginning = retrieval_kwargs.get('retrieval_at_beginning', False)
        if self.retrieval_at_beginning:
            if self.ret_frequency:
                self.ret_frequency = self.max_generation_len

    @property
    def use_retrieval(self):
        return self.ret_frequency > 0 or self.ret_boundary or self.use_gold

    @staticmethod
    def clean_retrieval(texts: List[str]):
        return ' '.join(texts).replace('\n', ' ')

    def retrieve(self, queries: List[str], is_question: bool = False):
        mql = None if (self.use_full_input_as_query and is_question) else self.max_query_length
        ctx_ids, ctx_texts = self.retriever.retrieve_and_prepare(
            decoder_texts=queries,
            topk=self.ret_topk,
            max_query_length=mql)
        return ctx_ids, ctx_texts

    def complete(
        self,
        queries: List[str],
        params: Dict[str, Any],
        max_num_req_per_min: int = 10,
        force_generate: int = None,
        forbid_generate: int = None,
    ) -> List[ApiReturn]:
        if 'max_tokens' in params:  # TODO: opt doesn't have this bug
            params['max_tokens'] = max(2, params['max_tokens'])  # openai returns nothing if set to 1
        min_sleep = 60 / max_num_req_per_min
        add_sleep = 3
        expbf = 2
        while True:
            try:
                logit_bias = dict()
                if force_generate:
                    logit_bias={f'{force_generate}': 2.0}
                elif forbid_generate:
                    logit_bias={f'{forbid_generate}': -100}
                responses = openai.Completion.create(
                    model=self.model,
                    prompt=queries,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    logprobs=0,
                    logit_bias=logit_bias,
                    **params)
                generations = [ApiReturn(
                    prompt=q,
                    text=r['text'],
                    tokens=r['logprobs']['tokens'],
                    probs=[np.exp(lp) for lp in r['logprobs']['token_logprobs']],
                    offsets=r['logprobs']['text_offset'],
                    finish_reason=r['finish_reason']) for r, q in zip(responses['choices'], queries)]
                if self.debug:
                    print('Prompt ->', queries[0])
                    print('Output ->', generations[0].text)
                    input('-' * 50)
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
        first_ret = True
        step_ind = 0
        while len(queries) and max_gen_len < self.max_generation_len:
            # retrieve
            look_aheads: List[str] = [''] * len(queries)
            if self.look_ahead_steps:  # generate a fixed number tokens for retrieval
                apireturns = self.complete(
                    [q.format(use_ctx=self.use_ctx) for i, q in queries],
                    params={'max_tokens': self.look_ahead_steps, 'stop': self.final_stop_sym})
                if self.look_ahead_truncate_at_boundary:
                    apireturns = [ar.truncate_at_boundary(self.look_ahead_truncate_at_boundary) for ar in apireturns]
                look_aheads = [ar.text for ar in apireturns]
            elif self.look_ahead_boundary:  # generate tokens until boundary for retrieval
                apireturns = self.complete(
                    [q.format(use_ctx=self.use_ctx) for i, q in queries],
                    params={'max_tokens': self.max_generation_len, 'stop': self.look_ahead_boundary})
                look_aheads = [ar.text for ar in apireturns]
            assert len(look_aheads) == len(queries)

            # send queries to index
            if generate_queries:  # some queries might be None which means no queries are generated
                assert len(generate_queries) == len(queries)
                queries_to_issue = [lh if self.only_use_look_ahead else (gq + lh)
                    for gq, lh in zip(generate_queries, look_aheads) if gq]
            else:
                # TODO: only use question
                #queries_to_issue = [lh if self.only_use_look_ahead else (q.case.split('\n')[0].split(':', 1)[1].strip() + lh)
                #    for (i, q), lh in zip(queries, look_aheads)]
                queries_to_issue = [lh if self.only_use_look_ahead else (q.case + lh)
                    for (i, q), lh in zip(queries, look_aheads)]
            if queries_to_issue and (not self.retrieval_at_beginning or first_ret):
                if self.debug:
                    print('Query ->', queries_to_issue[0])
                # (bs, ret_topk) * 2
                ctx_ids, ctx_texts = self.retrieve(queries_to_issue, is_question=first_ret)
                first_ret = False
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
                                q.update_retrieval(ret_text, method=self.ctx_increase)
                    else:
                        ret_id, ret_text = ctx_ids[_i].tolist(), self.clean_retrieval(ctx_texts[_i])
                        if self.append_retrieval:
                            final_retrievals[i].append(ret_id)
                            q.ctx = None
                            q.append_retrieval(ret_text, add_index=False)
                        else:
                            final_retrievals[i].append(ret_id)
                            q.update_retrieval(ret_text, method=self.ctx_increase)
            generate_queries = []

            # complete
            if self.ret_frequency:
                apireturns = self.complete(
                    [q.format(use_ctx=self.use_ctx) for i, q in queries],
                    params={'max_tokens': min(self.max_generation_len - max_gen_len, self.ret_frequency), 'stop': self.final_stop_sym})
                if self.truncate_at_prob > 0:
                    apireturns = [ar.truncate_at_prob(self.truncate_at_prob) for ar in apireturns]
                    max_gen_len += int(np.min([ar.num_tokens for ar in apireturns]))
                    #generate_queries = [ApiReturn.get_sent(ar.text, position='end')[0] for ar in apireturns]
                    generate_queries = [ar.text for ar in apireturns]
                elif self.truncate_at_boundary:
                    apireturns = [ar.truncate_at_boundary(self.truncate_at_boundary) for ar in apireturns]
                    max_gen_len += int(np.min([ar.num_tokens for ar in apireturns]))
                    generate_queries = [ar.text for ar in apireturns]
                else:
                    max_gen_len += self.ret_frequency
            elif self.ret_boundary:
                if self.forbid_generate_step and self.retrieval_trigers and step_ind > 0:  # start from the second step to forbid the force_generate token
                    apireturns = self.complete(
                        [q.format(use_ctx=self.use_ctx) for i, q in queries],
                        params={'max_tokens': min(self.max_generation_len - max_gen_len, self.forbid_generate_step), 'stop': self.final_stop_sym},
                        forbid_generate=self.force_generate)
                    for (i, query), ar in zip(queries, apireturns):
                        cont = ar.text
                        final_outputs[i] += cont
                        traces[i].append((ar.prompt, cont))
                        query.case += cont
                apireturns = self.complete(
                    [q.format(use_ctx=self.use_ctx) for i, q in queries],
                    params={'max_tokens': self.max_generation_len - max_gen_len, 'stop': self.ret_boundary},
                    force_generate=self.force_generate)
                # used to collect the generation with ret_boundary
                min_cont_len = 100000
                for i, ar in enumerate(apireturns):
                    cont, reason = ar.text, ar.finish_reason
                    if ar.has_endoftext:  # 003 stops proactively by returning endoftext
                        if self.retrieval_trigers:
                            generate_queries.append(None)
                    elif reason == 'stop' and self.final_stop_sym not in cont:  # stop at ret_boundary
                        remove_query = False
                        if self.retrieval_trigers:  # extract queries from generation
                            assert len(self.retrieval_trigers) == 1
                            # TODO: check if it stops at retrieval trigers
                            ret_tri_start = self.retrieval_trigers[0][0]
                            if ret_tri_start is None:  # use all generated tokens as query
                                generate_queries.append(cont)
                            else:
                                found_query = re.search(ret_tri_start, cont)
                                if found_query:
                                    generate_queries.append(cont[found_query.span()[1]:].strip())
                                    if self.forbid_generate_step:
                                        remove_query = True
                                        cont = cont[:found_query.span()[0]]  # remove queries
                                else:
                                    generate_queries.append(None)
                        assert len(self.ret_boundary) == 1
                        if not remove_query:
                            cont += self.ret_boundary[0]
                        reason = 'boundary'
                        #assert len(cont) > 0, 'empty generation will cause dead lock'
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
            if len(generate_queries):
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
                    if len(generate_queries):
                        new_generate_queries.append(generate_queries[_i])
                else:
                    raise ValueError
            queries = new_queries
            generate_queries = new_generate_queries
            step_ind += 1
        return final_outputs, final_retrievals, traces


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='strategyqa', choices=['strategyqa', 'hotpotqa', '2wikihop'])
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
    parser.add_argument('--debug', action='store_true')
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
    prompt_tokenizer.pad_token = prompt_tokenizer.eos_token
    retriever = BM25(
        tokenizer=prompt_tokenizer,
        dataset=(corpus, queries, qrels),
        index_name=args.index_name,
        use_decoder_input_ids=True,
        engine='elasticsearch',
        file_lock=FileLock(args.file_lock) if args.file_lock else None)
    retrieval_kwargs = {
        'retriever': retriever,
        'topk': 3,
        'use_ctx': True,
        'frequency': 128,
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
        'format_reference_method': 'default',
        'ctx_position': 'before_case',
        'prompt_type': 'cot_interleave_ret',
        'ctx_increase': 'replace',
        'add_ref_suffix': None,
        'add_ref_prefix': None,
        'debug': args.debug,
    }
    qagent = QueryAgent(
        model=args.model,
        tokenizer=prompt_tokenizer,
        max_generation_len=args.max_generation_len,
        retrieval_kwargs=retrieval_kwargs,
        temperature=args.temperature)
    if retrieval_kwargs['use_retrieval_instruction']:
        CtxPrompt.ret_instruction = RetrievalInstruction()
    CtxPrompt.format_reference_method = retrieval_kwargs['format_reference_method']
    CtxPrompt.ctx_position = retrieval_kwargs['ctx_position']
    CtxPrompt.add_ref_suffix = retrieval_kwargs['add_ref_suffix']
    CtxPrompt.add_ref_prefix = retrieval_kwargs['add_ref_prefix']

    # load data
    if args.dataset == 'strategyqa':
        data = StrategyQA(args.input, prompt_type=retrieval_kwargs['prompt_type'])
    elif args.dataset == 'hotpotqa':
        data = HotpotQA('validation', prompt_type=retrieval_kwargs['prompt_type'])
    elif args.dataset == '2wikihop':
        data = WikiMultiHopQA(args.input, prompt_type=retrieval_kwargs['prompt_type'])
        use_gold_func = WikiMultiHopQA.get_gold_ctxs
    else:
        raise NotImplementedError
    if qagent.use_ctx_for_examplars == 'gold':
        data.retrieval_augment_examplars(qagent, use_gold=use_gold_func)
    elif qagent.use_ctx_for_examplars == 'ret':
        data.retrieval_augment_examplars(qagent)
    elif qagent.use_ctx_for_examplars == False:
        pass
    else:
        raise NotImplementedError
    data.format(fewshot=args.fewshot)
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
