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
import copy
from operator import itemgetter
from collections import defaultdict, Counter
from filelock import FileLock
from multiprocessing import Process, Queue, Lock
from multiprocessing.managers import BaseManager
from transformers import AutoTokenizer, GPT2TokenizerFast
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search
import openai
from .retriever import BM25
from .templates import CtxPrompt, ApiReturn, RetrievalInstruction
from .datasets import StrategyQA, HotpotQA, WikiMultiHopQA, WikiSum, ELI5, WoW, WoWLong, ASQA, MMLU, LMData

logging.basicConfig(level=logging.INFO)


class KeyBrokenException(Exception):
    pass


class CustomManager(BaseManager):
    pass


class KeyManager:
    def __init__(self, keys: List[str]):
        assert len(keys)
        self.key2inuse = {key: False for key in keys}
        self.key2ind = {key: i for i, key in enumerate(keys)}
        self.keys = keys
        self.next_available_key_ind = 0
        self.key2times: Dict[str, List[float]] = defaultdict(list)

    def __len__(self):
        return len(self.keys)

    def get_key(self):
        logging.info(f'get key {self.next_available_key_ind}')
        # get key not in use
        to_return = self.keys[self.next_available_key_ind]
        assert not self.key2inuse[to_return]
        self.key2inuse[to_return] = True
        # set index
        found_key_not_inuse = False
        for _ in range(len(self)):
            self.next_available_key_ind = (self.next_available_key_ind + 1) % len(self)
            if not self.key2inuse[self.keys[self.next_available_key_ind]]:
                found_key_not_inuse = True
                break
        if not found_key_not_inuse:
            self.next_available_key_ind = None
        logging.info(f'get key {to_return[-5:]} next avai {self.next_available_key_ind}')
        return to_return

    def return_key(self, key, time_spent: float = None):
        logging.info('return key')
        # return key
        assert self.key2inuse[key]
        self.key2inuse[key] = False
        if time_spent:
            self.key2times[key].append(time_spent)
        # set index
        if self.next_available_key_ind is None:
            self.next_available_key_ind = self.key2ind[key]
        logging.info('return key done')

    def get_report(self):
        report: List[str] = []
        for key in self.keys:
            report.append(f'{key}\t{len(self.key2times[key])}\t{np.mean(self.key2times[key])}')
        return '\n'.join(report)


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
        self.final_stop_sym = retrieval_kwargs.get('final_stop_sym', '\n\n')
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
        self.look_ahead_filter_prob = retrieval_kwargs.get('look_ahead_filter_prob', 0)
        self.look_ahead_mask_prob = retrieval_kwargs.get('look_ahead_mask_prob', 0)
        self.look_ahead_pre_retrieval = retrieval_kwargs.get('look_ahead_pre_retrieval', False)
        assert self.look_ahead_pre_retrieval in {False, 'first', 'all'}
        self.max_query_length = retrieval_kwargs.get('max_query_length', None)
        self.use_full_input_as_query = retrieval_kwargs.get('use_full_input_as_query', False)
        self.only_use_look_ahead = retrieval_kwargs.get('only_use_look_ahead', False)
        self.retrieval_trigers = retrieval_kwargs.get('retrieval_trigers', [])
        for rts, rte in self.retrieval_trigers:
            assert rte in self.ret_boundary, 'end of retrieval trigers must be used as boundary'
        self.force_generate = retrieval_kwargs.get('force_generate', None)
        self.forbid_generate_step = retrieval_kwargs.get('forbid_generate_step', 0)
        self.use_gold_iterative = retrieval_kwargs.get('use_gold_iterative', False)
        self.append_retrieval = retrieval_kwargs.get('append_retrieval', False)

        self.ret_topk = retrieval_kwargs.get('topk', 1)
        self.debug = retrieval_kwargs.get('debug', False)

        self.retrieval_at_beginning = retrieval_kwargs.get('retrieval_at_beginning', False)
        if self.retrieval_at_beginning:
            if self.ret_frequency:
                self.ret_frequency = self.max_generation_len
        self.regenerate_at_end = retrieval_kwargs.get('regenerate_at_end', False)

        self.frequency_penalty = retrieval_kwargs.get('frequency_penalty', 0.0)
        self.frequency_penalty_in_prompt = retrieval_kwargs.get('frequency_penalty_in_prompt', 0.0)

        self.prefix_method = retrieval_kwargs.get('prefix_method', None)
        if self.prefix_method in {'sentence', 'all'} or (self.prefix_method and self.prefix_method.startswith('freq:')):  # no truncation when computing PPL
            self.truncate_at_prob = 0
            self.truncate_at_boundary = None
            self.max_generation_len = 100000  # no limit

    @property
    def use_retrieval(self):
        return self.ret_frequency > 0 or self.ret_boundary or self.use_gold

    def get_tokens(self, text: str, topk: int) -> Tuple[str, int]:
        assert topk >= 1
        tokenized = self.tokenizer(text, return_offsets_mapping=True)
        ids, offsets = tokenized['input_ids'][:topk], tokenized['offset_mapping'][:topk]
        last_position = offsets[-1][1]
        prefix = text[:last_position]
        return prefix, last_position

    def retrieve(self, queries: List[str], is_question: bool = False):
        mql = None if (self.use_full_input_as_query and is_question) else self.max_query_length
        ctx_ids, ctx_texts = self.retriever.retrieve_and_prepare(
            decoder_texts=queries,
            topk=self.ret_topk,
            max_query_length=mql)
        return ctx_ids, ctx_texts

    def complete(
        self,
        queries: List[CtxPrompt],
        params: Dict[str, Any],
        max_num_req_per_min: int = 10,
        max_retry: int = 5,  # retry before switching keys
        max_keys: int = 5,  # max number of keys tried before raising exceptions
        key_tried: int = 0,  # number of keys tried
        force_generate: int = None,
        forbid_generate: int = None,
        api_key: str = None,
        is_lookahead: bool = False,
    ) -> List[ApiReturn]:
        is_chat_model = 'turbo' in self.model

        # check num of keys tried
        if key_tried >= max_keys:
            logging.error(f'skip {len(queries)} examples because of key broken')
            raise KeyBrokenException()

        # init tracking variables
        max_num_req_per_min = 1000 if is_chat_model else 10
        min_sleep = 60 / max_num_req_per_min
        add_sleep = 3
        expbf = 2
        retry = 0

        # init param
        if 'max_tokens' in params:  # TODO: OPT doesn't have this bug
            params['max_tokens'] = max(2, params['max_tokens'])  # openai returns nothing if set to 1
        logit_bias = dict()

        if force_generate:
            logit_bias={f'{force_generate[0]}': force_generate[1]}
        elif forbid_generate:
            logit_bias={f'{forbid_generate[0]}': -100}

        # format to get the final prompt
        prompts: List[Tuple[str, int]] = [q.format(use_ctx=self.use_ctx) for q in queries]
        prompts_to_issue: List[str] = list(map(itemgetter(0), prompts))

        # get prefix
        echo = False
        use_prefix = (self.prefix_method and self.prefix_method.startswith('sentence_first:')) or (self.prefix_method and not is_lookahead)
        if use_prefix:  # prefix is not allowed to use in look ahead
            echo = True
            prefixes: List[Tuple[str, int]] = [q.get_prefix(qagent=self, prefix_method=self.prefix_method) for q in queries]
            to_gen_len = set(map(itemgetter(1), prefixes))
            if None in to_gen_len:  # generate follow original settings
                pass
            else:  # generate `to_gen_len` tokens
                params['max_tokens'] = max(to_gen_len)
            assert len(prompts_to_issue) == len(prefixes)
            for i in range(len(prompts_to_issue)):
                prompts_to_issue[i] += prefixes[i][0]

        # add penalty
        if self.frequency_penalty_in_prompt:
            assert len(queries) == 1, 'batching is not supported'
            current_cases: List[str] = [q[-l:] if l else '' for q, l in prompts]  # only use the generated content
            counter = Counter(sum(self.tokenizer(current_cases)['input_ids'], []))
            tokid2count: Dict[int, int] = dict(sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:200])  # penalize at most 200 tokens
            for tokid, count in tokid2count.items():
                if tokid not in logit_bias:
                    logit_bias[str(tokid)] = 0
                logit_bias[str(tokid)] -= self.frequency_penalty_in_prompt * count

        # init key
        ori_api_key = api_key
        get_key_func = return_key_func = None
        if type(ori_api_key) is tuple:  # get and return function
            get_key_func, return_key_func = ori_api_key
        else:  # a specific key or none
            api_key = ori_api_key or os.getenv('OPENAI_API_KEY')
        if get_key_func:  # get key
            start_t = time.time()
            api_key = get_key_func()

        # error-tolerant querying
        while True:
            try:
                logging.info(f'start query with key {api_key[-5:]}')
                if is_chat_model:
                    assert len(queries) == 1, ''
                    if 'max_tokens' in params:
                        params['max_tokens'] = max(1, params['max_tokens'])  # 0 is not allowed for chatgpt
                    responses = openai.ChatCompletion.create(
                        api_key=api_key,
                        model=self.model,
                        messages=[{'role': 'user', 'content': prompts_to_issue[0]}],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        logit_bias=logit_bias,
                        frequency_penalty=self.frequency_penalty,
                        **params)

                    generations = [ApiReturn(
                        prompt=q,
                        text=(prefixes[0][0] + ' ' + r['message']['content']) if echo else r['message']['content'],  # TODO: corner case where space does not work?
                        finish_reason='length' if echo else r['finish_reason'],  # never stop in echo mode
                        model=responses['model'],
                        skip_len=0) for r, (q, _) in zip(responses['choices'], prompts)]
                else:
                    responses = openai.Completion.create(
                        api_key=api_key,
                        model=self.model,
                        prompt=prompts_to_issue,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        logprobs=0,
                        logit_bias=logit_bias,
                        frequency_penalty=self.frequency_penalty,
                        echo=echo,
                        **params)

                    generations = [ApiReturn(
                        prompt=q,
                        text=r['text'],
                        tokens=r['logprobs']['tokens'],
                        probs=[np.exp(lp) if lp is not None else lp for lp in r['logprobs']['token_logprobs']],
                        offsets=r['logprobs']['text_offset'],
                        finish_reason='length' if echo else r['finish_reason'],  # never stop in echo mode
                        model=responses['model'],
                        skip_len=len(q) if echo else 0) for r, (q, _) in zip(responses['choices'], prompts)]

                logging.info(f'finish query with key {api_key[-5:]}')
                if self.debug:
                    print('Params ->', params)
                    print('Prompt ->', generations[0].prompt)
                    print('Output ->', generations[0].text)
                    print('Stop ->', generations[0].finish_reason)
                    print('Tokens ->', generations[0].tokens)
                    print('Probs ->', generations[0].token_probs)
                    print('Gold ->', queries[0].gold_output)
                    input('-' * 50 + '\n')
            except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.Timeout) as e:  # limit-related errors
                if retry >= max_retry:
                    if return_key_func:  # return key
                        end_t = time.time()
                        return_key_func(api_key, time_spent=end_t - start_t)
                    return self.complete(
                        queries,
                        params,
                        max_num_req_per_min=max_num_req_per_min,
                        max_retry=max_retry,
                        max_keys=max_keys,
                        key_tried=key_tried + 1,
                        force_generate=force_generate,
                        forbid_generate=forbid_generate,
                        api_key=ori_api_key)
                retry += 1
                logging.info(f'sleep for rate {add_sleep + min_sleep} with key {api_key[-5:]}')
                time.sleep(add_sleep + min_sleep)
                add_sleep = add_sleep * expbf
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:  # other errors
                if retry >= max_retry:
                    if return_key_func:  # return key
                        end_t = time.time()
                        return_key_func(api_key, time_spent=end_t - start_t)
                    return self.complete(
                        queries,
                        params,
                        max_num_req_per_min=max_num_req_per_min,
                        max_retry=max_retry,
                        max_keys=max_keys,
                        key_tried=key_tried + 1,
                        force_generate=force_generate,
                        forbid_generate=forbid_generate,
                        api_key=ori_api_key)
                retry += 1
                logging.info(f'sleep for error {min_sleep} with key {api_key[-5:]}')
                logging.info(e)
                time.sleep(min_sleep)
            else:
                break

        if return_key_func:  # return key
            end_t = time.time()
            return_key_func(api_key, time_spent=end_t - start_t)

        time.sleep(min_sleep)
        return generations

    def prompt(
        self,
        queries: List[CtxPrompt],
        api_key: str = None,
    ):
        # update retrieval for ctx
        for q in queries:
            for d in q.demo:
                d.update_retrieval(d.get_all_ctxs(), method=self.ctx_increase)
        if self.use_retrieval:
            return self.ret_prompt(queries, api_key=api_key)
        else:  # directly generate all without gold context
            ars = self.complete(
                queries,
                params={'max_tokens': self.max_generation_len, 'stop': self.final_stop_sym},
                api_key=api_key)
            outputs = [ar.text for ar in ars]
            probs = [ar.token_probs for ar in ars]
            traces = [[(ar.prompt, ar.text)] for ar in ars]
            return outputs, probs, None, traces

    def ret_prompt(
        self,
        queries: List[CtxPrompt],
        api_key: str = None,
        max_iteration: int = 10000,  # TODO: too high?
    ):
        batch_size = len(queries)
        final_retrievals: List[List[List[str]]] = [[] for _ in range(len(queries))]  # (bs, n_ret_steps, ret_topk)
        final_outputs: List[str] = [''] * len(queries)
        final_probs: List[List[float]] = [[] for _ in range(len(queries))]
        final_queries: List[CtxPrompt] = [None] * len(queries)
        traces: List[List[Tuple[str, str]]] = [[] for _ in range(len(queries))]
        queries: List[Tuple[int, CtxPrompt]] = [(i, q) for i, q in enumerate(queries)]  # to query
        max_gen_len = 0

        generate_queries: List[str] = []
        first_ret = True
        step_ind = 0
        while len(queries) and max_gen_len < self.max_generation_len and step_ind <= max_iteration:
            # retrieve
            look_aheads: List[str] = [''] * len(queries)
            if self.look_ahead_steps:  # generate a fixed number tokens for retrieval
                if (self.look_ahead_pre_retrieval == 'first' and step_ind == 0) or self.look_ahead_pre_retrieval == 'all':  # pre-retrieval for look ahead
                    ctx_ids, ctx_texts = self.retrieve([q.get_query_for_retrieval() for i, q in queries], is_question=first_ret)
                    for _i, (i, q) in enumerate(queries):
                        ret_id, ret_text = ctx_ids[_i].tolist(), ctx_texts[_i].tolist()
                        final_retrievals[i].append(ret_id)
                        if self.append_retrieval:
                            q.ctx = None
                            q.append_retrieval(ret_text, add_index=False)
                        else:
                            q.update_retrieval(ret_text, method=self.ctx_increase)
                apireturns = self.complete(
                    list(map(itemgetter(1), queries)),
                    params={'max_tokens': self.look_ahead_steps, 'stop': self.final_stop_sym},
                    api_key=api_key,
                    is_lookahead=True)
                if self.look_ahead_truncate_at_boundary:
                    apireturns = [ar.truncate_at_boundary(self.look_ahead_truncate_at_boundary) for ar in apireturns]
                look_aheads = [ar.use_as_query(low=self.look_ahead_filter_prob, mask=self.look_ahead_mask_prob) for ar in apireturns]
            elif self.look_ahead_boundary:  # generate tokens until boundary for retrieval
                apireturns = self.complete(
                    list(map(itemgetter(1), queries)),
                    params={'max_tokens': self.max_generation_len, 'stop': self.look_ahead_boundary},
                    api_key=api_key,
                    is_lookahead=True)
                look_aheads = [ar.text for ar in apireturns]
            assert len(look_aheads) == len(queries)

            # send queries to index
            if self.use_gold:
                for i, q in queries:
                    q.update_retrieval(q.get_all_ctxs(), method='replace')  # use all gold ctx
            else:
                if generate_queries:  # some queries might be None which means no queries are generated
                    assert len(generate_queries) == len(queries)
                    queries_to_issue = [lh if self.only_use_look_ahead else (gq + lh) for gq, lh in zip(generate_queries, look_aheads)]
                else:
                    # TODO: only use question
                    #queries_to_issue = [lh if self.only_use_look_ahead else (q.case.split('\n')[0].split(':', 1)[1].strip() + lh)
                    #    for (i, q), lh in zip(queries, look_aheads)]
                    queries_to_issue = [lh if self.only_use_look_ahead else (q.get_query_for_retrieval() + lh) for (i, q), lh in zip(queries, look_aheads)]
                if self.debug:
                    print(f'Query -> !{queries_to_issue[0]}!')
                nonemp_queries_to_issue = [q for q in queries_to_issue if q]
                if nonemp_queries_to_issue and (not self.retrieval_at_beginning or first_ret):
                    # (bs, ret_topk) * 2
                    ctx_ids, ctx_texts = self.retrieve(nonemp_queries_to_issue, is_question=first_ret)
                    first_ret = False
                    idx = -1
                    for _i, (i, q) in enumerate(queries):
                        if queries_to_issue[_i]:
                            idx += 1
                            if self.use_gold_iterative:
                                ret_id, ret_text = q.change_ctx()
                                ret_id = [ret_id]
                            else:
                                ret_id, ret_text = ctx_ids[idx].tolist(), ctx_texts[idx].tolist()
                            final_retrievals[i].append(ret_id)
                            if self.append_retrieval:
                                q.ctx = None
                                q.append_retrieval(ret_text, add_index=False)
                            else:
                                q.update_retrieval(ret_text, method=self.ctx_increase)
            generate_queries = []

            # complete
            if self.ret_frequency:
                apireturns = self.complete(
                    list(map(itemgetter(1), queries)),
                    params={'max_tokens': min(self.max_generation_len - max_gen_len, self.ret_frequency), 'stop': self.final_stop_sym},
                    api_key=api_key)
                if self.truncate_at_prob > 0:
                    apireturns = [ar.truncate_at_prob(self.truncate_at_prob) for ar in apireturns]
                    max_gen_len += int(np.min([ar.num_tokens for ar in apireturns]))
                    generate_queries = [ar.text for ar in apireturns]  # always use newly generated as query
                elif self.truncate_at_boundary:
                    apireturns = [ar.truncate_at_boundary(self.truncate_at_boundary) for ar in apireturns]
                    max_gen_len += int(np.min([ar.num_tokens for ar in apireturns]))
                    generate_queries = [ar.text for ar in apireturns]  # always use newly generated as query
                else:
                    max_gen_len += self.ret_frequency
                for ar in apireturns:  # check final sym
                    if self.final_stop_sym in ar.text:
                        ar.finish_reason = 'stop'
                    ar.truncate_at_substring(self.final_stop_sym)
            elif self.ret_boundary:
                if self.forbid_generate_step and self.retrieval_trigers and step_ind > 0:  # start from the second step to forbid the force_generate token
                    apireturns = self.complete(
                        list(map(itemgetter(1), queries)),
                        params={'max_tokens': min(self.max_generation_len - max_gen_len, self.forbid_generate_step), 'stop': self.final_stop_sym},
                        forbid_generate=self.force_generate,
                        api_key=api_key)
                    for (i, query), ar in zip(queries, apireturns):
                        cont = ar.text
                        final_outputs[i] += cont
                        final_probs[i].extend(ar.token_probs or [])
                        final_queries[i] = query
                        traces[i].append((ar.prompt, cont))
                        query.add_generation(cont)
                apireturns = self.complete(
                    list(map(itemgetter(1), queries)),
                    params={'max_tokens': self.max_generation_len - max_gen_len, 'stop': self.ret_boundary},
                    force_generate=self.force_generate,
                    api_key=api_key)
                # used to collect the generation with ret_boundary
                min_cont_len = 100000
                for i, ar in enumerate(apireturns):
                    cont, reason = ar.text, ar.finish_reason
                    if ar.has_endoftext:  # 003 stops proactively by returning endoftext
                        if self.retrieval_trigers:
                            generate_queries.append('')
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
                                    generate_queries.append('')
                        assert len(self.ret_boundary) == 1
                        if not remove_query:
                            cont += self.ret_boundary[0]
                        reason = 'boundary'
                        #assert len(cont) > 0, 'empty generation will cause dead lock'
                    else:
                        if self.retrieval_trigers:
                            generate_queries.append('')
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
                final_probs[i].extend(ar.token_probs)
                final_queries[i] = query
                traces[i].append((ar.prompt, cont))
                if reason == 'stop':
                    pass
                elif reason in {'length', 'boundary'}:
                    query.add_generation(cont)
                    new_queries.append((i, query))
                    if len(generate_queries):
                        new_generate_queries.append(generate_queries[_i])
                else:
                    raise ValueError
            queries = new_queries
            generate_queries = new_generate_queries
            step_ind += 1

        if self.regenerate_at_end:  # regenerate given retrieval results
            for query in final_queries:
                query.reset_generation()
            apireturns = self.complete(
                final_queries,
                params={'max_tokens': self.max_generation_len, 'stop': self.final_stop_sym},
                api_key=api_key)
            for i, (query, ar) in enumerate(zip(final_queries, apireturns)):
                cont, reason = ar.text, ar.finish_reason
                final_outputs[i] = cont
                final_probs[i] = ar.token_probs
                traces[i].append((ar.prompt, cont))

        return final_outputs, final_probs, final_retrievals, traces


def query_agent_worker(
    qagent: QueryAgent,
    key_manager: KeyManager,
    lock: Lock,
    input_queue: Queue,
    output_queue: Queue,
):
    def get_key_func():
        with lock:
            key = key_manager.get_key()
            return key
    def return_key_func(*args, **kwargs):
        with lock:
            key_manager.return_key(*args, **kwargs)

    while True:
        batch = input_queue.get()
        if type(batch) is str and batch == 'DONE':
            break
        prompts = [CtxPrompt.from_dict(example) for example in batch]
        generations, probs, retrievals, traces = qagent.prompt(prompts, api_key=(get_key_func, return_key_func))
        retrievals = retrievals or [None] * len(generations)
        traces = traces or [None] * len(generations)
        for example, generation, prob, retrieval, trace in zip(batch, generations, probs, retrievals, traces):
            example['output'] = generation
            example['output_prob'] = prob
            example['retrieval'] = retrieval
            example['trace'] = trace
            output_queue.put(example)


def write_worker(output_file: str, output_queue: Queue, size: int = None):
    with open(output_file, 'w') as fout, tqdm(total=size) as pbar:
        while True:
            example = output_queue.get()
            if type(example) is str and example == 'DONE':
                break
            pbar.update(1)
            fout.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='strategyqa', choices=
                        ['strategyqa', 'hotpotqa', '2wikihop', 'wikisum_all_beir', 'eli5', 'wow', 'wow_train_1k', 'asqa', 'mmlu', 'lmdata'])
    parser.add_argument('--model', type=str, default='code-davinci-002', choices=['code-davinci-002', 'text-davinci-002', 'text-davinci-003', 'gpt-3.5-turbo-0301'])
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--index_name', type=str, default='test')
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--file_lock', type=str, default=None)
    parser.add_argument('--openai_keys', type=str, default=[], help='openai keys', nargs='+')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_num_examples', type=int, default=None)
    parser.add_argument('--fewshot', type=int, default=0)
    parser.add_argument('--max_generation_len', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)

    parser.add_argument('--build_index', action='store_true')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    args.multiprocess = len(args.openai_keys) > 1
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load retrieval corpus and index
    corpus, queries, qrels = None, None, None
    if args.build_index:
        if args.input:
            corpus, queries, qrels = GenericDataLoader(data_folder=args.input).load(split='dev')
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
        'frequency': 64,
        'boundary': [],
        #'boundary': ['Intermediate answer:'],
        #'boundary': ['")]'],
        #'boundary': ['. '],
        'use_gold': False,
        'use_gold_iterative': False,
        'max_query_length': 64,
        'use_full_input_as_query': True,
        'retrieval_at_beginning': False,
        'look_ahead_steps': 64,
        'look_ahead_truncate_at_boundary': 'sentence',
        'look_ahead_filter_prob': None,
        'look_ahead_mask_prob': None,
        'look_ahead_boundary': [],
        'only_use_look_ahead': True,
        'retrieval_trigers': [],
        #'retrieval_trigers': [('Follow up:', 'Intermediate answer:')],
        #'retrieval_trigers': [('\[Search\("', '")]')],
        #'retrieval_trigers': [(None, '. ')],
        'force_generate': None,
        'forbid_generate_step': None,
        'truncate_at_prob': 0.0,
        'truncate_at_boundary': 'sentence',
        'append_retrieval': False,
        'use_ctx_for_examplars': 'ret',
        'use_retrieval_instruction': False,
        'format_reference_method': 'searchresults',
        'ctx_position': 'before_case',
        'prompt_type': 'cot',
        'ctx_increase': 'replace',
        'add_ref_suffix': None,
        'add_ref_prefix': None,
        'debug': args.debug,
    }
    retrieval_kwargs['final_stop_sym'] = '!@#$%^&*()\n\n)(*&^%$#@!' if args.dataset == 'lmdata' else '\n\n'
    qagent = QueryAgent(
        model=args.model,
        tokenizer=prompt_tokenizer,
        max_generation_len=args.max_generation_len,
        retrieval_kwargs=retrieval_kwargs,
        temperature=args.temperature)
    if retrieval_kwargs['use_retrieval_instruction']:
        CtxPrompt.ret_instruction = RetrievalInstruction(method=retrieval_kwargs['use_retrieval_instruction'])
    CtxPrompt.retrieval_kwargs = retrieval_kwargs
    CtxPrompt.format_reference_method = retrieval_kwargs['format_reference_method']
    CtxPrompt.ctx_position = retrieval_kwargs['ctx_position']
    CtxPrompt.add_ref_suffix = retrieval_kwargs['add_ref_suffix']
    CtxPrompt.add_ref_prefix = retrieval_kwargs['add_ref_prefix']

    if args.multiprocess:  # start query processes
        lock = Lock()
        CustomManager.register('KeyManager', KeyManager)
        manager = CustomManager()
        manager.start()
        key_manager = manager.KeyManager(args.openai_keys)
        logging.info(f'#keys {len(key_manager._getvalue())}')
        input_queue = Queue()
        output_queue = Queue()
        processes = []
        for _ in range(len(key_manager._getvalue())):
            p = Process(target=query_agent_worker, args=(qagent, key_manager, lock, input_queue, output_queue))
            p.daemon = True
            p.start()
            processes.append(p)
    else:
        key_manager = KeyManager(args.openai_keys)

    # load data
    if args.dataset == 'strategyqa':
        data = StrategyQA(args.input, prompt_type=retrieval_kwargs['prompt_type'])
    elif args.dataset == 'hotpotqa':
        data = HotpotQA('validation', prompt_type=retrieval_kwargs['prompt_type'])
        use_gold_func = HotpotQA.get_gold_ctxs
    elif args.dataset == '2wikihop':
        data = WikiMultiHopQA(args.input, prompt_type=retrieval_kwargs['prompt_type'])
        use_gold_func = WikiMultiHopQA.get_gold_ctxs
    elif args.dataset == 'eli5':
        data = ELI5(prompt_type=retrieval_kwargs['prompt_type'])
        use_gold_func = ELI5.get_gold_ctxs
    elif args.dataset == 'asqa':
        data = ASQA(json_file=args.input, prompt_type=retrieval_kwargs['prompt_type'])
        use_gold_func = ASQA.get_gold_ctxs
    elif args.dataset == 'wow':
        data = WoW(prompt_type=retrieval_kwargs['prompt_type'])
        use_gold_func = WoW.get_gold_ctxs
    elif args.dataset == 'wow_train_1k':
        data = WoWLong(jsonl_file=args.input, prompt_type=retrieval_kwargs['prompt_type'])
        use_gold_func = WoWLong.get_gold_ctxs
    elif args.dataset == 'wikisum_all_beir':
        data = WikiSum(args.input, prompt_type=retrieval_kwargs['prompt_type'])
        use_gold_func = WikiSum.get_gold_ctxs
    elif args.dataset == 'mmlu':
        tasks = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
        'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science',
        'college_mathematics', 'college_medicine', 'college_physics', 'computer_security',
        'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics',
        'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',
        'high_school_computer_science', 'high_school_european_history', 'high_school_geography',
        'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics',
        'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics',
        'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
        'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management',
        'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
        'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
        'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
        'sociology', 'us_foreign_policy', 'virology', 'world_religions']
        data = MMLU(tasks=tasks, prompt_type=retrieval_kwargs['prompt_type'])
        use_gold_func = MMLU.get_gold_ctxs
    elif args.dataset == 'lmdata':
        data = LMData(args.input, prompt_type=retrieval_kwargs['prompt_type'])
        use_gold_func = LMData.get_gold_ctxs
    else:
        raise NotImplementedError
    if qagent.use_ctx_for_examplars == 'gold':
        logging.info('gold ctx for examplars')
        data.retrieval_augment_examplars(qagent, use_gold=use_gold_func)
    elif qagent.use_ctx_for_examplars == 'ret':
        logging.info('retrieve ctx for examplars')
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

    # create dir
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if not args.multiprocess:  # query for one process
        key = key_manager.get_key()
        with tqdm(total=len(data)) as pbar, open(args.output, 'w') as fout:
            for b in range(0, len(data), args.batch_size):
                batch = data.select(range(b, min(b + args.batch_size, len(data))))
                prompts = [CtxPrompt.from_dict(example) for example in batch]
                generations, probs, retrievals, traces = qagent.prompt(prompts, api_key=key)
                retrievals = retrievals or [None] * len(generations)
                traces = traces or [None] * len(generations)
                for example, generation, prob, retrieval, trace in zip(batch, generations, probs, retrievals, traces):
                    example['output'] = generation
                    example['output_prob'] = prob
                    example['retrieval'] = retrieval
                    example['trace'] = trace
                    fout.write(json.dumps(example) + '\n')
                pbar.update(len(batch))
        key_manager.return_key(key)
    else:  # query for multi-process
        # start write process
        write_p = Process(target=write_worker, args=(args.output, output_queue, len(data)))
        write_p.daemon = True
        write_p.start()

        # feed data
        for b in range(0, len(data), args.batch_size):
            batch = data.select(range(b, min(b + args.batch_size, len(data))))
            input_queue.put(batch)

        # feed finish token
        for _ in processes:
            input_queue.put('DONE')
        for p in processes:
            p.join()
        output_queue.put('DONE')
        write_p.join()

        # report key performance
        logging.info('keys performance')
        logging.info(key_manager._getvalue().get_report())
        manager.shutdown()
