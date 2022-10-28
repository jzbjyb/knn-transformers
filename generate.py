from pickle import FALSE
from typing import List, Dict, Tuple, Any, Union, Callable
import contextlib
import argparse
import json
import re
import math
import logging
import sys
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils import setup_multi_gpu_slurm
from memtrans import MemTransWrapper
from models.t5 import T5ForConditionalGeneration

logger = logging.getLogger(__name__)
logger.setLevel(20)

class GenerationWrapper(object):
    def __init__(
        self, 
        model: AutoModelForSeq2SeqLM, 
        tokenizer: AutoTokenizer,
        args):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self._all_tokens = list(self.tokenizer.get_vocab().values())

        self.source_prefix: str = args.source_prefix
        self.source_suffix: str = args.source_suffix
        self.evidence_prefix: str = args.evidence_prefix
        self.evidence_suffix: str = args.evidence_suffix
        self.use_evidence: str = args.use_evidence
        self.add_question_mark: bool = True
        self.add_period: bool = True
        self.max_evidence_len: int = args.max_evidence_len
        self.target_as_prefix_len: int = args.target_as_prefix_len  # number of tokens in the target used as prefix

        self.batch_size: int = args.batch_size
        self.gen_args: Dict[str, Any] = {
            'max_length': args.max_gen_len,
        }

    @property
    def device(self):
        return self.model.device
    
    def clean_by_tokenizer(self, text: str, max_length: int = None):
        text = self.tokenizer.encode(text)
        if max_length:
            text = text[:max_length]
        return self.tokenizer.decode(text, skip_special_tokens=True)
    
    def split_by_tokenizer(self, text: str, split_length: int):
        text = self.tokenizer.encode(text, add_special_tokens=False)
        split_length = max(min(split_length, len(text) - 1), 0)  # at least keep one token for the second piece
        first, second = text[:split_length], text[split_length:]
        first = self.tokenizer.decode(first, skip_special_tokens=True)
        second = self.tokenizer.decode(second, skip_special_tokens=True)
        return first, second
    
    def load_data(
        self, 
        data_file: str, 
        shard_id: int = 0, 
        num_shards: int = 1,
        max_num_examples: int = None,
        process_exmaple_func: Callable = None,
        debug: bool = False) -> Tuple[List, List, List]:
        # TODO: for simplicity we reuse the en-zh translation dataset
        sources: List[str] = []
        targets: List[str] = []
        
        has_decoder_prefix = self.use_evidence in {'decoder_prefix', 'fixed'} or self.target_as_prefix_len > 0
        decoder_prefixes: List[str] = [] if has_decoder_prefix else None
        
        with open(data_file, 'r') as fin:
            prev_source = None
            for l in fin:
                
                example = json.loads(l)['translation']
                example = process_exmaple_func(example) if process_exmaple_func else example

                source = example['en'].strip()

                # skip duplicate source if there's no evidence or it's fixed
                if self.use_evidence in {'no', 'fixed'} and source == prev_source:
                    continue
                prev_source = source
                
                # process source
                if self.add_question_mark and re.search('[?!.]$', source) is None:
                    source += '?'
                source = self.source_prefix + source + self.source_suffix
                
                # process evidence
                dp = ''
                if self.use_evidence != 'no':
                    if self.use_evidence == 'fixed':  # fixed evidence (i.e., instruction)
                        dp += self.evidence_suffix  # TODO: use another argumenet?
                    else:  # specific evidence
                        evi = example['decoder_prefix'].strip()
                        if self.max_evidence_len:
                            evi = self.clean_by_tokenizer(evi, max_length=self.max_evidence_len)
                        if self.add_period and re.search('[?!.]$', evi) is None:
                            evi += '.'
                        evi = self.evidence_prefix + evi + self.evidence_suffix
                        if self.use_evidence == 'decoder_prefix':
                            dp += evi
                        elif self.use_evidence == 'encoder_suffix':
                            source = source + ' ' + evi
                        elif self.use_evidence == 'encoder_prefix':
                            source = evi + ' ' + source
                
                # process target
                target = example['zh'].strip()
                if self.target_as_prefix_len > 0:
                    target_prefix, target = self.split_by_tokenizer(target, split_length=self.target_as_prefix_len)
                    dp = f'{dp} {target_prefix}' if len(dp) else target_prefix

                # save
                sources.append(source)
                targets.append(target)
                if has_decoder_prefix:
                    decoder_prefixes.append(dp)
                
                if debug:
                    print('SOURCE\t', source)
                    print('TARGET\t', target)
                    print('PREFIX\t', dp)
                    input()

                if max_num_examples and len(sources) >= max_num_examples:
                    break

        total_count = len(sources)
        assert len(sources) == len(targets)
        if has_decoder_prefix:
            assert len(sources) == len(decoder_prefixes)

        # shard
        shard_size = math.ceil(total_count / num_shards)
        shard_start = shard_id * shard_size
        shard_end = min(shard_start + shard_size, total_count)
        sources = sources[shard_start:shard_end]
        targets = targets[shard_start:shard_end]
        if has_decoder_prefix:
            decoder_prefixes = decoder_prefixes[shard_start:shard_end]

        logger.info(f'loaded data "{data_file}" from {shard_start} to {shard_end}')

        return sources, targets, decoder_prefixes, (shard_start, shard_end)

    def generate_batch(
        self,
        sources: List[str],
        targets: List[str],
        decoder_prefixes: List[str] = None,
        label_padding: int = -100,
        only_evaluate: bool = False,
        dry_run: bool = False) -> List[str]:

        # decoder prefix function
        if decoder_prefixes:
            assert len(sources) == len(decoder_prefixes) == len(targets)
            prefix_tokens_ids = [self.tokenizer(prefix, add_special_tokens=False)['input_ids'] for prefix in decoder_prefixes]
            def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
                if input_ids.shape[-1] > len(prefix_tokens_ids[batch_id]):
                    return self._all_tokens
                return prefix_tokens_ids[batch_id][input_ids.shape[-1] - 1]
            targets = [f'{dp} {t}' for t, dp in zip(targets, decoder_prefixes)]  # prepend the prefix to targets
        else:
            prefix_allowed_tokens_fn = None

        # tokenize
        sources = self.tokenizer.batch_encode_plus(
            sources, return_tensors='pt', padding=True, truncation=True)
        sources = {k: v.to(self.device) for k, v in sources.items()}
        targets = self.tokenizer.batch_encode_plus(
            targets, return_tensors='pt', padding=True, truncation=True, max_length=self.gen_args['max_length'])
        labels = targets['input_ids'].to(self.device)
        labels[labels == self.tokenizer.pad_token_id] = label_padding

        if not dry_run:  # generate
            with torch.no_grad():
                if not only_evaluate:
                    output = self.model.generate(**sources, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, **self.gen_args)
                else:
                    output = []
                _ = self.model(**sources, labels=labels)
        else:
            return [(labels != label_padding).sum().item()]

        # detokenize
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return output
    
    def generate(
        self,
        sources: List[str],
        targets: List[str],
        decoder_prefixes: List[str] = None,
        output_file: str = None,
        only_evaluate: bool = False,
        dry_run: bool = False) -> List[str]:

        output: List[str] = []
        with open(output_file, 'w') if output_file else contextlib.nullcontext() as fout, tqdm(total=len(sources)) as pbar:
            for b in range(0, len(sources), self.batch_size):
                batch_s = sources[b : b + self.batch_size]
                batch_t = targets[b : b + self.batch_size] if targets else None
                batch_dp = decoder_prefixes[b : b + self.batch_size] if decoder_prefixes else None
                batch_o = self.generate_batch(batch_s, targets=batch_t, decoder_prefixes=batch_dp, only_evaluate=only_evaluate, dry_run=dry_run)

                if output_file:
                    for i, o in enumerate(batch_o):
                        # detokenized everything
                        s = self.clean_by_tokenizer(batch_s[i])
                        t = self.clean_by_tokenizer(batch_t[i]) if batch_t else ''
                        dp = self.clean_by_tokenizer(batch_dp[i]) if batch_dp else ''
                        fout.write(f'{s}\t{t}\t{o}\t{dp}\n')

                output.extend(batch_o)
                pbar.update(len(batch_s))
    
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('--data_file', type=str, required=True, help='data file')
    parser.add_argument('--out_file', type=str, default=None, help='output file')
    parser.add_argument('--source_prefix', type=str, default='', help='source prefix')
    parser.add_argument('--source_suffix', type=str, default='', help='source suffix')
    parser.add_argument('--evidence_prefix', type=str, default='', help='decoder prefix prefix')
    parser.add_argument('--evidence_suffix', type=str, default='', help='decoder prefix suffix')
    parser.add_argument('--use_evidence', type=str, default='no', 
        choices=['no', 'encoder_suffix', 'encoder_prefix', 'decoder_prefix', 'fixed'], help='use evidence in which position')
    parser.add_argument('--max_gen_len', type=int, default=256, help='max generation length')
    parser.add_argument('--max_evidence_len', type=int, default=128, help='max evidence length')
    parser.add_argument('--target_as_prefix_len', type=int, default=0, help='number of tokens in the target used as prefix')

    # datastore args
    parser.add_argument('--dstore_dir', type=str, default=None, help='datastore directory')
    parser.add_argument('--dstore_size', type=int, default=None, help='datastore size')

    # model args
    parser.add_argument('--model', type=str, required=True, help='model')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--stage', type=str, default='retrieve', choices=['save', 'retrieve'], help='save or retrieve')
    parser.add_argument('--retrieval_topk', type=int, default=0, help='topk tokens retrieved in decoder. 0 deactivates retreival')
    parser.add_argument('--retrieval_layers', type=str, default='[0]', help='python code of layers, e.g., list(range(24)) for all layers')
    parser.add_argument('--retrieval_track', type=str, default=False, help='file to track retrieval')
    parser.add_argument('--skip_retrieval_steps', type=int, default=0, help='number of steps to skip retrieval')
    parser.add_argument('--accum_retrieval_steps', type=int, default=0, help='number of accumulation steps for retrieval')
    parser.add_argument('--retrieval_every_steps', type=int, default=1, help='block-wise retrieval')
    parser.add_argument('--max_retrieval_times', type=int, default=None, help='max number of retrieval to perform')
    parser.add_argument('--filter_topk', type=int, default=0, help='filter_topk')
    parser.add_argument('--filter_order', type=str, default='original', help='filter_order')
    parser.add_argument('--only_use_head_idx', type=int, default=-1, help='head index to use')
    parser.add_argument('--num_ctxs', type=int, default=1, help='num of ctxs to retrieve')
    parser.add_argument('--ctx_order', type=str, default='parallel', help='how to ues multiple ctxs')
    args = parser.parse_args()
    args.is_save = args.stage == 'save'
    args.use_retrieval = args.is_save or args.retrieval_topk > 0
    args.retrieval_layers = eval(args.retrieval_layers)

    # logging config
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # setup slurm
    setup_multi_gpu_slurm(args)
    logger.info(args)

    # modify output path
    args.out_file = (args.out_file + f'.{args.global_rank}') if type(args.out_file) is str else args.out_file
    args.retrieval_track = (args.retrieval_track + f'.{args.global_rank}') if type(args.retrieval_track) is str else args.retrieval_track

    # dstore device
    args.dstore_device = torch.device('cpu') if len(args.retrieval_layers) > 3 else args.device

    # load model
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    wrapper = GenerationWrapper(model, tokenizer, args)

    # load data
    process_exmaple_func = None
    if args.is_save:  # use "decoder_prefix" as target
        def process_exmaple_func(example: Dict):
            example['zh'] = example['decoder_prefix']
            return example
    sources, targets, decoder_prefixes, (shard_start, shard_end) = wrapper.load_data(
        args.data_file, shard_id=args.global_rank, num_shards=args.world_size, process_exmaple_func=process_exmaple_func)

    # prepare for "save" stage
    only_evaluate = False
    if args.is_save:
        num_tokens = wrapper.generate(sources, targets=targets, decoder_prefixes=decoder_prefixes, dry_run=True)
        num_tokens = sum(num_tokens)
        logger.info(f'total eval tokens: {num_tokens}')
        args.dstore_size = num_tokens
        only_evaluate = True

    if args.use_retrieval:  # add retrieval
        ret_wrapper = MemTransWrapper(
            dstore_size=args.dstore_size, dstore_dir=args.dstore_dir,
            move_dstore_to_mem=True, device=args.dstore_device,
            recompute_dists=True, retrieval_layers=args.retrieval_layers,
            k=args.retrieval_topk, stage=args.stage, track=args.retrieval_track, 
            by_ids=False, cache_indices=True,  # TODO: debug
            skip_retrieval_steps=args.skip_retrieval_steps, 
            accum_retrieval_steps=args.accum_retrieval_steps, 
            retrieval_every_steps=args.retrieval_every_steps,
            max_retrieval_times=args.max_retrieval_times,
            skip_first_token=True, add_after_first=True, 
            filter_topk=args.filter_topk, filter_order=args.filter_order, 
            only_use_head_idx=args.only_use_head_idx, 
            shard_start=shard_start,
            num_ctxs=args.num_ctxs, ctx_order=args.ctx_order)
        ret_wrapper.break_into(model)

    # generate
    wrapper.generate(
        sources, 
        targets=targets, 
        decoder_prefixes=decoder_prefixes, 
        only_evaluate=only_evaluate,
        output_file=args.out_file)
    
    if args.is_save:  # build index
        ret_wrapper.build_index()
    
    if args.use_retrieval:
        ret_wrapper.break_out()
