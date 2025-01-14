from typing import Tuple, Union, List, Dict
import os
import logging
import time
import types
import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from pathlib import Path
import torch_scatter
import faiss
import faiss.contrib.torch_utils
from transformers.models.t5.modeling_t5 import T5Attention

from knnlm import get_dstore_path
from utils import StridedTensor

logger = logging.getLogger(__name__)
logger.setLevel(20)

def get_index_path(dstore_dir, model_type, dstore_size, dimension, head_idx):
    return f'{dstore_dir}/index_{model_type}_{dstore_size}_{dimension}_{head_idx}.indexed'

class FixedRetrieval(object):
    def __init__(self, line2did_file: str, fid_file: str):
        line2did: Dict[int, int] = {}
        did2line: Dict[int, int] = {}
        with open(line2did_file, 'r') as fin:
            for l in fin:
                line, did = l.strip().split()
                line, did = int(line), int(did)
                line2did[line] = did
                did2line[did] = line
        
        self._offset = 0
        with open(fid_file, 'r') as fin:
            self.examples: List[Dict] = json.load(fin)
            for example in self.examples:
                example['ctxs'] = [did2line[int(ctx['id'])] for ctx in example['ctxs']]
    
    def get_ctxs(self, batch_size) -> List[List[int]]:
        start = self._offset
        end = min(start + batch_size, len(self.examples))
        print(start, end)
        self._offset = end
        return [self.examples[i]['ctxs'] for i in range(start, end)]

fixed_retrieval = FixedRetrieval(
    line2did_file='data/wow/val_astarget_selfprov_evidence.json.beir_dedup_ans.fid/line2docid.tsv', 
    fid_file='data/wow/val_astarget_selfprov_evidence.json.beir_dedup_ans.fid/dev.json')  

class MemTransAttnCoordinator(object):
    def __init__(self):
        self._key_length: int = None  # track the decoding step
        self._indices: torch.LongTensor = None  # (batch_size, topk) track the retrieved indices
    
    def get_or_save_indices(
        self, 
        key_length: int,
        indices: torch.LongTensor = None):  # (batch_size, topk)
        # get
        if self._key_length == key_length:  # hit
            return self._indices
        else:  # not hit
            if indices is None:
                return None
            else:  # save
                self._key_length = key_length
                self._indices = indices
    
    def clear(self):
        self._key_length = self._indices = None

class MemTransDatastore(object):
    def __init__(
        self, directory: str, 
        model_type: str, 
        size: int, 
        dimension: int, 
        n_heads: int, 
        move_dstore_to_mem: bool = False, 
        device: torch.device = None):
        self.directory = directory
        self.model_type = model_type
        self.size = size
        self.dimension = dimension
        self.n_heads = n_heads
        self.move_dstore_to_mem = move_dstore_to_mem
        self.device = torch.device('cpu') if device is None else device
        self.cur_idx = 0
        self.precision = np.float32
        self.load_or_init_dstore()
        self.head2ids: Dict[int, List] = defaultdict(list)  # each item in list is (batch_size, final_topk)
    
    @property
    def use_cuda(self):
        return self.device.type == 'cuda'
    
    @property
    def num_docs(self):
        return len(self.lengths)

    def get_index_path(self, head_idx: int) -> str:
        return get_index_path(self.directory, self.model_type, self.size, self.dimension, head_idx=head_idx)
    
    def get_dstore_path(self) -> Tuple[str, str, str, str]:
        prefix = get_dstore_path(self.directory, self.model_type, self.size, self.dimension)
        key_file = f'{prefix}_keys.npy'
        val_file = f'{prefix}_vals.npy'
        tok_file = f'{prefix}_tokens.npy'
        id_file = f'{prefix}_ids.npy'
        return key_file, val_file, tok_file, id_file
    
    def load_or_init_dstore(self):
        start = time.time()
        key_file, val_file, tok_file, id_file = self.get_dstore_path()
        if os.path.exists(key_file):
            mode = 'r'
        else:
            mode = 'w+'
            Path(key_file).parent.mkdir(parents=True, exist_ok=True)
        self.keys = np.memmap(key_file, dtype=self.precision, mode=mode, shape=(self.n_heads, self.size, self.dimension))
        self.values = np.memmap(val_file, dtype=self.precision, mode=mode, shape=(self.n_heads, self.size, self.dimension))
        try:  # load tokens and ids if exists
            self.tokens = np.memmap(tok_file, dtype=np.int32, mode=mode, shape=(self.size))
            self.ids = np.memmap(id_file, dtype=np.int32, mode=mode, shape=(self.size))
            self.positions = np.arange(self.size, dtype=np.int32)  # TODO: make it a memmap?
        except:
            self.tokens = self.ids = self.positions = None
        logger.info(f'Loading dstore took {time.time() - start} s')
    
    def save_labels(
        self, 
        labels: torch.LongTensor,  # (batch_size, seq_length)
        decoder_input_ids: torch.LongTensor = None):  # (batch_size, seq_length)
        self._labels = labels
        self._decoder_input_ids = decoder_input_ids
    
    def get_labels(self):
        return self._labels, self._decoder_input_ids
    
    def save_decoder_input_ids(self, decoder_input_ids: torch.LongTensor):  # (batch_size, seq_length)
        self._decoder_input_ids = decoder_input_ids  # should be consistent with save_labels
    
    def get_decoder_input_ids(self):
        return self._decoder_input_ids

    def save_key_value(
        self,
        keys: torch.FloatTensor,  # (n_heads, n_tokens, dim_per_head)
        values: torch.FloatTensor,  # (n_heads, n_tokens, dim_per_head)
        tokens: torch.LongTensor = None,  # (n_tokens)
        ids: torch.LongTensor = None,  # (n_tokens)
        ):
        assert keys.size(1) == values.size(1) == tokens.size(0) == ids.size(0)
        nh, nt, dim = keys.size()
        assert nh == self.n_heads and dim == self.dimension, 'keys and values are in a wrong shape'

        # truncate
        assert self.cur_idx <= self.size, 'datastore overflow'
        if self.cur_idx + nt > self.size:
            nt = self.size - self.cur_idx
            keys = keys[:, :nt]
            values = values[:, :nt]
            tokens = tokens[:nt] if tokens is not None else tokens
            ids = ids[:nt] if ids is not None else ids
        
        # save to memmap
        try:
            self.keys[:, self.cur_idx:(nt + self.cur_idx)] = keys.cpu().numpy().astype(self.precision)
            self.values[:, self.cur_idx:(nt + self.cur_idx)] = values.cpu().numpy().astype(self.precision)
            if tokens is not None:
                self.tokens[self.cur_idx:(nt + self.cur_idx)] = tokens.cpu().numpy().astype(np.int32)
                self.ids[self.cur_idx:(nt + self.cur_idx)] = ids.cpu().numpy().astype(np.int32)
        except ValueError as ex:
            logger.error(f'Error saving datastore with mode {self.keys.mode}, did you try to save an already existing datastore?')
            logger.error(f'Delete the files {self.keys.filename} and {self.values.filename} and try again')
            raise ex

        self.cur_idx += nt

    def build_index(self, batch_size: int):
        self.indices = []
        for h in range(self.n_heads):
            index_name = self.get_index_path(head_idx=h)
            index = faiss.IndexFlatIP(self.dimension)
            self.indices.append(index)
            #index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)  # TODO: multi-gpu
            
            keys_one_head = self.keys[h][:self.cur_idx]  # remove unused slots
            if not index.is_trained:  # use all keys for training
                index.train(keys_one_head.astype(np.float32))
            for b in tqdm(range(0, len(keys_one_head), batch_size), desc='index adding'):
                batch = keys_one_head[b:b + batch_size].copy()
                index.add(torch.tensor(batch.astype(np.float32)))
            
            faiss.write_index(index, f'{index_name}')
    
    def load_index(self, build_offset: bool = False):
        # move dstore
        if self.move_dstore_to_mem:
            start = time.time()
            self.keys = torch.from_numpy(self.keys[:])
            self.values = torch.from_numpy(self.values[:])
            if self.tokens is not None:
                self.tokens = torch.from_numpy(self.tokens[:])
                self.ids = torch.from_numpy(self.ids[:])
                self.positions = torch.from_numpy(self.positions[:]).long()
            # TODO: move to gpu
            #self.keys = self.keys.to(self.device)
            #self.values = self.values.to(self.device)
            logger.info('Moving to memory took {} s'.format(time.time() - start))
        
        # build start/end offsets of each id (assuming ids are consecutive and start with 0)
        if build_offset:
            self.start_offsets = []  # inclusive
            self.end_offsets = []  # exclusive
            prev_id = -1
            for i, _id in enumerate(self.ids):
                if _id == prev_id + 1:  # new id
                    if len(self.start_offsets):  # not the first
                        self.end_offsets.append(i)
                    self.start_offsets.append(i)
                    prev_id = _id
                elif _id == prev_id:
                    pass
                else:
                    raise ValueError('ids are not consecutive')
            self.end_offsets.append(len(self.ids))
            assert len(self.start_offsets) == len(self.end_offsets)
            self.start_offsets = torch.tensor(self.start_offsets).to(self.device)
            self.end_offsets = torch.tensor(self.end_offsets).to(self.device)
            self.lengths = self.end_offsets - self.start_offsets

            # build strided tensor
            # (n_tokens, n_heads, dim_per_head)
            self.keys_strided = StridedTensor(self.keys.permute(1, 0, 2).contiguous(), self.lengths)
            # (n_tokens, n_heads, dim_per_head)
            self.values_strided = StridedTensor(self.values.permute(1, 0, 2).contiguous(), self.lengths)
            if self.tokens is not None and self.ids is not None:
                self.tokens_strided = StridedTensor(self.tokens, self.lengths)
                self.ids_strided = StridedTensor(self.ids, self.lengths)
                self.positions_strided = StridedTensor(self.positions, self.lengths)

        # load index
        self.indices = []
        res = faiss.StandardGpuResources()
        start = time.time()
        for h in range(self.n_heads):
            index_name = self.get_index_path(head_idx=h)
            cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
            if self.use_cuda:  # move index to gpu
                start = time.time()
                gpu_index = faiss.index_cpu_to_gpu(res, self.device.index, cpu_index)
            else:
                gpu_index = cpu_index
            self.indices.append(gpu_index)
        logger.info(f'Loading index took {time.time() - start} s')

    def _get_knns_single_head(
        self,
        ret_head_idx: int = None,  # head on which to perform knn retrieval
        select_head_idx: int = None,  # head from which to select keys and values
        queries: torch.FloatTensor = None,  # (n_heads, batch_size, dim)
        indices: torch.LongTensor = None,  # (batch_size, n_ctxs, topk)  used to avoid retrieval again
        topk: int = 0,
        return_all: bool = False,
        return_indices: bool = False):
        
        if indices is None:  # retreival
            index = self.indices[ret_head_idx]
            # TODO: add torch tensor (GPU) support
            indices = index.search(np.ascontiguousarray(queries[ret_head_idx].cpu().numpy()), topk)[1]  # (batch_size, topk)
            indices = torch.from_numpy(indices).to(self.keys.device).unsqueeze(1)  # (batch_size, 1 (n_ctxs), topk)
        if return_indices:
            return indices

        # select
        ret_k = self.keys[select_head_idx][indices]  # (batch_size, n_ctxs, topk, dim)
        ret_v = self.values[select_head_idx][indices]  # (batch_size, n_ctxs, topk, dim)
        ret_t = ret_id = None
        if return_all:
            ret_t = self.tokens[indices]  # (batch_size, n_ctxs, topk)
            ret_id = self.ids[indices]  # (batch_size, n_ctxs, topk)
        return ret_k, ret_v, ret_t, ret_id, indices
    
    def get_knns_by_indices(
        self,
        indices: torch.LongTensor,  # (batch_size, n_ctxs, topk), indices from previous layers
        device: torch.device,
        return_all: bool = False,
    ):
        ret_ks, ret_vs, ret_ts, ret_ids = [], [], [], []
        for h in range(self.n_heads):
            ret_k, ret_v, ret_t, ret_id, _ = self._get_knns_single_head(
                select_head_idx=h, indices=indices, return_all=return_all)
            ret_ks.append(ret_k)
            ret_vs.append(ret_v)
            ret_ts.append(ret_t)
            ret_ids.append(ret_id)

        ret_ks = torch.stack(ret_ks, dim=0).to(device)  # (n_heads, batch_size, n_ctxs, topk, dim)
        ret_vs = torch.stack(ret_vs, dim=0).to(device)  # (n_heads, batch_size, n_ctxs, topk, dim)
        if return_all:
            ret_ts = torch.stack(ret_ts, dim=0).to(device)  # (n_heads, batch_size, n_ctxs, topk)
            ret_ids = torch.stack(ret_ids, dim=0).to(device)  # (n_heads, batch_size, n_ctxs, topk)
            return ret_ks, ret_vs, ret_ts, ret_ids
        return ret_ks, ret_vs, None, None

    def get_knns_block(
        self, 
        queries: torch.FloatTensor,  # (n_heads, batch_size, seq_len, dim)
        topk: int,
        final_topk: int = 1,  # the final number of ctxs returned
        only_use_head_idx: Union[int, List[int]] = -1,  # only use a single head to retrieve
        skip_first_token: bool = False,
        return_all: bool = False,
        debug: bool = False):

        assert topk, 'topk should be positive'
        assert only_use_head_idx != -1, 'not implemented'

        only_use_head_idxs = [only_use_head_idx] if type(only_use_head_idx) is not list else only_use_head_idx
        assert len(only_use_head_idxs) == 1 or debug, 'can only use multiple heads in debug mode'

        ibs = 1  # inner batch size (to save memory)
        rerank_topk = topk  # num of docs to rerank  # TODO: use values different from topk
        doc_seq_len = topk  # TODO: use values different from topk

        all_queries = queries
        ori_device = all_queries.device
        nh, bs, sl, dim = all_queries.size()

        for only_use_head_idx in only_use_head_idxs:
            index = self.indices[only_use_head_idx]

            _queries = all_queries[only_use_head_idx].contiguous()  # (batch_size, seq_len, dim)
            _queries_to_faiss = _queries.view(-1, dim).cpu().numpy()  # (batch_size * seq_len, dim) TODO: add torch tensor (GPU) support
            _scores, _indices = index.search(_queries_to_faiss, topk)  # (batch_size * seq_len, topk) * 2
            _scores = torch.from_numpy(_scores).to(ori_device).view(bs, sl, topk)  # (batch_size, seq_len, topk)
            _indices = torch.from_numpy(_indices).to(self.ids.device).view(bs, sl, topk)  # (batch_size, seq_len, topk)
            _ret_ids = self.ids[_indices].to(ori_device)  # (batch_size, seq_len, topk)
            _ids = []  # list of ids for retrieval (i_batch_size, final_topk)
            #_ctxs: List[List[int]] = fixed_retrieval.get_ctxs(batch_size=bs)

            for b in range(0, bs, ibs):
                # get unique ids
                queries = _queries[b:b + ibs]  # (i_batch_size, seq_len, dim)
                scores = _scores[b:b + ibs]  # (i_batch_size, seq_len, topk)
                ret_ids = _ret_ids[b:b + ibs]  # (i_batch_size, seq_len, topk)
                unique_ret_ids, ret_ids = torch.unique(ret_ids, return_inverse=True)  # (uni,) (i_batch_size, seq_len, topk) uni is the number of unique ids
                uni = unique_ret_ids.size(0)
                #ctxs = _ctxs[b:b + ibs]
                
                # max
                lowest = scores.min()
                agg_scores = torch.zeros(ibs, sl, uni).to(ori_device) + lowest  # (i_batch_size, seq_len, uni)
                agg_mask = torch.zeros(ibs, sl, uni).to(ori_device)  # (i_batch_size, seq_len, uni)
                agg_scores = torch_scatter.scatter_max(scores, ret_ids, out=agg_scores, dim=-1)[0]
                agg_mask = torch_scatter.scatter_max(torch.ones_like(scores), ret_ids, out=agg_mask, dim=-1)[0]
                agg_scores = agg_scores * agg_mask  # assume zero for absence

                # sum
                agg_mask = agg_mask.any(1).to(agg_mask)  # (i_batch_size, uni)
                agg_scores = agg_scores.sum(1)  # (i_batch_size, uni)
                agg_scores = agg_scores + agg_mask.log()  # assume -inf for absence

                # rerank
                sort_indices = torch.topk(agg_scores, min(rerank_topk, uni), dim=-1).indices  # (i_batch_size, n_cand)
                cand_ret_ids = unique_ret_ids[sort_indices]  # (i_batch_size, n_cand)
                #cand_ret_ids = torch.tensor(ctxs).to(cand_ret_ids)
                n_cand = cand_ret_ids.size(-1)

                # get docs
                ret_ks, _, _, _, _, ret_mask = self.get_knns_by_ids(
                    cand_ret_ids, topk=doc_seq_len, skip_first_token=skip_first_token, return_all=False)  # (n_heads, i_batch_size, n_cand, doc_seq_len, dim), (i_batch_size, n_cand, doc_seq_len)
                ret_ks = ret_ks[only_use_head_idx]  # (i_batch_size, n_cand, doc_seq_len, dim)
                scores = torch.einsum('bqd,bckd->bcqk', queries, ret_ks)  # (i_batch_size, n_cand, seq_len, doc_seq_len)

                # max-sum TODO: add padding and masking
                scores = (scores + ret_mask.to(scores).log().unsqueeze(-2)).max(-1).values.sum(-1)  # (i_batch_size, n_cand)
                
                # final rank
                sort_indices = torch.topk(scores, min(n_cand, final_topk), dim=-1).indices  # (i_batch_size, final_topk)
                cand_ret_ids = torch.gather(cand_ret_ids, 1, sort_indices)  # (i_batch_size, final_topk)
                _ids.append(cand_ret_ids)

                if debug:
                    print(f'||{only_use_head_idx}||{cand_ret_ids[0, 0].item()}')

            _ids = torch.cat(_ids, 0)  # (batch_size, final_topk) TODO: different inner batches have different number of returned docs?
            if debug:
                self.head2ids[only_use_head_idx].append(_ids)

            # (n_heads, batch_size, final_topk, doc_seq_len, dim) * 2, (batch_size, final_topk, doc_seq_len) * 3
            ret_ks, ret_vs, ret_ts, ret_ids, indices, _ = self.get_knns_by_ids(
                _ids, topk=doc_seq_len, skip_first_token=skip_first_token, return_all=return_all)

        return ret_ks, ret_vs, ret_ts, ret_ids, indices  # return the results of the last head

    def get_knns(
        self, 
        queries: torch.FloatTensor,  # (n_heads, batch_size, dim) or (n_heads, batch_size, seq_len, dim)
        topk: int,
        final_topk: int = 1,
        only_use_head_idx: int = -1,  # only use a single head to retrieve
        skip_first_token: bool = False,
        return_all: bool = False):

        if len(queries.size()) == 4:
            return self.get_knns_block(
                queries, 
                topk=topk, 
                final_topk=final_topk,
                only_use_head_idx=only_use_head_idx, 
                skip_first_token=skip_first_token, 
                return_all=return_all)

        ori_device = queries.device
        nh, bs, dim = queries.size()
        if topk <= 0:  # return "empty" tensors
            ret_ks = torch.zeros(nh, bs, 1, 0, dim).to(queries)  # (n_heads, batch_size, 1 (n_ctxs), topk, dim)
            ret_vs = torch.zeros(nh, bs, 1, 0, dim).to(queries)  # (n_heads, batch_size, 1 (n_ctxs), topk, dim)
            indices = torch.zeros(bs, 1, 0).long().to(ori_device)  # (batch_size, 1 (n_ctxs), topk)
            if return_all:
                ret_ts = torch.zeros(nh, bs, 1, 0).long().to(ori_device)  # (n_heads, batch_size, 1 (n_ctxs), topk)
                ret_ids = torch.zeros(nh, bs, 1, 0).long().to(ori_device)  # (n_heads, batch_size, 1 (n_ctxs), topk)
                return ret_ks, ret_vs, ret_ts, ret_ids, indices
            return ret_ks, ret_vs, None, None, indices

        ret_ks, ret_vs, ret_ts, ret_ids = [], [], [], []

        if only_use_head_idx == -1:  # different heads retrieve separately
            for h in range(self.n_heads):
                ret_k, ret_v, ret_t, ret_id, indices = self._get_knns_single_head(
                    ret_head_idx=h, select_head_idx=h, queries=queries, topk=topk, return_all=return_all)
                ret_ks.append(ret_k)
                ret_vs.append(ret_v)
                ret_ts.append(ret_t)
                ret_ids.append(ret_id)
        else:
            indices = self._get_knns_single_head(
                ret_head_idx=only_use_head_idx, queries=queries, topk=topk, return_indices=True)
            for h in range(self.n_heads):
                ret_k, ret_v, ret_t, ret_id, _ = self._get_knns_single_head(
                    select_head_idx=h, indices=indices, return_all=return_all)
                ret_ks.append(ret_k)
                ret_vs.append(ret_v)
                ret_ts.append(ret_t)
                ret_ids.append(ret_id)

        ret_ks = torch.stack(ret_ks, dim=0).to(ori_device)  # (n_heads, batch_size, n_ctxs, topk, dim)
        ret_vs = torch.stack(ret_vs, dim=0).to(ori_device)  # (n_heads, batch_size, n_ctxs, topk, dim)
        if return_all:
            ret_ts = torch.stack(ret_ts, dim=0).to(ori_device)  # (n_heads, batch_size, n_ctxs, topk)
            ret_ids = torch.stack(ret_ids, dim=0).to(ori_device)  # (n_heads, batch_size, n_ctxs, topk)
            return ret_ks, ret_vs, ret_ts, ret_ids, indices
        return ret_ks, ret_vs, None, None, indices
    
    def filter_by_similarity(
        self,
        queries: torch.FloatTensor,  # (n_heads, batch_size, dim)
        ret_ks: torch.FloatTensor,  # (n_heads, batch_size, n_ctxs, seq_len, dim)
        ret_vs: torch.FloatTensor,  # (n_heads, batch_size, n_ctxs, seq_len, dim)
        ret_ts: torch.LongTensor = None,  # (n_heads, batch_size, n_ctxs, seq_len)
        ret_ids: torch.LongTensor = None,  # (n_heads, batch_size, n_ctxs, seq_len)
        indices: torch.LongTensor = None,  # (batch_size, n_ctxs, seq_len)
        topk: int = 1,
        order: str = 'original'):

        # merge multiple ctxs
        ret_ks = ret_ks.flatten(2, 3)
        ret_vs = ret_vs.flatten(2, 3)
        ret_ts = ret_ts.flatten(2, 3) if ret_ts is not None else None
        ret_ids = ret_ids.flatten(2, 3) if ret_ids is not None else None
        indices = indices.flatten(1, 2) if indices is not None else None

        seq_len, dim = ret_ks.shape[2:]
        topk = min(topk, seq_len)
        assert order in {'original', 'descending', 'ascending'}

        # compute similarity and choose topk
        scores = torch.einsum("nbd,nbsd->nbs", queries, ret_ks)  # (n_heads, batch_size, seq_len)
        
        # topk
        topk_inds = torch.topk(scores, topk, dim=-1, sorted=True)[1]  # (n_heads, batch_size, topk)
        if order == 'ascending':  # flip indices
            topk_inds = torch.flip(topk_inds, dims=[-1])
        elif order == 'original':  # follow the order of original appearance
            topk_inds = torch.sort(topk_inds, dim=-1)[0]

        # select
        if ret_ts is not None:
            ret_ts = torch.gather(ret_ts, 2, topk_inds)  # (n_heads, batch_size, topk)
        if ret_ids is not None:
            ret_ids = torch.gather(ret_ids, 2, topk_inds)  # (n_heads, batch_size, topk)
        if indices is not None:
            indices = torch.gather(indices, 1, topk_inds[0])  # (batch_size, topk)  TODO: support multi-head
        topk_inds = topk_inds.unsqueeze(-1).repeat(1, 1, 1, dim)  # (n_heads, batch_size, topk, dim)
        ret_ks = torch.gather(ret_ks, 2, topk_inds)  # (n_heads, batch_size, topk, dim)
        ret_vs = torch.gather(ret_vs, 2, topk_inds)  # (n_heads, batch_size, topk, dim)

        ret_ks = ret_ks.unsqueeze(2)
        ret_vs = ret_vs.unsqueeze(2)
        ret_ts = ret_ts.unsqueeze(2) if ret_ts is not None else None
        ret_ids = ret_ids.unsqueeze(2) if ret_ids is not None else None
        indices = indices.unsqueeze(1) if indices is not None else None

        return ret_ks, ret_vs, ret_ts, ret_ids, indices

    def get_knns_by_ids(
        self, 
        ids: torch.LongTensor,  # (batch_size) or (batch_size, n_cand)
        topk: int,
        skip_first_token: bool = False,
        return_all: bool = False):

        has_cand = len(ids.size()) > 1
        if has_cand:
            batch_size, n_cand = ids.size()
        ids = ids.view(-1)  # (batch_size) or (batch_size * n_cand)

        ret_ks = self.keys_strided.lookup(ids, output='padded')[0]  # (batch_size, seq_len, n_heads, dim)
        ret_vs = self.values_strided.lookup(ids, output='padded')[0]  # (batch_size, seq_len, n_heads, dim)
        indices, mask = self.positions_strided.lookup(ids, output='padded')  # (batch_size, seq_len) * 2
        if return_all:
            ret_ts = self.tokens_strided.lookup(ids, output='padded')[0]  # (batch_size, seq_len)
            ret_ids = self.ids_strided.lookup(ids, output='padded')[0]  # (batch_size, seq_len)

        if return_all:
            assert ret_ks.size(1) == ret_vs.size(1) == indices.size(1) == mask.size(1) == ret_ts.size(1) == ret_ids.size(1)
        else:
            assert ret_ks.size(1) == ret_vs.size(1) == indices.size(1) == mask.size(1)

        if skip_first_token:  # skip the first token which is usually the bos token (e.g., the pad token for T5)
            ret_ks = ret_ks[:, 1:]  # (batch_size, seq_len - 1, n_heads, dim)
            ret_vs = ret_vs[:, 1:]  # (batch_size, seq_len - 1, n_heads, dim)
            indices = indices[:, 1:]  # (batch_size, seq_len - 1)
            mask = mask[:, 1:]  # (batch_size, seq_len - 1)
            if return_all:
                ret_ts = ret_ts[:, 1:]  # (batch_size, seq_len - 1)
                ret_ids = ret_ids[:, 1:]  # (batch_size, seq_len - 1)

        seq_len = ret_ks.size(1)
        if seq_len > topk:  # truncate
            ret_ks = ret_ks[:, :topk]  # (batch_size, topk, n_heads, dim)
            ret_vs = ret_vs[:, :topk]  # (batch_size, topk, n_heads, dim)
            indices = indices[:, :topk]  # (batch_size, topk)
            mask = mask[:, :topk]  # (batch_size, topk)
            if return_all:
                ret_ts = ret_ts[:, :topk]  # (batch_size, topk)
                ret_ids = ret_ids[:, :topk]  # (batch_size, topk)

        ret_ks = ret_ks.permute(2, 0, 1, 3)  # (n_heads, batch_size, topk, dim)
        ret_vs = ret_vs.permute(2, 0, 1, 3)  # (n_heads, batch_size, topk, dim)
        if return_all:
            ret_ts = ret_ts.unsqueeze(0).repeat(self.n_heads, 1, 1)  # (n_heads, batch_size, topk)
            ret_ids = ret_ids.unsqueeze(0).repeat(self.n_heads, 1, 1)  # (n_heads, batch_size, topk)
        
        if has_cand:
            ret_ks = ret_ks.view(*ret_ks.shape[:1], batch_size, n_cand, *ret_ks.shape[2:])  # (n_heads, batch_size, n_cand, topk, dim)
            ret_vs = ret_vs.view(*ret_vs.shape[:1], batch_size, n_cand, *ret_vs.shape[2:])  # (n_heads, batch_size, n_cand, topk, dim)
            indices = indices.view(batch_size, n_cand, *indices.shape[1:])  # (batch_size, n_cand, topk)
            mask = mask.view(batch_size, n_cand, *mask.shape[1:])  # (batch_size, n_cand, topk)
            if return_all:
                ret_ts = ret_ts.view(batch_size, n_cand, *ret_ts.shape[1:])  # (batch_size, n_cand, topk)
                ret_ids = ret_ids.view(batch_size, n_cand, *ret_ids.shape[1:])  # (batch_size, n_cand, topk)

        if return_all:
            return ret_ks, ret_vs, ret_ts, ret_ids, indices, mask
        else:
            return ret_ks, ret_vs, None, None, indices, mask

class RetrievalTracker(object):
    def __init__(
        self,
        track_file: str,
        n_heads: int,
        topk: int,
        eos_token_id: int):
        self.track_file = track_file
        self.n_heads = n_heads
        self.topk = topk
        self.eos_token_id = eos_token_id
        self.handle = open(track_file + f'_h{n_heads}_k{topk}.txt', 'w') if track_file is not None else None
        # each item corresponds to one step
        self.predictions: List[torch.LongTensor] = []  # (seq_len, batch_size)
        self.retrieved_tokens: List[torch.LongTensor] = []  # (seq_len, batch_size, n_heads, topk)
        self.retrieved_ids: List[torch.LongTensor] = []  # (seq_len, batch_size, n_heads, topk)

    def add_single_step_batched(
        self, 
        prediction: torch.LongTensor,  # (batch_size)
        retrieved_token: torch.LongTensor,  # (batch_size, n_heads, topk)
        retrieved_id: torch.LongTensor):  # (batch_size, n_heads, topk)
        assert prediction.size(0) == retrieved_token.size(0) == retrieved_id.size(0)
        self.predictions.append(prediction)
        self.retrieved_tokens.append(retrieved_token)
        self.retrieved_ids.append(retrieved_id)
    
    def _write(self, text: str):
        if self.handle is None:
            print(text, end='')
        else:
            self.handle.write(text)

    def write(self):
        predictions = torch.stack(self.predictions, dim=1)  # (batch_size, seq_len)
    
        # pad with 0
        max_topk_idx = np.argmax([rt.size(-1) for rt in self.retrieved_tokens])
        max_topk = self.retrieved_tokens[max_topk_idx].size(-1)
        self.retrieved_tokens = [torch.zeros_like(self.retrieved_tokens[max_topk_idx]) if rt.size(-1) != max_topk else rt for rt in self.retrieved_tokens]
        self.retrieved_ids = [torch.zeros_like(self.retrieved_ids[max_topk_idx]) if ri.size(-1) != max_topk else ri for ri in self.retrieved_ids]

        # pack
        retrieved_tokens = torch.stack(self.retrieved_tokens, dim=1)  # (batch_size, seq_len, n_heads, topk)
        retrieved_ids = torch.stack(self.retrieved_ids, dim=1)  # (batch_size, seq_len, n_heads, topk)
        retrieved = torch.stack([retrieved_tokens, retrieved_ids], dim=-1)  # (batch_size, seq_len, n_heads, topk, 2)
        bs, sl, nh, topk = retrieved_tokens.size()
        agg = torch.cat([predictions.unsqueeze(-1), retrieved.flatten(2, 4)], dim=-1)  # (batch_size, seq_len, 1 + n_heads * topk * 2)
        for i in range(bs):
            for j in range(sl):
                if agg[i, j, 0].item() == self.eos_token_id:  # end of prediction
                    break
                s = agg[i, j].tolist()
                self._write(' '.join(map(str, s)) + '\n')
            self._write('\n')
        
        # clear cache
        self.predictions = []
        self.retrieved_tokens = []
        self.retrieved_ids = []

class MemTransAttn(object):
    def __init__(
        self, 
        dstore: MemTransDatastore, 
        topk: int, 
        eos_token_id: int,
        stage: str, 
        layer_index: int = 0,
        track: Union[bool, str] = False,  # track retrieved tokens by printing (bool) or writing to files (str)
        by_ids: bool = False,  # whether to retrieve documents by ids
        shard_start: int = 0,  # which id to start retrieving from
        skip_retrieval_steps: int = 0,  # number of decoding steps where retrieval is skipped
        accum_retrieval_steps: int = 0,  # accumulate steps to perform retrieval (i.e., block-wise retrieval)
        retrieval_for_next_step_at_layer: int = -1,  # retrieve indices for the next step at this layer
        retrieval_every_steps: int = 1,  # perform retrieval every retrieval_every_steps steps
        max_retrieval_times: int = None,  # max number of retrieval to performa
        skip_first_token: bool = False,  # skip the first token retrieved which is usually bos
        add_after_first: bool = False,  # add the retrieved tokens after the first token which is usually bos
        filter_topk: int = 0,
        filter_order: str = 'original',
        only_use_head_idx: int = -1,
        cache_indices: bool = False,
        mtac: MemTransAttnCoordinator = None,
        num_ctxs: int = 1,  # num of ctxs retrieved
        ctx_order: str = 'parallel',  # the ordering of multiple ctxs
        ):
        assert stage in {'save', 'retrieve'}
        assert ctx_order in {'parallel', 'near', 'far'}
        self.dstore = dstore
        self.topk = topk
        self.stage = stage
        self.layer_index = layer_index
        
        self.track = track
        if self.is_track:
            track_file = self.track if type(self.track) is str else None
            self.tracker = RetrievalTracker(track_file=track_file, n_heads=dstore.n_heads, topk=topk, eos_token_id=eos_token_id)
        
        self.by_ids = by_ids
        self.by_ids_cache = None  # cache the retrieved results so following decoding steps do not need to retrieve
        self.id_offset = shard_start  # example idx
        
        self.skip_retrieval_steps = skip_retrieval_steps
        self.accum_retrieval_steps = accum_retrieval_steps
        assert self.accum_retrieval_steps <= self.skip_retrieval_steps
        self.retrieval_for_next_step_at_layer = retrieval_for_next_step_at_layer
        self.skip_first_token = skip_first_token
        self.add_after_first = add_after_first
        self.filter_topk = filter_topk
        self.filter_order = filter_order

        self.only_use_head_idx = only_use_head_idx
        self.cache_indices = cache_indices
        self.mtac = mtac

        self.retrieval_every_steps = retrieval_every_steps
        self.max_retrieval_times = max_retrieval_times
        self._retrieval_cache = {
            'count': 0,
            'query': [],  # query states from previous steps
            'key': None,  # key states from previous retrieval
            'value': None,  # value states from previous retrieval
        }

        self.num_ctxs = num_ctxs
        self.ctx_order = ctx_order

    @property
    def is_track(self):
        return bool(self.track)

    def save(
        self,
        key_states: torch.FloatTensor,  # (batch_size, n_heads, seq_length, dim_per_head)
        value_states: torch.FloatTensor,  # (batch_size, n_heads, seq_length, dim_per_head)
    ):
        bs, _, sl, _ = key_states.size()

        # get mask
        labels, decoder_input_ids = self.dstore.get_labels()  # (batch, seq_length)
        labels, decoder_input_ids = labels.flatten(0, 1), decoder_input_ids.flatten(0, 1)  # (batch * seq_length)
        mask = labels != -100

        # get idx
        ids = self.id_offset + torch.arange(bs).unsqueeze(-1).repeat(1, sl).flatten(0, 1)  # (batch * seq_length)
        self.id_offset += bs
        
        # remove padding tokens
        key_states = key_states.permute(1, 0, 2, 3).flatten(1, 2)  # (n_heads, batch_size * seq_length, dim_per_head)
        value_states = value_states.permute(1, 0, 2, 3).flatten(1, 2)  # (n_heads, batch_size * seq_length, dim_per_head)
        key_states = key_states[:, mask, :]  # (n_heads, n_tokens, dim_per_head)
        value_states = value_states[:, mask, :]  # (n_heads, n_tokens, dim_per_head)
        tokens = decoder_input_ids[mask]  # (n_tokens)
        ids = ids[mask]  # (n_tokens)

        # save
        self.dstore.save_key_value(key_states, value_states, tokens=tokens, ids=ids)
    
    def save_for_accumlation(
        self,
        query_states: torch.FloatTensor,  # (n_heads, batch_size, dim)
        ret_ks: torch.FloatTensor,  # (n_heads, batch_size, n_cand, topk, dim)
    ):
        # init cache
        if not hasattr(self, '_accum_cache'):
            self._accum_cache = []
            self._accum_cache_all = []

        assert query_states.size(1) == ret_ks.size(1)
        attn_scores = torch.einsum("hbd,hbckd->hbck", query_states, ret_ks)  # (n_heads, batch_size, n_cand, topk)
        attn_scores = attn_scores.max(-1).values  # (n_heads, batch_size, n_cand)
        self._accum_cache.append(attn_scores.cpu())  # n_steps (n_heads, batch_size, n_cand)
        
        # dump cache
        if len(self._accum_cache) >= self.accum_retrieval_steps:
            self._accum_cache = torch.stack(self._accum_cache, 0)  # (n_steps, n_heads, batch_size, n_cand)
            self._accum_cache = self._accum_cache.mean(0)  # (n_heads, batch_size, n_cand)
            self._accum_cache_all.append(self._accum_cache)
            self._accum_cache = []
    
    def dump_save_for_accumlation(self, file_name: str):
        if not hasattr(self, '_accum_cache_all') or len(self._accum_cache_all) <= 0:
            return
        torch.save(torch.cat(self._accum_cache_all, 1), f'{file_name}{self.layer_index}.pt')
    
    def dump_retrieval(self, file_name: str):
        if not len(self.dstore.head2ids):
            return
        todump = {head: torch.cat(ids, 0) for head, ids in self.dstore.head2ids.items()}
        torch.save(todump, f'{file_name}{self.layer_index}.pt')

    def retrieve(
        self,
        query_states: torch.FloatTensor,  # (batch_size, n_heads, seq_length, dim_per_head)
        key_length: int,
        debug: bool = False,  # TODO: debug
    ):
        assert key_length > 0, 'key_length should be positive'
        bs, nh, sl, dph = query_states.size()
        topk = self.topk
        is_eval = sl > 1

        # eval mode is memory-intensive so we do not perform retrieval  (TODO: enable eval-mode retrieval)
        if is_eval:
            topk = 0
        # skip retrieval
        fake_retrieval = key_length <= self.accum_retrieval_steps
        skip_for_retrieval = key_length <= self.skip_retrieval_steps
        if skip_for_retrieval and not fake_retrieval:
            topk = 0

        # block-wise retrieval
        bw_ret = not is_eval and not skip_for_retrieval
        _bw_offset = key_length - self.skip_retrieval_steps
        bw_ret_new = bw_ret and _bw_offset % self.retrieval_every_steps == 0 and self._retrieval_cache['count'] < self.max_retrieval_times  # retrieve new results
        bw_ret_reuse = bw_ret and not bw_ret_new  # reuse previous retrieved results
        # TODO: debug check corner case retrieval_every_steps == 1
        bw_ret_reuse_and_for_next = bw_ret_reuse and self.retrieval_for_next_step_at_layer == self.layer_index and \
            self._retrieval_cache['count'] < self.max_retrieval_times and \
            (self.retrieval_every_steps == 1 or (_bw_offset + 1) % self.retrieval_every_steps == 0)  # next step is the retrieval step

        if debug:
            print(f'== step: {key_length} layer: {self.layer_index} ==')

        if bw_ret:
            self._retrieval_cache['query'].append(query_states)
        if bw_ret_new:
            query_states = torch.cat(self._retrieval_cache['query'], 2)  # (batch_size, n_heads, retrieval_every_steps, dim_per_head)
        if bw_ret_reuse_and_for_next:
            if debug:
                print('retrieve for next step')
            query_states = torch.cat(self._retrieval_cache['query'], 2)  # (batch_size, n_heads, retrieval_every_steps, dim_per_head)
            self._retrieve(query_states, key_length=key_length + 1, topk=topk)  # save indices for the next step
        if bw_ret_reuse:
            if self._retrieval_cache['key'] is not None:
                ret_ks, ret_vs = self._retrieval_cache['key'], self._retrieval_cache['value']
                if debug:
                    print(f'use horizontal cache {ret_ks.size()}')
                return ret_ks, ret_vs
            topk = 0

        if topk == 0:  # no need to perform retrieval, return "empty" tensors
            ret_ks = torch.zeros(bs, nh, sl, 1, 0, dph).to(query_states)  # (batch_size, n_heads, seq_length, 1 (n_ctxs), topk, dim_per_head)
            ret_vs = torch.zeros(bs, nh, sl, 1, 0, dph).to(query_states)  # (batch_size, n_heads, seq_length, 1 (n_ctxs), topk, dim_per_head)
            if debug:
                print(f'skip')
        else:  # perform retrieve
            if query_states.size(2) == 1:
                query_states = query_states.squeeze(2)
            ret_ks, ret_vs = self._retrieve(query_states, key_length=key_length, topk=topk)  # (batch_size, n_heads, n_ctx, topk, dim_per_head) * 2
            ret_ks = ret_ks.unsqueeze(2)  # (batch_size, n_heads, seq_length=1, n_ctx, topk, dim_per_head)
            ret_vs = ret_vs.unsqueeze(2)  # (batch_size, n_heads, seq_length=1, n_ctx, topk, dim_per_head)
            if debug:
                print(f'retrieve for current step {ret_ks.size()}')

            if bw_ret_new:  # update key and value
                self._retrieval_cache = {'count': self._retrieval_cache['count'] + 1, 'query': [], 'key': ret_ks, 'value': ret_vs}

        if is_eval:  # move to next batch
            self.id_offset += bs  # change offset
            self.by_ids_cache = None  # clear by_ids cache
            if self.mtac is not None:  # clear coordinator
                self.mtac.clear()
            self._retrieval_cache = {'count': 0, 'query': [], 'key': None, 'value': None}  # clear retrieval cache
            if self.is_track:  # write retrieval results
                self.tracker.write()

        return ret_ks, ret_vs

    def _retrieve(
        self,
        query_states: torch.FloatTensor,  # (batch_size, n_heads, dim_per_head) or (batch_size, n_heads, seq_length, dim_per_head)
        key_length: int,
        topk: int,
        ids: torch.LongTensor = None,  # (batch_size, n_ctxs)
        debug: bool = False,
    ):
        assert topk > 0, 'this function is called when retrieval is actually performed'
        bs = query_states.size(0)
        ori_device = query_states.device
        query_states = query_states.transpose(0, 1)  # (n_heads, batch_size, dim_per_head) or (n_heads, batch_size, seq_length, dim_per_head)

        fake_retrieval = key_length <= self.accum_retrieval_steps
        fake_retrieval_last_step = key_length == self.accum_retrieval_steps

        # ret_ks, ret_vs, ret_ts, ret_ids, indices
        # (n_heads, batch_size, n_ctxs, topk, dim_per_head) * 2, (n_heads, batch_size, n_ctxs, topk) * 3
        indices = None
        if self.cache_indices:  # use cached indices
            indices = self.mtac.get_or_save_indices(key_length=key_length)
            if indices is not None:
                if debug:
                    print('use indicies')
                ret_ks, ret_vs, ret_ts, ret_ids = self.dstore.get_knns_by_indices(indices=indices, device=ori_device, return_all=self.is_track)

        if not self.cache_indices or indices is None:  # perform retrieval
            if debug:
                print('use knn')
            if self.by_ids:
                if self.by_ids_cache:  # use horizontal cache
                    ret_ks, ret_vs, ret_ts, ret_ids, indices = self.by_ids_cache
                else:  # retrieval by ids
                    if ids is None:
                        if fake_retrieval:
                            ids = torch.arange(bs).to(ori_device) + self.id_offset
                            # (batch_size, accum) for the i-th example, retrieve i, i + 1, ..., i + accum - 1
                            ids = ids.unsqueeze(-1) + torch.arange(100).to(ori_device).unsqueeze(0)  # TODO: add argument
                            ids_maks = ids < self.dstore.num_docs  # (batch_size, accum)
                            ids = ids * ids_maks  # out-of-boundary ids are replaced with 0
                        else:
                            ids = torch.arange(bs).to(ori_device).unsqueeze(-1) + self.id_offset  # (batch_size, 1 (n_ctxs))
                    ret_ks, ret_vs, ret_ts, ret_ids, indices, _ = self.dstore.get_knns_by_ids(
                        ids, topk=topk, skip_first_token=self.skip_first_token, return_all=self.is_track)
                    self.by_ids_cache = (ret_ks, ret_vs, ret_ts, ret_ids, indices)

                if fake_retrieval:
                    # save for accumulation
                    self.save_for_accumlation(query_states, ret_ks)
                    # clear retreival
                    ret_ks, ret_vs, ret_ts, ret_ids, indices = self.dstore.get_knns(query_states, topk=0, return_all=self.is_track)
                
                if fake_retrieval_last_step:  # clear the cache
                    self.by_ids_cache = None

                if self.filter_topk:  # filter
                    ret_ks, ret_vs, ret_ts, ret_ids, indices = self.dstore.filter_by_similarity(
                        query_states, ret_ks, ret_vs, ret_ts, ret_ids, indices, topk=self.filter_topk, order=self.filter_order)

            else:
                ret_ks, ret_vs, ret_ts, ret_ids, indices = self.dstore.get_knns(
                    query_states, 
                    topk=topk, 
                    final_topk=self.num_ctxs,
                    only_use_head_idx=self.only_use_head_idx, 
                    skip_first_token=self.skip_first_token,
                    return_all=self.is_track)
        
        if self.cache_indices:  # cache indices
            self.mtac.get_or_save_indices(key_length=key_length, indices=indices)

        # track retrieval
        if self.is_track:
            input_ids = self.dstore.get_decoder_input_ids()  # (batch_size, seq_length)
            self.tracker.add_single_step_batched(
                prediction=input_ids.squeeze(-1), 
                retrieved_token=ret_ts.permute(1, 0, 2, 3).flatten(2, 3), 
                retrieved_id=ret_ids.permute(1, 0, 2, 3).flatten(2, 3))

        ret_ks = ret_ks.transpose(0, 1)  # (batch_size, n_heads, n_ctxs, topk, dim_per_head)
        ret_vs = ret_vs.transpose(0, 1)  # (batch_size, n_heads, n_ctxs, topk, dim_per_head)
        return ret_ks, ret_vs

    def update_mask_and_position_bias(
        self,
        ori_attn: T5Attention,
        mask: torch.FloatTensor,  # (batch_size, n_heads, seq_length, key_length)
        seq_length: int,
        real_seq_length: int,
        key_length: int,
        topk: int,
    ):
        # extend the mask
        if mask is not None:
            ext_size = mask.shape[:3] + (topk,)
            # (batch_size, n_heads, seq_length, topk + key_length)
            mask = torch.cat([torch.zeros(*ext_size).to(mask), mask], dim=3)
        
        # update relative positions
        position_bias = ori_attn.compute_bias(topk + real_seq_length, topk + key_length)
        # need to truncate because ret_topk > 0
        # (batch_size, n_heads, seq_length, topk + key_length)
        position_bias = position_bias[:, :, -seq_length:, :]
        if mask is not None:
            # (batch_size, n_heads, seq_length, topk + key_length)
            position_bias = position_bias + mask
        
        return position_bias
    
    @staticmethod
    def unshape(states):
        bs, nh, sl, d = states.size()
        return states.transpose(1, 2).contiguous().view(bs, sl, -1)
    
    def init_position_bias(
        self, 
        ori_attn: T5Attention, 
        past_key_value: torch.FloatTensor,
        mask: torch.FloatTensor,  # (batch_size, n_heads, seq_length, key_length)
        real_seq_length: int,
        key_length: int,
        seq_length: int,
        device: torch.device):
        self = ori_attn
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length), device=device, dtype=ori_attn.dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length, device=device)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            position_bias = position_bias[:, :, -seq_length :, :]

        if mask is not None:
            position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
        
        return position_bias

    def original_attn(
        self,
        ori_attn: T5Attention,
        query_states: torch.FloatTensor,  # (batch_size, n_heads, seq_length, dim_per_head)
        key_states: torch.FloatTensor,  # (batch_size, n_heads, key_length, dim_per_head)
        value_states: torch.FloatTensor,  # (batch_size, n_heads, key_length, dim_per_head)
        past_key_value: torch.FloatTensor,
        position_bias: torch.FloatTensor,  # (batch_size, n_heads, seq_length, key_length)
        mask: torch.FloatTensor,  # (batch_size, n_heads, seq_length, key_length)
        layer_head_mask,
        real_seq_length: int,
        key_length: int
    ):
        new_self = self
        self = ori_attn
        
        # === original attn code (start) ===
        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        
        if position_bias is None:
            position_bias = new_self.init_position_bias(
                ori_attn, past_key_value, mask, 
                real_seq_length=real_seq_length, key_length=key_length, seq_length=query_states.size(2), device=scores.device)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = new_self.unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)
        # === original attn code (end) ===

        return attn_weights, attn_output, position_bias

    def attn(
        self,
        ori_attn: T5Attention,
        query_states: torch.FloatTensor,  # (batch_size, n_heads, seq_length, dim_per_head)
        key_states: torch.FloatTensor,  # (batch_size, n_heads, key_length, dim_per_head)
        value_states: torch.FloatTensor,  # (batch_size, n_heads, key_length, dim_per_head)
        ret_ks: torch.FloatTensor,  # (batch_size, n_heads, seq_length, n_ctxs, topk, dim_per_head)
        ret_vs: torch.FloatTensor,  # (batch_size, n_heads, seq_length, n_ctxs, topk, dim_per_head)
        mask: torch.FloatTensor,  # (batch_size, n_heads, seq_length, key_length)
        layer_head_mask,
        real_seq_length: int,
        key_length: int
    ): 
        if layer_head_mask is not None:
            raise NotImplementedError()
        
        bs, nh, sl, d = query_states.size()
        kl = key_states.size(2)
        multi_token_eval = sl > 1  # multiple tokens per example in the query_states

        if self.ctx_order == 'near':
            # (batch_size, n_heads, seq_length, 1, n_ctxs * topk, dim_per_head) * 2
            ret_ks, ret_vs = ret_ks.flip([3]).flatten(3, 4).unsqueeze(3), ret_vs.flip([3]).flatten(3, 4).unsqueeze(3)
        elif self.ctx_order == 'far':
            # (batch_size, n_heads, seq_length, 1, n_ctxs * topk, dim_per_head) * 2
            ret_ks, ret_vs = ret_ks.flatten(3, 4).unsqueeze(3), ret_vs.flatten(3, 4).unsqueeze(3)
        elif self.ctx_order == 'parallel':
            pass
        else:
            raise NotImplementedError
        n_ctxs, topk = ret_ks.size(-3), ret_ks.size(-2)

        if multi_token_eval:
            # TODO: implement this
            #if self.add_after_first:
            #    raise NotImplementedError()

            assert sl == kl == real_seq_length == key_length, 'should be in eval mode'

            # compute the original scores over local context
            # (batch_size, n_heads, seq_length, key_length)
            scores = torch.matmul(query_states, key_states.transpose(3, 2))

            # compute the extended scores over the retrieved context
            # (batch_size, n_heads, seq_length, n_ctxs, topk)
            _scores = torch.einsum("bnqd,bnqckd->bnqck", query_states, ret_ks)

            # apply bias
            # (batch_size, n_heads, seq_length, topk + key_length)
            position_bias = self.update_mask_and_position_bias(
                ori_attn, mask, seq_length=sl, real_seq_length=real_seq_length, key_length=key_length, topk=topk)
            scores += position_bias[:, :, :, -kl:]
            _scores += position_bias[:, :, :, None, :topk]

            # combine scores
            # (batch_size, n_heads, seq_length, n_ctxs * topk + key_length)
            scores = torch.cat([_scores.flatten(3, 4), scores], dim=-1)

            # compute attn distribution
            # (batch_size, n_heads, seq_length, n_ctxs * topk + key_length)
            attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
            attn_weights = nn.functional.dropout(attn_weights, p=ori_attn.dropout, training=ori_attn.training)

            # compute output
            # (batch_size, n_heads, seq_length, dim_per_head)
            attn_output = torch.matmul(attn_weights[:, :, :, -kl:], value_states)
            attn_output += torch.einsum("bnqk,bnqkd->bnqd", attn_weights[:, :, :, :n_ctxs * topk], ret_vs.flatten(3, 4))
            attn_output = self.unshape(attn_output)  # (batch_size, seq_length, dim)
            attn_output = ori_attn.o(attn_output)
        
        else:
            assert sl == 1, 'should be in decoding mode'
            assert ret_ks.size(2) == ret_vs.size(2) == sl
            assert real_seq_length == key_length

            ret_ks, ret_vs = ret_ks.flatten(3, 4), ret_vs.flatten(3, 4)  # (batch_size, n_heads, seq_length, n_ctxs * topk, dim_per_head) * 2

            # prepend retrieved keys and values
            # (batch_size, n_heads, n_ctxs * topk + key_length, dim_per_head)
            if self.add_after_first and key_length > 1:  # always need to prepend for the first position
                key_states = torch.cat([key_states[:, :, :1], ret_ks.squeeze(2), key_states[:, :, 1:]], dim=2)
                value_states = torch.cat([value_states[:, :, :1], ret_vs.squeeze(2), value_states[:, :, 1:]], dim=2)
            else:
                key_states = torch.cat([ret_ks.squeeze(2), key_states], dim=2)
                value_states = torch.cat([ret_vs.squeeze(2), value_states], dim=2)

            # compute attn scores
            # (batch_size, n_heads, seq_length, n_ctxs * topk + key_length)
            scores = torch.matmul(query_states, key_states.transpose(3, 2))

            # apply bias
            # (batch_size, n_heads, seq_length, topk + key_length)
            position_bias = self.update_mask_and_position_bias(
                ori_attn, mask, seq_length=sl, real_seq_length=real_seq_length, key_length=key_length, topk=topk)
            _scores = scores[:, :, :, :n_ctxs * topk].view(bs, nh, sl, n_ctxs, topk)  # (batch_size, n_heads, seq_length, n_ctxs, topk)
            _scores = (_scores + position_bias[:, :, :, None, :topk]).flatten(3, 4)  # (batch_size, n_heads, seq_length, n_ctxs * topk)
            scores = scores[:, :, :, -kl:] + position_bias[:, :, :, -kl:]  # (batch_size, n_heads, seq_length, key_length)
            scores = torch.cat([_scores, scores], -1)  # (batch_size, n_heads, seq_length, n_ctxs * topk + key_length)

            # compute attn distribution
            # (batch_size, n_heads, seq_length, n_ctxs * topk + key_length)
            attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
            attn_weights = nn.functional.dropout(attn_weights, p=ori_attn.dropout, training=ori_attn.training)
            
            # compute output
            attn_output = self.unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
            attn_output = ori_attn.o(attn_output)

        return attn_weights, attn_output

def t5attetnion_forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
        assert (
            len(past_key_value) == 2
        ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
        real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
        """projection"""
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            else:
                # cross-attn
                hidden_states = past_key_value
        return hidden_states

    # get query states
    query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
    )
    value_states = project(
        hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
    )

    if self.mta.stage == 'save':
        self.mta.save(key_states, value_states)
        attn_weights, attn_output, position_bias = self.mta.original_attn(
            self, query_states, key_states, value_states, past_key_value,
            position_bias, mask, layer_head_mask, 
            real_seq_length=real_seq_length, key_length=key_length)
    elif self.mta.stage == 'retrieve':
        ret_ks, ret_vs = self.mta.retrieve(query_states, key_length=key_length)
        if position_bias is None:  # init position_bias in the first layer which is reused in following layers
            position_bias = self.mta.init_position_bias(
                self, past_key_value, mask, 
                real_seq_length=real_seq_length, key_length=key_length, seq_length=seq_length, device=query_states.device)
        attn_weights, attn_output = self.mta.attn(
            self, query_states, key_states, value_states, ret_ks, ret_vs, 
            mask, layer_head_mask, 
            real_seq_length=real_seq_length, key_length=key_length)
    else:  # original code
        attn_weights, attn_output, position_bias = self.mta.original_attn(
            self, query_states, key_states, value_states, past_key_value,
            position_bias, mask, layer_head_mask, 
            real_seq_length=real_seq_length, key_length=key_length)

    present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs


class MemTransWrapper(object):
    def __init__(
        self, 
        dstore_size: int, 
        dstore_dir: str, 
        recompute_dists: bool = False,
        retrieval_layers: List[int] = [-6],
        k: int = 1024, 
        stage: str = 'save',
        track: Union[bool, str] = False,
        by_ids: bool = False,
        shard_start: int = 0,  # used for sharded generated with by_ids = True
        skip_retrieval_steps: int = 0,
        accum_retrieval_steps: int = 0,
        retrieval_for_next_step_at_layer: int = -1,
        retrieval_every_steps: int = 1,
        max_retrieval_times: int = None,
        skip_first_token: bool = False,
        add_after_first: bool = False,
        filter_topk: int = 0,
        filter_order: str = 'original',
        only_use_head_idx: int = -1,
        cache_indices: bool = False,
        num_ctxs: int = 1,
        ctx_order: str = 'parallel',
        move_dstore_to_mem: bool = False, 
        device: torch.device = None):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.recompute_dists = recompute_dists
        self.retrieval_layers = retrieval_layers
        self.k = k
        self.stage = stage
        self.track = track
        self.by_ids = by_ids
        self.shard_start = shard_start
        self.skip_retrieval_steps = skip_retrieval_steps
        self.accum_retrieval_steps = accum_retrieval_steps
        self.retrieval_for_next_step_at_layer = retrieval_for_next_step_at_layer
        self.retrieval_every_steps = retrieval_every_steps
        self.max_retrieval_times = max_retrieval_times
        self.skip_first_token = skip_first_token
        self.add_after_first = add_after_first
        self.filter_topk = filter_topk
        self.filter_order = filter_order
        self.only_use_head_idx = only_use_head_idx
        self.cache_indices = cache_indices
        self.num_ctxs = num_ctxs
        self.ctx_order = ctx_order
        self.move_dstore_to_mem = move_dstore_to_mem
        self.device = torch.device('cpu') if device is None else device
    
    def get_layer(self, key: str = 'memtrans'):
        mt = self.model.config.model_type
        if key != 'memtrans':
            return self.CONFIG[mt][key](self.model)
        return [(layer_idx, self.CONFIG[mt][key](self.model, layer_idx)) for layer_idx in self.retrieval_layers]

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder
        eos_token_id = self.model.config.eos_token_id
        relative_attention_bias = self.get_layer(key='firstattn').relative_attention_bias

        # save labels for masking
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        # save decoder input for debugging
        self.original_decoder_forward_func = model.decoder.forward
        model.decoder.forward = self.pre_decoder_forward_hook
        
        attn_layers = self.get_layer()
        self.ori_t5attetnion_forwards = []
        self.dstores = []
        mtac = MemTransAttnCoordinator()
        for li, (layer_idx, attn_layer) in enumerate(attn_layers):
            # replace the attention layer with retrieval-augmented attention layer
            self.ori_t5attetnion_forwards.append(attn_layer.forward)
            attn_layer.forward = types.MethodType(t5attetnion_forward, attn_layer)

            # load dstore (and index if in retrieval stage)
            dstore_device = self.device
            if self.cache_indices and li > 0:  # load index to CPU except for the first one
                dstore_device = torch.device('cpu')
            dstore = MemTransDatastore(
                directory=os.path.join(self.dstore_dir, f'layer{layer_idx}'), 
                model_type=self.model.config.model_type, 
                size=self.dstore_size,
                dimension=self.model.config.d_kv,
                n_heads=self.model.config.num_heads,
                move_dstore_to_mem=self.move_dstore_to_mem,
                device=dstore_device)
            if self.stage == 'retrieve':
                dstore.load_index(build_offset=True)
            self.dstores.append(dstore)
            
            # inject MemTransAttn
            mta = MemTransAttn(
                dstore=dstore, 
                topk=self.k, 
                eos_token_id=eos_token_id, 
                stage=self.stage, 
                layer_index=layer_idx,
                track=self.track, 
                by_ids=self.by_ids, 
                shard_start=self.shard_start,
                skip_retrieval_steps=self.skip_retrieval_steps,
                accum_retrieval_steps=self.accum_retrieval_steps,
                retrieval_for_next_step_at_layer=self.retrieval_for_next_step_at_layer,
                retrieval_every_steps=self.retrieval_every_steps,
                max_retrieval_times=self.max_retrieval_times,
                skip_first_token=self.skip_first_token,
                add_after_first=self.add_after_first,
                filter_topk=self.filter_topk,
                filter_order=self.filter_order,
                only_use_head_idx=self.only_use_head_idx,
                cache_indices=self.cache_indices,
                mtac=mtac,
                num_ctxs=self.num_ctxs,
                ctx_order=self.ctx_order)
            attn_layer.mta = mta
            attn_layer.relative_attention_bias = relative_attention_bias

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels) if labels is not None else None
        for dstore in self.dstores:
            dstore.save_labels(labels, decoder_input_ids)
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)
    
    def pre_decoder_forward_hook(self, input_ids=None, **kwargs):
        for dstore in self.dstores:
            dstore.save_decoder_input_ids(input_ids)
        return self.original_decoder_forward_func(input_ids=input_ids, **kwargs)

    def break_out(self):
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            attn_layers = self.get_layer()
            for (layer_idx, attn_layer), ori_fwd in zip(attn_layers, self.ori_t5attetnion_forwards):
                attn_layer.forward = ori_fwd
                #attn_layer.mta.dump_save_for_accumlation('test')  # TODO: add argument
                attn_layer.mta.dump_retrieval('test')  # TODO: add argument
                del attn_layer.mta
                #del attn_layer.relative_attention_bias  # TODO: avoid deleting this for the first layer
            self.model.broken_into = None

    CONFIG = {
        't5': {
            'memtrans': lambda model, layer_idx: model.base_model.decoder.block[layer_idx].layer[0].SelfAttention,
            'firstattn': lambda model: model.base_model.decoder.block[0].layer[0].SelfAttention
        },
        'mt5': {
            'memtrans': lambda model, layer: model.base_model.decoder.block[layer].layer[0].SelfAttention,
            'firstattn': lambda model: model.base_model.decoder.block[0].layer[0].SelfAttention
        }
    }

    def build_index(self):
        for dstore in self.dstores:
            dstore.build_index(batch_size=1000000)
