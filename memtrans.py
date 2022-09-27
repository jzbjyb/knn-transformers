from typing import Tuple, Union, List
import os
import logging
import time
import types
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from pathlib import Path
import faiss
import faiss.contrib.torch_utils
from transformers.models.t5.modeling_t5 import T5Attention

from knnlm import get_dstore_path

logger = logging.getLogger(__name__)
logger.setLevel(20)

def get_index_path(dstore_dir, model_type, dstore_size, dimension, head_idx):
    return f'{dstore_dir}/index_{model_type}_{dstore_size}_{dimension}_{head_idx}.indexed'

class MemTransDatastore(object):
    def __init__(
        self, directory: str, 
        model_type: str, 
        size: int, 
        dimension: int, 
        n_heads: int, 
        move_dstore_to_mem: bool = False, 
        cuda_idx: int = -1):
        self.directory = directory
        self.model_type = model_type
        self.size = size
        self.dimension = dimension
        self.n_heads = n_heads
        self.move_dstore_to_mem = move_dstore_to_mem
        self.cuda_idx = cuda_idx
        self.cur_idx = 0
        self.precision = np.float32
        self.load_or_init_dstore()
    
    @property
    def use_cuda(self):
        return self.cuda_idx != -1
    
    @property
    def device(self):
        return torch.device(f'cuda:{self.cuda_idx}' if self.use_cuda else 'cpu')

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
        except:
            self.tokens = self.ids = None
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
            keys = keys[:nt]
            values = values[:nt]
            tokens = tokens[:nt] if tokens is not None else tokens
            ids = ids[:nt] if ids is not None else ids
        
        # save to memmap
        try:
            self.keys[:, self.cur_idx:(nt + self.cur_idx)] = keys.cpu().numpy().astype(self.precision)
            self.values[:, self.cur_idx:(nt + self.cur_idx)] = values.cpu().numpy().astype(self.precision)
            if tokens is not None:
                self.tokens[self.cur_idx:(nt + self.cur_idx)] = tokens.cpu().numpy().astype(np.int32)
            if ids is not None:
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
    
    def load_index(self):
        # move dstore
        if self.move_dstore_to_mem:
            start = time.time()
            self.keys = torch.from_numpy(self.keys[:])
            self.values = torch.from_numpy(self.values[:])
            if self.tokens is not None:
                self.tokens = torch.from_numpy(self.tokens[:])
            if self.ids is not None:
                self.ids = torch.from_numpy(self.ids[:])
            # TODO: move keys and values to gpu 
            #self.keys = self.keys.to(self.device)
            #self.values = self.values.to(self.device)
            logger.info('Moving to memory took {} s'.format(time.time() - start))
        
        # load index
        self.indices = []
        res = faiss.StandardGpuResources()
        for h in range(self.n_heads):
            start = time.time()
            index_name = self.get_index_path(head_idx=h)
            cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
            logger.info(f'Loading index took {time.time() - start} s')
            if self.use_cuda:  # move index to gpu
                start = time.time()
                gpu_index = faiss.index_cpu_to_gpu(res, self.cuda_idx, cpu_index)
                logger.info(f'Moving index to GPU took {time.time() - start} s')
            else:
                gpu_index = cpu_index
            self.indices.append(gpu_index)
    
    def get_knns(
        self, 
        queries: torch.FloatTensor,  # (n_heads, batch_size, dim)
        topk: int,
        return_all: bool = False):
        ori_device = queries.device
        queries = queries.to(self.device)
        ret_ks, ret_vs, ret_ts, ret_ids = [], [], [], []
        for h in range(self.n_heads):
            index = self.indices[h]
            dists, indices = index.search(queries[h].contiguous(), topk)  # (batch_size, topk)
            indices = indices.to(self.device)
            ret_ks.append(self.keys[h][indices])  # (batch_size, topk, dim)
            ret_vs.append(self.values[h][indices])  # (batch_size, topk, dim)
            if return_all:
                assert self.tokens is not None
                ret_ts.append(self.tokens[indices])  # (batch_size, topk)
                assert self.ids is not None
                ret_ids.append(self.ids[indices])  # (batch_size, topk)
            
        ret_ks = torch.stack(ret_ks, dim=0).to(ori_device)  # (n_heads, batch_size, topk, dim)
        ret_vs = torch.stack(ret_vs, dim=0).to(ori_device)  # (n_heads, batch_size, topk, dim)
        if return_all:
            ret_ts = torch.stack(ret_ts, dim=0).to(ori_device)  # (n_heads, batch_size, topk)
            ret_ids = torch.stack(ret_ids, dim=0).to(ori_device)  # (n_heads, batch_size, topk)
            return ret_ks, ret_vs, ret_ts, ret_ids
        return ret_ks, ret_vs

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

class MemTransAttn(object):
    def __init__(
        self, 
        dstore: MemTransDatastore, 
        topk: int, 
        eos_token_id: int,
        stage: str, 
        track: Union[bool, str] = False):  # track retrieved tokens by printing (bool) or writing to files (str)
        assert stage in {'save', 'retrieve'}
        self.dstore = dstore
        self.topk = topk
        self.stage = stage
        self.track = track
        if self.is_track:
            track_file = self.track if type(self.track) is str else None
            self.tracker = RetrievalTracker(track_file=track_file, n_heads=dstore.n_heads, topk=topk, eos_token_id=eos_token_id)
        self.id_offset = 0  # example idx

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

    def retrieve(
        self,
        query_states: torch.FloatTensor,  # (batch_size, n_heads, seq_length, dim_per_head)
    ):
        bs, nh, sl, d = query_states.size()
        query_states = query_states.permute(1, 0, 2, 3).flatten(1, 2)  # (n_heads, batch_size * seq_length, dim_per_head)
        
        if self.is_track:
            # (n_heads, batch_size * seq_length, topk, dim_per_head) * 2, (n_heads, batch_size * seq_length, topk) * 2
            ret_ks, ret_vs, ret_ts, ret_ids = self.dstore.get_knns(query_states, topk=self.topk, return_all=self.is_track)
            input_ids = self.dstore.get_decoder_input_ids()  # (batch_size, seq_length)
            if sl == 1:  # only track the generation process (not for the evaluation process)
                self.tracker.add_single_step_batched(
                    prediction=input_ids.squeeze(-1), 
                    retrieved_token=ret_ts.permute(1, 0, 2), 
                    retrieved_id=ret_ids.permute(1, 0, 2))
            else:  # write in evaluation process
                self.tracker.write()
        else:
            # (n_heads, batch_size * seq_length, topk, dim_per_head) * 2
            ret_ks, ret_vs = self.dstore.get_knns(query_states, topk=self.topk, return_all=self.is_track)
        
        ret_ks = ret_ks.view(nh, bs, sl, self.topk, d)  # (n_heads, batch_size, seq_length, topk, dim_per_head)
        ret_vs = ret_vs.view(nh, bs, sl, self.topk, d)  # (n_heads, batch_size, seq_length, topk, dim_per_head)
        ret_ks = ret_ks.permute(1, 0, 2, 3, 4)  # (batch_size, n_heads, seq_length, topk, dim_per_head)
        ret_vs = ret_vs.permute(1, 0, 2, 3, 4)  # (batch_size, n_heads, seq_length, topk, dim_per_head)
        return ret_ks, ret_vs
    
    def update_mask_and_position_bias(
        self,
        ori_attn: T5Attention,
        mask: torch.FloatTensor,  # (batch_size, n_heads, seq_length, key_length)
        seq_length: int,
        real_seq_length: int,
        key_length: int,
    ):
        # extend the mask
        if mask is not None:
            ext_size = mask.size()[:3] + (self.topk,)
            # (batch_size, n_heads, seq_length, topk + key_length)
            mask = torch.cat([torch.zeros(*ext_size).to(mask), mask], dim=3)
        
        # update relative positions
        position_bias = ori_attn.compute_bias(
            self.topk + real_seq_length, self.topk + key_length)
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
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -query_states.size(2) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

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

        return attn_weights, attn_output

    def attn(
        self,
        ori_attn: T5Attention,
        query_states: torch.FloatTensor,  # (batch_size, n_heads, seq_length, dim_per_head)
        key_states: torch.FloatTensor,  # (batch_size, n_heads, key_length, dim_per_head)
        value_states: torch.FloatTensor,  # (batch_size, n_heads, key_length, dim_per_head)
        ret_ks: torch.FloatTensor,  # (batch_size, n_heads, seq_length, topk, dim_per_head)
        ret_vs: torch.FloatTensor,  # (batch_size, n_heads, seq_length, topk, dim_per_head)
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

        if multi_token_eval:
            assert sl == kl, 'should be in eval mode'

            # compute the original scores over local context
            # (batch_size, n_heads, seq_length, key_length)
            scores = torch.matmul(query_states, key_states.transpose(3, 2))

            # compute the extended scores over the retrieved context
            # (batch_size, n_heads, seq_length, topk)
            _scores = torch.einsum("bnqd,bnqkd->bnqk", query_states, ret_ks)

            # combine scores
            # (batch_size, n_heads, seq_length, topk + key_length)
            scores = torch.cat([_scores, scores], dim=-1)

            # apply bias
            position_bias = self.update_mask_and_position_bias(
                ori_attn, mask, seq_length=sl, real_seq_length=real_seq_length, key_length=key_length)
            scores += position_bias

            # compute attn distribution
            # (batch_size, n_heads, seq_length, topk + key_length)
            attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
            attn_weights = nn.functional.dropout(attn_weights, p=ori_attn.dropout, training=ori_attn.training)

            # compute output
            # (batch_size, n_heads, seq_length, key_length, dim_per_head)
            value_states = value_states.unsqueeze(2).repeat(1, 1, sl, 1, 1)  
            # (batch_size, n_heads, seq_length, topk + key_length, dim_per_head)
            value_states = torch.cat([ret_vs, value_states], dim=3)
            # (batch_size, n_heads, seq_length, dim_per_head)
            attn_output = torch.einsum("bnqk,bnqkd->bnqd", attn_weights, value_states)  
            attn_output = self.unshape(attn_output)  # (batch_size, seq_length, dim)
            attn_output = ori_attn.o(attn_output)
        
        else:
            assert sl == 1, 'should be in decoding mode'
            assert ret_ks.size(2) == ret_vs.size(2) == sl
            assert real_seq_length == key_length

            # prepend retrieved keys and values
            # (batch_size, n_heads, topk + key_length, dim_per_head)
            key_states = torch.cat([ret_ks.squeeze(2), key_states], dim=2)
            value_states = torch.cat([ret_vs.squeeze(2), value_states], dim=2)

            # compute attn scores
            # (batch_size, n_heads, seq_length, topk + key_length)
            scores = torch.matmul(query_states, key_states.transpose(3, 2))

            # apply bias
            position_bias = self.update_mask_and_position_bias(
                ori_attn, mask, seq_length=sl, real_seq_length=real_seq_length, key_length=key_length)
            scores += position_bias

            # compute attn distribution
            # (batch_size, n_heads, seq_length, topk + key_length)
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
        attn_weights, attn_output = self.mta.original_attn(
            self, query_states, key_states, value_states, past_key_value,
            position_bias, mask, layer_head_mask, 
            real_seq_length=real_seq_length, key_length=key_length)
    elif self.mta.stage == 'retrieve':
        ret_ks, ret_vs = self.mta.retrieve(query_states)
        attn_weights, attn_output = self.mta.attn(
            self, query_states, key_states, value_states, 
            ret_ks, ret_vs, 
            mask, layer_head_mask, 
            real_seq_length=real_seq_length, key_length=key_length)
    else:  # original code
        attn_weights, attn_output = self.mta.original_attn(
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
        k: int = 1024, 
        stage: str = 'save',
        move_dstore_to_mem: bool = False, 
        cuda: bool = False):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.recompute_dists = recompute_dists
        self.k = k
        self.stage = stage
        self.move_dstore_to_mem = move_dstore_to_mem
        self.cuda = cuda and torch.cuda.is_available() and torch.cuda.device_count() > 0
        self.cuda_idx = 0 if self.cuda else -1
        self.device = torch.device(f'cuda:{self.cuda_idx}' if self.cuda else 'cpu')
    
    def get_layer(self, key: str = 'memtrans'):
        return MemTransWrapper.layer_to_capture[self.model.config.model_type][key](self.model)

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # save labels for masking
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        # save decoder input for debugging
        self.original_decoder_forward_func = model.decoder.forward
        model.decoder.forward = self.pre_decoder_forward_hook
        
        # replace the attention layer with retrieval-augmented attention layer
        attn_layer = self.get_layer()
        self.ori_t5attetnion_forward = attn_layer.forward
        attn_layer.forward = types.MethodType(t5attetnion_forward, attn_layer)

        # load dstore (and index if in retrieval stage)
        self.dstore = MemTransDatastore(
            directory=self.dstore_dir, 
            model_type=self.model.config.model_type, 
            size=self.dstore_size,
            dimension=self.model.config.d_kv,
            n_heads=self.model.config.num_heads,
            move_dstore_to_mem=self.move_dstore_to_mem,
            cuda_idx=self.cuda_idx)
        if self.stage == 'retrieve':
            self.dstore.load_index()
        
        # inject MemTransAttn
        eos_token_id = self.model.config.eos_token_id
        self.mta = MemTransAttn(dstore=self.dstore, topk=self.k, eos_token_id=eos_token_id, stage=self.stage)
        attn_layer.relative_attention_bias = self.get_layer(key='firstattn').relative_attention_bias
        attn_layer.mta = self.mta
    
    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels) if labels is not None else None
        self.dstore.save_labels(labels, decoder_input_ids)
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)
    
    def pre_decoder_forward_hook(self, input_ids=None, **kwargs):
        self.dstore.save_decoder_input_ids(input_ids)
        return self.original_decoder_forward_func(input_ids=input_ids, **kwargs)

    def break_out(self):
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            attn_layer = self.get_layer()
            attn_layer.forward = self.ori_t5attetnion_forward
            del attn_layer.relative_attention_bias
            del attn_layer.mta
            self.model.broken_into = None
    
    layer_to_capture = {
        't5': {
            'memtrans': lambda model: model.base_model.decoder.block[-6].layer[0].SelfAttention,  # TODO: debug
            'firstattn': lambda model: model.base_model.decoder.block[0].layer[0].SelfAttention
        },
        'mt5': {
            'memtrans': lambda model: model.base_model.decoder.block[-3].layer[0].SelfAttention,
            'firstattn': lambda model: model.base_model.decoder.block[0].layer[0].SelfAttention
        }
    }

    def build_index(self):
        self.dstore.build_index(batch_size=1000000)
