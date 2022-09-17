from typing import Tuple
import os
import logging
import time
import types
import functools
from xml.sax.handler import property_declaration_handler
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from pathlib import Path
import faiss
import faiss.contrib.torch_utils

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
        self.precision = np.float16  # TODO: 32?
        self.load_or_init_dstore()
    
    @property
    def use_cuda(self):
        return self.cuda_idx != -1
    
    @property
    def device(self):
        return torch.device(f'cuda:{self.cuda_idx}' if torch.cuda.is_available() else 'cpu')

    def get_index_path(self, head_idx: int) -> str:
        return get_index_path(self.directory, self.model_type, self.size, self.dimension, head_idx=head_idx)
    
    def get_dstore_path(self) -> Tuple[str, str]:
        prefix = get_dstore_path(self.directory, self.model_type, self.size, self.dimension)
        key_file = f'{prefix}_keys.npy'
        val_file = f'{prefix}_vals.npy'
        return key_file, val_file
    
    def load_or_init_dstore(self):
        start = time.time()
        key_file, val_file = self.get_dstore_path()
        if os.path.exists(key_file) and os.path.exists(val_file):
            mode = 'r'
        else:
            mode = 'w+'
            Path(key_file).parent.mkdir(parents=True, exist_ok=True)
        self.keys = np.memmap(key_file, dtype=self.precision, mode=mode, shape=(self.n_heads, self.size, self.dimension))
        self.values = np.memmap(val_file, dtype=self.precision, mode=mode, shape=(self.n_heads, self.size, self.dimension))
        logger.info(f'Loading dstore took {time.time() - start} s')
    
    def save_labels(
        self, 
        labels: torch.LongTensor):  # (batch_size, seq_length)
        self._labels = labels
    
    def get_labels(self):
        return self._labels

    def save_key_value(
        self,
        keys: torch.FloatTensor,  # (n_heads, n_tokens, dim_per_head)
        values: torch.FloatTensor):  # (n_heads, n_tokens, dim_per_head)
        nh, nt, dim = keys.size()
        assert nh == self.n_heads and dim == self.dimension, 'keys and values are in a wrong shape'

        assert self.cur_idx <= self.size, 'datastore overflow'
        if self.cur_idx + nt > self.size:
            nt = self.size - self.cur_idx
            keys = keys[:nt]
            values = values[:nt]
        
        try:
            self.keys[:, self.cur_idx:(nt + self.cur_idx)] = keys.cpu().numpy().astype(self.precision)
            self.values[:, self.cur_idx:(nt + self.cur_idx)] = values.cpu().numpy().astype(self.precision)
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
            self.keys = self.keys.to(self.device)
            self.values = self.values.to(self.device)
            logger.info('Moving to memory took {} s'.format(time.time() - start))
        
        # load index
        self.indices = []
        for h in range(self.n_heads):
            start = time.time()
            index_name = self.get_index_path(head_idx=h)
            cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
            logger.info(f'Loading index took {time.time() - start} s')
            if self.use_cuda:  # move index to gpu
                start = time.time()
                gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.cuda_idx, cpu_index)
                logger.info(f'Moving index to GPU took {time.time() - start} s')
            else:
                gpu_index = cpu_index
            self.indices.append(gpu_index)
    
    def get_knns(
        self, 
        queries: torch.FloatTensor,  # (n_heads, batch_size, dim)
        topk: int):
        queries = queries.to(self.device)
        ret_ks, ret_vs = [], []
        for h in range(self.n_heads):
            index = self.indices[h]
            dists, indices = index.search(queries[h].contiguous(), topk)  # (batch_size, topk)
            indices = indices.to(self.device)
            ret_ks.append(self.keys[h][indices])  # (batch_size, topk, dim)
            ret_vs.append(self.values[h][indices])  # (batch_size, topk, dim)
        ret_ks = torch.cat(ret_ks, dim=0)  # (n_heads, batch_size, topk, dim)
        ret_vs = torch.cat(ret_vs, dim=0)  # (n_heads, batch_size, topk, dim)
        return ret_ks, ret_vs

def memtrans_save(
    self,
    key_states: torch.FloatTensor,  # (batch_size, n_heads, seq_length, dim_per_head)
    value_states: torch.FloatTensor,  # (batch_size, n_heads, seq_length, dim_per_head)
    dstore: MemTransDatastore = None,
):
    # get mask
    labels = dstore.get_labels().flatten(0, 1)  # (batch * seq_length)
    mask = labels != -100
    
    # remove padding tokens
    key_states = key_states.permute(1, 0, 2, 3).flatten(1, 2)  # (n_heads, batch_size * seq_length, dim_per_head)
    value_states = value_states.permute(1, 0, 2, 3).flatten(1, 2)  # (n_heads, batch_size * seq_length, dim_per_head)
    key_states = key_states[:, mask, :]  # (n_heads, n_tokens, dim_per_head)
    value_states = value_states[:, mask, :]  # (n_heads, n_tokens, dim_per_head)

    # save
    dstore.save_key_value(key_states, value_states)

def memtrans_retrieve(
    self,
    query_states: torch.FloatTensor,  # (batch_size, n_heads, seq_length, dim_per_head)
    dstore: MemTransDatastore,
    topk: int,
):
    bs, nh, sl, d = query_states.size()
    query_states = query_states.permute(1, 0, 2, 3).flatten(1, 2)  # (n_heads, batch_size * seq_length, dim_per_head)
    ret_ks, ret_vs = dstore.get_knns(query_states, topk=topk)  # (n_heads, batch_size * seq_length, topk, dim_per_head)
    ret_ks = ret_ks.view(nh, bs, sl, topk, d)  # (n_heads, batch_size, seq_length, topk, dim_per_head)
    ret_vs = ret_vs.view(nh, bs, sl, topk, d)  # (n_heads, batch_size, seq_length, topk, dim_per_head)
    ret_ks = ret_ks.permute(1, 0, 2, 3, 4).flatten(2, 3)  # (batch_size, n_heads, seq_length * topk, dim_per_head)
    ret_vs = ret_vs.permute(1, 0, 2, 3, 4).flatten(2, 3)  # (batch_size, n_heads, seq_length * topk, dim_per_head)
    return ret_ks, ret_vs

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

    ori_position_bias = ori_key_states = ori_value_states = None
    if self.memtrans_stage == 'save':
        self.memtrans_save(key_states, value_states)
    elif self.memtrans_stage == 'retrieve':
        ret_ks, ret_vs = self.memtrans_retrieve(query_states)

        # concat the retrieved key/value and the original key/value so retrieved is more far away
        ret_topk = ret_ks.size(2)
        real_seq_length += ret_topk
        key_length += ret_topk
        ori_key_states = key_states
        ori_value_states = value_states
        key_states = torch.cat([ret_ks, key_states], dim=2)
        value_states = torch.cat([ret_vs, value_states], dim=2)

        # update mask and position_bias
        if mask is not None:  # extend the mask
            ext_size = mask.size()[:3] + (ret_topk,)
            mask = torch.cat([torch.zeros(*ext_size).to(mask), mask], dim=3)
        if position_bias is not None:  # update relative positions
            ori_position_bias = position_bias
            position_bias = self.compute_bias(real_seq_length, key_length, device=position_bias.device)
            if past_key_value is not None or ret_topk > 0:
                # for the first generation position where past_key_value is None, we need to truncate position_bias if ret_topk > 0
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]
            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    else:
        raise ValueError()
    
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
            position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

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

    attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    if self.memtrans_stage == 'retrieve':
        if ori_position_bias is not None:  # recover
            position_bias = ori_position_bias
        if ori_key_states is not None:
            key_states = ori_key_states
            value_states = ori_value_states

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
        self.device = torch.device(f'cuda:{self.cuda_idx}' if torch.cuda.is_available() else 'cpu')
    
    def get_layer(self, key: str = 'memtrans'):
        return MemTransWrapper.layer_to_capture[self.model.config.model_type][key](self.model)

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # save labels for masking
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook
        
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
        
        # inject functions and variables
        attn_layer.relative_attention_bias = self.get_layer(key='firstattn').relative_attention_bias
        attn_layer.memtrans_stage = self.stage
        attn_layer.memtrans_save = types.MethodType(
            functools.partial(memtrans_save, dstore=self.dstore), attn_layer)
        attn_layer.memtrans_retrieve = types.MethodType(
            functools.partial(memtrans_retrieve, dstore=self.dstore, topk=self.k), attn_layer)
    
    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.dstore.save_labels(labels)
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def break_out(self):
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            attn_layer = self.get_layer()
            attn_layer.forward = self.ori_t5attetnion_forward
            del attn_layer.relative_attention_bias
            del attn_layer.memtrans_stage
            del attn_layer.memtrans_save
            del attn_layer.memtrans_retrieve
            self.model.broken_into = None
    
    layer_to_capture = {
        't5': {
            'memtrans': lambda model: model.base_model.decoder.block[-3].layer[0].SelfAttention,
            'firstattn': lambda model: model.base_model.decoder.block[0].layer[0].SelfAttention
        }
    }

    def build_index(self):
        self.dstore.build_index(batch_size=1000000)
