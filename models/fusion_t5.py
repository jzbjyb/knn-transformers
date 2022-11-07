# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model."""


import copy
import warnings
from typing import Optional, Tuple, Union, List, Dict, Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from deepspeed import checkpointing as ds_checkpointing

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config

from transformers.models.t5.modeling_t5 import (
    _CONFIG_FOR_DOC, 
    load_tf_weights_in_t5,
    PARALLELIZE_DOCSTRING,
    DEPARALLELIZE_DOCSTRING,
    T5LayerFF,
    T5_START_DOCSTRING,
    T5_INPUTS_DOCSTRING,
    __HEAD_MASK_WARNING_MSG,
    T5Attention,
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    T5Block,
    T5PreTrainedModel,
    T5Stack
)


logger = logging.get_logger(__name__)


class FusionT5Config(T5Config):
    def __init__(
        self,
        ctx_attention_loss: Dict[str, Any] = None,
        **kwargs
    ):  
        super().__init__(**kwargs)
        self.ctx_attention_loss = ctx_attention_loss
        if ctx_attention_loss is not None:
            assert ctx_attention_loss['layer'] < self.num_decoder_layers
            assert ctx_attention_loss['head'] < self.num_heads

    @staticmethod
    def parse_ctx_attention_loss(ctx_attention_loss: str = None):  # 'block:8-layer:0-head:9-loss:hard-alpha:8'
        if ctx_attention_loss is None:
            return None
        parsed = dict(tuple(field.split(':')) for field in ctx_attention_loss.strip().split('-'))
        parsed = {k: (int(v) if k not in {'loss'} else v) for k, v in parsed.items()}
        return parsed

class FusionT5Attention(T5Attention):
    def __init__(
        self, 
        config: FusionT5Config,
        has_relative_attention_bias: bool = False,
        layer_index: int = -1):
        super().__init__(config, has_relative_attention_bias=has_relative_attention_bias)
        self.attn_specific = config.ctx_attention_loss is not None
        self.compute_loss = self.attn_specific and config.ctx_attention_loss['layer'] == layer_index
        if self.attn_specific:
            self.block_size = config.ctx_attention_loss['block']
            self.use_head = config.ctx_attention_loss['head']
            self.loss_type = config.ctx_attention_loss['loss']

    def add_position_bias(
        self, 
        extended_seq_length: int,  # the total length of query
        extended_key_length: int,  # the total length of key
        scores: torch.Tensor,  # (batch_size, n_heads, seq_length, [n_ctxs,] key_length)
        mask: torch.Tensor = None,  # (batch_size, n_heads, seq_length, [n_ctxs,] key_length)
        ):
        # (batch_size, n_heads, extended_seq_length, extended_key_length)
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros((1, self.n_heads, extended_seq_length, extended_key_length), device=scores.device, dtype=scores.dtype)
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(extended_seq_length, extended_key_length, device=scores.device)

        # truncate position bias based on size of scores
        if scores.dim() == 4:
            sl, kl = scores.shape[-2:]
        elif scores.dim() == 5:
            sl, n_ctxs, kl = scores.shape[-3:]
        else:
            raise ValueError('unexpected size of scores')
        position_bias = position_bias[:, :, -sl:, :kl]  # (batch_size, n_heads, seq_length, key_length)

        # add scores, position_bias, and mask
        if scores.dim() == 5:
            position_bias = position_bias.unsqueeze(-2)  # (batch_size, n_heads, seq_length, 1, key_length)
        if mask is not None:
            position_bias = position_bias + mask   # (batch_size, n_heads, seq_length, [n_ctxs,] key_length)
        scores = scores + position_bias

        return scores, position_bias

    def forward(
        self,
        hidden_states,  # (batch_size, seq_length, dim)
        mask=None,  # (batch_size, n_heads, seq_length, key_length)
        key_value_states=None,  # (batch_size, encoder_seq_length, dim)
        position_bias=None,  # (batch_size, n_heads, seq_length, key_length)
        past_key_value=None,  # (batch_size, n_heads, q_len - 1, dim_per_head)
        ctx_hidden_states: torch.Tensor = None,  # (batch_size, n_ctxs, ctx_seq_length, dim)
        ctx_past_key_value: Tuple[torch.Tensor, torch.Tensor] = None,  # (batch_size, n_ctxs, n_heads, ctx_seq_length, dim_per_head) * 2
        ctx_self_mask: torch.Tensor = None,  # (batch_size, n_ctxs, n_heads, ctx_seq_length, ctx_seq_length) autoregressive mask
        ctx_all_mask: torch.Tensor = None,  # (batch_size, n_ctxs, n_heads, seq_length, ctx_seq_length) full mask
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        in_generation = ctx_past_key_value is not None or past_key_value is not None

        if ctx_hidden_states is None and ctx_past_key_value is None:  # no ctx, run original attn
            attention_output = super().forward(
                hidden_states=hidden_states,
                mask=mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                past_key_value=past_key_value,
                layer_head_mask=layer_head_mask,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions)
            attention_output = ((attention_output[0], None), (attention_output[1], None)) + attention_output[2:]
            return attention_output
        
        batch_size = hidden_states.size(0)

        if ctx_past_key_value is None:  # encode ctxs
            # always compute position bias on the fly 
            # TODO: support layer_head_mask
            n_ctxs = ctx_hidden_states.size(1)
            ctx_hidden_states, ctx_past_key_value = super().forward(
                hidden_states=ctx_hidden_states.flatten(0, 1),  # (batch_size * n_ctxs, ctx_seq_length, dim)
                mask=ctx_self_mask.flatten(0, 1),  # (batch_size * n_ctxs, n_heads, ctx_seq_length, ctx_seq_length)
                use_cache=True,  # always return key value
                output_attentions=False,  # always not return attn
            )[:2]
            ctx_hidden_states = ctx_hidden_states.view(batch_size, n_ctxs, *ctx_hidden_states.shape[1:])
            ctx_key, ctx_value = ctx_past_key_value  # (batch_size * n_ctxs, n_heads, ctx_seq_length, dim_per_head) * 2
            # (batch_size, n_ctxs, n_heads, ctx_seq_length, dim_per_head)
            ctx_key: torch.Tensor = ctx_key.view(batch_size, n_ctxs, *ctx_key.shape[1:])
            # (batch_size, n_ctxs, n_heads, ctx_seq_length, dim_per_head)
            ctx_value: torch.Tensor = ctx_value.view(batch_size, n_ctxs, *ctx_value.shape[1:])
            ctx_past_key_value = (ctx_key, ctx_value)
        else:  # cache from previous steps
            ctx_key, ctx_value = ctx_past_key_value

        ctx_key = ctx_key.transpose(1, 2).contiguous()  # (batch_size, n_heads, n_ctxs, ctx_seq_length, dim_per_head)
        ctx_value = ctx_value.transpose(1, 2).contiguous()  # (batch_size, n_heads, n_ctxs, ctx_seq_length, dim_per_head)

        seq_length = hidden_states.size(1)
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

        # compute scores
        # (batch_size, n_heads, seq_length, key_length)
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        if position_bias is None:
            scores, position_bias = self.add_position_bias(
                extended_seq_length=real_seq_length, extended_key_length=key_length, scores=scores, mask=mask)
        else:
            scores += position_bias

        # compute scores over ctx
        # (batch_size, n_heads, seq_length, n_ctxs, ctx_seq_length)
        ctx_scores = torch.einsum('bnqd,bnckd->bnqck', query_states, ctx_key)
        n_ctxs, ctx_seq_length = ctx_scores.shape[-2:]

        ctx_all_mask = ctx_all_mask.permute(0, 2, 3, 1, 4)  # (batch_size, n_heads, seq_length, n_ctxs, ctx_seq_length)
        ctx_scores, _ = self.add_position_bias(
            extended_seq_length=ctx_seq_length + real_seq_length, extended_key_length=ctx_seq_length + key_length, scores=ctx_scores, mask=ctx_all_mask)
        
        two_dists = None
        if self.attn_specific and not in_generation:  # TODO: implement for inference
            if self.compute_loss:  # compute loss
                assert self.block_size > 0, 'computing ctx attention loss requires block_ctx_attention > 0'

                first_block_attn = ctx_scores[:, self.use_head, :self.block_size]  # (batch_size, first_block_size, n_ctxs, ctx_seq_length)
                # ues -1 since mask is autoregressive so the last position has the complete mask
                first_block_mask = mask[:, 0, -1, :self.block_size].eq(0).to(first_block_attn)  # (batch_size, first_block_size)

                second_block_attn = ctx_scores[:, self.use_head, self.block_size:]  # (batch_size, second_block_size, n_ctxs, ctx_seq_length)
                # ues -1 since mask is autoregressive so the last position has the complete mask
                second_block_mask = mask[:, 0, -1, self.block_size:].eq(0).to(second_block_attn)  # (batch_size, second_block_size)

                first_dist = (first_block_attn.max(-1).values * first_block_mask.unsqueeze(-1)).mean(1).log_softmax(-1)  # (batch_size, n_ctxs)
                if self.loss_type == 'soft':
                    second_dist = (second_block_attn.max(-1).values * second_block_mask.unsqueeze(-1)).mean(1).softmax(-1)  # (batch_size, n_ctxs)
                elif self.loss_type == 'hard':
                    second_dist = torch.zeros_like(first_dist)  # (batch_size, n_ctxs)
                    second_dist[:, 0] = 1.0
                else:
                    raise NotImplementedError

                two_dists = torch.stack([first_dist, second_dist], 0)  # (2, batch_size, n_ctxs)

            # skip the first block
            ctm_shifted_right = torch.cat([torch.ones_like(ctx_all_mask)[..., :1] * torch.finfo(ctx_all_mask.dtype).min, ctx_all_mask[..., :-1]], -1)
            # (batch_size, n_heads, seq_length, n_ctxs, ctx_seq_length)
            block_mask = ctm_shifted_right.eq(0) & ctx_all_mask.eq(0)  # mask all ctx tokens except the first token
            # only do mask for the first block of sequence
            if block_mask.size(2) == 1:
                block_mask = block_mask.repeat(1, 1, ctx_scores.size(2), 1, 1)  # (batch_size, n_heads, seq_length, n_ctxs, ctx_seq_length)
            block_mask[:, :, self.block_size:] = False
            block_mask = block_mask.to(ctx_all_mask) * torch.finfo(ctx_all_mask.dtype).min
            ctx_scores += block_mask

        cat_scores = torch.cat([ctx_scores.flatten(3, 4), scores], -1)  # (batch_size, n_heads, seq_length, n_ctxs * ctx_seq_length + key_length)
        cat_attn_weights = nn.functional.softmax(cat_scores.float(), dim=-1).type_as(cat_scores)  # (batch_size, n_heads, seq_length, n_ctxs * ctx_seq_length + key_length)
        cat_attn_weights = nn.functional.dropout(cat_attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            cat_attn_weights = cat_attn_weights * layer_head_mask

        cat_value_states = torch.cat([ctx_value.flatten(2, 3), value_states], -2)  # (batch_size, n_heads, n_ctxs * ctx_seq_length + key_length, dim_per_head)
        attn_output = unshape(torch.matmul(cat_attn_weights, cat_value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        ctx_past_key_value = ctx_past_key_value if (self.is_decoder and use_cache) else None
        outputs = ((attn_output, ctx_hidden_states),) + ((present_key_value_state, ctx_past_key_value),) + (position_bias,)

        if output_attentions:
            if two_dists is not None:
                cat_attn_weights = two_dists  # avoid checkpoint bug TODO: better workaround?
            outputs = outputs + (cat_attn_weights,)
        return outputs


class FusionT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(
        self, 
        config, 
        has_relative_attention_bias: bool = False,
        layer_index: int = -1):
        super().__init__(config, has_relative_attention_bias=has_relative_attention_bias)
        self.SelfAttention = FusionT5Attention(
            config, 
            has_relative_attention_bias=has_relative_attention_bias, 
            layer_index=layer_index)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        ctx_hidden_states: torch.Tensor = None,
        ctx_past_key_value: Tuple[torch.Tensor, torch.Tensor] = None,
        ctx_self_mask: torch.Tensor = None,
        ctx_all_mask: torch.Tensor = None,
        use_cache=False,
        output_attentions=False,
    ):
        # whether we need to compute hidden_states
        run_ctx_hidden_states = ctx_hidden_states is not None

        # layer norm
        normed_hidden_states = self.layer_norm(hidden_states)
        normed_ctx_hidden_states = self.layer_norm(ctx_hidden_states) if run_ctx_hidden_states else None

        # attn
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            ctx_hidden_states=normed_ctx_hidden_states,
            ctx_past_key_value=ctx_past_key_value,
            ctx_self_mask=ctx_self_mask,
            ctx_all_mask=ctx_all_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # residual
        hidden_states = hidden_states + self.dropout(attention_output[0][0])
        ctx_hidden_states = (ctx_hidden_states + self.dropout(attention_output[0][1])) if run_ctx_hidden_states else None
        outputs = ((hidden_states, ctx_hidden_states),) + attention_output[1:]
        return outputs


class FusionT5LayerCrossAttention(T5LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        hidden_states,  # (batch_size, seq_length, dim)
        key_value_states: torch.Tensor,  # (batch_size, encoder_seq_length, dim)
        attention_mask: torch.Tensor = None,  # (batch_size, 1, 1, encoder_seq_length)
        position_bias: torch.Tensor = None,  # (batch_size, 1, 1, encoder_seq_length)
        layer_head_mask=None,
        past_key_value: torch.Tensor = None,  # (batch_size, n_heads, encoder_seq_length, dim_per_head)
        ctx_hidden_states: torch.Tensor = None,  # (batch_size, n_ctxs, ctx_seq_length, dim)
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        # no need to compute cross attention between ctx and key_value_states for decoding except the first step
        run_ctx = past_key_value is None and ctx_hidden_states is not None

        # layer norm
        normed_hidden_states = self.layer_norm(hidden_states)
        # attn
        query_length_for_target = query_length  # extend query_length because of ctx
        if ctx_hidden_states is not None:
            query_length_for_target = ctx_hidden_states.size(2) if query_length is None else (query_length + ctx_hidden_states.size(2))
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length_for_target,
            output_attentions=output_attentions,
        )
        # residual
        hidden_states = hidden_states + self.dropout(attention_output[0])

        if run_ctx:
            # duplicate
            bs, n_ctxs = ctx_hidden_states.shape[:2]
            ctx_hidden_states = ctx_hidden_states.flatten(0, 1)  # (batch_size * n_ctxs, ctx_seq_length, dim)
            _attention_mask = attention_mask.unsqueeze(1).repeat(1, n_ctxs, 1, 1, 1).flatten(0, 1)  # (batch_size * n_ctxs, 1, 1, encoder_seq_length)
            _key_value_states = key_value_states.unsqueeze(1).repeat(1, n_ctxs, 1, 1).flatten(0, 1)  # (batch_size * n_ctxs, encoder_seq_length, dim)
            _past_key_value = past_key_value.unsqueeze(1).repeat(1, n_ctxs, 1, 1, 1).flatten(0, 1) if past_key_value is not None else None  # (batch_size * n_ctxs, n_heads, encoder_seq_length, dim_per_head)
            # TODO: position_bias
            # layer norm
            normed_ctx_hidden_states = self.layer_norm(ctx_hidden_states)
            # attn
            ctx_attention_output = self.EncDecAttention(
                normed_ctx_hidden_states,
                mask=_attention_mask,
                key_value_states=_key_value_states,
                position_bias=None,
                layer_head_mask=layer_head_mask,
                past_key_value=_past_key_value,
                use_cache=use_cache,
                query_length=None,
                output_attentions=output_attentions,
            )
            # residual
            ctx_hidden_states = ctx_hidden_states + self.dropout(ctx_attention_output[0])
            # reshape
            ctx_hidden_states = ctx_hidden_states.view(bs, n_ctxs, *ctx_hidden_states.shape[1:])
        
        outputs = ((hidden_states, ctx_hidden_states),) + attention_output[1:]
        return outputs


class FusionT5Block(T5Block):
    def __init__(
        self, 
        config, 
        has_relative_attention_bias: bool = False,
        layer_index: int = -1):
        super().__init__(config, has_relative_attention_bias=has_relative_attention_bias)
        if self.is_decoder:
            self.layer = nn.ModuleList()
            self.layer.append(FusionT5LayerSelfAttention(
                config, 
                has_relative_attention_bias=has_relative_attention_bias,
                layer_index=layer_index))
            self.layer.append(FusionT5LayerCrossAttention(config))
            self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        ctx_hidden_states: torch.Tensor = None,
        ctx_past_key_value: Tuple[torch.Tensor, torch.Tensor] = None,
        ctx_self_mask: torch.Tensor = None,
        ctx_all_mask: torch.Tensor = None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            ctx_hidden_states=ctx_hidden_states,
            ctx_past_key_value=ctx_past_key_value,
            ctx_self_mask=ctx_self_mask,
            ctx_all_mask=ctx_all_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        (hidden_states, ctx_hidden_states), (present_key_value_state, ctx_past_key_value) = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        def clamp_fp16(hs):
            if hs is None:
                return None
            if hs.dtype == torch.float16 and torch.isinf(hs).any():
                clamp_value = torch.finfo(hs.dtype).max - 1000
                hs = torch.clamp(hs, min=-clamp_value, max=clamp_value)
            return hs
        hidden_states, ctx_hidden_states = clamp_fp16(hidden_states), clamp_fp16(ctx_hidden_states)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention if using past key value states
            # TODO: is it a bug to use present_key_value_state instead of self_attn_past_key_value to get query_length?
            query_length = None if present_key_value_state is None else present_key_value_state[0].shape[2]
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                ctx_hidden_states=ctx_hidden_states,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states, ctx_hidden_states = cross_attention_outputs[0]
            # clamp inf values to enable fp16 training
            hidden_states, ctx_hidden_states = clamp_fp16(hidden_states), clamp_fp16(ctx_hidden_states)
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]
            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        hidden_states = clamp_fp16(hidden_states)
        if ctx_hidden_states is not None:
            ctx_hidden_states = self.layer[-1](ctx_hidden_states)
            ctx_hidden_states = clamp_fp16(ctx_hidden_states)

        outputs = (hidden_states, ctx_hidden_states,)  # cannot be nested due to gradient checkpointing

        if use_cache:
            outputs = outputs + ((present_key_value_state, ctx_past_key_value),) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class FusionT5PreTrainedModel(T5PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FusionT5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["T5Block", "FusionT5Block"]

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (T5Attention, T5Stack, FusionT5Attention, FusionT5Stack)):
            module.gradient_checkpointing = value


class FusionT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)
        self.block = nn.ModuleList(
            [FusionT5Block(
                config, 
                has_relative_attention_bias=True,  # has_relative_attention_bias for all layers
                layer_index=i) for i in range(config.num_layers)])
        rab = self.block[0].layer[0].SelfAttention.relative_attention_bias
        for block in self.block:  # copy relative_attention_bias to all layers to enable on-the-fly computation of position_bias 
            block.layer[0].SelfAttention.relative_attention_bias = rab
        # Initialize weights and apply final processing
        # TODO: any issue when executed twice?
        self.post_init()
    
    def _copy_relative_attention_bias(self):
        if hasattr(self, '_copied') and self._copied:
            return
        rab = self.block[0].layer[0].SelfAttention.relative_attention_bias
        for block in self.block:
            block.layer[0].SelfAttention.relative_attention_bias = rab
        self._copied = True
    
    def forward(
        self,
        input_ids=None,  # (batch_size, seq_length)
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,  # (batch_size, n_heads, q_len - 1, dim_per_head) * 2, (batch_size, n_heads, encoder_seq_length, dim_per_head) * 2, (batch_size, n_ctxs, n_heads, ctx_seq_length, dim_per_head) * 2
        decoder_ctx_input_ids: torch.Tensor = None,  # (batch_size, n_ctxs, ctx_seq_length)
        decoder_ctx_attention_mask: torch.Tensor = None,  # (batch_size, n_ctxs, ctx_seq_length)
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):        
        # copy
        self._copy_relative_attention_bias()

        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ctx_input_ids, ctx_attention_mask = decoder_ctx_input_ids, decoder_ctx_attention_mask
        has_ctx = ctx_input_ids is not None
        has_ctx_and_run_hidden_states = has_ctx and (past_key_values is None or past_key_values[0][1] is None)

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if has_ctx:  # get ctx shape
            assert ctx_input_ids.size(0) == batch_size
            _, n_ctxs, ctx_seq_length = ctx_input_ids.size()

        ctx_inputs_embeds = None
        if has_ctx_and_run_hidden_states:  # emb ctx
            ctx_inputs_embeds = self.embed_tokens(ctx_input_ids)

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)  # (batch_size, mask_seq_length)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )  # (batch_size, encoder_seq_length)

        if has_ctx and ctx_attention_mask is None:
            ctx_attention_mask = torch.ones(batch_size, n_ctxs, ctx_seq_length).to(inputs_embeds.device)  # (batch_size, n_ctxs, ctx_seq_length)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [(None, None) for _ in range(len(self.block))]

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)  # (batch_size, 1, mask_seq_length, mask_seq_length)
        ctx_self_mask = ctx_all_mask = None
        if has_ctx:
            ctx_attention_mask = ctx_attention_mask.flatten(0, 1)  # (batch_size * n_ctxs, ctx_seq_length)
            ctx_self_mask = self.get_extended_attention_mask(ctx_attention_mask, ctx_attention_mask.size())  # (batch_size * n_ctxs, 1, ctx_seq_length, ctx_seq_length)
            ctx_self_mask = ctx_self_mask.view(batch_size, n_ctxs, *ctx_self_mask.shape[1:])  # (batch_size, n_ctxs, 1, ctx_seq_length, ctx_seq_length)
            ctx_all_mask = self.invert_attention_mask(ctx_attention_mask)  # (batch_size * n_ctxs, 1, 1, ctx_seq_length)
            ctx_all_mask = ctx_all_mask.view(batch_size, n_ctxs, *ctx_all_mask.shape[1:])  # (batch_size, n_ctxs, 1, 1, ctx_seq_length)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)  # (batch_size, 1, 1, encoder_sequence_length)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)

        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        ctx_hidden_states = None
        if has_ctx_and_run_hidden_states:
            ctx_hidden_states = self.dropout(ctx_inputs_embeds)

        for i, (layer_module, (past_key_value, ctx_past_key_value)) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if has_ctx_and_run_hidden_states:
                    ctx_hidden_states = ctx_hidden_states.to(hidden_states.device)
                if has_ctx:
                    ctx_self_mask = ctx_self_mask.to(hidden_states.device)
                    ctx_all_mask = ctx_all_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))
                    return custom_forward

                #layer_outputs = checkpoint(
                layer_outputs = ds_checkpointing.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    ctx_hidden_states,
                    None,
                    ctx_self_mask,
                    ctx_all_mask,
                )

            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    ctx_hidden_states=ctx_hidden_states,
                    ctx_past_key_value=ctx_past_key_value,
                    ctx_self_mask=ctx_self_mask,
                    ctx_all_mask=ctx_all_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:2] + ((None, None),) + layer_outputs[2:]

            hidden_states, ctx_hidden_states, (present_key_value_state, ctx_past_key_value) = layer_outputs[:3]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[3]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 4]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + ((present_key_value_state, ctx_past_key_value),)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[4],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[6],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


@dataclass
class FusionSeq2SeqLMOutput(Seq2SeqLMOutput):
    ctx_attention_loss: Optional[torch.FloatTensor] = None
    rerank_accuracy: Optional[torch.FloatTensor] = None
    ctx_pred_dist: Optional[torch.FloatTensor] = None
    ctx_gold_dist: Optional[torch.FloatTensor] = None


@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class FusionT5ForConditionalGeneration(FusionT5PreTrainedModel):  # TODO: multiple inheritance
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: FusionT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = FusionT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.attn_specific = config.ctx_attention_loss is not None
        if self.attn_specific:
            self.loss_alpha = config.ctx_attention_loss['alpha']
            self.loss_layer = config.ctx_attention_loss['layer']

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        decoder_ctx_input_ids: Optional[torch.LongTensor] = None,  # (batch_size, n_ctxs, ctx_seq_length)
        decoder_ctx_attention_mask: Optional[torch.FloatTensor] = None,  # (batch_size, n_ctxs, ctx_seq_length)
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if decoder_ctx_input_ids is not None:
                decoder_ctx_input_ids = decoder_ctx_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
            if decoder_ctx_attention_mask is not None:
                decoder_ctx_attention_mask = decoder_ctx_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            decoder_ctx_input_ids=decoder_ctx_input_ids,
            decoder_ctx_attention_mask=decoder_ctx_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=True,  # TODO: debug
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = ctx_pred_dist = ctx_gold_dist = ctx_attention_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            
            if self.attn_specific:
                two_dists = decoder_outputs.attentions[self.loss_layer]
                ctx_pred_dist, ctx_gold_dist = two_dists[0], two_dists[1]
                kldiv = torch.nn.KLDivLoss(reduction='batchmean')
                ctx_attention_loss = kldiv(ctx_pred_dist, ctx_gold_dist.detach())
                loss = loss + self.loss_alpha * ctx_attention_loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return FusionSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            ctx_pred_dist=ctx_pred_dist,
            ctx_gold_dist=ctx_gold_dist,
            ctx_attention_loss=ctx_attention_loss,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        decoder_ctx_input_ids=None,
        past=None,
        attention_mask=None,
        decoder_ctx_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_ctx_input_ids": decoder_ctx_input_ids,
            "decoder_ctx_attention_mask": decoder_ctx_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


def prepare(
    tokenizer, 
    questions: List[str], 
    answers: List[str], 
    ctxs: List[List[str]], 
    question_max_length: int = 50,
    answer_max_length: int = 50,
    ctx_max_length: int = 50,
    add_to_ctx: str = '<pad> ',
    add_to_answers: str = 'Answer: ',
    for_generation: bool = False,
    ctx_to_answer: bool = False):

    questions = [
        'Definition: Given a question, generate a descriptive answer. Question: Who is the president of the US?',
        'Definition: Given a question, generate a descriptive answer. Question: who lives in the imperial palace in tokyo?'
    ]
    ctxs = [
        ['Evidence: Frank Xu is the president of the US. Answer '],
        ['Evidence: The Tokyo Imperial Palace is the primary residence of the Emperor of Japan. Answer ']
    ]
    answers = [
        'Barack Obama.',
        'the Imperial Family.'
    ]
    
    # question
    questions = tokenizer(
        questions,
        truncation=True,
        padding=True,
        max_length=question_max_length,
        return_tensors='pt'
    )

    # context
    bs, n_ctxs = len(ctxs), len(ctxs[0])
    tokenizer.padding_side = 'left'  # put ctx on the right to be close to the answer
    ctxs = tokenizer(
        [add_to_ctx + ctx for ctx in sum(ctxs, [])],
        truncation=True,
        padding=True,
        max_length=ctx_max_length,
        return_tensors='pt',
        add_special_tokens=False,  # avoid eos
    )
    tokenizer.padding_side = 'right'
    decoder_ctx_input_ids = ctxs.input_ids.view(bs, n_ctxs, -1)  # (batch_size, n_ctxs, ctx_seq_length)
    decoder_ctx_attention_mask = ctxs.attention_mask.view(bs, n_ctxs, -1)  # (batch_size, n_ctxs, ctx_seq_length)
    
    # answers
    answers = tokenizer(
        [add_to_answers + ans for ans in answers],
        truncation=True,
        padding=True,
        max_length=answer_max_length,
        return_tensors='pt'
    )
    decoder_input_ids = answers.input_ids  # (batch_size, seq_length)
    decoder_attention_mask = answers.attention_mask  # (batch_size, seq_length)

    # convert answers to labels
    assert len(add_to_answers) > 0
    assert tokenizer.pad_token_id == 0
    # remove the added "special" token to the answer to make it labels
    labels = torch.zeros_like(decoder_input_ids)
    labels[..., :-1] = decoder_input_ids[..., 1:].clone()
    labels.masked_fill_(labels == 0, -100)

    if ctx_to_answer:  # combine ctxs and answers
        decoder_input_ids = torch.cat([decoder_ctx_input_ids[:, 0, :], decoder_input_ids], -1)
        decoder_attention_mask = torch.cat([decoder_ctx_attention_mask[:, 0, :], decoder_attention_mask], -1)
        labels = torch.cat([torch.ones_like(decoder_ctx_input_ids[:, 0, :]) * -100, labels], -1)

    batch = {
        'input_ids': questions.input_ids,
        'attention_mask': questions.attention_mask,
        'decoder_ctx_input_ids': decoder_ctx_input_ids,
        'decoder_ctx_attention_mask': decoder_ctx_attention_mask,
    }

    if not for_generation:
        batch['labels'] = labels  # decoder_input_ids will be created based on labels
        batch['decoder_input_ids'] = decoder_input_ids
        batch['decoder_attention_mask'] = decoder_attention_mask 

    return batch
