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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from collections import defaultdict
import pickle

import numpy as np
import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
import torch.linalg
from tqdm import trange

import evaluate
import transformers
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from qa_trainer import QuestionAnsweringSeq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import ShardSampler
from transformers.trainer_utils import EvalPrediction

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search
from deepspeed import checkpointing as ds_checkpointing
from models.fusion_t5 import FusionT5Config, FusionT5ForConditionalGeneration, FusionSeq2SeqLMOutput
from models.retriever import BM25
from knnlm import KNNWrapper, KEY_TYPE, DIST

from em_eval import ems

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    torch_dtype: Optional[str] = field(default=None)
    decode_offset: Optional[int] = field(default=160)
    ctx_attn_add_bias: Optional[bool] = field(default=False)
    ctx_position_shift_right: Optional[bool] = field(default=False)
    ctx_attention_loss: Optional[str] = field(default=None)
    bos_attention: Optional[str] = field(default=None)
    ctx_topk: Optional[int] = field(default=0)
    encode_retrieval_in: Optional[str] = field(default='decoder')
    knnlm: Optional[bool] = field(default=False)

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )
        self.ctx_attention_loss = FusionT5Config.parse_ctx_attention_loss(self.ctx_attention_loss)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    beir_dir: Optional[str] = field(default=None, metadata={"help": "The beir directory"})
    beir_index_name: Optional[str] = field(default=None, metadata={"help": "Index name"})
    predict_file: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    use_context: Optional[bool] = field(default=True, metadata={"help": "Whether to use context in training"})
    context_bos: Optional[bool] = field(default=True, metadata={"help": "Whether to prepend bos to contexts"})
    answer_bos: Optional[bool] = field(default=False, metadata={"help": "Whether to prepend bos to answers"})
    max_question_len: Optional[int] = field(default=128)
    max_context_len: Optional[int] = field(default=128)
    max_answer_len: Optional[int] = field(default=128)
    generation_prefix_len: Optional[int] = field(default=0)
    question_prefix: Optional[str] = field(default='Definition: Given a question, generate a descriptive answer. Question: ')
    context_prefix: Optional[str] = field(default='Evidence: ')
    answer_prefix: Optional[str] = field(default='Answer: ')
    encoder_input_for_context: Optional[str] = field(default='Definition: Given a question, generate a descriptive answer.')

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    depth: Optional[int] = field(default=20)
    use_target_answer: Optional[bool] = field(default=False)


@dataclass
class TrainingArgs(Seq2SeqTrainingArguments):
    extract_attention: bool = field(default=False)
    do_eval_special: str = field(default=None)
    output_file: Optional[str] = field(default=None)


@dataclass
class KnnlmArguments:
    dstore_size: int = field(default=None)
    dstore_dir: str = field(default=None)


@dataclass
class DataCollatorForFusion:

    model: transformers.AutoModelForSeq2SeqLM
    tokenizer: transformers.PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = None
    use_context: bool = True
    context_bos: bool = True  # prepend bos to contexts
    answer_bos: bool = False  # prepend bos to answers
    encode_retrieval_in: bool = 'decoder'
    max_question_len: int = None
    max_context_len: int = None
    max_answer_len: int = None
    question_prefix: str = None
    answer_prefix: str = None
    context_prefix: str = None
    encoder_input_for_context: str = None  # the input to encoder that contexts attend to (through cross-attention)
    _train: bool = True

    def eval(self):
        self._train = False

    def train(self):
        self._train = True

    @property
    def all_tokens(self):
        if hasattr(self, '_all_tokens'):
            return self._all_tokens
        self._all_tokens = list(self.tokenizer.get_vocab().values())

    def get_real_decoder_start_token_id(self):
        if not self.answer_bos:  # use the first token of the answer prefix to start generation if bos is not prepended to the answer
            return self.tokenizer.encode(self.answer_prefix)[0]
        return self.model.config.decoder_start_token_id

    def get_labels_from_decoder_input_ids(self, decoder_input_ids):
        assert self.tokenizer.pad_token_id == 0
        # remove the first token of answers (treated as generation trigger token) to make it labels
        labels = torch.zeros_like(decoder_input_ids)
        labels[..., :-1] = decoder_input_ids[..., 1:].clone()
        labels.masked_fill_(labels == 0, -100)
        return labels

    def encode_context(self, ctxs: List[str]):
        decoder_start_token = self.tokenizer.convert_ids_to_tokens([self.model.config.decoder_start_token_id])[0]

        # setting
        if self.encode_retrieval_in == 'encoder':
            # normal padding
            padding_side = 'right'
            add_special_tokens = True
            context_bos = ''
        elif self.encode_retrieval_in == 'decoder':
            # put ctx on the right to be close to the answer
            padding_side = 'left'
            add_special_tokens = False
            context_bos = f'{decoder_start_token} '
        else:
            raise NotImplementedError

        # tokenize
        # TODO: problem with multiple processes?
        ori_ps = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        encoded = self.tokenizer(
            [context_bos + self.context_prefix + ctx for ctx in ctxs],
            truncation=True,
            padding=True,
            max_length=self.max_context_len,
            add_special_tokens=add_special_tokens,  # avoid eos
            return_tensors='pt')
        self.tokenizer.padding_side = ori_ps
        return encoded

    def __call__(
        self,
        examples: List[Dict],
        debug: bool = False):
        decoder_start_token = self.tokenizer.convert_ids_to_tokens([self.model.config.decoder_start_token_id])[0]

        # question ids
        qids = np.array([e['qid'] for e in examples])

        # questions
        questions = self.tokenizer(
            [self.question_prefix + e['question'] for e in examples],
            truncation=True,
            padding=True,
            max_length=self.max_question_len,
            return_tensors='pt')
        input_ids = questions.input_ids  # (batch_size, encoder_seq_length)
        attention_mask = questions.attention_mask  # (batch_size, encoder_seq_length)

        # encoder input for context
        if self.encoder_input_for_context:
            eifc = self.tokenizer(
                [self.encoder_input_for_context],
                truncation=True,
                padding=True,
                max_length=self.max_question_len,
                return_tensors='pt')
            input_ids_for_ctx = eifc.input_ids.unsqueeze(1)  # (1 (batch_size), 1 (n_ctxs), encoder_seq_length_for_context)
            attention_mask_for_ctx = eifc.attention_mask.unsqueeze(1)  # (1 (batch_size), 1 (n_ctxs), encoder_seq_length_for_context)

        # ctxs
        bs, n_ctxs = len(examples), len(examples[0]['ctxs'])
        if bs > 1 and len(np.unique([len(e['ctxs']) for e in examples])) > 1:
            raise Exception('num of ctxs is inconsistent')
        ctxs = self.encode_context([ctx for e in examples for ctx in e['ctxs']])
        decoder_ctx_input_ids = ctxs.input_ids.view(bs, n_ctxs, -1)  # (batch_size, n_ctxs, ctx_seq_length)
        decoder_ctx_attention_mask = ctxs.attention_mask.view(bs, n_ctxs, -1)  # (batch_size, n_ctxs, ctx_seq_length)

        # answers
        dst = f'{decoder_start_token} ' if self.answer_bos else ''
        answers = self.tokenizer(
            [dst + self.answer_prefix + e['answers'] for e in examples],
            truncation=True,
            padding=True,
            max_length=self.max_answer_len,
            return_tensors='pt'
        )
        decoder_input_ids = answers.input_ids  # (batch_size, seq_length)
        decoder_attention_mask = answers.attention_mask  # (batch_size, seq_length)
        # convert answers to labels
        labels = self.get_labels_from_decoder_input_ids(decoder_input_ids)

        batch = {
            'idxs': qids,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels
        }
        if self.use_context:
            batch['decoder_ctx_input_ids'] = decoder_ctx_input_ids
            batch['decoder_ctx_attention_mask'] = decoder_ctx_attention_mask
            if self.encoder_input_for_context:
                batch['input_ids_for_ctx'] = input_ids_for_ctx
                batch['attention_mask_for_ctx'] = attention_mask_for_ctx

        if debug:
            print(batch)
            print(decoder_ctx_input_ids[0, :3])
            print(decoder_ctx_attention_mask[0, :3])
            input()

        return batch


def _load_data_file(path: str, max_num_samples: int = 0):
    if path.endswith('json'):
        with open(path) as f:
            data = json.load(f)
    else:
        data = torch.load(path)
    if max_num_samples:
        max_num_samples = min(len(data), max_num_samples)
        data = data[:max_num_samples]
    return data


class CacheManager:
    def __init__(
        self,
        get_cache: bool = False,
        save_cache: bool = False,
        cache_file: str = None
    ):
        self.get_cache = get_cache
        self.save_cache = save_cache
        self.step = 0
        self.rank = dist.get_rank()
        self.cache_file = f'{cache_file}.{self.rank}' if cache_file else cache_file
        if self.use_cache and self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as fin:
                self.cache = pickle.load(fin)
                logger.info(f'Retrieval cache size {len(self.cache)}')
        else:
            self.cache = {}

    @property
    def use_cache(self):
        return self.get_cache or self.save_cache

    def save(self, val: Any):
        if not self.save_cache:
            return
        self.cache[(self.rank, self.step)] = val
        self.step += 1

    def get(self):
        if not self.get_cache:
            return None
        val = self.cache[(self.rank, self.step)]
        self.step += 1
        return val

    def dump(self):
        if self.save_cache and self.cache_file and not os.path.exists(self.cache_file):
            with open(self.cache_file, 'wb') as fout:
                pickle.dump(self.cache, fout)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArgs, KnnlmArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, knnlm_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, knnlm_args = parser.parse_args_into_dataclasses()

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: Seq2SeqTrainingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        'torch_dtype': model_args.torch_dtype,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if 'opt' in model_args.model_name_or_path or 'gpt2' in model_args.model_name_or_path:
        config_specific_kwargs = {
            'decode_offset': model_args.decode_offset,
            'ctx_attn_add_bias': model_args.ctx_attn_add_bias,
            'ctx_position_shift_right': model_args.ctx_position_shift_right
        }
    elif 't5' in model_args.model_name_or_path:
        config_specific_kwargs = {}
        for key in ['ctx_attention_loss', 'bos_attention', 'ctx_topk', 'encode_retrieval_in']:
            if getattr(model_args, key) is not None:
                config_specific_kwargs[key] = getattr(model_args, key)
    else:
        config_specific_kwargs = {}

    if 'opt' in model_args.model_name_or_path:
        _config_class = FusionOPTConfig
        _model_class = FusionOPTForCausalLM
    elif 'gpt2' in model_args.model_name_or_path:
        _config_class = FusionGPT2Config
        _model_class = FusionGPT2LMHeadModel
    elif 't5' in model_args.model_name_or_path:
        _config_class = FusionT5Config
        _model_class = FusionT5ForConditionalGeneration
    else:
        raise ValueError

    if model_args.config_name:
        config = _config_class.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        # get original config
        _dict = _config_class.get_config_dict(model_args.model_name_or_path, **config_kwargs)[0]
        # update config with new fields
        _dict.update(config_specific_kwargs)
        config = _config_class.from_dict(_dict)
    else:
        raise ValueError

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "use_cache": False,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer: transformers.PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        config.pad_token_id = tokenizer.pad_token_id

    if model_args.model_name_or_path:
        model = _model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = _model_class.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))

    if model_args.knnlm:  # load knnlm
        knn_wrapper = KNNWrapper(
            dstore_size=knnlm_args.dstore_size,
            dstore_dir=knnlm_args.dstore_dir,
            dimension=model.config.hidden_size,
            knn_sim_func=DIST.l2,
            knn_keytype=KEY_TYPE.last_ffn_input,
            no_load_keys=True,
            move_dstore_to_mem=True,
            knn_gpu=training_args.local_rank,
            recompute_dists=False,
            k=1,
            lmbda=0.25,
            knn_temp=50,
            probe=32)
        knn_wrapper.break_into(model)

    if 'opt' in model_args.model_name_or_path or 'gpt2' in model_args.model_name_or_path:
        def format_data_dict(examples: List[Dict]):
            formated = []
            for example in examples:
                ctxs = [
                    f'question: {x["question"]}\ntitle: {c["title"]} | context: {c["text"]}\n'
                    for c in example['ctxs'][:data_args.depth]
                ]
                if data_args.add_answer_prefix:
                    if data_args.long_prefix:
                        a_pfx = 'the answer is: '
                    else:
                        a_pfx = 'answer: '
                else:
                    a_pfx = ''
                answers = [f"{a_pfx}{example['target']}{tokenizer.eos_token}"] if 'target' in example else [f'{a_pfx}{a}{tokenizer.eos_token}' for a in example['answers']]
                formated.append({'ctxs': ctxs, 'answers': answers})
            return formated
        def compute_metrics(output):
            predictions=output.predictions
            references=output.label_ids
            scores = [ems(p, r['answers']) for p, r in zip(predictions, references)]
            return {'em_score': sum(scores) / len(scores)}
        def post_processing_function(references, features, outputs, stage='eval'):
            preds = outputs.predictions
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            if data_args.add_answer_prefix:
                if data_args.long_prefix:
                    formatted_predictions = [x[len('the answer is: '):].strip() for x in decoded_preds]
                else:
                    formatted_predictions = [x[len('answer: '):].strip() for x in decoded_preds]
            else:
                formatted_predictions = [x.strip() for x in decoded_preds]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    elif 't5' in model_args.model_name_or_path:
        def format_data_dict(examples: List[Dict]):
            formatted = []
            for example in examples:
                question = example['question']
                ctxs = [ctx['text'] for ctx in example['ctxs'][:data_args.depth]]
                answers = [ans[0] if type(ans) is list else ans for ans in example['answers']][0]  # the first alias of the first answer
                formatted.append({'qid': example['id'], 'question': question, 'ctxs': ctxs, 'answers': answers})
            return formatted
        def compute_metrics(output):
            predictions=output.predictions
            references=output.label_ids  # TODO: use all answers in metric
            metric_func = evaluate.load('rouge')
            metrics = metric_func.compute(predictions=predictions, references=references)
            return metrics
        def post_processing_function(
            references,
            features,
            outputs,
            stage: str = 'eval',
            debug: bool = False,
            remove_decoding_prefix: bool = True):
            decode_ids = outputs.predictions
            gold_ids = outputs.label_ids
            assert tokenizer.pad_token_id == 0
            gold_ids[gold_ids == -100] = 0
            skip_prefix = remove_decoding_prefix and data_args.generation_prefix_len > 0
            if skip_prefix:
                decode_ids = decode_ids[:, 1 + data_args.generation_prefix_len:]  # skip bos + prefix
                gold_ids = gold_ids[:, data_args.generation_prefix_len:]  # skip prefix
            decoded_preds = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)
            gold_preds = tokenizer.batch_decode(gold_ids, skip_special_tokens=True)
            if skip_prefix:
                decoded_preds = [x.strip() for x in decoded_preds]
                gold_preds = [x.strip() for x in gold_preds]
            else:
                decoded_preds = [x.strip().lstrip(data_args.answer_prefix).strip() for x in decoded_preds]
                gold_preds = [x.strip().lstrip(data_args.answer_prefix).strip() for x in gold_preds]
            if debug:
                print(outputs.predictions[:2])
                print(decoded_preds[:2])
                print(outputs.label_ids[:2])
                print(gold_preds[:2])
                input()
            return EvalPrediction(predictions=decoded_preds, label_ids=gold_preds)
    else:
        raise ValueError

    if training_args.do_train:
        train_dataset = format_data_dict(_load_data_file(data_args.train_file, data_args.max_train_samples))

    if training_args.do_eval:
        validation_examples = _load_data_file(data_args.validation_file, data_args.max_eval_samples)
        validation_dataset = format_data_dict(validation_examples)

    collator = DataCollatorForFusion(
        model=model,
        tokenizer=tokenizer,
        max_question_len=data_args.max_question_len,
        max_context_len=data_args.max_context_len,
        max_answer_len=data_args.max_answer_len,
        question_prefix=data_args.question_prefix,
        answer_prefix=data_args.answer_prefix,
        context_prefix=data_args.context_prefix,
        use_context=data_args.use_context,
        context_bos=data_args.context_bos,
        answer_bos=data_args.answer_bos,
        encoder_input_for_context=data_args.encoder_input_for_context,
        encode_retrieval_in=model_args.encode_retrieval_in)

    gen_kwargs = {
        'max_length': data_args.max_answer_len,
        'num_beams': 1,
        'decoder_start_token_id': collator.get_real_decoder_start_token_id(),
        'prefix_len': data_args.generation_prefix_len,
    }

    # Initialize our Trainer
    trainer = QuestionAnsweringSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        eval_examples=validation_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        post_process_function=post_processing_function)

    if training_args.deepspeed:
        ds_checkpointing.configure(None, training_args.hf_deepspeed_config.config)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval and training_args.do_eval_special:
        logger.info("*** Evaluate special ***")
        model.eval()
        finals = []

        sampler = ShardSampler(
            validation_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            num_processes=training_args.world_size,
            process_index=training_args.process_index)

        dataloader = DataLoader(
            validation_dataset,
            sampler=sampler,
            batch_size=training_args.eval_batch_size,
            collate_fn=collator,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory)
        # TODO: if there is not enough examples for the last batch the examples at the beginning will be reused

        corpus, queries, qrels = GenericDataLoader(data_folder=data_args.beir_dir).load(split='dev')

        if not data_args.beir_index_name:
            data_args.beir_index_name = 'test'
            if not trainer.is_world_process_zero():
                dist.barrier()
            if trainer.is_world_process_zero():  # TODO: only initialize when necessary
                logger.info("Build BM25 index")
                BM25Search(index_name=data_args.beir_index_name, hostname='localhost', initialize=True, number_of_shards=1).index(corpus)
                time.sleep(5)
                dist.barrier()

        retriever = BM25(
            tokenizer=tokenizer,
            collator=collator,
            corpus=corpus,
            index_name=data_args.beir_index_name,
            use_encoder_input_ids=True,
            use_decoder_input_ids=True)
        decoder_retrieval_kwargs = {
            'retriever': retriever,
            'topk': data_args.depth,
            'frequency': 1 if data_args.use_context else 0
        }
        model.cache = CacheManager(get_cache=False, save_cache=False, cache_file='.cache')

        for step, batch in tqdm(enumerate(dataloader)):
            batch = trainer._prepare_inputs(batch)

            # generate
            gen_outputs = None
            if 'generate' in training_args.do_eval_special:
                with torch.no_grad():
                    # remove model inputs irrelevant to generation
                    gen_batch = dict(batch)
                    for key in ['labels', 'decoder_input_ids', 'decoder_attention_mask']:
                        if key in gen_batch:
                            del gen_batch[key]

                    _gen_kwargs = dict(gen_kwargs)
                    # prepare prefix function
                    prefix_len = _gen_kwargs.pop('prefix_len', 0)
                    prefix_allowed_tokens_fn = None
                    if prefix_len:
                        prefix_ids = batch['labels'][:, :prefix_len]
                        def prefix_allowed_tokens_fn(batch_id: int, gen_ids: torch.Tensor) -> List[int]:
                            if gen_ids.shape[-1] > len(prefix_ids[batch_id]):
                                return collator.all_tokens
                            return prefix_ids[batch_id][gen_ids.shape[-1] - 1]

                    # generate and use generated outputs to overwrite inputs
                    gen_batch['decoder_retrieval_kwargs'] = decoder_retrieval_kwargs
                    gen_outputs = model.generate(
                        **gen_batch,
                        **_gen_kwargs,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                        output_scores=True,
                        return_dict_in_generate=True)

                    if 'rerank' in training_args.do_eval_special:  # use the generated to run evaluation
                        batch['decoder_input_ids'] = gen_outputs.sequences
                        batch['decoder_attention_mask'] = torch.ones_like(gen_outputs.sequences)
                        batch['labels'] = collator.get_labels_from_decoder_input_ids(gen_outputs.sequences)

                    # process scores
                    scores = torch.stack(gen_outputs.scores, 1)  # (bs, seq_len - 1, vocab_size) without bos
                    scores = torch.cat([scores, torch.zeros(scores.size(0), 1, scores.size(2)).to(scores)], 1)  # (bs, seq_len, vocab_size) with bos at the end
                    gen_outputs.scores = scores

                    # process retrieval sequences
                    if gen_outputs.retrieval_sequences is not None:
                        retrieval_sequences = list(filter(lambda x: x is not None, gen_outputs.retrieval_sequences))
                        retrieval_sequences = np.stack(retrieval_sequences, 1) if len(retrieval_sequences) else None  # (bs, <=seq_len, n_ctxs)
                        gen_outputs.retrieval_sequences = retrieval_sequences

            eval_outputs = None
            if 'rerank' in training_args.do_eval_special:  # compute retrieval attention for reranking
                with torch.no_grad():
                    eval_outputs = model(**batch)
            elif 'perplexity' in training_args.do_eval_special:  # compute perplexity
                if 'generate' not in training_args.do_eval_special:
                    with torch.no_grad():
                        eval_outputs = model(**batch)
            elif 'gradient' in training_args.do_eval_special or 'gradient-batch' in training_args.do_eval_special:  # bp
                eval_outputs = model(**batch, output_embeddings=True)  # TODO: use deepspeed
            elif 'generate' == training_args.do_eval_special:
                pass
            else:
                raise NotImplementedError

            def rerank(
                eval_outputs,
                gen_outputs,
                use_gold: bool = False
            ):
                # gather
                ctx_pred_scores = eval_outputs.ctx_gold_scores if use_gold else eval_outputs.ctx_pred_scores
                ctx_pred_scores = trainer._pad_across_processes(ctx_pred_scores, pad_index=-1e10)  # (batch_size, n_ctx, n_used_layers, n_used_heads)
                ctx_pred_scores = trainer._nested_gather(ctx_pred_scores)  # (gathered_batch_size, n_ctx, n_used_layers, n_used_heads)
                # rank
                ranks = torch.sort(ctx_pred_scores, descending=True, dim=1).indices  # (gathered_batch_size, n_ctx, n_used_layers, n_used_heads)
                top1_accs = ranks[:, 0, ...].eq(0).float()  # (gathered_batch_size, n_used_layers, n_used_heads)
                return top1_accs

            def perplexity(
                eval_outputs,
                gen_outputs,
                evaluate_retrieval: Dict = {'percentiles': [0.25, 0.5, 0.75, 1.0]},
                output_all: bool = True,
            ):
                qids = batch['idxs']
                labels = batch['labels']
                labels[labels == -100] = tokenizer.pad_token_id

                predictions = gen_outputs.sequences
                labels_of_pred = collator.get_labels_from_decoder_input_ids(predictions)
                seqlen = labels_of_pred.ne(-100).sum(-1)  # (batch_size,)

                # decode labels and predictions
                # skip generation prefix
                skip_prefix = data_args.generation_prefix_len > 0
                if skip_prefix:
                    predictions = predictions[:, 1 + data_args.generation_prefix_len:]  # skip bos + prefix
                    labels = labels[:, data_args.generation_prefix_len:]  # skip prefix
                predictions: List[str] = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                labels: List[str] = tokenizer.batch_decode(labels, skip_special_tokens=True)
                # remove answer prefix
                if skip_prefix:
                    predictions = [x.strip() for x in predictions]
                    labels = [x.strip() for x in labels]
                else:
                    predictions = [x.strip().lstrip(data_args.answer_prefix).strip() for x in predictions]
                    labels = [x.strip().lstrip(data_args.answer_prefix).strip() for x in labels]

                # compute log prob
                logits = gen_outputs.scores
                loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
                lp = -loss_fct(logits.view(-1, logits.size(-1)), labels_of_pred.view(-1)).view(*labels_of_pred.size())  # (batch_size, seq_len)
                slp = lp.sum(-1)  # (batch_size,)

                # compute retrieval accuracy
                dids = accs = None
                if evaluate_retrieval and gen_outputs.retrieval_sequences is not None:
                    dids = gen_outputs.retrieval_sequences  # (bs, seq_len, n_ctxs)
                    accs: List[List[float]] = []
                    assert len(qids) == len(dids) == len(seqlen)
                    for qid, _dids, sl in zip(qids, dids, seqlen):
                        accs.append([])
                        rel_dids: List[str] = np.array([d for d, r in qrels[qid].items() if r])
                        rels = np.isin(_dids, rel_dids).any(-1)[:sl.item()]  # (rel_seq_len)
                        for pt in evaluate_retrieval['percentiles']:
                            pt = int(len(rels) * pt)
                            acc = rels[:pt].mean()
                            accs[-1].append(acc)
                    accs = torch.tensor(accs).to(logits.device)  # (bs, n_percentiles)
                    accs = trainer._nested_gather(trainer._pad_across_processes(accs, pad_index=0))  # (gathered_batch_size, n_percentiles)
                    if output_all:
                        # (gathered_batch_size, seq_len, n_ctxs)
                        dids = trainer._nested_gather(trainer._pad_across_processes(torch.tensor(dids.astype(np.int32)).to(labels_of_pred.device), pad_index=-1))

                # aggregate
                # (gathered_batch_size,)
                slp = trainer._nested_gather(trainer._pad_across_processes(slp, pad_index=0))
                # (gathered_batch_size,)
                seqlen = trainer._nested_gather(trainer._pad_across_processes(seqlen, pad_index=0))

                if output_all:
                    # (gathered_batch_size,)
                    qids = trainer._nested_gather(trainer._pad_across_processes(torch.tensor(qids.astype(np.int32)).to(labels_of_pred.device), pad_index=-1))
                    # (gathered_batch_size, seq_len)
                    lp = trainer._nested_gather(trainer._pad_across_processes(lp, pad_index=0))
                    # (gathered_batch_size, seq_len)
                    labels_of_pred = trainer._nested_gather(trainer._pad_across_processes(labels_of_pred, pad_index=-100))
                else:
                    qids = lp = labels_of_pred = None

                return slp, seqlen, accs, qids, lp, labels_of_pred, dids, predictions, labels

            def generate(
                eval_outputs,
                gen_outputs,
            ):
                predictions = gen_outputs.sequences
                labels = batch['labels']

                labels[labels == -100] = tokenizer.pad_token_id

                # skip generation prefix
                skip_prefix = data_args.generation_prefix_len > 0
                if skip_prefix:
                    predictions = predictions[:, 1 + data_args.generation_prefix_len:]  # skip bos + prefix
                    labels = labels[:, data_args.generation_prefix_len:]  # skip prefix

                predictions: List[str] = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                labels: List[str] = tokenizer.batch_decode(labels, skip_special_tokens=True)

                # remove answer prefix
                if skip_prefix:
                    predictions = [x.strip() for x in predictions]
                    labels = [x.strip() for x in labels]
                else:
                    predictions = [x.strip().lstrip(data_args.answer_prefix).strip() for x in predictions]
                    labels = [x.strip().lstrip(data_args.answer_prefix).strip() for x in labels]

                return predictions, labels

            def gradient(
                eval_outputs,
                gen_outputs,
                agg: str = 'mean',
                remove_ctx_dim: bool = False
            ):
                eval_outputs.loss.backward()
                #trainer.deepspeed.backward(eval_outputs.loss)
                ctx_embeddings = eval_outputs.ctx_embeddings  # (bs, n_ctxs, ctx_seq_length, dim)
                # compute grad norm
                grad = torch.linalg.norm(ctx_embeddings.grad, dim=-1)  # (bs, n_ctxs, ctx_seq_length)
                if agg == 'mean':
                    grad = grad.mean(-1)  # (bs, n_ctxs)
                elif agg == 'max':
                    grad = grad.max(-1).values  # (bs, n_ctxs)
                elif agg == 'min':
                    grad = grad.min(-1).values  # (bs, n_ctxs)
                else:
                    raise NotImplementedError
                if remove_ctx_dim:
                    grad = grad[:, 0]  # (bs, n_ctxs)
                grad = trainer._pad_across_processes(grad, pad_index=-1)  # (bs, n_ctxs) or (bs)
                grad = trainer._nested_gather(grad)  # (gathered_batch_size, n_ctxs) or (gathered_batch_size)
                if not remove_ctx_dim:
                    ranks = torch.sort(grad, descending=True, dim=1).indices  # (gathered_batch_size, n_ctxs)
                    top1_accs = ranks[:, 0].eq(0).float()  # (gathered_batch_size)
                    return top1_accs
                return grad

            if 'rerank' in training_args.do_eval_special:
                finals.append(rerank(eval_outputs, gen_outputs))
            elif 'perplexity' in training_args.do_eval_special:
                finals.append(perplexity(eval_outputs, gen_outputs))
            elif 'gradient' in training_args.do_eval_special:
                finals.append(gradient(eval_outputs, gen_outputs, agg='mean', remove_ctx_dim=True))
            elif 'gradient-batch' in training_args.do_eval_special:
                finals.append(gradient(eval_outputs, gen_outputs, agg='mean', remove_ctx_dim=False))
            elif 'generate' in training_args.do_eval_special:
                finals.append(generate(eval_outputs, gen_outputs))
            else:
                raise NotImplementedError

        if trainer.is_world_process_zero():
            if 'rerank' in training_args.do_eval_special:
                finals = torch.cat(finals, 0)  # (num_examples, n_used_layers, n_used_heads)
                acc = finals.mean(0)  # (n_used_layers, n_used_heads)
                print(f'#examples {len(finals)}, accuracy')
                for layer in acc:
                    print('\t'.join(map(str, layer.tolist())))
            elif 'gradient-batch' in training_args.do_eval_special:
                finals = torch.cat(finals, 0)  # (num_examples)
                acc = finals.mean(0).item()
                print(f'#examples {len(finals)}, accuracy {acc}')
            elif 'generate_perplexity' in training_args.do_eval_special:
                slp, seqlen, accuracy, qids, lp, labels_of_pred, dids, predictions, labels = zip(*finals)
                slp = torch.cat(slp)[:len(validation_dataset)]
                seqlen = torch.cat(seqlen)[:len(validation_dataset)]
                accuracy = torch.cat(accuracy, 0)[:len(validation_dataset)].mean(0) if accuracy[0] is not None else None
                if qids[0] is not None:
                    torch.save({'labels': labels_of_pred, 'logprobs': lp, 'docids': dids, 'qids': qids}, os.path.join(training_args.output_dir, 'logprob_label.pt'))
                print(f'#examples {len(slp)}, #tokens {seqlen.sum().item()}, perplexity: {torch.exp(-slp.sum() / seqlen.sum()).item()}, retrieval accuracy: {accuracy}')
                predictions, labels = sum(predictions, [])[:len(validation_dataset)], sum(labels, [])[:len(validation_dataset)]
                n_chars_pred = np.mean([len(p) for p in predictions])
                n_chars_labels = np.mean([len(l) for l in labels])
                metric_func = evaluate.load('rouge')
                metrics = metric_func.compute(predictions=predictions, references=labels)
                print(f'#examples {len(validation_dataset)}, metric:')
                print('\t'.join(metrics.keys()) + '\t#chars_pred\t#chars_gold')
                print('\t'.join(map(str, metrics.values())) + f'\t{n_chars_pred}\t{n_chars_labels}')
                with open(os.path.join(training_args.output_dir, training_args.output_file), 'w') as fout:
                    for example, pred, label in zip(validation_dataset, predictions, labels):
                        example['output'] = pred
                        example['gold'] = label
                        fout.write(json.dumps(example) + '\n')
            elif 'perplexity' in training_args.do_eval_special or 'gradient' in training_args.do_eval_special:
                qids = [example['qid'] for example in validation_dataset]
                scores = torch.cat(finals, 0)[:len(qids)].tolist()  # (num_examples,)
                assert len(qids) == len(scores)
                qid2scores: Dict[str, List[float]] = defaultdict(list)
                for qid, score in zip(qids, scores):
                    qid2scores[qid].append(score)
                acc = np.mean([np.argmax(scores) == 0 for qid, scores in qid2scores.items()])
                print(f'#exampels {len(qid2scores)}, acc {acc}')
            elif 'generate' in training_args.do_eval_special:
                predictions, labels = zip(*finals)
                predictions, labels = sum(predictions, [])[:len(validation_dataset)], sum(labels, [])[:len(validation_dataset)]
                n_chars_pred = np.mean([len(p) for p in predictions])
                n_chars_labels = np.mean([len(l) for l in labels])
                metric_func = evaluate.load('rouge')
                metrics = metric_func.compute(predictions=predictions, references=labels)
                with open(os.path.join(training_args.output_dir, training_args.output_file), 'w') as fout:
                    for example, pred, label in zip(validation_dataset, predictions, labels):
                        example['output'] = pred
                        example['gold'] = label
                        fout.write(json.dumps(example) + '\n')
                print(f'#examples {len(validation_dataset)}, metric:')
                print('\t'.join(metrics.keys()) + '\t#chars_pred\t#chars_gold')
                print('\t'.join(map(str, metrics.values())) + f'\t{n_chars_pred}\t{n_chars_labels}')
            else:
                raise NotImplementedError

        model.cache.dump()
        exit()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(**gen_kwargs, metric_key_prefix='eval')
        print(f'#examples {len(validation_dataset)}, metric:')
        print('\t'.join(metrics.keys()))
        print('\t'.join(map(str, metrics.values())))

    if training_args.do_predict:
        logger.info("*** Predict ***")
        num_beams = training_args.generation_num_beams
        predict_data = format_data_dict(_load_data_file(data_args.predict_file))

        generations = []
        for i in trange(0, len(predict_data), training_args.eval_batch_size):
            batch = collator(predict_data[i: i+training_args.eval_batch_size])
            with torch.no_grad():
                if training_args.fp16:
                    with torch.cuda.amp.autocast():
                        # TODO: decoder_start_token_id
                        outputs = model.generate(
                            **batch, num_beams=num_beams, max_length=data_args.max_answer_len)
                else:
                    outputs = model.generate(
                        **batch, num_beams=num_beams, max_length=data_args.max_answer_len)
            generations.append(outputs)

        if trainer.is_world_process_zero():
            predictions = []
            for gg in generations:
                predictions.extend(
                    tokenizer.batch_decode(
                        gg, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(predictions))

if __name__ == "__main__":
    main()
