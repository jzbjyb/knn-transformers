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
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from tqdm import tqdm

import numpy as np
import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from tqdm import trange

import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    DataCollatorWithPadding,
    is_torch_tpu_available,
    set_seed,
)
from qa_trainer import QuestionAnsweringSeq2SeqTrainer
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import ShardSampler
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction

from deepspeed import checkpointing as ds_checkpointing
from models.fusion_t5 import FusionT5Config, FusionT5ForConditionalGeneration

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

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


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
    max_question_len: Optional[int] = field(default=128)
    max_context_len: Optional[int] = field(default=128)
    max_answer_len: Optional[int] = field(default=128)
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
class DataCollatorForFusion:

    model: transformers.AutoModelForSeq2SeqLM
    tokenizer: transformers.PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = None
    use_context: bool = True
    encoder_input_for_context: str = None  # the input to encoder that contexts attend to (through cross-attention)
    max_question_len: int = None
    max_context_len: int = None
    max_answer_len: int = None
    answer_prefix: str = None
    _train: bool = True

    def eval(self):
        self._train = False

    def train(self):
        self._train = True
    
    def get_real_decoder_start_token_id(self):
        if self.use_context:  # use the first token of answer prefix to start generation
            return self.tokenizer.encode(self.answer_prefix)[0]
        return self.model.config.decoder_start_token_id

    def __call__(
        self, 
        examples: List[Dict], 
        debug: True = False):
        decoder_start_token = self.tokenizer.convert_ids_to_tokens([self.model.config.decoder_start_token_id])[0]

        # questions
        questions = self.tokenizer(
            [e['question'] for e in examples],
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
        # TODO: problem with multiple processes?
        # put ctx on the right to be close to the answer
        self.tokenizer.padding_side = 'left'
        # add decoder_start_token to ctx
        dst = f'{decoder_start_token} ' if self.use_context else ''
        ctxs = self.tokenizer(
            [dst + ctx for e in examples for ctx in e['ctxs']],
            truncation=True,
            padding=True,
            max_length=self.max_context_len,
            add_special_tokens=False,  # avoid eos
            return_tensors='pt')
        self.tokenizer.padding_side = 'right'
        decoder_ctx_input_ids = ctxs.input_ids.view(bs, n_ctxs, -1)  # (batch_size, n_ctxs, ctx_seq_length)
        decoder_ctx_attention_mask = ctxs.attention_mask.view(bs, n_ctxs, -1)  # (batch_size, n_ctxs, ctx_seq_length)
        
        # answers
        dst = f'{decoder_start_token} ' if not self.use_context else ''
        answers = self.tokenizer(
            [dst + random.choice(e['answers']) for e in examples],
            truncation=True,
            padding=True,
            max_length=self.max_answer_len,
            return_tensors='pt'
        )
        decoder_input_ids = answers.input_ids  # (batch_size, seq_length)
        decoder_attention_mask = answers.attention_mask  # (batch_size, seq_length)
        # convert answers to labels
        assert self.tokenizer.pad_token_id == 0
        # remove the first token of answers (treated as generation trigger token) to make it labels
        labels = torch.zeros_like(decoder_input_ids)
        labels[..., :-1] = decoder_input_ids[..., 1:].clone()
        labels.masked_fill_(labels == 0, -100)

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels
        }
        if self.use_context:
            batch['decoder_ctx_input_ids'] = decoder_ctx_input_ids
            batch['decoder_ctx_attention_mask'] = decoder_ctx_attention_mask
        if self.encoder_input_for_context and self._train:  # TODO: add inference support
            batch['input_ids_for_ctx'] = input_ids_for_ctx
            batch['attention_mask_for_ctx'] = attention_mask_for_ctx
        
        if debug:
            print(batch)
            print(decoder_ctx_input_ids[0, 0])
            print(decoder_ctx_attention_mask[0, 0])
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

@dataclass
class TrainingArgs(Seq2SeqTrainingArguments):
    extract_attention: bool = field(default=False)
    do_eval_rerank: bool = field(default=False)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
        config_specific_kwargs = {'ctx_attention_loss': FusionT5Config.parse_ctx_attention_loss(model_args.ctx_attention_loss)}
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
                question = data_args.question_prefix + example['question']
                ctxs = [data_args.context_prefix + ctx['text'] for ctx in example['ctxs'][:data_args.depth]]
                answers = [data_args.answer_prefix + ans for ans in example['answers']]
                formatted.append({'question': question, 'ctxs': ctxs, 'answers': answers})
            return formatted
        def compute_metrics(output):
            predictions=output.predictions
            references=[example['answers'][0] for example in output.label_ids]  # TODO: use all answers in metric
            metric_func = evaluate.load('rouge')
            metrics = metric_func.compute(predictions=predictions, references=references)
            return metrics
        def post_processing_function(references, features, outputs, stage='eval', debug=False):
            decoded_preds = tokenizer.batch_decode(outputs.predictions, skip_special_tokens=True)
            formatted_predictions = [x.strip().lstrip(data_args.answer_prefix).strip() for x in decoded_preds]
            if debug:
                print(outputs.predictions[:2])
                print(decoded_preds[:2])
                print([example['answers'][0] for example in references[:2]])
                input()
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)
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
        answer_prefix=data_args.answer_prefix,
        use_context=data_args.use_context,
        encoder_input_for_context=data_args.encoder_input_for_context)

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
    
    if training_args.do_eval and training_args.do_eval_rerank:
        logger.info("*** Evaluate reranking ***")
        model.eval()
        accuracies = []

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

        for step, batch in tqdm(enumerate(dataloader)):
            batch = trainer._prepare_inputs(batch)
            with torch.no_grad():
                outputs = model(**batch)
            # gather
            ctx_pred_dist = outputs.ctx_pred_dist
            ctx_pred_dist = trainer._pad_across_processes(ctx_pred_dist)
            ctx_pred_dist = trainer._nested_gather(ctx_pred_dist)
            
            ranks = torch.sort(ctx_pred_dist, descending=True, dim=-1).indices
            top1_accs = ranks[:, 0].eq(0).float()
            accuracies.append(top1_accs)

        if trainer.is_world_process_zero():
            acc = torch.cat(accuracies).mean()
            print(f'#examples {len(accuracies)}, accuracy: {acc}')
        exit()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=data_args.max_answer_len, 
            num_beams=1,
            metric_key_prefix='eval')

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
