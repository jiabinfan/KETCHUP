#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
import datetime
custom_transformers_path = os.path.join(current_dir, 'transformers')
sys.path.insert(0, custom_transformers_path)
import logging
import sys
import math
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import datasets
import numpy as np
import evaluate
import nltk
nltk.download('punkt_tab')
from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.bleu import Bleu
from utils.accuracy import Acc
import transformers
import logging
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    get_scheduler,
)
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import random

from utils.helper import n_step_reward_shift_upgrade, calc_returns, shift_left, shift_right

from Mymodel import Net
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.43.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

# Set the global logging level to WARNING
logging.basicConfig(level=logging.WARNING)

# Set the logging level for configuration_utils to ERROR
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_type: Optional[str] = field(
        default=None, metadata={"help": "training loss, sft, bso, semi-bso"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    context_column: Optional[str] = field(
        default="context",
        metadata={"help": "The name of the column in the datasets containing the contexts (for question answering)."},
    )
    question_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    answer_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "Answer or Summarization"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=100,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_answer_length: int = field(
        default=100,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )
    val_max_answer_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_answer_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when"
                " batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU)."
            )
        },
    )
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "The threshold used to select the null answer: if the best answer has a score that is less than "
                "the score of the null answer minus this threshold, the null answer is selected for this example. "
                "Only useful when `version_2_with_negative=True`."
            )
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    with_tracking: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable experiment trackers for logging."
        },
    )
    max_train_steps: int = field(
        default=None,
        metadata={"help": "Total number of training steps to perform. If provided, overrides num_train_epochs."
        },
    )
    num_warmup_steps: int = field(
        default=0,
        metadata={
            "help": "Number of steps for the warmup in the lr scheduler."
        },
    )
    checkpointing_steps: str = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
        },
    )
    n_step: int = field(default=1, metadata={"help": "n-step reward"})
    denom: int = field(default=1, metadata={"help": "denominator for n-step reward"})
    entropy_coef: float = field(default=0.001, metadata={"help": "entropy coefficient"})
    reward_clip: int = field(default=50, metadata={"help": "reward clipping"})
    softmax: bool = field(default=False, metadata={"help": "use softmax for next state value"})
    step_per_update: int = field(default=5, metadata={"help": "step per update"})
    teacher_ratio: float = field(default=0.8, metadata={"help": "teacher ratio"})
    
    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_answer_length is None:
            self.val_max_answer_length = self.max_answer_length


question_answering_column_name_mapping = {
    "xsum": ("document", "summary")
}

def reward_fn(reward_model, input_ids, output_ids, args, pad_token_id=None, n_step = 1):
    with torch.no_grad():
        pad_token_id = pad_token_id or reward_model.config.pad_token_id
        reward_model.eval()
        input_ids = input_ids.repeat(1, 1)
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': input_ids != pad_token_id
        }
        start_tokens = torch.zeros((output_ids.shape[0], 1)).to(output_ids)
        start_tokens.fill_(reward_model.config.decoder_start_token_id)
        
        model_inputs['decoder_input_ids'] = torch.cat([start_tokens, output_ids[..., :-1]], dim=-1)
        model_inputs['decoder_attention_mask'] = output_ids != pad_token_id
        model_inputs['use_cache'] = False
        with torch.no_grad():
            outputs = reward_model(**model_inputs)
        
        logits = outputs.logits
        logits = logits - torch.mean(logits, dim=-1, keepdim=True)
        selection_value = torch.gather(logits, -1, output_ids[..., None]).squeeze(-1)
        selection_value.masked_fill_(output_ids == pad_token_id, 0.0)
        next_logits = torch.roll(logits, -n_step, 1)
        if args.softmax:
            next_state_value = torch.sum(torch.softmax(next_logits, dim=-1) * next_logits, dim=-1)
        else:
            next_state_value = next_logits.max(dim=-1)[0]
        next_state_value.masked_fill_(output_ids == pad_token_id, 0.0)
        next_state_value.masked_fill_(output_ids == reward_model.config.eos_token_id, 0.0)

        reward = selection_value - next_state_value
            
        return reward, selection_value, next_state_value

def main():
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("Model Arguments:", model_args)
    print("Data Training Arguments:", data_args)
    # print("Training Arguments:", training_args)
    # sys.exit()
    
    #BATCH PER DEVICE
    # training_args.per_device_train_batch_size = 32
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_seq2seq", model_args, data_args)
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator_log_kwargs = {}

    if data_args.with_tracking:
        accelerator_log_kwargs["log_with"] = training_args.report_to
        accelerator_log_kwargs["project_dir"] = training_args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_error()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_warning() # Or set to ERROR

    log_level = training_args.get_process_log_level()

    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
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
    
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    # config.dropout_rate = 0.0                # General dropout used in various parts
    # config.layer_norm_epsilon = 1e-12         # Epsilon to avoid division by zero in layer normalization
    # config.feed_forward_proj_dropout_rate = 0.0  # Dropout for the feed-forward network
    # config.attention_dropout_rate = 0.0   
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    #seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained("/mnt/nvme/jiabin/bso_accelerate/outputs2", use_safetensors=True)

    #teacher directory t0 for now
    teacher = AutoModelForSeq2SeqLM.from_pretrained('/mnt/nvme/guoqing/flan-t5-xl')
    teacher.requires_grad_(False)

    # teacher = Teacher()
    # student = Student()
    # model = Net(seq2seq_model, teacher=teacher, student=student)
    model = Net(seq2seq_model, teacher=teacher, args=training_args, 
                max_length=data_args.val_max_answer_length)

    #tokenizer = AutoTokenizer.from_pretrained("/mnt/nvme/jiabin/bso_accelerate/outputs")

    if seq2seq_model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Preprocessing the datasets.
    # We need to generate and tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = question_answering_column_name_mapping.get(data_args.dataset_name, None)
    if data_args.question_column is None:
        question_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        question_column = data_args.question_column
        if question_column not in column_names:
            raise ValueError(
                f"--question_column' value '{data_args.question_column}' needs to be one of: {', '.join(column_names)}"
            )
        
    if data_args.answer_column is None:
        answer_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        answer_column = data_args.answer_column
        if answer_column not in column_names:
            raise ValueError(
                f"--answer_column' value '{data_args.answer_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_answer_length for training.
    max_answer_length = data_args.max_answer_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(seq2seq_model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{seq2seq_model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"Seq2seq_model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.source_prefix.startswith('summarize') or data_args.source_prefix.startswith('Translate'):
        prefix = data_args.source_prefix
    else: #gsm8k
        prefix=open('data/gsm8k/few-shot-examples.txt','r').read().strip()
        prefix=prefix+'\n\n'

    def preprocess_batch(
        examples,
        question_column: str,
        answer_column: str,
    ) -> Tuple[List[str], List[str]]:
        questions = examples[question_column]
        # contexts = examples[context_column]
        answers = examples[answer_column]

        #prompt can be changed into different prompt
        def generate_input(_question):
            return "".join([prefix, _question.lstrip()])
        
        inputs = [generate_input(question) for question in questions]
        targets = [answer if len(answer) > 0 else "" for answer in answers]
        return inputs, targets

    def preprocess_function(examples):
        inputs, targets = preprocess_batch(examples, question_column, answer_column)

        model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)
        # Tokenize targets with text_target=...
        labels = tokenizer(text_target=targets, max_length=max_answer_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -0) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # Create train feature from dataset
        with accelerator.main_process_first():
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            # # Save the preprocessed dataset to a directory
            # train_dataset.save_to_disk("/home/dongheng/LLMR/accelerate/preprocessed_train_dataset")
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(len(eval_examples), data_args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        with accelerator.main_process_first():
            eval_dataset = eval_examples.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        with accelerator.main_process_first():
            predict_dataset = predict_examples.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Data collator
    label_pad_token_id = -0 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size)

    if training_args.do_predict:
        
        predict_dataloader = DataLoader(
            predict_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.seq2seq_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.seq2seq_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if data_args.max_train_steps is None:
        data_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=data_args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=data_args.max_train_steps
        if overrode_max_train_steps
        else data_args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        data_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(data_args.max_train_steps / num_update_steps_per_epoch)
    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = data_args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if data_args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(data_args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = training_args.lr_scheduler_type
            accelerator.init_trackers("run_seq2seq", experiment_config)
    
    #metric = Bleu()
    if 'xsum' in data_args.train_file:
        metric = evaluate.load("utils/rouge.py")
    elif 'wmt' in data_args.train_file:
        metric = Bleu()
    elif 'gsm8k' in data_args.train_file:
        # metric = evaluate.load("utils/rouge.py")
        metric = Acc()
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        # rougeLSum expects newline after each sentence
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != 0, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != 0, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    
    # Train!
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {data_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(int(data_args.max_train_steps)), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    best_score = 0
    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            checkpoint_path = training_args.resume_from_checkpoint
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * training_args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // training_args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    debug_str = "10"
    eval_metrics = {}

    for epoch in range(starting_epoch, training_args.num_train_epochs):
        
        if data_args.with_tracking:
            total_loss = 0
        if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        cum_loss = 0

        for step, batch in enumerate(active_dataloader):
            device = batch["input_ids"].device
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            model.eval()
            batch_indices = list(range(input_ids.shape[0])) # Get sample indices in the batch
            random.shuffle(batch_indices)

            # Split indices for student and teacher
            split_point = math.ceil(len(batch_indices) * data_args.teacher_ratio)  # For teacher:student ratio
            teacher_indices = batch_indices[:split_point]
            student_indices = batch_indices[split_point:]

            # Prepare input subsets
            student_input_ids = input_ids[student_indices]
            student_attention_mask = attention_mask[student_indices]

            teacher_input_ids = input_ids[teacher_indices]
            teacher_attention_mask = attention_mask[teacher_indices]

            if student_input_ids.shape[0]!=0:
                # Student generates outputs for its subset
                stu_output_ids = model.generate(
                    input_ids=student_input_ids,
                    attention_mask=student_attention_mask,
                    do_sample=False,
                    num_return_sequences=1,
                    max_length=512,
                )[:, 1:]

            # Teacher generates outputs for its subset
            teacher.eval()
            output_ids= teacher.generate(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                max_length=512,
                num_beams=1,
                length_penalty=1.0,
                do_sample=False,
                num_return_sequences=1,
            )[:,1:]
            
            max_output_length = max(output_ids.shape[-1], stu_output_ids.shape[-1])
            output_ids = F.pad(output_ids, (0, max_output_length - output_ids.shape[-1]), value=config.pad_token_id)
            stu_output_ids = F.pad(stu_output_ids, (0, max_output_length - stu_output_ids.shape[-1]), value=config.pad_token_id)

            # Concatenate padded sequences
            output_ids = torch.cat((output_ids, stu_output_ids), dim=0)
            input_ids = torch.cat((teacher_input_ids, student_input_ids), dim=0)
            attention_mask = torch.cat((teacher_attention_mask, student_attention_mask), dim=0)

            del stu_output_ids, teacher_input_ids, student_input_ids, teacher_attention_mask, student_attention_mask
            # input_ids= input_ids.repeat(1+1, 1)
            # #shuffle them
            # shuffled_indices = torch.randperm(input_ids.shape[0])
            # input_ids = input_ids[shuffled_indices]
            # output_ids = output_ids[shuffled_indices]
            
            rewards = []
            reward, selection_value, next_state_value=reward_fn(
                        reward_model=teacher,
                        input_ids=input_ids,
                        output_ids=output_ids,
                        args=data_args,
                        pad_token_id=config.pad_token_id,
                        n_step=data_args.n_step,
                    )
            rewards.append(reward)
            rewards = torch.stack(rewards, dim=0).mean(0)
            if data_args.n_step==1:
                returns = calc_returns(rewards)
            else:
                returns = n_step_reward_shift_upgrade(reward, 
                                                      selection_value, 
                                                      next_state_value, 
                                                      n_step=data_args.n_step)
            model.train()

            #net class return tuple (Seq2SeqLMOutput, loss)
            loss = model.reinforce(
                input_ids=input_ids,
                output_ids=output_ids,
                config=config,
                returns=returns,
                denom=data_args.denom,
                reward_clip=data_args.reward_clip,
                entropy_coef=data_args.entropy_coef,
            )

            device = loss.get_device()
            # We keep track of the loss at each epoch
            if data_args.with_tracking:
                total_loss += loss.detach().float()
                cum_loss += loss.detach().float()
            loss = loss / training_args.gradient_accumulation_steps

            accelerator.backward(loss)

            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if "0" in str(device) :
                    total_loss=cum_loss/training_args.gradient_accumulation_steps
                    print("total_loss:", loss.item())
                cum_loss = 0
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= data_args.max_train_steps:
                break
            if step!=0 and step % (training_args.gradient_accumulation_steps*data_args.step_per_update) == 0 or step == len(train_dataloader) - 1:
                # print(99999, step, len(active_dataloader))
                model.eval()
                gen_kwargs = {
                    "max_length": data_args.val_max_answer_length if data_args is not None else config.max_length,
                    "num_beams": data_args.num_beams,
                }

                samples_seen = 0

                gen_path=f"{training_args.output_dir}/steps_{str(completed_steps)}"
                if not os.path.exists(gen_path):
                    os.makedirs(gen_path)
                gen_file = open(f"{gen_path}.gen", "w")

                all_preds =[]
                all_indices=[]
                for step_e, batch in enumerate(eval_dataloader):
                    accelerator.wait_for_everyone()

                    with torch.no_grad():
                        try:
                            accelerator.wait_for_everyone()
                            generated_tokens = accelerator.unwrap_model(model).generate(
                                batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                **gen_kwargs,
                            )

                        except Exception as e:
                            logging.error(f"Error during generate: {e}")
                            logging.error(f"Failed gen_kwargs: {gen_kwargs}")
                            raise e

                        labels = batch["labels"]
                        if not data_args.pad_to_max_length:
                            accelerator.wait_for_everyone()
                            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                        if data_args.ignore_pad_token_for_loss:
                            labels = np.where(labels != 0, labels, tokenizer.pad_token_id)

                        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                        if accelerator.num_processes > 1:
                            if step_e == len(eval_dataloader):
                                decoded_preds = decoded_preds[: len(eval_dataloader.dataset) - samples_seen]
                                decoded_labels = decoded_labels[: len(eval_dataloader.dataset) - samples_seen]
                            else:
                                samples_seen += len(decoded_labels)

                        accelerator.wait_for_everyone()
                        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                        gen_file.write("\n".join(decoded_preds) + "\n")

                eval_metric = metric.compute()
                print("metric:", eval_metric)
                
                if 'xsum' in data_args.train_file:
                    use_metric='rouge1'
                elif 'wmt' in data_args.train_file:
                    use_metric='bleu'
                elif 'gsm8k' in data_args.train_file:
                    use_metric='acc'

                eval_metrics[f"steps_{str(completed_steps)}"]=eval_metric
                with open(training_args.output_dir+"/results.json", "w") as f:
                    json.dump(eval_metrics, f, indent=2)

                if data_args.with_tracking:
                    accelerator.log(
                        {
                            "metric": eval_metric,
                            "train_loss": total_loss.item() / len(train_dataloader),
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

                if training_args.output_dir is not None and eval_metric[use_metric] > best_score:
                    best_score = eval_metric[use_metric] 
                    # accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model).seq2seq_model
                    unwrapped_model.save_pretrained(
                        training_args.output_dir+"/steps_"+str(completed_steps), is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                    tokenizer.save_pretrained(
                        training_args.output_dir+"/steps_"+str(completed_steps), is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )

    if data_args.with_tracking:
        accelerator.end_training()
    
if __name__ == "__main__":
    main()
