#!/usr/bin/env python3
"""Train Teacher model with real CoT reasoning from QwQ-32B traces.

Based on train_beauty_RA.py but uses think_content column from
convert_traces_to_cot_parquet.py instead of the fixed template.

The assistant content format:
    🧐
    [extract_evidence]
    ...real reasoning from QwQ-32B...
    [model_preferences]
    ...real reasoning...
    [infer_intent]
    ...real reasoning...
    [recommend]
    ...real reasoning...

    <|sid_begin|>...<|sid_end|>

Usage (8-GPU DeepSpeed ZeRO-2):
    cd R3/pipeline/latent_opd

    python train_beauty_RA_cot.py \
        --model_name_or_path /pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/checkpoints/beauty_RA/epoch_2 \
        --data_path /pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/data/beauty/training_RA_cot_train.parquet \
        --output_dir /pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/checkpoints/beauty_RA_cot/epoch_3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-6 \
        --num_train_epochs 3 \
        --warmup_ratio 0.05 \
        --bf16 \
        --deepspeed ds_config_zero2.json \
        --logging_steps 10 \
        --save_strategy epoch \
        --eval_strategy epoch \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --greater_is_better false
"""

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer,
    TrainingArguments, HfArgumentParser,
)
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/checkpoints/beauty_RA/epoch_2",
        metadata={"help": "Path to pretrained model (RA baseline)"},
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to CoT training parquet (with think_content column)"},
    )
    val_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation data (auto-inferred if not set)"},
    )
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA"})
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )


SYSTEM_MESSAGE = (
    "You are a professional recommendation expert who needs to recommend the next possible "
    "purchase for users based on their purchase history. Please predict the most likely next "
    "product that the user will purchase based on the user's historical purchase information."
)

THINK_START = "🧐"
THINK_END = "\n\n"


def prepare_chat_dataset(data_path, sample_size=None, local_rank=0):
    if local_rank == 0:
        print(f"Loading parquet file: {data_path}")
    data_pq = pd.read_parquet(data_path)
    if local_rank == 0:
        print(f"Data shape: {data_pq.shape}")
        print(f"Columns: {list(data_pq.columns)}")

    if sample_size is not None and len(data_pq) > sample_size:
        if local_rank == 0:
            print(f"Sampling {sample_size} from {len(data_pq)}")
        data_pq = data_pq.head(sample_size)

    texts = []

    for _, row in data_pq.iterrows():
        think_content = row.get('think_content', '')

        if think_content and isinstance(think_content, str) and think_content.strip():
            assistant_content = f"{THINK_START}\n{think_content}\n{THINK_END}{row['groundtruth']}"
        elif 'title' in row.keys() and row['title'] is not None:
            categories = row['categories']
            assistant_content = (
                f"{THINK_START}\n"
                f"Based on the user's purchase history, the user has shown interest in {categories}. "
                f"Considering the patterns in their preferences, the next likely item would be in a related category.\n"
                f"{THINK_END}{row['groundtruth']}"
            )
        else:
            assistant_content = (
                f"{THINK_START}\n"
                f"The user is likely to buy items in a related category based on their purchase history.\n"
                f"{THINK_END}{row['groundtruth']}"
            )

        formatted_text = (
            f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n{row['description']}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_content}<|im_end|>\n"
        )
        texts.append(formatted_text)

    if local_rank == 0:
        print(f"Total texts: {len(texts)}")
        has_cot = sum(1 for t in texts if '[extract_evidence]' in t)
        print(f"  With real CoT: {has_cot}, Fallback template: {len(texts) - has_cot}")
        for i, text in enumerate(texts[:2]):
            print(f"\n  [{i}] Length: {len(text)} chars")
            print(f"  [{i}] Preview: {text[:800]}...")

    return Dataset.from_dict({'text': texts})


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['text'],
        padding='longest',
        truncation=True,
        max_length=4096,
        add_special_tokens=True,
        return_attention_mask=True,
    )


def get_special_tokens():
    special_tokens = ['<|sid_begin|>', '<|sid_end|>']
    for prefix in ['s_a', 's_b', 's_c', 's_d']:
        for i in range(256):
            special_tokens.append(f'<{prefix}_{i}>')
    return special_tokens


class CustomDataCollator:
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        max_length = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        labels = []

        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_length - len(ids)
            padded_ids = [self.tokenizer.pad_token_id] * padding_length + ids
            padded_mask = [0] * padding_length + mask

            label = padded_ids.copy()
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            user_start_pos = text.find("<|im_start|>user")

            if user_start_pos != -1:
                user_start_tokens = self.tokenizer.encode(
                    "<|im_start|>user", add_special_tokens=False
                )
                for j in range(len(ids) - len(user_start_tokens) + 1):
                    if ids[j:j + len(user_start_tokens)] == user_start_tokens:
                        pad_and_prefix = padding_length + j
                        for k in range(pad_and_prefix):
                            label[k] = -100
                        break
                else:
                    for k in range(len(label)):
                        label[k] = -100
            else:
                for k in range(len(label)):
                    label[k] = -100

            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            labels.append(label)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names = ["labels"]

    if training_args.local_rank == 0:
        print(f"Model: {model_args.model_name_or_path}")
        print(f"Data: {model_args.data_path}")
        print(f"Output: {training_args.output_dir}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    special_tokens = get_special_tokens()
    tokenized_special_tokens = tokenizer.convert_tokens_to_ids(special_tokens)
    valid_special_token_ids = [
        tid for tid in tokenized_special_tokens
        if tid != tokenizer.unk_token_id
    ]

    if training_args.local_rank == 0:
        print(f"Valid special tokens: {len(valid_special_token_ids)}")

    if model_args.use_lora:
        target_modules = model_args.lora_target_modules.split(",")
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            trainable_token_indices={
                'embed_tokens': valid_special_token_ids,
                'lm_head': valid_special_token_ids,
            },
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        if training_args.local_rank == 0:
            print("Full parameter training")

    train_dataset = prepare_chat_dataset(
        model_args.data_path, local_rank=training_args.local_rank
    )

    val_data_path = model_args.val_data_path
    if not val_data_path:
        val_data_path = str(
            Path(model_args.data_path).parent / 'training_RA_cot_val.parquet'
        )
        if not Path(val_data_path).exists():
            for ancestor in Path(model_args.data_path).parents:
                candidate = ancestor / 'training_RA_cot_val.parquet'
                if candidate.exists():
                    val_data_path = str(candidate)
                    break
            if not Path(val_data_path).exists():
                val_data_path = str(
                    Path(model_args.data_path).parent / 'training_RA_val.parquet'
                )
    if training_args.local_rank == 0:
        print(f"Val data: {val_data_path}")
    val_dataset = prepare_chat_dataset(val_data_path, local_rank=training_args.local_rank)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val",
    )

    data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.local_rank == 0:
        print(f"Model saved to {training_args.output_dir}")
