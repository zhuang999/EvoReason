from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src' / 'latent'))
from layers import LatentModel
from onerec.modeling import latent_model_class
from onerec import OneRecLatentDataset, ensure_thought_token, set_reproducible_seed, write_run_manifest

def parse_args():
    parser = argparse.ArgumentParser(description='LatentR3 SFT on canonical OneRec SID data')
    parser.add_argument('--base-model', required=True, help='Shared expanded Qwen base or matched SID checkpoint')
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--eval-data', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--micro-batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=float, default=3)
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--max-sequence-length', type=int, default=512)
    parser.add_argument('--end-k', type=int, default=-1)
    parser.add_argument('--prompt-protocol', choices=['onerec', 'legacy-r3'], default='onerec')
    parser.add_argument('--freeze-backbone', action='store_true', help='Preserve the SID recommender and train only LatentRATT.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume-from-checkpoint')
    parser.add_argument('--bf16', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    world_size = int(__import__('os').environ.get('WORLD_SIZE', '1'))
    denominator = args.micro_batch_size * world_size
    if args.batch_size % denominator:
        raise ValueError('batch-size must be divisible by micro-batch-size * world-size')
    gradient_accumulation = args.batch_size // denominator
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    thought_id = ensure_thought_token(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    dtype = torch.bfloat16 if args.bf16 else None
    model_class = latent_model_class(args.base_model, LatentModel)
    model = model_class.from_pretrained(args.base_model, torch_dtype=dtype)
    model.resize_token_embeddings(len(tokenizer))
    model.attention.end_k = args.end_k
    model.config.thought_token_id = thought_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.attention.parameters():
            parameter.requires_grad = True
    trainable = sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    if trainable == 0:
        raise ValueError('SFT has no trainable parameters')
    print(f'Trainable parameters: {trainable}')
    dataset_args = {'tokenizer': tokenizer, 'train': True, 'max_sequence_length': args.max_sequence_length, 'prompt_protocol': args.prompt_protocol, 'limit': args.limit}
    train_dataset = OneRecLatentDataset(args.train_data, **dataset_args)
    eval_dataset = OneRecLatentDataset(args.eval_data, **dataset_args)
    training_args = TrainingArguments(output_dir=args.output_dir, per_device_train_batch_size=args.micro_batch_size, per_device_eval_batch_size=args.micro_batch_size, gradient_accumulation_steps=gradient_accumulation, num_train_epochs=args.epochs, learning_rate=args.learning_rate, warmup_steps=20, optim='adamw_torch', bf16=args.bf16, logging_steps=10, eval_strategy='epoch', save_strategy='epoch', save_total_limit=3, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, ddp_find_unused_parameters=False if world_size > 1 else None, report_to='none', remove_unused_columns=False, seed=args.seed, data_seed=args.seed)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True))
    write_run_manifest(args.output_dir, stage='latentr3_onerec_sft', args=args, extra={'protocol': 'onerec-sid-matched', 'paper': 'arXiv:2505.19092', 'train_data_sha256': train_dataset.source_sha256, 'eval_data_sha256': eval_dataset.source_sha256, 'trainable_parameters': trainable, 'backbone_frozen': args.freeze_backbone})
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
if __name__ == '__main__':
    main()
