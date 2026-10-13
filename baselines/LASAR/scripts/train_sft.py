from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from lasar import LasarCollator, LasarForCausalLM, OneRecLasarDataset, set_reproducible_seed, write_run_manifest
LATENT_TOKENS = ['<|lasar_start|>', '<|lasar_thought|>', '<|lasar_end|>']

class LasarTrainer(Trainer):

    def _save(self, output_dir: str | None=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        self.model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, Path(output_dir) / 'training_args.bin')

def parse_args():
    parser = argparse.ArgumentParser(description='LASAR Stage-2 semantic-aligned SFT')
    parser.add_argument('--model', required=True, help='OneRec Stage-1 grounded recommendation checkpoint')
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--eval-data', required=True)
    parser.add_argument('--train-anchors', required=True)
    parser.add_argument('--eval-anchors', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-steps', type=int, default=8)
    parser.add_argument('--align-weight', type=float, default=0.1)
    parser.add_argument('--policy-weight', type=float, default=0.1)
    parser.add_argument('--epochs', type=float, default=20)
    parser.add_argument('--learning-rate', type=float, default=5e-05)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=32)
    parser.add_argument('--deepspeed')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--eval-limit', type=int, help='Limit validation rows independently while retaining the full training set.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-sequence-length', type=int, default=512)
    parser.add_argument('--max-answer-length', type=int, default=32)
    parser.add_argument('--resume-from-checkpoint')
    parser.add_argument('--load-best-model-at-end', action=argparse.BooleanOptionalAction, default=True, help='Restore the checkpoint with the lowest validation loss before the final save.')
    parser.add_argument('--init-latent-specials', action='store_true', help='Initialize LASAR special-token embeddings from stable existing chat/eos tokens.')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze the causal LM backbone and train only LASAR heads/new modules.')
    parser.add_argument('--latent-adapter', action='store_true', help='Train an identity-initialized residual adapter on latent feedback.')
    parser.add_argument('--latent-adapter-scale', type=float, default=0.001)
    parser.add_argument('--trainable-markers', action='store_true')
    parser.add_argument('--lora-r', type=int, default=0, help='Enable paper-compatible LoRA Stage-2 when greater than zero.')
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--fixed-train-steps', type=int)
    parser.add_argument('--teacher-force-steps', action='store_true', help="Use CoT segment count for N instead of the paper's policy argmax")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.freeze_backbone and args.lora_r:
        raise ValueError('Use either --freeze-backbone or --lora-r, not both')
    set_reproducible_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({'additional_special_tokens': LATENT_TOKENS})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.bf16 else None
    backbone = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    backbone.resize_token_embeddings(len(tokenizer))
    backbone.config.pad_token_id = tokenizer.pad_token_id
    if args.init_latent_specials:
        embeddings = backbone.get_input_embeddings().weight
        with torch.no_grad():
            fallback_id = tokenizer.eos_token_id
            start_source = tokenizer.convert_tokens_to_ids('<|im_start|>')
            end_source = tokenizer.convert_tokens_to_ids('<|im_end|>')
            if not isinstance(start_source, int) or start_source < 0:
                start_source = fallback_id
            if not isinstance(end_source, int) or end_source < 0:
                end_source = fallback_id
            init_pairs = {LATENT_TOKENS[0]: start_source, LATENT_TOKENS[1]: fallback_id, LATENT_TOKENS[2]: end_source}
            for token, source_id in init_pairs.items():
                target_id = tokenizer.convert_tokens_to_ids(token)
                embeddings[target_id].copy_(embeddings[source_id])
    if args.freeze_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
    if args.lora_r:
        from peft import LoraConfig, TaskType, get_peft_model
        backbone = get_peft_model(backbone, LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias='none', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    model = LasarForCausalLM(backbone, latent_start_id=tokenizer.convert_tokens_to_ids(LATENT_TOKENS[0]), latent_thought_id=tokenizer.convert_tokens_to_ids(LATENT_TOKENS[1]), latent_end_id=tokenizer.convert_tokens_to_ids(LATENT_TOKENS[2]), max_latent_steps=args.max_steps, align_weight=args.align_weight, policy_weight=args.policy_weight, use_latent_adapter=args.latent_adapter, latent_adapter_scale=args.latent_adapter_scale, use_trainable_markers=args.trainable_markers)
    model.policy_head.requires_grad_(args.policy_weight != 0.0)
    model.latent_adapter.requires_grad_(args.latent_adapter)
    model.latent_start_embedding.requires_grad_(args.trainable_markers)
    model.latent_end_embedding.requires_grad_(args.trainable_markers)
    trainable_parameters = sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    dataset_args = {'max_sequence_length': args.max_sequence_length, 'max_answer_length': args.max_answer_length, 'max_latent_steps': args.max_steps, 'limit': args.limit}
    train_dataset = OneRecLasarDataset(args.train_data, tokenizer, args.train_anchors, **dataset_args)
    eval_dataset_args = {**dataset_args, 'limit': args.eval_limit or args.limit}
    eval_dataset = OneRecLasarDataset(args.eval_data, tokenizer, args.eval_anchors, **eval_dataset_args)
    collator = LasarCollator(tokenizer.pad_token_id, args.max_steps, teacher_force_steps=args.teacher_force_steps, fixed_steps=args.fixed_train_steps)
    training_args = TrainingArguments(output_dir=args.output_dir, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate, lr_scheduler_type='cosine', warmup_ratio=0.08, weight_decay=0.01, max_grad_norm=1.0, bf16=args.bf16, deepspeed=args.deepspeed, eval_strategy='epoch', save_strategy='epoch', logging_steps=10, save_total_limit=3, load_best_model_at_end=args.load_best_model_at_end, metric_for_best_model='eval_loss', greater_is_better=False, remove_unused_columns=False, prediction_loss_only=True, report_to='none', seed=args.seed, data_seed=args.seed)
    trainer = LasarTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collator, processing_class=tokenizer)
    write_run_manifest(args.output_dir, stage='stage2_sft', args=args, extra={'train_data_sha256': train_dataset.source_sha256, 'eval_data_sha256': eval_dataset.source_sha256, 'paper': 'arXiv:2605.10207v1', 'trainable_parameters': trainable_parameters, 'paper_compatible_lora': bool(args.lora_r)})
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    selection = {'load_best_model_at_end': args.load_best_model_at_end, 'best_model_checkpoint': trainer.state.best_model_checkpoint, 'best_eval_loss': trainer.state.best_metric, 'final_global_step': trainer.state.global_step, 'final_epoch': trainer.state.epoch}
    (Path(args.output_dir) / 'checkpoint_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
if __name__ == '__main__':
    main()
