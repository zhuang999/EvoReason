from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import torch
import trl
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src' / 'latent'))
sys.path.insert(0, str(ROOT / 'src' / 'grpo_attention_tuning'))
from layers import LatentModel
from grpo_trainer import NoiseGRPORecTrainer
from onerec.modeling import latent_model_class
from onerec import OneRecLatentDataset, build_sid_trie, ensure_thought_token, set_reproducible_seed, write_run_manifest

def parse_args():
    parser = argparse.ArgumentParser(description='Official modified GRPO for LatentR3 on OneRec SID')
    parser.add_argument('--model', required=True, help='LatentR3 OneRec SFT checkpoint')
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--eval-data', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--epochs', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--micro-batch-size', type=int, default=8)
    parser.add_argument('--num-generations', type=int, default=8)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--num-iterations', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--epsilon-high', type=float, default=0.28)
    parser.add_argument('--noise-std', type=float, default=1.5)
    parser.add_argument('--reward-transform', choices=['negative-ppl', 'mean-logp'], default='negative-ppl')
    parser.add_argument('--advantage-mode', choices=['clean-batch', 'group-zscore'], default='clean-batch')
    parser.add_argument('--max-sequence-length', type=int, default=512)
    parser.add_argument('--end-k', type=int, default=-1)
    parser.add_argument('--prompt-protocol', choices=['onerec', 'legacy-r3'], default='onerec')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume-from-checkpoint')
    return parser.parse_args()

def to_hf_dataset(dataset: OneRecLatentDataset) -> HFDataset:
    rows = [dataset[index] for index in range(len(dataset))]
    return HFDataset.from_list(rows)

def main() -> None:
    args = parse_args()
    if trl.__version__ != '0.16.1':
        raise RuntimeError(f'Official modified GRPO requires trl==0.16.1, found {trl.__version__}. Create the dedicated environment from baseline-R3/environment.yaml.')
    set_reproducible_seed(args.seed)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size != 2:
        raise ValueError('LatentR3 modified GRPO requires exactly 2 processes; its cross-device advantage pairing is not defined for other world sizes')
    denominator = args.micro_batch_size * world_size
    if args.batch_size % denominator:
        raise ValueError('batch-size must be divisible by micro-batch-size * world-size')
    if args.micro_batch_size % args.num_generations:
        raise ValueError('micro-batch-size must contain complete num-generations groups')
    gradient_accumulation = args.batch_size // denominator
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ensure_thought_token(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model_class = latent_model_class(args.model, LatentModel)
    model = model_class.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.attention.end_k = args.end_k
    model.config.use_cache = False
    for parameter in model.model.parameters():
        parameter.requires_grad = False
    for parameter in model.lm_head.parameters():
        parameter.requires_grad = False
    trainable = sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    print(f'Trainable LatentRATT parameters: {trainable}')
    dataset_args = {'tokenizer': tokenizer, 'train': True, 'max_sequence_length': args.max_sequence_length, 'prompt_protocol': args.prompt_protocol, 'limit': args.limit}
    train_dataset = OneRecLatentDataset(args.train_data, **dataset_args)
    eval_dataset = OneRecLatentDataset(args.eval_data, **dataset_args)
    train_hf = to_hf_dataset(train_dataset)
    eval_hf = to_hf_dataset(eval_dataset)
    trie = build_sid_trie(args.metadata, tokenizer)
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        del batch_id
        return trie.allowed(input_ids.tolist()) or [im_end_id]

    def paper_ppl_reward_contract(completions, **kwargs):
        del kwargs
        return [0.0 for _ in completions]
    trainer = NoiseGRPORecTrainer(prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, model=model, reward_funcs=[paper_ppl_reward_contract], args=trl.GRPOConfig(warmup_steps=200, num_generations=args.num_generations, max_prompt_length=args.max_sequence_length, temperature=1.0, max_completion_length=32, learning_rate=args.learning_rate, beta=args.beta, num_iterations=args.num_iterations, epsilon=args.epsilon, epsilon_high=args.epsilon_high, num_train_epochs=args.epochs, output_dir=args.output_dir, per_device_train_batch_size=args.micro_batch_size, per_device_eval_batch_size=args.micro_batch_size, seed=args.seed, gradient_accumulation_steps=gradient_accumulation, lr_scheduler_type='cosine', eval_strategy='no', save_steps=200, save_total_limit=2, ddp_find_unused_parameters=True, report_to='none', bf16=True, logging_steps=10), train_dataset=train_hf, eval_dataset=eval_hf, processing_class=tokenizer)
    trainer.noise_std = args.noise_std
    trainer.reward_transform = args.reward_transform
    trainer.advantage_mode = args.advantage_mode
    write_run_manifest(args.output_dir, stage='latentr3_onerec_rl', args=args, extra={'protocol': 'onerec-sid-matched', 'paper': 'arXiv:2505.19092', 'train_data_sha256': train_dataset.source_sha256, 'eval_data_sha256': eval_dataset.source_sha256, 'trainable_parameters': trainable, 'official_two_gpu_advantage_pairing': True})
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.log_metrics('train', result.metrics)
    trainer.save_metrics('train', result.metrics)
    trainer.save_state()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
if __name__ == '__main__':
    main()
