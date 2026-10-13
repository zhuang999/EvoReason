from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DeepSpeedPlugin
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from lasar import LasarCollator, LasarForCausalLM, OneRecLasarDataset, set_reproducible_seed, write_run_manifest
from lasar.trie import TrieLogitsProcessor, build_sid_trie

def parse_args():
    parser = argparse.ArgumentParser(description='LASAR GRPO + REINFORCE + Terminal-KL training')
    parser.add_argument('--model', required=True, help='LASAR Stage-2 checkpoint')
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--train-anchors', required=True)
    parser.add_argument('--metadata', required=True, help='OneRec *.pretrain.json for SID trie')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-05)
    parser.add_argument('--num-generations', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=7)
    parser.add_argument('--clip-ratio', type=float, default=0.2)
    parser.add_argument('--kl-beta', type=float, default=0.001)
    parser.add_argument('--terminal-kl-weight', type=float, default=1e-05)
    parser.add_argument('--reinforce-coef', type=float, default=0.01)
    parser.add_argument('--step-penalty', type=float, default=0.0005)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--ema-decay', type=float, default=0.95)
    parser.add_argument('--mixed-precision', choices=['no', 'fp16', 'bf16'], default='bf16')
    parser.add_argument('--zero-stage', type=int, choices=[0, 1, 2], default=2)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-sequence-length', type=int, default=512)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--resume-from-checkpoint')
    parser.add_argument('--train-backbone', action='store_true', help='Also update the Stage-1 SID backbone. Disabled by default for numerical stability.')
    return parser.parse_args()

def sequence_log_probs(logits, target_ids, target_mask, logits_processor=None):
    values = []
    for token_index in range(target_ids.shape[1]):
        token_logits = logits[:, token_index, :]
        if logits_processor is not None:
            token_logits = logits_processor(token_logits, target_ids[:, :token_index])
        token_log_prob = F.log_softmax(token_logits.float(), dim=-1).gather(1, target_ids[:, token_index:token_index + 1]).squeeze(1)
        values.append(token_log_prob)
    token_log_probs = torch.stack(values, dim=1)
    return (token_log_probs * target_mask).sum(dim=-1)

def rewards_for_group(candidates, targets, target_masks, group_size):
    batch_size = targets.shape[0]
    rewards = torch.zeros(batch_size, group_size, device=candidates.device)
    ranks = torch.arange(group_size, device=candidates.device, dtype=torch.float32)
    position_weights = -1.0 / torch.log2(ranks + 2.0)
    normalized_weights = position_weights / position_weights.sum()
    for batch_index in range(batch_size):
        length = int(target_masks[batch_index].sum().item())
        target = targets[batch_index, :length]
        group = candidates[batch_index * group_size:(batch_index + 1) * group_size, :length]
        if candidates.shape[1] < length:
            hits = torch.zeros(group_size, dtype=torch.bool, device=candidates.device)
        else:
            hits = group.eq(target.unsqueeze(0)).all(dim=1)
        if hits.any():
            rewards[batch_index] = normalized_weights
            rewards[batch_index, hits] = 1.0
    return rewards

def repeat_batch(tensor, count):
    return tensor.repeat_interleave(count, dim=0)

def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    deepspeed_plugin = None
    if args.zero_stage > 0:
        deepspeed_plugin = DeepSpeedPlugin(zero_stage=args.zero_stage, gradient_accumulation_steps=args.gradient_accumulation_steps, gradient_clipping=1.0)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision, deepspeed_plugin=deepspeed_plugin)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.mixed_precision == 'bf16' else torch.float16 if args.mixed_precision == 'fp16' else torch.float32
    model = LasarForCausalLM.from_pretrained(args.model, torch_dtype=dtype, adapter_trainable=args.train_backbone)
    reference = LasarForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    embedding_rows = model.get_input_embeddings().num_embeddings
    max_token_id = max(tokenizer.get_vocab().values())
    if max_token_id >= embedding_rows:
        raise ValueError(f'Tokenizer id {max_token_id} exceeds model vocabulary rows {embedding_rows}; resize and save the Stage-2 checkpoint before RL.')
    if not args.train_backbone:
        model.backbone.requires_grad_(False)
    model.policy_head.requires_grad_(True)
    model.latent_adapter.requires_grad_(model.use_latent_adapter)
    model.latent_start_embedding.requires_grad_(model.use_trainable_markers)
    model.latent_end_embedding.requires_grad_(model.use_trainable_markers)
    reference.requires_grad_(False).eval().to(accelerator.device)
    dataset = OneRecLasarDataset(args.train_data, tokenizer, args.train_anchors, max_sequence_length=args.max_sequence_length, max_latent_steps=model.max_latent_steps, limit=args.limit)
    collator = LasarCollator(tokenizer.pad_token_id, model.max_latent_steps)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError('RL has no trainable parameters')
    optimizer = AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=0.01)
    total_steps = max(1, math.ceil(len(loader) * args.epochs / args.gradient_accumulation_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.08), num_training_steps=total_steps)
    trie = build_sid_trie(args.metadata, tokenizer)
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    logits_processor = TrieLogitsProcessor(trie, fallback_token_id=im_end_id)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    if accelerator.is_main_process:
        write_run_manifest(args.output_dir, stage='rl', args=args, extra={'train_data_sha256': dataset.source_sha256, 'trainable_parameters': sum((parameter.numel() for parameter in trainable_parameters)), 'paper': 'arXiv:2605.10207v1'})
    reward_baseline = torch.tensor(0.0, device=accelerator.device)
    global_step = 0
    start_epoch = 0
    resume_batch = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        state_path = Path(args.resume_from_checkpoint) / 'rl_state.json'
        state = json.loads(state_path.read_text(encoding='utf-8'))
        global_step = int(state['global_step'])
        start_epoch = int(state['epoch'])
        resume_batch = int(state['next_batch'])
        reward_baseline.fill_(float(state['reward_baseline']))
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loader = loader
        first_batch = 0
        if epoch == start_epoch and resume_batch:
            epoch_loader = accelerator.skip_first_batches(loader, resume_batch)
            first_batch = resume_batch
        for batch_index, batch in enumerate(epoch_loader, start=first_batch):
            with accelerator.accumulate(model):
                batch_size = batch['prompt_ids'].shape[0]
                with torch.no_grad():
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.eval()
                    prompt_state, _ = unwrapped.encode_prompt(batch['prompt_ids'], batch['prompt_attention_mask'])
                    initial_policy_logits = unwrapped.policy_head(prompt_state)
                    sampled_steps = torch.distributions.Categorical(logits=initial_policy_logits).sample() + 1
                    beam_candidates, _, _ = unwrapped.beam_search(batch['prompt_ids'], batch['prompt_attention_mask'], max_new_tokens=args.max_new_tokens, num_beams=args.num_generations, step_counts=sampled_steps, logits_processor=logits_processor)
                    candidates = beam_candidates.reshape(batch_size * args.num_generations, args.max_new_tokens)
                    group_rewards = rewards_for_group(candidates, batch['answer_ids'], batch['answer_attention_mask'], args.num_generations)
                    advantages = (group_rewards - group_rewards.mean(dim=1, keepdim=True)) / (group_rewards.std(dim=1, keepdim=True, unbiased=False) + 1e-06)
                    unwrapped.train()
                repeated_prompts = repeat_batch(batch['prompt_ids'], args.num_generations)
                repeated_prompt_masks = repeat_batch(batch['prompt_attention_mask'], args.num_generations)
                repeated_steps = repeat_batch(sampled_steps, args.num_generations)
                candidate_mask = candidates.ne(tokenizer.pad_token_id).long()
                repeated_anchors = repeat_batch(batch['cot_anchors'], args.num_generations)
                repeated_anchor_mask = repeat_batch(batch['anchor_mask'], args.num_generations)
                output = model(prompt_ids=repeated_prompts, prompt_attention_mask=repeated_prompt_masks, answer_ids=candidates, answer_attention_mask=candidate_mask, step_counts=repeated_steps, cot_anchors=repeated_anchors, anchor_mask=repeated_anchor_mask, terminal_only_alignment=True)
                new_log_probs = sequence_log_probs(output.logits, candidates, candidate_mask, logits_processor)
                with torch.no_grad():
                    ref_output = reference(prompt_ids=repeated_prompts, prompt_attention_mask=repeated_prompt_masks, answer_ids=candidates, answer_attention_mask=candidate_mask, step_counts=repeated_steps)
                    ref_log_probs = sequence_log_probs(ref_output.logits, candidates, candidate_mask, logits_processor)
                log_ratio = new_log_probs - ref_log_probs
                ratio = log_ratio.exp()
                flat_advantages = advantages.reshape(-1)
                clipped_ratio = ratio.clamp(1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                policy_objective = torch.minimum(ratio * flat_advantages, clipped_ratio * flat_advantages)
                approx_kl = log_ratio.exp() - log_ratio - 1.0
                grpo_loss = -policy_objective.mean() + args.kl_beta * approx_kl.mean()
                policy_logits = output.policy_logits.view(batch_size, args.num_generations, -1)[:, 0]
                distribution = torch.distributions.Categorical(logits=policy_logits)
                group_mean_reward = group_rewards.mean(dim=1)
                reinforce_signal = group_mean_reward - reward_baseline - args.step_penalty * sampled_steps.float()
                reinforce_loss = -(reinforce_signal.detach() * distribution.log_prob(sampled_steps - 1)).mean()
                reinforce_loss -= args.entropy_coef * distribution.entropy().mean()
                terminal_kl = output.alignment_loss
                loss = grpo_loss + args.terminal_kl_weight * terminal_kl + args.reinforce_coef * reinforce_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                reward_baseline = args.ema_decay * reward_baseline + (1.0 - args.ema_decay) * group_mean_reward.mean().detach()
                if accelerator.sync_gradients:
                    global_step += 1
                if accelerator.is_main_process and accelerator.sync_gradients and (global_step % 10 == 0):
                    print(f'epoch={epoch + 1} step={global_step} loss={loss.item():.5f} grpo={grpo_loss.item():.5f} terminal_kl={terminal_kl.item():.5f} reinforce={reinforce_loss.item():.5f} reward={group_mean_reward.mean().item():.4f} hit={group_rewards.eq(1.0).any(dim=1).float().mean().item():.4f} policy_kl={approx_kl.mean().item():.5f} N={sampled_steps.float().mean().item():.2f}')
                if accelerator.sync_gradients and args.save_steps > 0 and (global_step % args.save_steps == 0):
                    checkpoint_dir = Path(args.output_dir) / f'checkpoint-{global_step}'
                    accelerator.save_state(checkpoint_dir)
                    if accelerator.is_main_process:
                        (checkpoint_dir / 'rl_state.json').write_text(json.dumps({'global_step': global_step, 'epoch': epoch, 'next_batch': batch_index + 1, 'reward_baseline': reward_baseline.item()}, indent=2) + '\n', encoding='utf-8')
        resume_batch = 0
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
if __name__ == '__main__':
    main()
