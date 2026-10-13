from __future__ import annotations
import torch

def reward_from_logps(per_token_logps: torch.Tensor, target_lengths: torch.Tensor, transform: str) -> torch.Tensor:
    mean_logp = per_token_logps.sum(dim=-1) / target_lengths
    if transform == 'mean-logp':
        return mean_logp.to(torch.float32)
    if transform == 'negative-ppl':
        return -(-mean_logp).exp().to(torch.float32)
    raise ValueError(f'Unsupported reward transform: {transform}')

def normalize_advantages(rewards: torch.Tensor, num_generations: int, mode: str) -> torch.Tensor:
    groups = rewards.view(-1, num_generations)
    if mode == 'clean-batch':
        centered = rewards - groups[:, 0].mean()
        return centered / (torch.norm(centered) + 1e-06)
    if mode == 'group-zscore':
        centered = groups - groups.mean(dim=1, keepdim=True)
        scale = centered.std(dim=1, keepdim=True, unbiased=False)
        return (centered / (scale + 1e-06)).reshape_as(rewards)
    raise ValueError(f'Unsupported advantage mode: {mode}')
