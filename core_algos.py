import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis=None) -> torch.Tensor:
    if axis is not None:
        return (values * mask).sum(axis) / mask.sum(axis).clamp(min=1)
    return (values * mask).sum() / mask.sum().clamp(min=1)


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str) -> torch.Tensor:
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss

    Copied from SEED: verl/trainer/ppo/core_algos.py
    """
    if loss_agg_mode == "token-mean":
        loss = masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1).clamp(min=1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
    return loss


def compute_opd_loss(
    log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    opd_step_mask: torch.Tensor = None,
    gate_beta: float = 5.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute OPD-style confidence-gated teacher distillation loss.

    This keeps the teacher log-probs and the gate detached, so gradients flow
    only through the current policy log-probs:
        gate = sigmoid(beta * (teacher_log_prob - log_prob.detach()))
        loss = gate * (teacher_log_prob - log_prob)

    Args:
        log_prob: Current policy log-probabilities, shape (batch_size, response_length).
        teacher_log_prob: Teacher/enhanced-prompt log-probabilities, shape (batch_size, response_length).
        response_mask: Mask for valid response tokens, shape (batch_size, response_length).
        opd_step_mask: Optional sample-level or token-level mask selecting OPD-supervised steps.
        gate_beta: Sigmoid sharpness for the teacher-student gap gate.
        loss_agg_mode: Aggregation mode for `agg_loss`.

    Returns:
        opd_loss, opd_active_token_ratio, opd_gate_mean,
        opd_gate_active_ratio, opd_teacher_gap_mean

    Copied from SEED: verl/trainer/ppo/core_algos.py
    """
    if log_prob.shape != teacher_log_prob.shape:
        raise ValueError(
            f"log_prob shape {tuple(log_prob.shape)} does not match "
            f"teacher_log_prob shape {tuple(teacher_log_prob.shape)}"
        )
    if log_prob.shape != response_mask.shape:
        raise ValueError(
            f"log_prob shape {tuple(log_prob.shape)} does not match "
            f"response_mask shape {tuple(response_mask.shape)}"
        )

    if opd_step_mask is None:
        opd_mask = response_mask
    else:
        opd_step_mask = opd_step_mask.to(device=log_prob.device, dtype=response_mask.dtype)
        if opd_step_mask.dim() == 1:
            if opd_step_mask.shape[0] != response_mask.shape[0]:
                raise ValueError(
                    f"opd_step_mask batch size {opd_step_mask.shape[0]} does not match "
                    f"response_mask batch size {response_mask.shape[0]}"
                )
            opd_mask = response_mask * opd_step_mask.unsqueeze(-1)
        elif opd_step_mask.dim() == 2:
            if opd_step_mask.shape != response_mask.shape:
                raise ValueError(
                    f"opd_step_mask shape {tuple(opd_step_mask.shape)} does not match "
                    f"response_mask shape {tuple(response_mask.shape)}"
                )
            opd_mask = response_mask * opd_step_mask
        else:
            raise ValueError(f"opd_step_mask must be 1D or 2D, got shape {tuple(opd_step_mask.shape)}")

    opd_loss = log_prob.new_tensor(0.0)
    opd_active_token_ratio = log_prob.new_tensor(0.0)
    opd_gate_mean = log_prob.new_tensor(0.0)
    opd_gate_active_ratio = log_prob.new_tensor(0.0)
    opd_teacher_gap_mean = log_prob.new_tensor(0.0)
    if not torch.any(opd_mask > 0):
        return (
            opd_loss,
            opd_active_token_ratio,
            opd_gate_mean,
            opd_gate_active_ratio,
            opd_teacher_gap_mean,
        )

    teacher_log_prob = teacher_log_prob.detach()
    teacher_gap = (teacher_log_prob - log_prob.detach()).detach()
    opd_gate = torch.sigmoid(float(gate_beta) * teacher_gap).detach()
    opd_loss_mat = opd_gate * (teacher_log_prob - log_prob)

    opd_loss = agg_loss(
        loss_mat=opd_loss_mat,
        loss_mask=opd_mask,
        loss_agg_mode=loss_agg_mode,
    )
    opd_active_token_ratio = (opd_mask > 0).float().mean()
    opd_gate_mean = masked_mean(opd_gate, opd_mask)
    opd_gate_active_ratio = masked_mean((opd_gate > 0.5).float(), opd_mask)
    opd_teacher_gap_mean = masked_mean(teacher_gap, opd_mask)
    return (
        opd_loss,
        opd_active_token_ratio,
        opd_gate_mean,
        opd_gate_active_ratio,
        opd_teacher_gap_mean,
    )


def compute_forward_kl_mc(
    teacher_log_prob: torch.Tensor,
    student_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    kl_token_clip: float = 0.0,
) -> torch.Tensor:
    """
    Forward KL (Monte Carlo): KL(P_T || P_S) ≈ E_{y~P_T}[log P_T(y) - log P_S(y)].

    Faithful to TRD's compute_forward_kl_monte_carlo in kl_utils.py:
      kl = (teacher_logprobs - student_logprobs) * mask

    With optional per-token KL clipping (TRD kl_token_clip): caps each
    token's KL contribution to prevent outlier tokens from dominating
    the gradient.

    Args:
        teacher_log_prob: Teacher per-token log probs, shape (B, L). Detached internally.
        student_log_prob: Student per-token log probs, shape (B, L). Has gradient.
        response_mask: Valid token mask, shape (B, L).
        loss_agg_mode: Aggregation mode.
        kl_token_clip: Per-token KL upper bound. 0 disables.
    Returns:
        kl_loss: scalar KL divergence.
    """
    teacher_log_prob = teacher_log_prob.detach()
    kl_mat = teacher_log_prob - student_log_prob

    if kl_token_clip and kl_token_clip > 0:
        kl_mat = kl_mat.clamp(max=kl_token_clip)

    return agg_loss(loss_mat=kl_mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

# ──────────────────────────────────────────────────────────
# GRPO-style advantage: learn reward token-level + group normalization
# ──────────────────────────────────────────────────────────

def compute_grpo_group_advantage(
    sid_token_rewards: torch.Tensor,
    group_ids: torch.Tensor,
    response_mask: torch.Tensor,
    bias: float = 0.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute GRPO-style group-based advantage from token-level rewards.
    Each sample's reward is sum of token rewards. Within the group, normalize
    by subtracting mean and dividing by standard deviation.

    Args:
        sid_token_rewards: shape (bs, dec_len), per-token correctness: 1 if correct, 0 if not.
        group_ids: shape (bs,), group membership of each sample. Same=one GRPO batch group.
        response_mask: shape (bs, dec_len), 1=valid token position.
    Returns:
        advantage: shape (bs, dec_len), broadcast token-level advantage.

    Advantage formula:
        R_i = sum_j reward_i,j  (scalar per sample)
        mu = mean(R)
        sigma = std(R)
        A_i = (R_i - mu) / sigma   (same advantage broadcast to all tokens)
    """
    B, dec_len = sid_token_rewards.shape
    scores = sid_token_rewards.sum(dim=-1) + bias
    advantages = torch.zeros_like(sid_token_rewards)
    unique_groups = group_ids.unique()
    for gid in unique_groups:
        mask_g = (group_ids == gid)
        r_group = scores[mask_g]
        if r_group.numel() <= 1:
            continue
        mu = r_group.mean()
        sigma = r_group.std(unbiased=False)
        if sigma < epsilon:
            sigma = epsilon
        adv = (scores[mask_g] - mu) / sigma
        advantages[mask_g] = adv.unsqueeze(-1).expand(-1, dec_len) * response_mask[mask_g].float()
    return advantages
