import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from core_algos import compute_opd_loss, compute_forward_kl_mc, agg_loss as agg_loss_opd
from kl_utils import compute_kl_divergence
from rkv_utils import cal_similarity, compute_attention_scores
from dataset_codi import THINK_END, SYSTEM_MESSAGE, THINK_START

REFINE_PROMPT = "Please re-examine the user's purchase history and provide the correct recommendation."


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_q(q, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin)


def get_embed_layer(model):
    if hasattr(model, 'get_base_model'):
        base = model.get_base_model()
    else:
        base = model
    if hasattr(base, 'model') and hasattr(base.model, 'embed_tokens'):
        return base.model.embed_tokens
    if hasattr(base, 'embed_tokens'):
        return base.embed_tokens
    raise AttributeError(f"Cannot find embed_tokens in {type(model)}")


def _get_base_model(model):
    if hasattr(model, 'get_base_model'):
        return model.get_base_model()
    return model


def _get_kv_config(model):
    base = _get_base_model(model)
    config = base.config
    H_kv = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    num_layers = len(base.model.layers)
    return base, config, H_kv, head_dim, num_layers


def _sinkhorn(log_alpha: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """
    Log-domain Sinkhorn iterations for differentiable optimal transport.
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return log_alpha.exp()


class LatentRecCoDi(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_latent: int = 8,
        num_jacobi_steps: int = 3,
        use_prj: bool = True,
        prj_dim: int = 1024,
        prj_dropout: float = 0.0,
        prj_no_ln: bool = False,
        distill_loss_factor: float = 10.0,
        distill_loss_type: str = "smooth_l1",
        distill_loss_div_std: bool = False,
        use_lora: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        kv_distill_factor: float = 1.0,
        mix_lambda: float = 0.07,
        kv_window_size: int = 4,
        kv_loss_p: int = 1,
        opd_kl_factor: float = 1.0,
        opd_gate_beta: float = 5.0,
        opd_loss_agg_mode: str = "token-mean",
        kl_token_clip: float = 0.0,
        ref_loss_factor: float = 0.0,
        div_factor: float = 0.02,
        use_primitive_distill: bool = False,
        primitive_distill_factor: float = 2.0,
        kv_compress_mode: str = "rkv_whole",
        tokens_per_primitive: int = 1,
    ):
        super().__init__()
        self.model_name = model_name_or_path
        self.num_latent = num_latent
        self.num_jacobi_steps = max(1, num_jacobi_steps)
        self.distill_loss_factor = distill_loss_factor
        self.distill_loss_div_std = distill_loss_div_std
        self.distill_loss_type = distill_loss_type
        self.use_prj = use_prj
        self.kv_distill_factor = kv_distill_factor
        self.mix_lambda = mix_lambda
        self.kv_window_size = kv_window_size
        self.kv_loss_p = kv_loss_p
        self.opd_kl_factor = opd_kl_factor
        self.opd_gate_beta = opd_gate_beta
        self.opd_loss_agg_mode = opd_loss_agg_mode
        self.kl_token_clip = kl_token_clip
        self.ref_loss_factor = ref_loss_factor
        self.div_factor = div_factor
        self.use_primitive_distill = use_primitive_distill
        self.primitive_distill_factor = primitive_distill_factor
        self.kv_compress_mode = kv_compress_mode
        self.tokens_per_primitive = tokens_per_primitive

        self.codi = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # ── Add SID tokens (if not already present) ──
        sid_tokens = ['<|sid_begin|>', '<|sid_end|>']
        for prefix in ['s_a', 's_b', 's_c', 's_d']:
            for i in range(256):
                sid_tokens.append(f'<{prefix}_{i}>')
        sid_ids = self.tokenizer.convert_tokens_to_ids(sid_tokens)
        has_sid = all(tid != self.tokenizer.unk_token_id for tid in sid_ids)
        if not has_sid:
            missing = [t for t in sid_tokens
                       if self.tokenizer.convert_tokens_to_ids(t) == self.tokenizer.unk_token_id]
            self.tokenizer.add_special_tokens({"additional_special_tokens": missing})

        # ── Ensure the tokenizer has a real pad token, then reserve bot/eot slots ──
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        base_vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bot_id = base_vocab_size
        self.eot_id = base_vocab_size + 1

        # `bot_id` / `eot_id` are runtime-only numeric slots and are not tokenizer tokens.
        self.codi.resize_token_embeddings(base_vocab_size + 2)

        self.dim = self.codi.config.hidden_size

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
            )
            self.codi = get_peft_model(self.codi, lora_config)

        if use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(prj_dropout),
                nn.Linear(self.dim, prj_dim),
                nn.GELU(),
                nn.Linear(prj_dim, self.dim),
            )
            if not prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))
            self.prj = self.prj.to(torch.bfloat16)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise ValueError(f"Unknown distill_loss_type: {distill_loss_type}")

        self.primitive_slot_embeddings = None
        if use_primitive_distill:
            from dataset_codi import NUM_PRIMITIVES, LATENT_PER_PRIMITIVE
            _num_slots = NUM_PRIMITIVES * LATENT_PER_PRIMITIVE
            self.primitive_slot_embeddings = nn.Parameter(
                torch.randn(_num_slots, self.dim) * 0.02
            )

        self.identifier_token_ids: list = []
        if kv_compress_mode == "identifier":
            from dataset_codi import NUM_PRIMITIVES
            _new_toks = [f"<ptool_end_{i}>" for i in range(NUM_PRIMITIVES)]
            self.tokenizer.add_special_tokens({"additional_special_tokens": _new_toks})
            self.codi.resize_token_embeddings(len(self.tokenizer))
            self.identifier_token_ids = self.tokenizer.convert_tokens_to_ids(_new_toks)

    @property
    def config(self):
        return self.codi.config

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.codi.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.codi.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        self.codi.enable_input_require_grads()

    @torch.no_grad()
    def cot_forward(self, ref_input_ids, ref_attention_mask=None):
        """
        Teacher CoT mode forward（共享 codi 参数，no_grad）。
        输入完整序列（encoder + CoT + decoder），返回 logits / past_key_values / hidden_states。
        """
        outputs = self.codi(
            input_ids=ref_input_ids,
            attention_mask=ref_attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        return outputs

    def _get_student_token_log_probs(
        self, logits: torch.Tensor, labels: torch.Tensor, enc_seq_len: int, M: int
    ) -> torch.Tensor:
        """
        从 student logits 中提取 decoder 位置的 per-token log_prob。
        Returns: [B, dec_len-1]，与 response_mask = (labels[:, 1:] != -100) 对齐。
        """
        dec_len = labels.shape[1]
        dec_start = enc_seq_len + M
        dec_logits = logits[:, dec_start:dec_start + dec_len - 1, :]
        log_probs = F.log_softmax(dec_logits.float(), dim=-1)
        targets = labels[:, 1:].clamp(min=0)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.to(logits.dtype)

    def _get_teacher_token_log_probs(
        self,
        ref_logits: torch.Tensor,
        labels: torch.Tensor,
        enc_lengths: torch.Tensor,
        cot_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        从 teacher ref_logits 中提取 decoder 位置的 per-token log_prob。
        ref 序列结构: [enc | CoT | dec]，decoder 起始于 enc_l + cot_l。
        Returns: [B, dec_len-1]，与 response_mask = (labels[:, 1:] != -100) 对齐。
        """
        B = ref_logits.shape[0]
        dec_len = labels.shape[1]
        result = torch.zeros(B, dec_len - 1, device=ref_logits.device, dtype=ref_logits.dtype)
        for b in range(B):
            enc_l = enc_lengths[b].item()
            cot_l = cot_lengths[b].item()
            t_start = enc_l + cot_l
            t_end = min(t_start + dec_len - 1, ref_logits.shape[1])
            avail_len = t_end - t_start
            if avail_len <= 0:
                continue
            t_logits = ref_logits[b, t_start:t_end, :]
            t_log_probs = F.log_softmax(t_logits.float(), dim=-1)
            targets_b = labels[b, 1:avail_len + 1].clamp(min=0)
            token_lp = t_log_probs.gather(-1, targets_b.unsqueeze(-1)).squeeze(-1)
            result[b, :avail_len] = token_lp.to(result.dtype)
        return result

    def forward(
        self,
        encoder_input_ids, decoder_input_ids, ref_input_ids, labels, ref_labels,
        encoder_attention_mask=None, ref_attention_mask=None,
        ref_answer_position=None, model_answer_position=None,
        enc_lengths=None, cot_lengths=None, dec_lengths=None,
        global_step=None,
        primitive_token_ranges=None, identifier_positions=None, **kwargs,
    ):
        if "qwen" in self.model_name.lower():
            model_answer_position = model_answer_position + 1
            ref_answer_position = ref_answer_position + 1
        model_answer_position = model_answer_position - 1
        ref_answer_position = ref_answer_position - 1

        B = encoder_input_ids.shape[0]
        enc_seq_len = encoder_input_ids.shape[1]
        M = self.num_latent
        T = self.num_jacobi_steps
        device = encoder_input_ids.device

        # ── Teacher CoT forward ────────────────────────────────────────────────
        # ref_loss_factor != 0（Stage 1 SFT）：带梯度，训练模型预测 CoT+SID 全序列
        # ref_loss_factor == 0（Stage 2 OPD）：no_grad，只作为 teacher signal
        ref_ce_loss = torch.tensor(0.0, device=device)
        if self.ref_loss_factor != 0:
            ref_outputs = self.codi(
                input_ids=ref_input_ids,
                attention_mask=ref_attention_mask,
                use_cache=True,
                output_hidden_states=True,
            )
            ref_logits_shift = ref_outputs.logits[:, :-1, :].reshape(-1, ref_outputs.logits.size(-1))
            ref_target_ids = ref_labels[:, 1:].reshape(-1)
            ref_ce_loss = self.loss_fct(ref_logits_shift, ref_target_ids)
        else:
            ref_outputs = self.cot_forward(ref_input_ids, ref_attention_mask)

        # ── Student Latent forward（Jacobi 迭代）─────────────────────────────
        embed_fn = get_embed_layer(self.codi)
        enc_embeds = embed_fn(encoder_input_ids)
        dec_embeds = embed_fn(decoder_input_ids)
        dec_len = dec_embeds.shape[1]

        init_raw = torch.zeros(B, M, self.dim, device=device, dtype=enc_embeds.dtype)
        if self.primitive_slot_embeddings is not None and M == self.primitive_slot_embeddings.shape[0]:
            init_raw = self.primitive_slot_embeddings.unsqueeze(0).expand(B, -1, -1).to(init_raw.dtype)
        if self.use_prj:
            latent_embeds = self.prj(init_raw)
        else:
            latent_embeds = init_raw

        enc_mask = encoder_attention_mask if encoder_attention_mask is not None \
            else torch.ones(B, enc_seq_len, device=device, dtype=torch.long)
        full_mask = torch.cat([
            enc_mask,
            torch.ones(B, M + dec_len, device=device, dtype=torch.long),
        ], dim=1)

        full_pos = torch.cat([
            torch.arange(enc_seq_len, device=device).unsqueeze(0).expand(B, -1),
            torch.arange(enc_seq_len, enc_seq_len + M + dec_len, device=device).unsqueeze(0).expand(B, -1),
        ], dim=1)

        logits = None
        student_latent_kvs = []
        ce_loss = torch.tensor(0.0, device=device)
        distill_loss = torch.tensor(0.0, device=device)
        slot_orth_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
        slot_dir_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
        lat_orth_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
        primitive_distill_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
        latent_embeds_final = None

        for t in range(T):
            is_last = (t == T - 1)
            full_embeds = torch.cat([enc_embeds, latent_embeds, dec_embeds], dim=1)

            if not is_last:
                with torch.no_grad():
                    jac_out = self.codi(
                        inputs_embeds=full_embeds,
                        attention_mask=full_mask,
                        position_ids=full_pos,
                        output_hidden_states=True,
                    )
                    raw_latent = jac_out.hidden_states[-1][:, enc_seq_len:enc_seq_len + M, :]
                if self.use_prj:
                    latent_embeds = self.prj(raw_latent)
                else:
                    latent_embeds = raw_latent
            else:
                jac_out = self.codi(
                    inputs_embeds=full_embeds,
                    attention_mask=full_mask,
                    position_ids=full_pos,
                    use_cache=True,
                    output_hidden_states=True,
                )

                student_latent_kvs = [
                    (
                        jac_out.past_key_values.key_cache[l][:, :, enc_seq_len:enc_seq_len + M, :],
                        jac_out.past_key_values.value_cache[l][:, :, enc_seq_len:enc_seq_len + M, :],
                    )
                    for l in range(len(jac_out.past_key_values.key_cache))
                ]

                logits = jac_out.logits
                dec_start = enc_seq_len + M
                effective_logits = logits[:, dec_start:dec_start + dec_len - 1, :].reshape(-1, logits.size(-1))
                target_ids = labels[:, 1:].reshape(-1)
                ce_loss = self.loss_fct(effective_logits, target_ids)

                # ── Hidden state distill loss（per-layer, answer position）──────
                student_pos = enc_seq_len + M + model_answer_position
                distill_loss = torch.tensor(0.0, device=device)
                for l_idx, (out_h, ref_h) in enumerate(
                    zip(jac_out.hidden_states, ref_outputs.hidden_states)
                ):
                    ref_sel = ref_h.gather(
                        1,
                        ref_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_h.size(-1)),
                    )
                    out_sel = out_h.gather(
                        1,
                        student_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out_h.size(-1)),
                    )
                    dloss_tmp = self.distill_loss_fct(out_sel, ref_sel.detach())
                    if self.distill_loss_div_std:
                        dloss_tmp = dloss_tmp / (ref_sel.std() + 1e-8)
                    distill_loss = distill_loss + dloss_tmp
                distill_loss = distill_loss / len(jac_out.hidden_states)

                # ── Primitive distill loss (R-KV KV compression alignment) ──
                if self.use_primitive_distill and primitive_token_ranges is not None and student_latent_kvs:
                    _prim_kvs = self._compress_teacher_kv_cyclic(
                        ref_outputs.past_key_values,
                        ref_outputs.hidden_states,
                        enc_lengths,
                        primitive_token_ranges,
                        self.tokens_per_primitive,
                        dec_lengths=dec_lengths,
                    )
                    _prim_kvs = self._align_kv_to_M(_prim_kvs, M)
                    primitive_distill_loss = self._kv_distill_loss(student_latent_kvs, _prim_kvs).to(torch.bfloat16)

                if self.use_primitive_distill and self.primitive_slot_embeddings is not None:
                    _num_slots = self.primitive_slot_embeddings.shape[0]
                    slot_norm = F.normalize(self.primitive_slot_embeddings.float(), dim=-1)
                    slot_gram = slot_norm @ slot_norm.T
                    slot_off_diag = slot_gram * (1 - torch.eye(_num_slots, device=device))
                    slot_orth_loss = (slot_off_diag ** 2).sum().to(torch.bfloat16)

                if M > 1:
                    _lat_h = jac_out.hidden_states[-1][:, enc_seq_len:enc_seq_len + M, :].float()
                    _lat_h_norm = F.normalize(_lat_h, dim=-1)
                    _lat_gram = torch.bmm(_lat_h_norm, _lat_h_norm.transpose(-1, -2))
                    _lat_off_diag = _lat_gram * (1 - torch.eye(M, device=device).unsqueeze(0))
                    lat_orth_loss = _lat_off_diag.pow(2).mean().to(torch.bfloat16)

                latent_embeds_final = jac_out.hidden_states[-1][:, enc_seq_len:enc_seq_len + M, :]
                if self.use_prj:
                    latent_embeds_final = self.prj(latent_embeds_final)

        # ── KV Distill loss（R-KV 压缩 teacher CoT KV → 对齐 student latent KV）────
        kv_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
        if enc_lengths is not None and student_latent_kvs:
            try:
                teacher_kvs = None
                if self.kv_compress_mode == "rkv_whole" and cot_lengths is not None:
                    teacher_kvs = self._compress_teacher_kv(
                        ref_outputs.past_key_values,
                        ref_outputs.hidden_states,
                        enc_lengths,
                        cot_lengths,
                        dec_lengths=dec_lengths,
                        M=M,
                    )
                elif self.kv_compress_mode == "rkv_cyclic" and primitive_token_ranges is not None:
                    teacher_kvs = self._compress_teacher_kv_cyclic(
                        ref_outputs.past_key_values,
                        ref_outputs.hidden_states,
                        enc_lengths,
                        primitive_token_ranges,
                        self.tokens_per_primitive,
                        dec_lengths=dec_lengths,
                    )
                elif self.kv_compress_mode == "identifier" and identifier_positions is not None:
                    teacher_kvs = self._extract_identifier_kvs(
                        ref_outputs.past_key_values,
                        identifier_positions,
                    )

                if teacher_kvs is not None and self.kv_distill_factor != 0:
                    kv_loss = self._kv_distill_loss(student_latent_kvs, teacher_kvs)
            except Exception as e:
                import traceback
                print(f"  [WARN] KV distill failed: {e}")
                traceback.print_exc()

        # ── OPD loss（SEED compute_opd_loss，置信度门控）────────────────────────
        opd_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
        opd_active_ratio = torch.tensor(0.0, device=device)
        opd_gate_mean = torch.tensor(0.0, device=device)
        opd_teacher_gap = torch.tensor(0.0, device=device)
        if (
            self.opd_kl_factor != 0
            and logits is not None
            and enc_lengths is not None
            and cot_lengths is not None
        ):
            try:
                student_log_prob = self._get_student_token_log_probs(logits, labels, enc_seq_len, M)
                teacher_log_prob = self._get_teacher_token_log_probs(
                    ref_outputs.logits, labels, enc_lengths, cot_lengths
                )
                response_mask = (labels[:, 1:] != -100).float()

                opd_loss, opd_active_ratio, opd_gate_mean, _, opd_teacher_gap = compute_opd_loss(
                    log_prob=student_log_prob,
                    teacher_log_prob=teacher_log_prob,
                    response_mask=response_mask,
                    gate_beta=self.opd_gate_beta,
                    loss_agg_mode=self.opd_loss_agg_mode,
                )
                opd_loss = opd_loss.to(torch.bfloat16)
            except Exception as e:
                print(f"  [WARN] OPD loss failed: {e}")

        # ── Diversity loss（防 latent collapse）──────────────────────────────
        div_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
        lat_ev1 = 0.0
        lat_eff_rank = 0.0
        if self.div_factor != 0 and latent_embeds_final is not None:
            M_div = latent_embeds_final.shape[1]
            if M_div > 1:
                eye = torch.eye(M_div, device=device).unsqueeze(0)
                _margin = 0.3
                l_norm = F.normalize(latent_embeds_final.float(), dim=-1)
                l_gram = torch.bmm(l_norm, l_norm.transpose(-1, -2))
                l_off_diag = l_gram * (1.0 - eye)
                div_loss = F.relu(l_off_diag - _margin).pow(2).mean().to(torch.bfloat16)

                with torch.no_grad():
                    _lat_gram = torch.bmm(l_norm, l_norm.transpose(-1, -2))
                    _eigs = torch.linalg.eigvalsh(_lat_gram.float()).clamp(min=0)
                    _eigs_sum = _eigs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    _eigs_norm = _eigs / _eigs_sum
                    lat_ev1 = _eigs_norm[:, -1].mean().item()
                    lat_eff_rank = torch.exp(
                        -(_eigs_norm * (_eigs_norm + 1e-8).log()).sum(dim=-1)
                    ).mean().item()

        # ── 总 loss：L = L_SID + λ1*L_distill + λ2*L_KV + λ3*L_OPD + λ4*L_div + auxiliary ──
        loss = (
            ce_loss
            + ref_ce_loss * self.ref_loss_factor
            + distill_loss * self.distill_loss_factor
            + primitive_distill_loss * self.primitive_distill_factor
            + slot_orth_loss * 0.01
            + slot_dir_loss * 0.5
            + lat_orth_loss * 0.01
            + kv_loss * self.kv_distill_factor
            + opd_loss * self.opd_kl_factor
            + div_loss * self.div_factor
        )

        return {
            "loss": loss,
            "logits": logits,
            "ce_loss": ce_loss.detach().item() if torch.is_tensor(ce_loss) else 0,
            "ref_ce_loss": (ref_ce_loss * self.ref_loss_factor).detach().item() if torch.is_tensor(ref_ce_loss) else 0,
            "distill_loss": (distill_loss * self.distill_loss_factor).detach().item() if torch.is_tensor(distill_loss) else 0,
            "primitive_distill_loss": (primitive_distill_loss * self.primitive_distill_factor).detach().item() if torch.is_tensor(primitive_distill_loss) else 0,
            "slot_orth_loss": slot_orth_loss.detach().item() if torch.is_tensor(slot_orth_loss) else 0,
            "slot_dir_loss": slot_dir_loss.detach().item() if torch.is_tensor(slot_dir_loss) else 0,
            "lat_orth_loss": lat_orth_loss.detach().item() if torch.is_tensor(lat_orth_loss) else 0,
            "kv_loss": kv_loss.detach().item() if torch.is_tensor(kv_loss) else 0,
            "opd_loss": opd_loss.detach().item() if torch.is_tensor(opd_loss) else 0,
            "div_loss": div_loss.detach().item() if torch.is_tensor(div_loss) else 0,
            "opd_active_ratio": opd_active_ratio.detach().item() if torch.is_tensor(opd_active_ratio) else 0,
            "opd_gate_mean": opd_gate_mean.detach().item() if torch.is_tensor(opd_gate_mean) else 0,
            "opd_teacher_gap": opd_teacher_gap.detach().item() if torch.is_tensor(opd_teacher_gap) else 0,
            "lat_ev1": lat_ev1,
            "lat_eff_rank": lat_eff_rank,
        }

    @torch.no_grad()
    def _compress_teacher_kv(self, ref_past_kvs, ref_hidden_states, enc_lengths, cot_lengths, M, dec_lengths=None, return_indices: bool = False):
        """
        Compress teacher CoT KV cache N_C -> M using R-KV algorithm (KAVA Section 3.2).
        Uses past_key_values from use_cache=True (ZeRO-2 compatible).
        Importance I (Eq. 5-6): computed with ALL N_A answer tokens, Q with RoPE applied.
        Redundancy R: cal_similarity on CoT keys (R-KV convention).
        Score Z = lambda*I - (1-lambda)*R
        Select top-(M-w) + always keep last w tokens (R-KV window convention).
        When return_indices=True, also returns per-sample selected token positions (last layer).
        """
        B = enc_lengths.shape[0]
        ref_model = self.codi
        base, config, H_kv, head_dim, num_layers = _get_kv_config(ref_model)
        w = self.kv_window_size

        compressed = []
        sample_positions = [] if return_indices else None

        for l in range(num_layers):
            K_full = ref_past_kvs.key_cache[l]
            V_full = ref_past_kvs.value_cache[l]
            d = K_full.shape[3]

            K_comp_list, V_comp_list = [], []
            for b in range(B):
                enc_l = enc_lengths[b].item()
                cot_l = cot_lengths[b].item()

                K_cot = K_full[b:b+1, :, enc_l : enc_l + cot_l, :]
                V_cot = V_full[b:b+1, :, enc_l : enc_l + cot_l, :]
                NC = K_cot.shape[2]

                idx = None
                if NC <= M:
                    pad = M - NC
                    K_b = F.pad(K_cot, (0, 0, 0, pad))
                    V_b = F.pad(V_cot, (0, 0, 0, pad))
                else:
                    ans_start = enc_l + cot_l
                    ref_total = ref_hidden_states[l + 1].shape[1]

                    if dec_lengths is not None:
                        dec_l = dec_lengths[b].item()
                        ans_end = min(ans_start + dec_l, ref_total)
                    else:
                        ans_end = ref_total

                    NA = ans_end - ans_start

                    if NA > 0:
                        h_ans_bf16 = ref_hidden_states[l + 1][b:b+1, ans_start:ans_end, :]
                        attn_layer = base.model.layers[l].self_attn
                        num_q_heads = getattr(attn_layer, 'num_heads', config.num_attention_heads)

                        q_raw = attn_layer.q_proj(h_ans_bf16)
                        q_raw = q_raw.view(1, NA, num_q_heads, d)
                        if hasattr(attn_layer, 'q_norm'):
                            q_raw = attn_layer.q_norm(q_raw)
                        q_ans = q_raw.transpose(1, 2)

                        try:
                            rotary_emb = base.model.rotary_emb
                            ans_pos_ids = torch.arange(
                                ans_start, ans_end, device=h_ans_bf16.device
                            ).unsqueeze(0)
                            cos, sin = rotary_emb(h_ans_bf16, ans_pos_ids)
                            q_ans = _apply_rotary_pos_emb_q(q_ans, cos.to(q_ans.dtype), sin.to(q_ans.dtype))
                        except Exception:
                            pass

                        attn_w = compute_attention_scores(q_ans, K_cot)
                        I = torch.softmax(
                            attn_w[:, :, :, :-w], dim=-1
                        ).mean(dim=-2)
                        I = F.max_pool1d(I.float(), kernel_size=7, padding=3, stride=1).to(q_ans.dtype)

                        R = cal_similarity(K_cot)[:, :-w].unsqueeze(0)

                        Z = I * self.mix_lambda - R * (1 - self.mix_lambda)

                        budget = max(M - w, 1)
                        idx = Z.topk(budget, dim=-1).indices
                        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, d)
                        K_comp = K_cot[:, :, :-w, :].gather(dim=2, index=idx_exp)
                        V_comp = V_cot[:, :, :-w, :].gather(dim=2, index=idx_exp)
                        K_b = torch.cat([K_comp, K_cot[:, :, -w:, :]], dim=2)
                        V_b = torch.cat([V_comp, V_cot[:, :, -w:, :]], dim=2)

                        if K_b.shape[2] > M:
                            K_b = K_b[:, :, :M, :]
                            V_b = V_b[:, :, :M, :]
                        elif K_b.shape[2] < M:
                            pad = M - K_b.shape[2]
                            K_b = F.pad(K_b, (0, 0, 0, pad))
                            V_b = F.pad(V_b, (0, 0, 0, pad))
                    else:
                        R = cal_similarity(K_cot)[:, :-w].unsqueeze(0)
                        Z = -R
                        budget = min(M - w, Z.shape[-1])
                        idx = Z.topk(budget, dim=-1).indices
                        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, d)
                        K_b = torch.cat(
                            [K_cot[:, :, :-w, :].gather(dim=2, index=idx_exp), K_cot[:, :, -w:, :]],
                            dim=2,
                        )
                        V_b = torch.cat(
                            [V_cot[:, :, :-w, :].gather(dim=2, index=idx_exp), V_cot[:, :, -w:, :]],
                            dim=2,
                        )
                        if K_b.shape[2] > M:
                            K_b = K_b[:, :, :M, :]
                            V_b = V_b[:, :, :M, :]
                        elif K_b.shape[2] < M:
                            pad = M - K_b.shape[2]
                            K_b = F.pad(K_b, (0, 0, 0, pad))
                            V_b = F.pad(V_b, (0, 0, 0, pad))

                K_comp_list.append(K_b)
                V_comp_list.append(V_b)

                if l == num_layers - 1 and sample_positions is not None:
                    if idx is None:
                        sample_positions.append(None)
                    else:
                        _non_win = idx[0, 0, :].long().cpu()
                        _win = torch.arange(NC - w, NC, dtype=torch.long)
                        _pos = torch.cat([_non_win, _win])[:M]
                        sample_positions.append(_pos)

            K_comp = torch.cat(K_comp_list, dim=0).to(K_full.dtype)
            V_comp = torch.cat(V_comp_list, dim=0).to(V_full.dtype)
            compressed.append((K_comp, V_comp))

        if return_indices:
            return compressed, sample_positions
        return compressed

    def _kv_distill_loss(self, student_latent_kvs, teacher_kvs_compressed):
        """
        L_KV = mean_over_layers [(||sg[K_t] - K_s||_p + ||sg[V_t] - V_s||_p) / 2]
        Follows KAVA paper Eq. 7.
        """
        total = torch.tensor(0.0, device=student_latent_kvs[0][0].device)
        num_layers = len(student_latent_kvs)
        p = self.kv_loss_p

        for l in range(num_layers):
            K_s, V_s = student_latent_kvs[l]
            K_t, V_t = teacher_kvs_compressed[l]

            K_s_f = K_s.float()
            V_s_f = V_s.float()
            K_t_f = K_t.float().detach()
            V_t_f = V_t.float().detach()

            if p == 1:
                loss_K = F.l1_loss(K_s_f, K_t_f)
                loss_V = F.l1_loss(V_s_f, V_t_f)
            else:
                loss_K = F.mse_loss(K_s_f, K_t_f)
                loss_V = F.mse_loss(V_s_f, V_t_f)

            total = total + (loss_K + loss_V) / 2

        return total / num_layers

    @staticmethod
    def _align_kv_to_M(kv_list, M):
        aligned = []
        for K, V in kv_list:
            n = K.shape[2]
            if n == M:
                aligned.append((K, V))
            elif n < M:
                K = F.pad(K, (0, 0, 0, M - n))
                V = F.pad(V, (0, 0, 0, M - n))
                aligned.append((K, V))
            else:
                aligned.append((K[:, :, :M, :], V[:, :, :M, :]))
        return aligned

    @torch.no_grad()
    def _compress_teacher_kv_cyclic(
        self, ref_past_kvs, ref_hidden_states, enc_lengths,
        primitive_token_ranges, tokens_per_primitive, dec_lengths=None,
    ):
        """
        Scheme A — Cyclic R-KV: compress each ptool segment independently.
        Returns list[(K,V)] each [B, H_kv, NUM_PRIM*tokens_per_primitive, head_dim].
        """
        B = enc_lengths.shape[0]
        ref_model = self.codi
        base, config, H_kv, head_dim, num_layers = _get_kv_config(ref_model)
        w = self.kv_window_size
        K_per = tokens_per_primitive
        NUM_PRIM = primitive_token_ranges.shape[1]

        compressed = []
        for l in range(num_layers):
            K_full = ref_past_kvs.key_cache[l]
            V_full = ref_past_kvs.value_cache[l]
            d = K_full.shape[3]

            batch_K, batch_V = [], []
            for b in range(B):
                dec_l = dec_lengths[b].item() if dec_lengths is not None else 1
                seg_K_list, seg_V_list = [], []

                for i in range(NUM_PRIM):
                    seg_s = primitive_token_ranges[b, i, 0].item()
                    seg_e = primitive_token_ranges[b, i, 1].item()

                    if seg_s < 0 or seg_e <= seg_s:
                        seg_K_list.append(torch.zeros(1, H_kv, K_per, d, device=K_full.device, dtype=K_full.dtype))
                        seg_V_list.append(torch.zeros(1, H_kv, K_per, d, device=V_full.device, dtype=V_full.dtype))
                        continue

                    K_seg = K_full[b:b+1, :, seg_s:seg_e, :]
                    V_seg = V_full[b:b+1, :, seg_s:seg_e, :]
                    NC = K_seg.shape[2]

                    if NC <= K_per:
                        K_b = F.pad(K_seg, (0, 0, 0, K_per - NC))
                        V_b = F.pad(V_seg, (0, 0, 0, K_per - NC))
                    else:
                        ref_total = ref_hidden_states[l + 1].shape[1]
                        ans_start = seg_e
                        ans_end = min(ans_start + dec_l, ref_total)
                        NA = ans_end - ans_start
                        w_eff = min(w, NC - 1) if NC > 1 else 0

                        if NA > 0:
                            h_ans = ref_hidden_states[l + 1][b:b+1, ans_start:ans_end, :]
                            attn_layer = base.model.layers[l].self_attn
                            num_q_heads = getattr(attn_layer, 'num_heads', config.num_attention_heads)
                            q_raw = attn_layer.q_proj(h_ans).view(1, NA, num_q_heads, d)
                            if hasattr(attn_layer, 'q_norm'):
                                q_raw = attn_layer.q_norm(q_raw)
                            q_ans = q_raw.transpose(1, 2)
                            try:
                                ans_pos_ids = torch.arange(ans_start, ans_end, device=h_ans.device).unsqueeze(0)
                                cos, sin = base.model.rotary_emb(h_ans, ans_pos_ids)
                                q_ans = _apply_rotary_pos_emb_q(q_ans, cos.to(q_ans.dtype), sin.to(q_ans.dtype))
                            except Exception:
                                pass
                            if w_eff > 0:
                                I = torch.softmax(
                                    compute_attention_scores(q_ans, K_seg)[:, :, :, :-w_eff], dim=-1
                                ).mean(dim=-2)
                                I = F.max_pool1d(I.float(), kernel_size=7, padding=3, stride=1).to(q_ans.dtype)
                                R = cal_similarity(K_seg)[:, :-w_eff].unsqueeze(0)
                            else:
                                I = torch.softmax(
                                    compute_attention_scores(q_ans, K_seg), dim=-1
                                ).mean(dim=-2)
                                I = F.max_pool1d(I.float(), kernel_size=7, padding=3, stride=1).to(q_ans.dtype)
                                R = cal_similarity(K_seg).unsqueeze(0)
                        else:
                            R = cal_similarity(K_seg)
                            R = (R[:, :-w_eff] if w_eff > 0 else R).unsqueeze(0)
                            I = torch.zeros_like(R)

                        Z = I * self.mix_lambda - R * (1 - self.mix_lambda)
                        budget_sel = max(K_per - w_eff, 1)
                        idx = Z.topk(min(budget_sel, Z.shape[-1]), dim=-1).indices
                        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, d)

                        if w_eff > 0:
                            K_b = torch.cat([
                                K_seg[:, :, :-w_eff, :].gather(2, idx_exp),
                                K_seg[:, :, -w_eff:, :],
                            ], dim=2)
                            V_b = torch.cat([
                                V_seg[:, :, :-w_eff, :].gather(2, idx_exp),
                                V_seg[:, :, -w_eff:, :],
                            ], dim=2)
                        else:
                            K_b = K_seg.gather(2, idx_exp)
                            V_b = V_seg.gather(2, idx_exp)

                        if K_b.shape[2] > K_per:
                            K_b, V_b = K_b[:, :, :K_per, :], V_b[:, :, :K_per, :]
                        elif K_b.shape[2] < K_per:
                            K_b = F.pad(K_b, (0, 0, 0, K_per - K_b.shape[2]))
                            V_b = F.pad(V_b, (0, 0, 0, K_per - V_b.shape[2]))

                    seg_K_list.append(K_b)
                    seg_V_list.append(V_b)

                batch_K.append(torch.cat(seg_K_list, dim=2))
                batch_V.append(torch.cat(seg_V_list, dim=2))

            compressed.append((
                torch.cat(batch_K, dim=0).to(K_full.dtype),
                torch.cat(batch_V, dim=0).to(V_full.dtype),
            ))

        return compressed

    @torch.no_grad()
    def _extract_identifier_kvs(self, ref_past_kvs, identifier_positions):
        """
        Scheme B — Identifier 1:1 KV extraction.
        identifier_positions: [B, M] — token positions of <ptool_end_i> tokens in ref sequence.
        Returns list[(K,V)] each [B, H_kv, M, head_dim].
        """
        ref_model = self.codi
        _, _, H_kv, head_dim, num_layers = _get_kv_config(ref_model)
        B, M = identifier_positions.shape

        compressed = []
        for l in range(num_layers):
            K_full = ref_past_kvs.key_cache[l]
            V_full = ref_past_kvs.value_cache[l]
            pos = identifier_positions.to(K_full.device)
            pos_exp = pos.unsqueeze(1).unsqueeze(-1).expand(B, H_kv, M, head_dim)
            compressed.append((K_full.gather(2, pos_exp), V_full.gather(2, pos_exp)))

        return compressed

    @torch.no_grad()
    def latent_generate(
        self,
        encoder_input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor = None,
        eot_token_id: int = None,  # defaults to self.eot_id inside method
        max_new_tokens: int = 30,
        num_beams: int = 10,
        num_return_sequences: int = 10,
        prefix_allowed_tokens_fn=None,
    ):
        """
        Inference: T Jacobi iterations over [enc | latent | eot], then
        HF generate() with inputs_embeds for constrained beam search.
        """
        B = encoder_input_ids.shape[0]
        enc_seq_len = encoder_input_ids.shape[1]
        M = self.num_latent
        T = self.num_jacobi_steps
        device = encoder_input_ids.device

        if eot_token_id is None:
            eot_token_id = self.eot_id

        embed_fn = get_embed_layer(self.codi)

        if M == 0:
            enc_embeds = embed_fn(encoder_input_ids)
            enc_mask = encoder_attention_mask if encoder_attention_mask is not None \
                else torch.ones(B, enc_seq_len, device=device, dtype=torch.long)
            parts = [enc_embeds]
            mask_parts = [enc_mask]
            if eot_token_id is not None:
                eot_ids = torch.full((B, 1), eot_token_id, device=device, dtype=torch.long)
                parts.append(embed_fn(eot_ids))
                mask_parts.append(torch.ones(B, 1, device=device, dtype=torch.long))
            gen_embeds = torch.cat(parts, dim=1)
            gen_mask = torch.cat(mask_parts, dim=1)
            generate_kwargs = dict(
                inputs_embeds=gen_embeds,
                attention_mask=gen_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
            )
            if num_beams > 1:
                generate_kwargs["early_stopping"] = True
            if prefix_allowed_tokens_fn is not None:
                generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
            return self.codi.generate(**generate_kwargs)

        eot_embed = None
        if eot_token_id is not None:
            eot_ids = torch.full((B, 1), eot_token_id, device=device, dtype=torch.long)
            eot_embed = embed_fn(eot_ids)

        init_raw = torch.zeros(B, M, self.dim, device=device, dtype=torch.bfloat16)
        if self.primitive_slot_embeddings is not None and M == self.primitive_slot_embeddings.shape[0]:
            init_raw = self.primitive_slot_embeddings.unsqueeze(0).expand(B, -1, -1).to(init_raw.dtype)
        if self.use_prj:
            latent_embeds = self.prj(init_raw)
        else:
            latent_embeds = init_raw

        enc_mask = encoder_attention_mask if encoder_attention_mask is not None \
            else torch.ones(B, enc_seq_len, device=device, dtype=torch.long)

        for t in range(T):
            enc_embeds = embed_fn(encoder_input_ids)
            parts = [enc_embeds, latent_embeds]
            mask_parts = [enc_mask, torch.ones(B, M, device=device, dtype=torch.long)]
            if eot_embed is not None:
                parts.append(eot_embed)
                mask_parts.append(torch.ones(B, 1, device=device, dtype=torch.long))
            full_embeds = torch.cat(parts, dim=1)
            full_mask = torch.cat(mask_parts, dim=1)
            total_len = full_embeds.shape[1]
            full_pos = torch.arange(total_len, device=device).unsqueeze(0).expand(B, -1)

            jac_out = self.codi(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                position_ids=full_pos,
                output_hidden_states=True,
            )
            raw_latent = jac_out.hidden_states[-1][:, enc_seq_len:enc_seq_len + M, :]
            if self.use_prj:
                latent_embeds = self.prj(raw_latent)
            else:
                latent_embeds = raw_latent

        gen_embeds = full_embeds
        gen_mask = full_mask

        generate_kwargs = dict(
            inputs_embeds=gen_embeds,
            attention_mask=gen_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
        if num_beams > 1:
            generate_kwargs["early_stopping"] = True
        if prefix_allowed_tokens_fn is not None:
            generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        return self.codi.generate(**generate_kwargs)

    def forward_opd(
        self,
        encoder_input_ids,
        decoder_input_ids,
        labels,
        yr_cot_ids,
        encoder_attention_mask=None,
        yr_cot_attention_mask=None,
        enc_lengths=None,
        cot_lengths=None,
        dec_lengths=None,
        model_answer_position=None,
        primitive_token_ranges=None,
        use_ppo=False,
        old_log_prob=None,
        advantages=None,
        clip_ratio=0.2,
        ppo_coef=1.0,
        opd_coef=1.0,
        **kwargs,
    ):
        """
        TRD-style OPD forward (aligned with SEED actor update):
        1. Student: [enc | latent(M) | dec] -> Jacobi -> SID logits, CE loss
        2. Teacher: [enc | latent(M).detach() | refine_prompt | yr_cot | dec] -> SID logits (no_grad)
        3. Forward KL (MC) on SID tokens
        4. Primitive distill loss (rkv_cyclic KV alignment)
        5. If use_ppo: PPO clipped loss + SEED OPD gated loss

        When use_ppo=True, the total loss becomes:
          loss = ppo_coef * pg_loss + opd_coef * opd_gated_loss + primitive_distill_loss
                 + ce_loss + div_loss
        When use_ppo=False (legacy):
          loss = ce_loss + opd_kl_factor * forward_kl_loss + div_loss
        """
        B = encoder_input_ids.shape[0]
        enc_seq_len = encoder_input_ids.shape[1]
        M = self.num_latent
        T = self.num_jacobi_steps
        device = encoder_input_ids.device

        if "qwen" in self.model_name.lower():
            model_answer_position = model_answer_position + 1
        model_answer_position = model_answer_position - 1

        embed_fn = get_embed_layer(self.codi)
        enc_embeds = embed_fn(encoder_input_ids)
        dec_embeds = embed_fn(decoder_input_ids)
        dec_len = dec_embeds.shape[1]

        init_raw = torch.zeros(B, M, self.dim, device=device, dtype=enc_embeds.dtype)
        if self.primitive_slot_embeddings is not None and M == self.primitive_slot_embeddings.shape[0]:
            init_raw = self.primitive_slot_embeddings.unsqueeze(0).expand(B, -1, -1).to(init_raw.dtype)
        if self.use_prj:
            latent_embeds = self.prj(init_raw)
        else:
            latent_embeds = init_raw

        enc_mask = encoder_attention_mask if encoder_attention_mask is not None \
            else torch.ones(B, enc_seq_len, device=device, dtype=torch.long)

        full_mask = torch.cat([
            enc_mask,
            torch.ones(B, M + dec_len, device=device, dtype=torch.long),
        ], dim=1)

        full_pos = torch.cat([
            torch.arange(enc_seq_len, device=device).unsqueeze(0).expand(B, -1),
            torch.arange(enc_seq_len, enc_seq_len + M + dec_len, device=device).unsqueeze(0).expand(B, -1),
        ], dim=1)

        logits = None
        ce_loss = torch.tensor(0.0, device=device)
        div_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
        lat_ev1 = 0.0
        lat_eff_rank = 0.0
        latent_embeds_final = None

        student_latent_kvs = []
        primitive_distill_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
        teacher_past_kvs = None

        for t in range(T):
            is_last = (t == T - 1)
            full_embeds = torch.cat([enc_embeds, latent_embeds, dec_embeds], dim=1)

            if not is_last:
                with torch.no_grad():
                    jac_out = self.codi(
                        inputs_embeds=full_embeds,
                        attention_mask=full_mask,
                        position_ids=full_pos,
                        output_hidden_states=True,
                    )
                    raw_latent = jac_out.hidden_states[-1][:, enc_seq_len:enc_seq_len + M, :]
                if self.use_prj:
                    latent_embeds = self.prj(raw_latent)
                else:
                    latent_embeds = raw_latent
            else:
                use_cache_student = self.use_primitive_distill and primitive_token_ranges is not None
                jac_out = self.codi(
                    inputs_embeds=full_embeds,
                    attention_mask=full_mask,
                    position_ids=full_pos,
                    use_cache=use_cache_student,
                    output_hidden_states=True,
                )

                if use_cache_student:
                    student_latent_kvs = [
                        (
                            jac_out.past_key_values.key_cache[l][:, :, enc_seq_len:enc_seq_len + M, :],
                            jac_out.past_key_values.value_cache[l][:, :, enc_seq_len:enc_seq_len + M, :],
                        )
                        for l in range(len(jac_out.past_key_values.key_cache))
                    ]

                logits = jac_out.logits
                dec_start = enc_seq_len + M
                effective_logits = logits[:, dec_start:dec_start + dec_len - 1, :].reshape(-1, logits.size(-1))
                target_ids = labels[:, 1:].reshape(-1)
                ce_loss = self.loss_fct(effective_logits, target_ids)

                latent_embeds_final = jac_out.hidden_states[-1][:, enc_seq_len:enc_seq_len + M, :]
                if self.use_prj:
                    latent_embeds_final = self.prj(latent_embeds_final)

                if self.div_factor != 0 and M > 1:
                    _lat_h = jac_out.hidden_states[-1][:, enc_seq_len:enc_seq_len + M, :].float()
                    _lat_h_norm = F.normalize(_lat_h, dim=-1)
                    _lat_gram = torch.bmm(_lat_h_norm, _lat_h_norm.transpose(-1, -2))
                    _lat_off_diag = _lat_gram * (1 - torch.eye(M, device=device).unsqueeze(0))
                    div_loss = F.relu(_lat_off_diag - 0.3).pow(2).mean().to(torch.bfloat16)

                    with torch.no_grad():
                        _eigs = torch.linalg.eigvalsh(_lat_gram.float()).clamp(min=0)
                        _eigs_sum = _eigs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        _eigs_norm = _eigs / _eigs_sum
                        lat_ev1 = _eigs_norm[:, -1].mean().item()
                        lat_eff_rank = torch.exp(
                            -(_eigs_norm * (_eigs_norm + 1e-8).log()).sum(dim=-1)
                        ).mean().item()

                if self.use_primitive_distill and primitive_token_ranges is not None and student_latent_kvs:
                    num_prims = primitive_token_ranges.shape[1] if primitive_token_ranges.numel() > 0 else 0
                    if num_prims > 0:
                        pass  # computed after teacher forward below

        # ── Teacher forward (based on student latent + refine prompt + yr_cot) ──
        use_cache_teacher = self.use_primitive_distill and primitive_token_ranges is not None

        refine_ids = self.tokenizer(REFINE_PROMPT, add_special_tokens=False)["input_ids"]
        refine_len = len(refine_ids)
        refine_ids_t = torch.tensor([refine_ids] * B, device=device, dtype=torch.long)
        refine_embeds = embed_fn(refine_ids_t)

        cot_embeds = embed_fn(yr_cot_ids)
        cot_mask = yr_cot_attention_mask if yr_cot_attention_mask is not None \
            else yr_cot_ids.ne(self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            teacher_embeds = torch.cat([
                enc_embeds,
                latent_embeds_final.detach(),
                refine_embeds,
                cot_embeds,
                dec_embeds,
            ], dim=1)

            teacher_mask = torch.cat([
                enc_mask,
                torch.ones(B, M + refine_len, device=device, dtype=torch.long),
                cot_mask,
                torch.ones(B, dec_len, device=device, dtype=torch.long),
            ], dim=1)

            total_teacher_len = teacher_embeds.shape[1]
            teacher_pos = torch.arange(total_teacher_len, device=device).unsqueeze(0).expand(B, -1)

            teacher_out = self.codi(
                inputs_embeds=teacher_embeds,
                attention_mask=teacher_mask,
                position_ids=teacher_pos,
                use_cache=use_cache_teacher,
                output_hidden_states=True,
            )

            if use_cache_teacher:
                teacher_past_kvs = teacher_out.past_key_values
                teacher_hidden_states = teacher_out.hidden_states

        # ── Primitive distill loss (after teacher forward has KV cache) ──
        if self.use_primitive_distill and primitive_token_ranges is not None and student_latent_kvs and teacher_past_kvs is not None:
            num_prims = primitive_token_ranges.shape[1] if primitive_token_ranges.numel() > 0 else 0
            if num_prims > 0:
                try:
                    abs_ranges = primitive_token_ranges.clone()
                    for b in range(B):
                        offset = enc_lengths[b].item() + M + refine_len
                        mask_b = abs_ranges[b, :, 0] >= 0
                        abs_ranges[b, mask_b, 0] += offset
                        abs_ranges[b, mask_b, 1] += offset
                    _prim_kvs = self._compress_teacher_kv_cyclic(
                        teacher_past_kvs,
                        teacher_hidden_states,
                        enc_lengths,
                        abs_ranges,
                        self.tokens_per_primitive,
                        dec_lengths=dec_lengths,
                    )
                    _prim_kvs = self._align_kv_to_M(_prim_kvs, M)
                    primitive_distill_loss = self._kv_distill_loss(student_latent_kvs, _prim_kvs).to(torch.bfloat16)
                except Exception as e:
                    print(f"  [WARN] primitive distill failed in forward_opd: {e}")

        # ── Extract student/teacher log probs ──
        student_log_prob = self._get_student_token_log_probs(logits, labels, enc_seq_len, M)

        teacher_log_prob = self._get_teacher_sid_log_probs_opd(
            teacher_out.logits, labels, enc_seq_len, M, refine_len, cot_lengths
        )

        response_mask = (labels[:, 1:] != -100).float()

        # ── Loss computation ──
        pg_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
        opd_gated_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
        forward_kl_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)

        if use_ppo and old_log_prob is not None and advantages is not None:
            # ── PPO clipped loss (SEED compute_policy_loss) ──
            ratio = torch.exp(student_log_prob - old_log_prob)
            pg_losses1 = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            pg_losses = torch.max(pg_losses1, pg_losses2)
            pg_loss = agg_loss_opd(pg_losses, response_mask, self.opd_loss_agg_mode).to(torch.bfloat16)

            # ── SEED OPD gated loss (confidence-gated distillation) ──
            opd_gated_loss, _, _, _, _ = compute_opd_loss(
                log_prob=student_log_prob,
                teacher_log_prob=teacher_log_prob,
                response_mask=response_mask,
                gate_beta=self.opd_gate_beta,
                loss_agg_mode=self.opd_loss_agg_mode,
            )
            opd_gated_loss = opd_gated_loss.to(torch.bfloat16)

            loss = (
                ppo_coef * pg_loss
                + opd_coef * opd_gated_loss
                + self.primitive_distill_factor * primitive_distill_loss
                + ce_loss
                + self.div_factor * div_loss
            )
        else:
            # ── Legacy: Forward KL (MC) on SID tokens ──
            forward_kl_loss = compute_forward_kl_mc(
                teacher_log_prob=teacher_log_prob,
                student_log_prob=student_log_prob,
                response_mask=response_mask,
                loss_agg_mode=self.opd_loss_agg_mode,
                kl_token_clip=self.kl_token_clip,
            )
            forward_kl_loss = forward_kl_loss.to(torch.bfloat16)

            loss = (
                ce_loss
                + self.opd_kl_factor * forward_kl_loss
                + self.primitive_distill_factor * primitive_distill_loss
                + self.div_factor * div_loss
            )

        return {
            "loss": loss,
            "logits": logits,
            "student_log_prob": student_log_prob.detach(),
            "teacher_log_prob": teacher_log_prob.detach(),
            "response_mask": response_mask.detach(),
            "ce_loss": ce_loss.detach().item() if torch.is_tensor(ce_loss) else 0,
            "opd_loss": (opd_gated_loss if use_ppo else forward_kl_loss).detach().item() if torch.is_tensor(opd_gated_loss if use_ppo else forward_kl_loss) else 0,
            "pg_loss": pg_loss.detach().item() if torch.is_tensor(pg_loss) and use_ppo else 0,
            "primitive_distill_loss": (primitive_distill_loss * self.primitive_distill_factor).detach().item() if torch.is_tensor(primitive_distill_loss) else 0,
            "div_loss": div_loss.detach().item() if torch.is_tensor(div_loss) else 0,
            "lat_ev1": lat_ev1,
            "lat_eff_rank": lat_eff_rank,
        }

    def _get_teacher_sid_log_probs_opd(
        self,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        enc_seq_len: int,
        M: int,
        refine_len: int,
        cot_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        From teacher logits in forward_opd, extract SID token log probs.
        Teacher sequence: [enc | latent(M) | refine_prompt | CoT | dec]
        SID starts at enc_seq_len + M + refine_len + cot_lengths[b].
        """
        B = teacher_logits.shape[0]
        dec_len = labels.shape[1]
        result = torch.zeros(B, dec_len - 1, device=teacher_logits.device, dtype=teacher_logits.dtype)
        for b in range(B):
            cot_l = cot_lengths[b].item()
            t_start = enc_seq_len + M + refine_len + cot_l
            t_end = min(t_start + dec_len - 1, teacher_logits.shape[1])
            avail_len = t_end - t_start
            if avail_len <= 0:
                continue
            t_logits = teacher_logits[b, t_start:t_end, :]
            t_log_probs = F.log_softmax(t_logits.float(), dim=-1)
            targets_b = labels[b, 1:avail_len + 1].clamp(min=0)
            token_lp = t_log_probs.gather(-1, targets_b.unsqueeze(-1)).squeeze(-1)
            result[b, :avail_len] = token_lp.to(result.dtype)
        return result

    @torch.no_grad()
    def latent_refine_generate(
        self,
        encoder_input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor = None,
        eot_token_id: int = None,
        max_new_tokens: int = 512,
        num_beams: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        prefix_allowed_tokens_fn=None,
    ):
        """
        Teacher refine generate based on student latent + refine prompt.
        1. Jacobi iterations -> latent_embeds
        2. Concat [enc | latent | refine_prompt | eot] -> generate CoT + SID
        """
        B = encoder_input_ids.shape[0]
        enc_seq_len = encoder_input_ids.shape[1]
        M = self.num_latent
        T = self.num_jacobi_steps
        device = encoder_input_ids.device

        if eot_token_id is None:
            eot_token_id = self.eot_id

        embed_fn = get_embed_layer(self.codi)
        enc_embeds = embed_fn(encoder_input_ids)

        eot_embed = None
        if eot_token_id is not None:
            eot_ids = torch.full((B, 1), eot_token_id, device=device, dtype=torch.long)
            eot_embed = embed_fn(eot_ids)

        init_raw = torch.zeros(B, M, self.dim, device=device, dtype=torch.bfloat16)
        if self.primitive_slot_embeddings is not None and M == self.primitive_slot_embeddings.shape[0]:
            init_raw = self.primitive_slot_embeddings.unsqueeze(0).expand(B, -1, -1).to(init_raw.dtype)
        if self.use_prj:
            latent_embeds = self.prj(init_raw)
        else:
            latent_embeds = init_raw

        enc_mask = encoder_attention_mask if encoder_attention_mask is not None \
            else torch.ones(B, enc_seq_len, device=device, dtype=torch.long)

        for t in range(T):
            enc_embeds_t = embed_fn(encoder_input_ids)
            parts = [enc_embeds_t, latent_embeds]
            mask_parts = [enc_mask, torch.ones(B, M, device=device, dtype=torch.long)]
            if eot_embed is not None:
                parts.append(eot_embed)
                mask_parts.append(torch.ones(B, 1, device=device, dtype=torch.long))
            full_embeds = torch.cat(parts, dim=1)
            full_mask = torch.cat(mask_parts, dim=1)
            total_len = full_embeds.shape[1]
            full_pos = torch.arange(total_len, device=device).unsqueeze(0).expand(B, -1)

            jac_out = self.codi(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                position_ids=full_pos,
                output_hidden_states=True,
            )
            raw_latent = jac_out.hidden_states[-1][:, enc_seq_len:enc_seq_len + M, :]
            if self.use_prj:
                latent_embeds = self.prj(raw_latent)
            else:
                latent_embeds = raw_latent

        refine_ids = self.tokenizer(REFINE_PROMPT, add_special_tokens=False)["input_ids"]
        refine_ids_t = torch.tensor([refine_ids] * B, device=device, dtype=torch.long)
        refine_embeds = embed_fn(refine_ids_t)
        refine_len = refine_embeds.shape[1]

        gen_parts = [enc_embeds, latent_embeds, refine_embeds]
        gen_mask_parts = [enc_mask, torch.ones(B, M, device=device, dtype=torch.long),
                          torch.ones(B, refine_len, device=device, dtype=torch.long)]
        if eot_embed is not None:
            gen_parts.append(eot_embed)
            gen_mask_parts.append(torch.ones(B, 1, device=device, dtype=torch.long))

        gen_embeds = torch.cat(gen_parts, dim=1)
        gen_mask = torch.cat(gen_mask_parts, dim=1)

        generate_kwargs = dict(
            inputs_embeds=gen_embeds,
            attention_mask=gen_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
        if num_beams > 1:
            generate_kwargs["early_stopping"] = True
        if prefix_allowed_tokens_fn is not None:
            generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

        return self.codi.generate(**generate_kwargs)
