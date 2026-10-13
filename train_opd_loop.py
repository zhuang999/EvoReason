#!/usr/bin/env python3
"""On-Policy OPD Training Loop — aligned with SEED's fit() method.

Unifies generate_yo, generate_yr, PPO+OPD training, and ptool evolution
into a single DeepSpeed training loop.

Reference: SEED-main/verl/trainer/ppo/ray_trainer.py fit()

Training loss (aligned with SEED actor update):
  total_loss = ppo_coef * pg_loss + opd_coef * opd_gated_loss
               + primitive_distill_factor * primitive_distill_loss
               + ce_loss + div_factor * div_loss

Usage:
    WANDB_MODE=disabled deepspeed --num_gpus=8 train_opd_loop.py \
        --model_name_or_path /path/to/beauty_RA_cot/epoch_3 \
        --data_path /path/to/training_RA_cot_train.parquet \
        --output_dir checkpoints/opd_loop \
        --report_path reasoning_primitives/benchmarks/rec/induction_out/original/rec/report.json
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_MODE", "disabled")

import argparse
import datetime
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from model_codi import LatentRecCoDi
from dataset_codi import SYSTEM_MESSAGE, THINK_START, THINK_END
from dataset_opd import LatentRecOPDDataset, LatentRecOPDCollator
from core_algos import (
    compute_opd_loss,
    compute_forward_kl_mc,
    agg_loss,
)
from experiment_tracking import ExperimentTracker

SID_RE_LOOP = re.compile(r'<\|sid_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)><s_d_(\d+)><\|sid_end\|>')


def _parse_sid_tuple(sid_str):
    """Return 4-tuple(a,b,c,d) for SID match count, or None if invalid."""
    if not sid_str:
        return None
    m = SID_RE_LOOP.search(sid_str)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))


def _count_matching_sid(pred_sid, gt_sid):
    """Count how many SID tokens (a,b,c,d) match between pred and ground truth."""
    pt = _parse_sid_tuple(pred_sid)
    gt = _parse_sid_tuple(gt_sid)
    if pt is None or gt is None:
        return 0
    return sum(1 for p, g in zip(pt, gt) if p == g)


def build_sid_trie(tokenizer):
    """
    Efficient SID prefix trie for beam search.
    
    SID format: <|sid_begin|> <s_a_X> <s_b_Y> <s_c_Z> <s_d_W> <|sid_end|>
    
    Instead of building 256^4 entries, we use a step-based approach:
    - step 0 (just <|sid_begin|>): only s_a tokens allowed
    - step 1 (after <s_a_X>): only s_b tokens allowed
    - step 2 (after <s_b_Y>): only s_c tokens allowed
    - step 3 (after <s_c_Z>): only s_d tokens allowed
    - step 4 (after <s_d_W>): only <|sid_end|> allowed
    """
    sid_begin_id = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
    sid_end_id = tokenizer.convert_tokens_to_ids("<|sid_end|>")
    s_a_ids = {tokenizer.convert_tokens_to_ids(f"<s_a_{i}>") for i in range(256)}
    s_b_ids = {tokenizer.convert_tokens_to_ids(f"<s_b_{i}>") for i in range(256)}
    s_c_ids = {tokenizer.convert_tokens_to_ids(f"<s_c_{i}>") for i in range(256)}
    s_d_ids = {tokenizer.convert_tokens_to_ids(f"<s_d_{i}>") for i in range(256)}

    # Remove unk token IDs (tokens not in vocab)
    unk_id = tokenizer.unk_token_id
    allowed_sets = [
        s_a_ids - {unk_id},
        s_b_ids - {unk_id},
        s_c_ids - {unk_id},
        s_d_ids - {unk_id},
        {sid_end_id},
    ]
    sid_token_ids = {sid_begin_id, sid_end_id} | s_a_ids | s_b_ids | s_c_ids | s_d_ids

    class TriePrefixAllowedTokens:
        def __init__(self, allowed_sets, sid_begin_id, sid_token_ids, batch_size):
            self.allowed_sets = allowed_sets
            self.sid_begin_id = sid_begin_id
            self.sid_token_ids = sid_token_ids
            self.batch_size = batch_size

        def __call__(self, batch_id, input_ids):
            # Count how many SID tokens already generated
            sid_count = 0
            for tid in input_ids.tolist():
                if tid == self.sid_begin_id:
                    sid_count = 1
                elif sid_count > 0 and tid in self.sid_token_ids:
                    sid_count += 1
            # sid_count ranges from 0 to 5 (begin + a + b + c + d)
            # After begin(0), we need s_a (1), then s_b (2), etc.
            # After 5 sid tokens (begin+a+b+c+d), only end is allowed.
            step_idx = sid_count - 1  # 0->s_a, 1->s_b, ...
            if 0 <= step_idx < len(self.allowed_sets):
                return list(self.allowed_sets[step_idx])
            return list(self.allowed_sets[0]) if self.allowed_sets else []

    return TriePrefixAllowedTokens(allowed_sets, sid_begin_id, sid_token_ids, 1)


def extract_sid(token_ids, tokenizer):
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    match = re.search(r'<\|sid_begin\|>(.*?)<\|sid_end\|>', text)
    if match:
        return "<|sid_begin|>" + match.group(1) + "<|sid_end|>"
    return None


def _as_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "detach"):
        return float(value.detach().item())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class OPDTrainer:
    def __init__(self, args):
        self.args = args
        self.global_step = 0
        self.epoch = 0

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.world_size = world_size

        dist.init_process_group(
            "nccl",
            timeout=datetime.timedelta(seconds=7200),
            device_id=self.device,
        )

        self.model = LatentRecCoDi(
            model_name_or_path=args.model_name_or_path,
            num_latent=args.num_latent,
            num_jacobi_steps=args.num_jacobi_steps,
            use_prj=args.use_prj,
            prj_dim=args.prj_dim,
            kv_compress_mode=args.kv_compress_mode,
            use_primitive_distill=args.use_primitive_distill,
            opd_kl_factor=args.opd_kl_factor,
            opd_gate_beta=args.opd_gate_beta,
            opd_loss_agg_mode=args.opd_loss_agg_mode,
            div_factor=args.div_factor,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            tokens_per_primitive=args.tokens_per_primitive,
        )

        prj_path = os.path.join(args.model_name_or_path, "prj.pt")
        if os.path.exists(prj_path) and args.use_prj:
            self.model.prj.load_state_dict(
                torch.load(prj_path, map_location="cpu", weights_only=True)
            )

        self.tokenizer = self.model.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        self.trie_fn = build_sid_trie(self.tokenizer)

        self.data_collator = LatentRecOPDCollator(
            tokenizer=self.tokenizer,
            max_length=args.max_length,
            max_cot_length=args.max_cot_length,
            bot_id=self.model.bot_id,
            eot_id=self.model.eot_id,
        )

        self.base_df = pd.read_parquet(args.data_path)
        if args.max_samples:
            self.base_df = self.base_df.head(args.max_samples)

        self.report_path = args.report_path
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.ref_model = None
        if args.use_kl_penalty:
            self.ref_model = LatentRecCoDi(
                model_name_or_path=args.model_name_or_path, use_lora=False,
                num_latent=args.num_latent, num_jacobi_steps=args.num_jacobi_steps,
                use_prj=args.use_prj, prj_dim=args.prj_dim,
                kv_compress_mode="rkv_whole", use_primitive_distill=False,
                div_factor=0.0,
            )
            self.ref_model = self.ref_model.to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        self.tracker = None
        if self.local_rank == 0:
            run_name = args.run_name or Path(self.output_dir).name
            self.tracker = ExperimentTracker(
                output_dir=self.output_dir,
                run_name=run_name,
                method_name=args.method_name,
                moving_avg_window=args.moving_avg_window,
                auto_plot=args.auto_plot_dynamics,
            )

        num_update_steps_per_epoch = len(self.base_df) // (
            world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        total_num_steps = num_update_steps_per_epoch * args.num_epochs
        warmup_steps = int(total_num_steps * args.warmup_ratio)

        ds_config = {
            "bf16": {"enabled": args.bf16},
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
            },
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_clipping": args.max_grad_norm,
            "train_batch_size": world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": args.weight_decay,
                },
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0.0,
                    "warmup_max_lr": args.learning_rate,
                    "warmup_num_steps": warmup_steps,
                },
            },
            "zero_force_ds_cpu_optimizer": False,
        }

        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            config=ds_config,
        )

        if local_rank == 0:
            print(f"OPD Loop Trainer initialized")
            print(f"  model: {args.model_name_or_path}")
            print(f"  data: {len(self.base_df)} samples")
            print(f"  world_size: {world_size}")
            print(f"  total_steps: {total_num_steps}, warmup_steps: {warmup_steps}")
            print(f"  ppo_coef={args.ppo_coef}, opd_coef={args.opd_coef}")
            print(f"  clip_ratio={args.clip_ratio}")

    def fit(self):
        """On-policy OPD training loop — epoch-level generate then train (aligned with SEED)."""
        for epoch in range(self.args.start_epoch, self.args.num_epochs):
            self.epoch = epoch
            if self.local_rank == 0:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{self.args.num_epochs}")
                print(f"{'='*60}")

            records = self.base_df.to_dict("records")
            gen_bs = self.args.yo_batch_size

            # ── Phase 1: Generate yo/yr distributed across all ranks ──
            total = len(records)
            per_rank = (total + self.world_size - 1) // self.world_size
            start = self.local_rank * per_rank
            end = min(start + per_rank, total)
            my_records = records[start:end]

            all_yo, all_yr_cot, all_yr_sid = [], [], []
            if self.local_rank == 0:
                print(f"[Generate] Generating yo/yr for {total} samples ({per_rank}/rank)...")
            for idx in tqdm(range(0, len(my_records), gen_bs),
                           desc=f"gen-r{self.local_rank}",
                           disable=self.local_rank != 0):
                batch = my_records[idx: idx + gen_bs]
                yo_sid = self._gen_yo_batch(batch)
                yr_cot, yr_sid = self._gen_yr_batch(batch, yo_sid)
                all_yo.extend(yo_sid)
                all_yr_cot.extend(yr_cot)
                all_yr_sid.extend(yr_sid)

            # All-gather: each rank sends its chunk, then assemble in order
            gather_data = [None] * self.world_size
            my_chunk = [all_yo, all_yr_cot, all_yr_sid]
            dist.all_gather_object(gather_data, my_chunk)
            all_yo, all_yr_cot, all_yr_sid = [], [], []
            for r in range(self.world_size):
                r_start = r * per_rank
                r_end = min(r_start + per_rank, total)
                r_records = records[r_start:r_end]
                chunk = gather_data[r]
                expected = r_end - r_start
                all_yo.extend(chunk[0][:expected])
                all_yr_cot.extend(chunk[1][:expected])
                all_yr_sid.extend(chunk[2][:expected])

            # ── Phase 2: Train on all generated data ──
            if self.local_rank == 0:
                print(f"[Train] Training on {len(records)} samples...")

            all_sid_scores = torch.tensor([
                _count_matching_sid(all_yo[i], records[i]["groundtruth"])
                for i in range(len(records))
            ], device=self.device, dtype=torch.float32)

            _eps = 1e-6
            mu = all_sid_scores.mean()
            sigma = all_sid_scores.std(unbiased=False)
            if sigma < _eps:
                sigma = 1.0
            grpo_adv = (all_sid_scores - mu) / (sigma + _eps)

            train_bs = self.args.per_device_train_batch_size
            epoch_metrics = defaultdict(float)
            n_steps = 0

            for idx in tqdm(range(0, len(records), train_bs),
                           desc="train", disable=self.local_rank != 0):
                batch = records[idx: idx + train_bs]
                yo_sid = all_yo[idx: idx + train_bs]
                yr_cot = all_yr_cot[idx: idx + train_bs]
                yr_sid = all_yr_sid[idx: idx + train_bs]
                batch_grpo = grpo_adv[idx: idx + train_bs]

                step_metrics = self._train_on_batch(batch, yo_sid, yr_cot, yr_sid, batch_grpo)

                for k, v in step_metrics.items():
                    epoch_metrics[k] += v
                n_steps += 1
                self.global_step += 1

                if self.global_step % self.args.logging_steps == 0 and self.local_rank == 0:
                    avg = {k: v / max(n_steps, 1) for k, v in epoch_metrics.items()}
                    print(f"  step={self.global_step} loss={avg.get('loss', 0):.4f} "
                          f"ce={avg.get('ce_loss', 0):.4f} opd={avg.get('opd_loss', 0):.4f} "
                          f"pg={avg.get('pg_loss', 0):.4f} prim={avg.get('primitive_distill_loss', 0):.4f}")
                    self.tracker.maybe_plot_training_dynamics()

            # ── Save yo/yr results + checkpoint ──
            flat_yo = all_yo
            flat_yr_cot = all_yr_cot
            flat_yr_sid = all_yr_sid
            yo_df = self._build_yo_df(records, flat_yo)
            yr_df = self._build_yr_df(records, flat_yo, flat_yr_cot, flat_yr_sid)

            if self.local_rank == 0:
                yo_df.to_parquet(os.path.join(self.output_dir, f"epoch_{epoch}_yo.parquet"))
                yr_df.to_parquet(os.path.join(self.output_dir, f"epoch_{epoch}_yr.parquet"))
                print(f"\n--- save checkpoint ---")
                yo_acc = (yo_df["yo_sid"] == yo_df["groundtruth"]).mean()
                yr_acc = (yr_df["yr_sid"] == yr_df["groundtruth"]).mean()
                print(f"Epoch {epoch + 1}: yo_acc={yo_acc:.4f} yr_acc={yr_acc:.4f}")
                avg_m = {k: v / max(n_steps, 1) for k, v in epoch_metrics.items()}
                for k, v in avg_m.items():
                    print(f"  {k}: {v:.4f}")
            self._save_checkpoint(epoch)
            if self.local_rank == 0 and self.tracker is not None:
                self.tracker.maybe_plot_training_dynamics()

            dist.barrier(device_ids=[self.local_rank])

            if self.args.single_epoch:
                if self.local_rank == 0:
                    print(f"\nsingle_epoch mode: exiting after epoch {epoch}")
                break

    # ── per-batch helpers ──────────────────────────────────────

    def _gen_yo_batch(self, batch):
        """Generate yo_sid for a single batch of records."""
        model = self.model_engine.module
        model.eval()
        descriptions = [str(r["description"]) for r in batch]
        encoder_strs = [
            f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n{desc}<|im_end|>\n"
            f"<|im_start|>assistant\n{THINK_START}\n"
            for desc in descriptions
        ]
        enc = self.tokenizer(encoder_strs, return_tensors="pt", padding="longest",
                             truncation=True, max_length=2048, add_special_tokens=True)
        enc_ids = enc["input_ids"].to(self.device)
        enc_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            pred_ids = model.latent_generate(
                encoder_input_ids=enc_ids, encoder_attention_mask=enc_mask,
                max_new_tokens=30, num_beams=1, num_return_sequences=1,
                prefix_allowed_tokens_fn=self.trie_fn,
            )
        yo_sid = []
        for b_idx in range(len(batch)):
            seq = pred_ids[b_idx].cpu().tolist()
            yo_sid.append(extract_sid(seq, self.tokenizer) or "")
        return yo_sid

    def _gen_yr_batch(self, batch, yo_sid):
        """Generate yr_cot and yr_sid for a single batch."""
        model = self.model_engine.module
        model.eval()
        encoder_strs = []
        for r in batch:
            encoder_strs.append(
                f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
                f"<|im_start|>user\n{r['description']}<|im_end|>\n"
                f"<|im_start|>assistant\n{THINK_START}\n"
            )
        enc = self.tokenizer(encoder_strs, return_tensors="pt", padding="longest",
                             truncation=True, max_length=2048, add_special_tokens=True)
        enc_ids = enc["input_ids"].to(self.device)
        enc_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out_ids = model.latent_refine_generate(
                encoder_input_ids=enc_ids, encoder_attention_mask=enc_mask,
                max_new_tokens=self.args.max_cot_length,
                num_beams=1, temperature=0.7, top_p=0.95,
            )
        yr_cot, yr_sid = [], []
        for b_idx in range(len(batch)):
            raw = self.tokenizer.decode(out_ids[b_idx].cpu().tolist(), skip_special_tokens=False)
            m = re.search(r'<\|sid_begin\|>(.*?)<\|sid_end\|>', raw)
            sid = ("<|sid_begin|>" + m.group(1) + "<|sid_end|>") if m else ""
            cot_pat = re.escape(THINK_END) + r"\n"
            cm = re.search(r"(.*?)" + cot_pat, raw, re.DOTALL)
            cot = cm.group(1).strip() if cm else ""
            yr_cot.append(cot)
            yr_sid.append(sid)
        return yr_cot, yr_sid

    def _train_on_batch(self, batch, yo_sid, yr_cot, yr_sid, grpo_adv):
        """
        Collate → old_log_probs → advantages → PPO+OPD+primitive distill training.
        grpo_adv: pre-computed GRPO advantage for this batch (per-sample scalar).
        Returns dict of metrics for this step.
        """
        self.model_engine.train()

        coll_records = []
        for i, r in enumerate(batch):
            coll_records.append({
                "description": r["description"],
                "groundtruth": r["groundtruth"],
                "yo_sid": yo_sid[i],
                "yr_cot": yr_cot[i],
                "yr_sid": yr_sid[i],
            })

        model_inputs = self.data_collator(coll_records)
        model_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in model_inputs.items()}

        with torch.no_grad():
            O = self.model_engine.module.forward_opd(**model_inputs)
            old_slp = O["student_log_prob"]
            resp_mask = O["response_mask"]

        kl_val = torch.tensor(0.0, device=self.device)
        if self.ref_model is not None:
            with torch.no_grad():
                ref_out = self.ref_model.forward_opd(**model_inputs)
                ref_slp = ref_out["student_log_prob"]
                ref_lp = (ref_slp * resp_mask).sum(dim=-1) / resp_mask.sum(dim=-1).clamp(min=1)
            old_lp_scalar = (old_slp * resp_mask).sum(dim=-1) / resp_mask.sum(dim=-1).clamp(min=1)
            kl_val = F.kl_div(old_lp_scalar.float().log(), ref_lp.float(),
                              reduction='batchmean', log_target=True)

        B = len(batch)
        adv_per_token = grpo_adv.unsqueeze(-1).expand_as(resp_mask) * resp_mask.float()

        outputs = self.model_engine.module.forward_opd(
            **model_inputs,
            use_ppo=True,
            old_log_prob=old_slp,
            advantages=adv_per_token,
            clip_ratio=self.args.clip_ratio,
            ppo_coef=self.args.ppo_coef,
            opd_coef=self.args.opd_coef,
        )
        loss = outputs["loss"] + kl_val * self.args.kl_coef
        self.model_engine.backward(loss)
        self.model_engine.step()

        yo_success_rate = float((grpo_adv > 0).float().mean().item())
        yr_success_rate = float(np.mean([
            1.0 if yr_sid[i] == batch[i]["groundtruth"] else 0.0 for i in range(B)
        ]))
        episode_lengths = [
            len(self.tokenizer.encode(cot or "", add_special_tokens=False))
            for cot in yr_cot
        ]
        episode_length_mean = float(np.mean(episode_lengths)) if episode_lengths else 0.0
        current_lr = 0.0
        if self.optimizer is not None and getattr(self.optimizer, "param_groups", None):
            current_lr = float(self.optimizer.param_groups[0]["lr"])

        if self.local_rank == 0 and self.tracker is not None:
            self.tracker.log_train_metrics({
                "epoch": self.epoch,
                "global_step": self.global_step + 1,
                "batch_size": B,
                "rollout_success_rate": yo_success_rate,
                "teacher_success_rate": yr_success_rate,
                "episode_length_mean": episode_length_mean,
                "reward_mean": float(sid_scores.mean().item()),
                "lr": current_lr,
                "loss": _as_float(outputs.get("loss", 0)),
                "ce_loss": _as_float(outputs.get("ce_loss", 0)),
                "opd_loss": _as_float(outputs.get("opd_loss", 0)),
                "pg_loss": _as_float(outputs.get("pg_loss", 0)),
                "primitive_distill_loss": _as_float(outputs.get("primitive_distill_loss", 0)),
                "div_loss": _as_float(outputs.get("div_loss", 0)),
            })

        return {k: outputs.get(k, 0) if isinstance(outputs.get(k), (int, float)) else outputs.get(k, 0).detach().item()
                for k in ["loss", "ce_loss", "opd_loss", "pg_loss", "primitive_distill_loss", "div_loss"]}

    def _build_yo_df(self, records, yo_sid_list):
        rows = []
        for r, sid in zip(records, yo_sid_list):
            rows.append({
                "user_id": r.get("user_id", ""),
                "description": r["description"],
                "groundtruth": r["groundtruth"],
                "title": r.get("title", ""),
                "categories": r.get("categories", ""),
                "yo_sid": sid,
            })
        return pd.DataFrame(rows)

    def _build_yr_df(self, records, yo_sid_list, yr_cot_list, yr_sid_list):
        rows = []
        for r, yos, cot, sid in zip(records, yo_sid_list, yr_cot_list, yr_sid_list):
            rows.append({
                "user_id": r.get("user_id", ""),
                "description": r["description"],
                "groundtruth": r["groundtruth"],
                "title": r.get("title", ""),
                "categories": r.get("categories", ""),
                "yo_sid": yos,
                "yr_cot": cot,
                "yr_sid": sid,
            })
        return pd.DataFrame(rows)

    def _update_ptools(self, yo_df, yr_df):
        report_path = self.report_path
        if not report_path or not os.path.exists(report_path):
            print("  WARNING: report_path not found, skipping")
            return

        with open(report_path) as f:
            report = json.load(f)

        correct_mask = yo_df["yo_sid"] == yo_df["groundtruth"]
        correct_items = self._df_to_items(yo_df[correct_mask], correct=True)
        failed_items = self._df_to_items(yo_df[~correct_mask], correct=False)

        existing_names = [p.get("func_name", "") for p in report.get("ptools", [])]
        total_ptools = (
            len(report.get("ptools", []))
            + len(report.get("success_ptools", []))
            + len(report.get("avoidance_ptools", []))
        )

        if total_ptools >= self.args.max_total_ptools:
            print(f"  ptool limit reached ({total_ptools}/{self.args.max_total_ptools})")
            return

        bench_dir = os.path.join(os.path.dirname(__file__), "reasoning_primitives", "benchmarks", "rec")
        sys.path.insert(0, os.path.join(os.path.dirname(bench_dir), "src"))
        sys.path.insert(0, bench_dir)
        from ptool_updater import PtoolUpdater

        updater = PtoolUpdater(
            existing_ptool_names=existing_names,
            max_total_ptools=self.args.max_total_ptools,
            max_new_per_update=1,
        )
        updated = updater.update(
            correct_items=correct_items,
            failed_items=failed_items,
            report=report,
        )

        if updated:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

    def _df_to_items(self, df, correct):
        items = []
        for _, row in df.iterrows():
            desc = str(row.get("description", ""))
            for pos, para in enumerate(desc.split("\n")):
                para = para.strip()
                if para:
                    items.append({
                        "case": str(row.get("user_id", "unknown")),
                        "pos": pos,
                        "text": para,
                        "correct": correct,
                    })
        return items

    def _save_checkpoint(self, epoch):
        ckpt_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model_engine.save_checkpoint(ckpt_dir)
        if self.local_rank == 0:
            self.tokenizer.save_pretrained(ckpt_dir)
            if self.model_engine.module.prj is not None:
                torch.save(
                    self.model_engine.module.prj.state_dict(),
                    os.path.join(ckpt_dir, "prj.pt"),
                )
            print(f"  Saved checkpoint to {ckpt_dir}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="On-Policy OPD Training Loop")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/opd_loop")
    parser.add_argument("--report_path", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_cot_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--num_latent", type=int, default=8)
    parser.add_argument("--num_jacobi_steps", type=int, default=3)
    parser.add_argument("--use_prj", type=str2bool, default=True)
    parser.add_argument("--prj_dim", type=int, default=1024)
    parser.add_argument("--kv_compress_mode", type=str, default="rkv_cyclic",
                        choices=["rkv_whole", "rkv_cyclic", "identifier"])
    parser.add_argument("--use_primitive_distill", type=str2bool, default=True)
    parser.add_argument("--opd_kl_factor", type=float, default=1.0)
    parser.add_argument("--opd_gate_beta", type=float, default=5.0)
    parser.add_argument("--opd_loss_agg_mode", type=str, default="token-mean",
                        choices=["token-mean", "seq-mean-token-sum",
                                 "seq-mean-token-mean", "seq-mean-token-sum-norm"])
    parser.add_argument("--div_factor", type=float, default=0.02)

    parser.add_argument("--use_lora", type=str2bool, default=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Start from this epoch (for resuming)")
    parser.add_argument("--single_epoch", type=str2bool, default=False,
                        help="Run only one epoch then exit (for shell-level orchestration)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)

    parser.add_argument("--ppo_coef", type=float, default=1.0,
                        help="Weight for PPO clipped loss")
    parser.add_argument("--opd_coef", type=float, default=1.0,
                        help="Weight for SEED OPD gated loss")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                        help="PPO clip range epsilon")
    parser.add_argument("--use_kl_penalty", type=str2bool, default=False,
                        help="Enable KL penalty (ref_policy)")
    parser.add_argument("--kl_coef", type=float, default=0.01,
                        help="KL penalty coefficient")

    parser.add_argument("--yo_batch_size", type=int, default=8)
    parser.add_argument("--yr_batch_size", type=int, default=4)

    parser.add_argument("--update_ptools", type=str2bool, default=True)
    parser.add_argument("--max_total_ptools", type=int, default=7)

    parser.add_argument("--tokens_per_primitive", type=int, default=4,
                        help="KV tokens per primitive in cyclic compression")
    parser.add_argument("--method_name", type=str, default="SEED-OPD")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--auto_plot_dynamics", type=str2bool, default=True)
    parser.add_argument("--moving_avg_window", type=int, default=13)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank passed by DeepSpeed launcher")

    args = parser.parse_args()
    trainer = OPDTrainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
