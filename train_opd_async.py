#!/usr/bin/env python3
"""
SEED-style async OPD training: generation (rollout) runs in a separate
process, training runs in parallel via torchrun on the remaining GPUs.

Architecture (aligned with SEED ray_trainer.py):
  ┌──────────────────────┐    mp.Queue        ┌───────────────────────┐
  │ GenProcess (GPU 2)   │ ─────────────────> │ TrainProcesses (GPU 3-7) │
  │ gen_yo → gen_yr      │ (pickled batches)  │ rank0: dequeue → dist │
  │ (actor_rollout)      │                    │   .broadcast to all   │
  └──────────────────────┘                    └───────────────────────┘

Usage:
  python train_opd_async.py \
      --model_name_or_path /pub/.../beauty_RA_cot/epoch_3 \
      --data_path /pub/.../training_RA_cot_train.parquet \
      --output_dir checkpoints/opd_async \
      --gen_gpu 2 --train_gpus 3,4,5,6,7 \
      --num_epochs 3 --per_device_train_batch_size 2
"""

import os, sys, re, time, argparse, datetime, pickle, queue, json, traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from model_codi import LatentRecCoDi
from dataset_codi import SYSTEM_MESSAGE, THINK_START, THINK_END
from dataset_opd import LatentRecOPDCollator
from core_algos import compute_opd_loss
from experiment_tracking import ExperimentTracker

SID_RE = re.compile(r'<\|sid_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)><s_d_(\d+)><\|sid_end\|>')


def _parse_sid_tuple(sid_str):
    """Return 4-tuple(a,b,c,d) for SID match count, or None if invalid."""
    if not sid_str:
        return None
    m = SID_RE.search(sid_str)
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


def extract_sid(token_ids, tokenizer):
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    m = re.search(r'<\|sid_begin\|>(.*?)<\|sid_end\|>', text)
    return "<|sid_begin|>" + m.group(1) + "<|sid_end|>" if m else ""


def str2bool(v):
    if isinstance(v, bool): return v
    return v.lower() in ("yes", "true", "t", "1")


def _setup_runtime_tokenizer(tokenizer, model=None):
    """
    Keep async/sync entrypoints aligned: generation and collation both require
    a usable pad token for batch padding.
    """
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if model is not None:
                model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = "left"
    if model is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def _as_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "detach"):
        return float(value.detach().item())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _move_latent_codi_to_device(model, device):
    """
    Move the full wrapper module, not only `model.codi`.
    `LatentRecCoDi` also owns `prj` and optional primitive slot parameters.
    """
    return model.to(device)


# ── Generation Process (SEED: actor_rollout_wg, GPU 2) ──

def gen_process(data_queue, gen_gpu, model_path, data_path, max_cot_length,
                yo_batch_size, yr_batch_size, num_epochs, stop_event):
    """Generates yo/yr on a dedicated GPU, puts into queue for training."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gen_gpu)
    device = torch.device("cuda")
    try:
        model = LatentRecCoDi(model_name_or_path=model_path, use_lora=True)
        _setup_runtime_tokenizer(model.tokenizer, model.codi)
        model = _move_latent_codi_to_device(model, device)
        model.eval()

        base_df = pd.read_parquet(data_path)
        records = base_df.to_dict("records")

        for epoch in range(num_epochs):
            if stop_event.is_set(): break
            for idx in range(0, len(records), yo_batch_size):
                if stop_event.is_set(): break
                batch = records[idx: idx + yo_batch_size]

                # gen_yo
                descriptions = [str(r["description"]) for r in batch]
                encoder_strs = [
                    f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
                    f"<|im_start|>user\n{desc}<|im_end|>\n"
                    f"<|im_start|>assistant\n{THINK_START}\n" for desc in descriptions
                ]
                enc = model.tokenizer(encoder_strs, return_tensors="pt", padding="longest",
                                      truncation=True, max_length=2048, add_special_tokens=True)
                enc_ids, enc_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
                with torch.no_grad():
                    pred_ids = model.latent_generate(
                        encoder_input_ids=enc_ids, encoder_attention_mask=enc_mask,
                        max_new_tokens=30, num_beams=1, num_return_sequences=1,
                    )
                yo_sid = [extract_sid(pred_ids[bi].cpu().tolist(), model.tokenizer) or ""
                          for bi in range(len(batch))]

                # gen_yr
                yr_cot, yr_sid = [], []
                for y_idx in range(0, len(batch), yr_batch_size):
                    yb = batch[y_idx:y_idx + yr_batch_size]
                    y_enc_strs = [
                        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
                        f"<|im_start|>user\n{r['description']}<|im_end|>\n"
                        f"<|im_start|>assistant\n{THINK_START}\n" for r in yb
                    ]
                    ye = model.tokenizer(y_enc_strs, return_tensors="pt", padding="longest",
                                         truncation=True, max_length=2048, add_special_tokens=True)
                    y_ids, y_mask = ye["input_ids"].to(device), ye["attention_mask"].to(device)
                    with torch.no_grad():
                        out_ids = model.latent_refine_generate(
                            encoder_input_ids=y_ids, encoder_attention_mask=y_mask,
                            max_new_tokens=max_cot_length, num_beams=1, temperature=0.7, top_p=0.95,
                        )
                    for bi in range(len(yb)):
                        raw = model.tokenizer.decode(out_ids[bi].cpu().tolist(), skip_special_tokens=False)
                        m = re.search(r'<\|sid_begin\|>(.*?)<\|sid_end\|>', raw)
                        sid = ("<|sid_begin|>" + m.group(1) + "<|sid_end|>") if m else ""
                        cm = re.search(r"(.*?)" + re.escape(THINK_END) + r"\n", raw, re.DOTALL)
                        cot = cm.group(1).strip() if cm else ""
                        yr_cot.append(cot)
                        yr_sid.append(sid)

                data_queue.put({
                    "batch": batch, "yo_sid": yo_sid,
                    "yr_cot": yr_cot, "yr_sid": yr_sid, "epoch": epoch,
                })
    except Exception:
        stop_event.set()
        data_queue.put({"__error__": traceback.format_exc()})
    finally:
        data_queue.put(None)


# ── Training Process (SEED: ray_trainer, GPU 3-7) ──

def train_process(rank, world_size, data_queue, args):
    """Training process group. Rank 0 reads from queue, broadcasts to all."""
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=7200))
    try:
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)

        # ── Actor model ──
        model = LatentRecCoDi(
            model_name_or_path=args.model_name_or_path, use_lora=True,
            num_latent=args.num_latent, num_jacobi_steps=args.num_jacobi_steps,
            use_prj=args.use_prj, prj_dim=args.prj_dim,
            kv_compress_mode=args.kv_compress_mode,
            use_primitive_distill=args.use_primitive_distill,
            opd_kl_factor=args.opd_kl_factor, opd_gate_beta=args.opd_gate_beta,
            opd_loss_agg_mode=args.opd_loss_agg_mode,
            div_factor=args.div_factor,
            tokens_per_primitive=args.tokens_per_primitive,
        )
        tokenizer = model.tokenizer
        _setup_runtime_tokenizer(tokenizer, model.codi)

        collator = LatentRecOPDCollator(
            tokenizer=tokenizer, max_length=args.max_length,
            max_cot_length=args.max_cot_length,
            bot_id=model.bot_id, eot_id=model.eot_id,
        )

        # ── Reference policy (SEED: ref_policy_wg, frozen base model) ──
        ref_model = None
        if args.use_kl_penalty:
            ref_model = LatentRecCoDi(
                model_name_or_path=args.model_name_or_path, use_lora=False,
                num_latent=args.num_latent, num_jacobi_steps=args.num_jacobi_steps,
                use_prj=args.use_prj, prj_dim=args.prj_dim,
                kv_compress_mode="rkv_whole", use_primitive_distill=False,
                div_factor=0.0,
            )
            ref_model = _move_latent_codi_to_device(ref_model, device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False

        # ── DeepSpeed engine ──
        total_num_steps = args.num_epochs * (len(pd.read_parquet(args.data_path)) //
            (world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps))
        warmup_steps = int(total_num_steps * args.warmup_ratio)

        ds_config = {
            "bf16": {"enabled": args.bf16},
            "zero_optimization": {
                "stage": 2, "contiguous_gradients": True,
                "overlap_comm": True, "reduce_scatter": True,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
            },
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_clipping": args.max_grad_norm,
            "train_batch_size": world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": args.learning_rate, "betas": [0.9, 0.999],
                           "eps": 1e-8, "weight_decay": args.weight_decay},
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {"warmup_min_lr": 0.0, "warmup_max_lr": args.learning_rate,
                           "warmup_num_steps": warmup_steps},
            },
            "zero_force_ds_cpu_optimizer": False,
        }

        import deepspeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model, config=ds_config,
        )

        tracker = None
        if rank == 0:
            run_name = args.run_name or Path(args.output_dir).name
            tracker = ExperimentTracker(
                output_dir=args.output_dir,
                run_name=run_name,
                method_name=args.method_name,
                moving_avg_window=args.moving_avg_window,
                auto_plot=args.auto_plot_dynamics,
            )

        # ── Training loop ──
        model_engine.train()
        global_step = 0
        epoch_metrics = defaultdict(float)
        n_steps = 0

        while True:
            # Rank 0: dequeue from generation process (SEED: gen_batch from queue)
            data = None
            if rank == 0:
                try:
                    data = data_queue.get(timeout=30)
                except queue.Empty:
                    data = None

            # Broadcast data to all ranks (SEED: batch.union)
            data_list = [data]
            dist.broadcast_object_list(data_list, src=0)
            data = data_list[0]

            if data is None:
                break
            if "__error__" in data:
                raise RuntimeError(f"Generation process failed:\n{data['__error__']}")

            batch = data["batch"]; yo_sid = data["yo_sid"]
            yr_cot = data["yr_cot"]; yr_sid = data["yr_sid"]

            # ── Collate ──
            coll_records = []
            for i, r in enumerate(batch):
                coll_records.append({
                    "description": r["description"], "groundtruth": r["groundtruth"],
                    "yo_sid": yo_sid[i], "yr_cot": yr_cot[i], "yr_sid": yr_sid[i],
                })
            model_inputs = collator(coll_records)
            model_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in model_inputs.items()}

            B = len(batch)
            dec_len = model_inputs["labels"].shape[1] - 1

            # ── old_log_probs (no_grad, SEED: π_θold) ──
            with torch.no_grad():
                O = model_engine.module.forward_opd(**model_inputs)
                old_slp = O["student_log_prob"]
                rmask = O["response_mask"]

            # ── ref_log_prob (SEED: ref_policy, frozen base model) ──
            kl_val = torch.tensor(0.0, device=device)
            if ref_model is not None:
                with torch.no_grad():
                    ref_out = ref_model.forward_opd(**model_inputs)
                    ref_slp = ref_out["student_log_prob"]
                    ref_lp = (ref_slp * rmask).sum(dim=-1) / rmask.sum(dim=-1).clamp(min=1)
                old_lp_scalar = (old_slp * rmask).sum(dim=-1) / rmask.sum(dim=-1).clamp(min=1)
                kl_val = F.kl_div(old_lp_scalar.float().log(), ref_lp.float(),
                                  reduction='batchmean', log_target=True)

            # ── GRPO group-based advantage from SID matching ──
            sid_scores = torch.tensor([
                _count_matching_sid(yo_sid[i], batch[i]["groundtruth"])
                for i in range(B)
            ], device=device, dtype=torch.float32)

            _eps = 1e-6
            if B > 1:
                mu = sid_scores.mean()
                sigma = sid_scores.std(unbiased=False)
                if sigma < _eps:
                    sigma = 1.0
                grpo_base = (sid_scores - mu) / (sigma + _eps)
            else:
                grpo_base = sid_scores

            adv_per_token = grpo_base.unsqueeze(-1).expand_as(rmask) * rmask.float()

            # ── PPO + OPD + primitive distill forward ──
            outputs = model_engine.module.forward_opd(
                **model_inputs,
                use_ppo=True,
                old_log_prob=old_slp,
                advantages=adv_per_token,
                clip_ratio=args.clip_ratio,
                ppo_coef=args.ppo_coef,
                opd_coef=args.opd_coef,
            )
            loss = outputs["loss"] + kl_val * args.kl_coef
            model_engine.backward(loss)
            model_engine.step()

            yo_success_rate = float((sid_scores > 0).float().mean().item())
            yr_success_rate = float(np.mean([
                1.0 if yr_sid[i] == batch[i]["groundtruth"] else 0.0 for i in range(B)
            ]))
            episode_lengths = [
                len(tokenizer.encode(cot or "", add_special_tokens=False))
                for cot in yr_cot
            ]
            episode_length_mean = float(np.mean(episode_lengths)) if episode_lengths else 0.0
            current_lr = 0.0
            if optimizer is not None and getattr(optimizer, "param_groups", None):
                current_lr = float(optimizer.param_groups[0]["lr"])

            for k in ["loss", "ce_loss", "opd_loss", "pg_loss", "primitive_distill_loss", "div_loss"]:
                v = outputs.get(k, 0)
                epoch_metrics[k] += v if isinstance(v, float) else v.item()
            n_steps += 1; global_step += 1

            if tracker is not None:
                tracker.log_train_metrics({
                    "epoch": data.get("epoch", 0),
                    "global_step": global_step,
                    "batch_size": B,
                    "rollout_success_rate": float((sid_scores > 0).float().mean().item()),
                    "teacher_success_rate": yr_success_rate,
                    "episode_length_mean": episode_length_mean,
                    "sid_score_mean": float(sid_scores.mean().item()),
                    "reward_mean": float(sid_scores.mean().item()),
                    "kl_value": float(kl_val.item()),
                    "lr": current_lr,
                    "loss": _as_float(outputs.get("loss", 0)),
                    "ce_loss": _as_float(outputs.get("ce_loss", 0)),
                    "opd_loss": _as_float(outputs.get("opd_loss", 0)),
                    "pg_loss": _as_float(outputs.get("pg_loss", 0)),
                    "primitive_distill_loss": _as_float(outputs.get("primitive_distill_loss", 0)),
                    "div_loss": _as_float(outputs.get("div_loss", 0)),
                })

            if rank == 0 and global_step % args.logging_steps == 0:
                avg = {k: v / max(n_steps, 1) for k, v in epoch_metrics.items()}
                print(f"  step={global_step} loss={avg.get('loss', 0):.4f} "
                      f"ce={avg.get('ce_loss', 0):.4f} opd={avg.get('opd_loss', 0):.4f} "
                      f"pg={avg.get('pg_loss', 0):.4f} prim={avg.get('primitive_distill_loss', 0):.4f} "
                      f"kl={kl_val.item():.4f}")
                tracker.maybe_plot_training_dynamics()

        if rank == 0:
            print(f"Training completed. {n_steps} steps.")
            if tracker is not None:
                tracker.maybe_plot_training_dynamics()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ── Main: spawn gen_process + train_processes ──

def main():
    parser = argparse.ArgumentParser(description="SEED-style async OPD training")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/opd_async")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_cot_length", type=int, default=1024)
    parser.add_argument("--num_latent", type=int, default=8)
    parser.add_argument("--num_jacobi_steps", type=int, default=3)
    parser.add_argument("--use_prj", type=str2bool, default=True)
    parser.add_argument("--prj_dim", type=int, default=1024)
    parser.add_argument("--kv_compress_mode", type=str, default="rkv_cyclic")
    parser.add_argument("--use_primitive_distill", type=str2bool, default=True)
    parser.add_argument("--opd_kl_factor", type=float, default=1.0)
    parser.add_argument("--opd_gate_beta", type=float, default=5.0)
    parser.add_argument("--opd_loss_agg_mode", type=str, default="token-mean")
    parser.add_argument("--div_factor", type=float, default=0.02)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--ppo_coef", type=float, default=1.0)
    parser.add_argument("--opd_coef", type=float, default=1.0)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--yo_batch_size", type=int, default=8)
    parser.add_argument("--yr_batch_size", type=int, default=4)
    parser.add_argument("--tokens_per_primitive", type=int, default=4)
    parser.add_argument("--gen_gpu", type=int, default=2, help="GPU for generation")
    parser.add_argument("--train_gpus", type=str, default="3,4,5,6,7",
                        help="GPUs for training (comma-separated)")
    parser.add_argument("--use_kl_penalty", type=str2bool, default=True)
    parser.add_argument("--kl_coef", type=float, default=0.01)
    parser.add_argument("--method_name", type=str, default="SEED-OPD")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--auto_plot_dynamics", type=str2bool, default=True)
    parser.add_argument("--moving_avg_window", type=int, default=13)
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    data_queue = mp.Queue(maxsize=4)
    stop_event = mp.Event()

    # ── Start generation process (SEED: actor_rollout_wg) ──
    gen_proc = mp.Process(target=gen_process, args=(
        data_queue, args.gen_gpu, args.model_name_or_path, args.data_path,
        args.max_cot_length, args.yo_batch_size, args.yr_batch_size,
        args.num_epochs, stop_event,
    ))
    gen_proc.start()

    # ── Start training processes (SEED: ray_trainer) ──
    train_gpu_list = [int(g) for g in args.train_gpus.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.train_gpus
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"

    train_procs = []
    for rank in range(len(train_gpu_list)):
        p = mp.Process(target=train_process, args=(rank, len(train_gpu_list), data_queue, args))
        p.start()
        train_procs.append(p)

    gen_proc.join()
    for p in train_procs:
        p.join()
    if gen_proc.exitcode not in (0, None):
        raise SystemExit(f"gen_process exited with code {gen_proc.exitcode}")
    failed_train = [idx for idx, p in enumerate(train_procs) if p.exitcode not in (0, None)]
    if failed_train:
        raise SystemExit(f"train_processes failed at ranks {failed_train}")
    print("Async OPD training completed.")


if __name__ == "__main__":
    main()
