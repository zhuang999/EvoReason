#!/usr/bin/env python3
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import datetime
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import deepspeed
from torch.utils.data import DataLoader, DistributedSampler

from model_codi import LatentRecCoDi
from dataset_opd import LatentRecOPDDataset, LatentRecOPDCollator


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to yr_responses.parquet from generate_yr.py")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/latent_opd")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_cot_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--num_latent", type=int, default=8)
    parser.add_argument("--num_jacobi_steps", type=int, default=3)
    parser.add_argument("--use_prj", type=str2bool, default=True)
    parser.add_argument("--prj_dim", type=int, default=1024)
    parser.add_argument("--prj_dropout", type=float, default=0.0)
    parser.add_argument("--prj_no_ln", type=str2bool, default=False)

    parser.add_argument("--opd_kl_factor", type=float, default=1.0,
                        help="Weight for Forward KL (MC) loss")
    parser.add_argument("--opd_loss_agg_mode", type=str, default="token-mean",
                        choices=["token-mean", "seq-mean-token-sum", "seq-mean-token-mean",
                                 "seq-mean-token-sum-norm"])
    parser.add_argument("--div_factor", type=float, default=0.02,
                        help="Weight for diversity loss (anti-latent-collapse)")
    parser.add_argument("--kl_token_clip", type=float, default=0.0,
                        help="Per-token KL upper bound (TRD kl_token_clip). 0 disables.")
    parser.add_argument("--kv_compress_mode", type=str, default="rkv_cyclic",
                        choices=["rkv_whole", "rkv_cyclic", "identifier"],
                        help="KV distillation compression mode.")
    parser.add_argument("--use_primitive_distill", type=str2bool, default=True,
                        help="Enable per-primitive KV distillation loss.")

    parser.add_argument("--use_lora", type=str2bool, default=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600), device_id=torch.device(f"cuda:{local_rank}"))
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        print(f"TRD-style Latent OPD training")
        print(f"Loading model: {args.model_name_or_path}")

    model = LatentRecCoDi(
        model_name_or_path=args.model_name_or_path,
        num_latent=args.num_latent,
        num_jacobi_steps=args.num_jacobi_steps,
        use_prj=args.use_prj,
        prj_dim=args.prj_dim,
        prj_dropout=args.prj_dropout,
        prj_no_ln=args.prj_no_ln,
        distill_loss_factor=0.0,
        kv_distill_factor=0.0,
        opd_kl_factor=args.opd_kl_factor,
        opd_gate_beta=0.0,
        opd_loss_agg_mode=args.opd_loss_agg_mode,
        kl_token_clip=args.kl_token_clip,
        ref_loss_factor=0.0,
        div_factor=args.div_factor,
        kv_compress_mode=args.kv_compress_mode,
        use_primitive_distill=args.use_primitive_distill,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    model.codi.config.use_cache = True

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = LatentRecOPDDataset(
        args.data_path, max_samples=args.max_samples,
    )
    data_collator = LatentRecOPDCollator(
        tokenizer=tokenizer, max_length=args.max_length,
        max_cot_length=args.max_cot_length,
        bot_id=model.bot_id, eot_id=model.eot_id,
    )

    if local_rank == 0:
        print(f"Train dataset: {len(train_dataset)} samples")

    num_update_steps_per_epoch = len(train_dataset) // (world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps)
    total_num_steps = num_update_steps_per_epoch * args.num_train_epochs
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

    if local_rank == 0:
        print(f"  total_steps={total_num_steps}, warmup_steps={warmup_steps}")
        print(f"  lr={args.learning_rate}, weight_decay={args.weight_decay}")
        print(f"  num_epochs={args.num_train_epochs}, batch_size={args.per_device_train_batch_size}")
        print(f"  gradient_accumulation_steps={args.gradient_accumulation_steps}")
        print(f"  gradient_checkpointing={args.gradient_checkpointing}")
        print(f"  opd_kl_factor={args.opd_kl_factor}, div_factor={args.div_factor}")
        print(f"\nStarting training...")

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    global_step = 0
    checkpoint_dirs = []

    for epoch in range(args.num_train_epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_opd = 0.0
        epoch_div = 0.0
        epoch_lat_ev1 = 0.0
        epoch_lat_eff_rank = 0.0
        num_steps = 0
        total_steps_epoch = len(train_dataloader)

        for step, batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model_engine.module.forward_opd(**batch)
            loss = outputs["loss"]
            model_engine.backward(loss)
            model_engine.step()

            epoch_loss += loss.item()
            epoch_ce += outputs["ce_loss"]
            epoch_opd += outputs["opd_loss"]
            epoch_div += outputs.get("div_loss", 0)
            epoch_lat_ev1 += outputs.get("lat_ev1", 0)
            epoch_lat_eff_rank += outputs.get("lat_eff_rank", 0)
            num_steps += 1
            global_step += 1

            if global_step % args.logging_steps == 0 and local_rank == 0:
                avg_loss = epoch_loss / num_steps
                avg_ce = epoch_ce / num_steps
                avg_opd = epoch_opd / num_steps
                current_lr = optimizer.param_groups[0]["lr"]
                progress = (step + 1) / total_steps_epoch * 100
                print(
                    f"Epoch {epoch+1}/{args.num_train_epochs} | "
                    f"{step+1}/{total_steps_epoch} [{progress:.1f}%] | "
                    f"loss={loss.item():.4f} avg={avg_loss:.4f} | "
                    f"ce={avg_ce:.4f} opd={avg_opd:.4f} div={epoch_div/num_steps:.4f} | "
                    f"lat_ev1={epoch_lat_ev1/num_steps:.3f} lat_rank={epoch_lat_eff_rank/num_steps:.2f} | "
                    f"lr={current_lr:.2e} step={global_step}"
                )

            if global_step % args.save_steps == 0:
                ckpt_dir = f"{args.output_dir}/checkpoint-{global_step}"
                model_engine.save_checkpoint(ckpt_dir)
                if local_rank == 0:
                    checkpoint_dirs.append(ckpt_dir)
                    if len(checkpoint_dirs) > args.save_total_limit:
                        old_dir = checkpoint_dirs.pop(0)
                        if Path(old_dir).exists():
                            shutil.rmtree(old_dir)

        if local_rank == 0:
            print(
                f"Epoch {epoch+1} done, avg_loss={epoch_loss/num_steps:.4f} "
                f"ce={epoch_ce/num_steps:.4f} opd={epoch_opd/num_steps:.4f} "
                f"div={epoch_div/num_steps:.4f} "
                f"lat_ev1={epoch_lat_ev1/num_steps:.3f} lat_rank={epoch_lat_eff_rank/num_steps:.2f}"
            )

        dist.barrier(device_ids=[local_rank])
        epoch_ckpt_dir = f"{args.output_dir}/epoch_{epoch+1}"
        model_engine.save_checkpoint(epoch_ckpt_dir)

    if local_rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    dist.barrier()
    model_engine.save_checkpoint(args.output_dir)

    if local_rank == 0:
        tokenizer.save_pretrained(args.output_dir)
        config_out = {
            "num_latent": args.num_latent,
            "use_prj": args.use_prj,
            "prj_dim": args.prj_dim,
            "opd_kl_factor": args.opd_kl_factor,
            "div_factor": args.div_factor,
            "mode": "trd_latent_opd",
        }
        with open(os.path.join(args.output_dir, "codi_config.json"), "w") as f:
            json.dump(config_out, f, indent=2)
        print(f"Model saved to {args.output_dir}")
        print("Training completed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
