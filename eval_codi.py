#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from model_codi import LatentRecCoDi, get_embed_layer
from dataset_codi import SYSTEM_MESSAGE, THINK_START, THINK_END
from experiment_tracking import append_csv, append_jsonl


def build_sid_prefix_fn_factory(tokenizer, data_path):
    import re as _re
    from collections import defaultdict as _dd

    sid_begin_id = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
    sid_end_id = tokenizer.convert_tokens_to_ids("<|sid_end|>")

    sid_pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'

    df = pd.read_parquet(data_path)
    valid_sids = set()
    for _, row in df.iterrows():
        for m in _re.findall(sid_pattern, str(row.get('description', ''))):
            valid_sids.add(m)
        gt = _re.search(sid_pattern, str(row.get('groundtruth', '')))
        if gt:
            valid_sids.add(gt.group(0))
    print(f"Found {len(valid_sids)} unique SIDs in test set")

    sid_token_sequences = []
    for sid in valid_sids:
        tokens = tokenizer.encode(sid, add_special_tokens=False)
        sid_token_sequences.append(tokens)

    exact_trie = _dd(lambda: _dd(set))
    for seq in sid_token_sequences:
        for pos in range(len(seq)):
            current_token = seq[pos]
            if pos + 1 < len(seq):
                exact_trie[pos][current_token].add(seq[pos + 1])
            else:
                eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                exact_trie[pos][current_token].add(eos_id)

    allowed_tokens = {}
    for pos in exact_trie:
        allowed_tokens[pos] = {}
        for token_id in exact_trie[pos]:
            allowed_tokens[pos][token_id] = list(exact_trie[pos][token_id])

    print(f"SID trie: {len(allowed_tokens)} positions")
    for pos in sorted(allowed_tokens.keys())[:6]:
        keys = list(allowed_tokens[pos].keys())
        print(f"  Position {pos}: {len(keys)} possible tokens, keys[:5]={keys[:5]}")
        if pos == 0 and keys:
            first_key = keys[0]
            vals = allowed_tokens[pos][first_key]
            print(f"    pos=0, key={first_key} -> vals[:5]={vals[:5]}")

    def make_prefix_fn(prompt_len):
        def prefix_fn(batch_id, sentence):
            ids = sentence.tolist()
            n_generated = len(ids) - prompt_len

            if n_generated <= 0:
                if 0 in allowed_tokens:
                    return list(allowed_tokens[0].keys())
                return [sid_begin_id]

            last_token = ids[-1]
            prev_pos = n_generated - 1

            if prev_pos in allowed_tokens and last_token in allowed_tokens[prev_pos]:
                return allowed_tokens[prev_pos][last_token]

            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            return [eos_id]

        return prefix_fn

    return make_prefix_fn


def extract_sid_from_ids(token_ids: List[int], tokenizer) -> Optional[str]:
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return extract_sid_from_text(text)


def extract_sid_from_text(text: str) -> Optional[str]:
    pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return None


def compute_metrics(all_predictions: List[List[str]], all_groundtruths: List[str],
                    ks=(5, 10)):
    metrics = {}
    for k in ks:
        hits, ndcgs = [], []
        for preds, gt in zip(all_predictions, all_groundtruths):
            top_k = preds[:k]
            hit = int(gt in top_k)
            hits.append(hit)
            if hit:
                rank = top_k.index(gt) + 1
                ndcgs.append(1.0 / np.log2(rank + 1))
            else:
                ndcgs.append(0.0)
        metrics[f"hit@{k}"] = float(np.mean(hits))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs))
    return metrics


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="eval_codi_results.json")
    parser.add_argument("--num_latent", type=int, default=8)
    parser.add_argument("--use_prj", type=str2bool, default=True)
    parser.add_argument("--prj_dim", type=int, default=1024)
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--num_return_sequences", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--base_model", type=str,
        default="/pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/checkpoints/beauty_RA/epoch_2")
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--record_file", type=str, default=None,
                        help="Optional csv/jsonl file to append sample efficiency results.")
    parser.add_argument("--method_name", type=str, default="SEED-OPD")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--data_fraction", type=float, default=None,
                        help="Training data fraction used for this checkpoint, e.g. 0.6")
    parser.add_argument("--score_metric", type=str, default="hit@10",
                        help="Metric to record into sample efficiency registry.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model architecture from {args.base_model}")
    model = LatentRecCoDi(
        model_name_or_path=args.base_model,
        num_latent=args.num_latent,
        num_jacobi_steps=3,
        use_prj=args.use_prj,
        prj_dim=args.prj_dim,
        use_lora=False,
        use_memory_decoder=True,
        num_memory_tokens=4,
        mem_decoder_layers=2,
        use_primitive_distill=True,
        primitive_distill_factor=1.0,
        identity_factor=0.05,
        identity_temperature=0.15,
        div_factor=0.1,
    )

    print(f"Loading trained weights from {args.model_path}")
    index_file = os.path.join(args.model_path, "pytorch_model.bin.index.json")
    single_file = os.path.join(args.model_path, "pytorch_model.bin")

    if os.path.exists(index_file) or os.path.exists(single_file):
        state_dict = {}
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                index = json.load(f)
            shard_files = sorted(set(index.get("weight_map", {}).values()))
            for shard in shard_files:
                shard_path = os.path.join(args.model_path, shard)
                print(f"  Loading {shard}...")
                state_dict.update(torch.load(shard_path, map_location="cpu", weights_only=True))
        else:
            print(f"  Loading pytorch_model.bin...")
            state_dict = torch.load(single_file, map_location="cpu", weights_only=True)

        state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
        mapped = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                mapped[k[len("model."):]] = v
            else:
                mapped[k] = v
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        loaded = len(mapped) - len(unexpected)
        print(f"Loaded {loaded}/{len(mapped)} params, missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  First 5 missing: {missing[:5]}")
        if unexpected:
            print(f"  First 5 unexpected: {unexpected[:5]}")
    else:
        print("WARNING: No pytorch_model files found, using base model weights")

    model = model.to(device)
    model.eval()

    make_prefix_fn = build_sid_prefix_fn_factory(tokenizer, args.data_path)

    df = pd.read_parquet(args.data_path)
    if args.sample_offset > 0:
        df = df.iloc[args.sample_offset:].reset_index(drop=True)
    if args.max_samples:
        df = df.head(args.max_samples)
    records = df.to_dict("records")
    print(f"GPU {args.gpu_id}: Evaluating {len(records)} samples (offset={args.sample_offset})")

    all_predictions = []
    all_groundtruths = []

    for idx in tqdm(range(0, len(records), args.batch_size), desc="Evaluating"):
        batch_rows = records[idx: idx + args.batch_size]
        descriptions = [str(r["description"]) for r in batch_rows]
        groundtruths = [str(r["groundtruth"]) for r in batch_rows]
        groundtruth_sids = [extract_sid_from_text(gt) or gt for gt in groundtruths]

        encoder_strs = [
            f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n{desc}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{THINK_START}\n"
            for desc in descriptions
        ]

        enc = tokenizer(
            encoder_strs,
            return_tensors="pt",
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=2048,
        )
        encoder_input_ids = enc["input_ids"].to(device)
        encoder_attention_mask = enc["attention_mask"].to(device)

        bsz = len(batch_rows)

        num_beams = args.num_beams
        num_return_sequences = args.num_return_sequences

        with torch.no_grad():
            pred_ids = model.latent_generate(
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                prefix_allowed_tokens_fn=make_prefix_fn(0),
            )

        pred_ids_list = pred_ids.cpu().tolist()

        if args.debug and idx == 0:
            print(f"\n[DEBUG] pred_ids shape: {pred_ids.shape}")
            for j in range(min(3, len(pred_ids_list))):
                seq = pred_ids_list[j]
                text = tokenizer.decode(seq, skip_special_tokens=False)
                sid = extract_sid_from_ids(seq, tokenizer)
                print(f"  beam {j}: len={len(seq)} sid={sid}")
                print(f"    text[:200]={text[:200]}")
            print(f"[DEBUG] gt_sid={groundtruth_sids[0]}")

        for b_idx in range(bsz):
            group = pred_ids_list[b_idx * args.num_return_sequences: (b_idx + 1) * args.num_return_sequences]
            preds = []
            for seq in group:
                sid = extract_sid_from_ids(seq, tokenizer)
                if sid and sid not in preds:
                    preds.append(sid)
            all_predictions.append(preds)
            all_groundtruths.append(groundtruth_sids[b_idx])

    metrics = compute_metrics(all_predictions, all_groundtruths)

    print("\n===== Evaluation Results =====")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    result = {
        "model": args.model_path,
        "num_latent": args.num_latent,
        "n_test": len(records),
        "metrics": metrics,
    }
    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output_file}")

    if args.record_file:
        record = {
            "method": args.method_name,
            "run_name": args.run_name or os.path.basename(os.path.abspath(args.model_path)),
            "model_path": args.model_path,
            "data_path": args.data_path,
            "data_fraction": args.data_fraction,
            "score_metric": args.score_metric,
            "score": metrics.get(args.score_metric),
            "hit@5": metrics.get("hit@5"),
            "ndcg@5": metrics.get("ndcg@5"),
            "hit@10": metrics.get("hit@10"),
            "ndcg@10": metrics.get("ndcg@10"),
            "n_test": len(records),
        }
        if args.record_file.endswith(".jsonl"):
            append_jsonl(Path(args.record_file), record)
        elif args.record_file.endswith(".csv"):
            append_csv(Path(args.record_file), record)
        else:
            raise ValueError("--record_file must end with .csv or .jsonl")
        print(f"Recorded sample efficiency result to {args.record_file}")


if __name__ == "__main__":
    main()
