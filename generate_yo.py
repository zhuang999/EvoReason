#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict

import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from model_codi import LatentRecCoDi
from dataset_codi import SYSTEM_MESSAGE, THINK_START


def build_sid_trie(tokenizer):
    sid_begin_id = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
    sid_end_id = tokenizer.convert_tokens_to_ids("<|sid_end|>")
    s_a_ids = [tokenizer.convert_tokens_to_ids(f"<s_a_{i}>") for i in range(256)]
    s_b_ids = [tokenizer.convert_tokens_to_ids(f"<s_b_{i}>") for i in range(256)]
    s_c_ids = [tokenizer.convert_tokens_to_ids(f"<s_c_{i}>") for i in range(256)]
    s_d_ids = [tokenizer.convert_tokens_to_ids(f"<s_d_{i}>") for i in range(256)]

    trie = defaultdict(dict)
    for a in s_a_ids:
        if a == tokenizer.unk_token_id:
            continue
        for b in s_b_ids:
            if b == tokenizer.unk_token_id:
                continue
            for c in s_c_ids:
                if c == tokenizer.unk_token_id:
                    continue
                for d in s_d_ids:
                    if d == tokenizer.unk_token_id:
                        continue
                    path = [sid_begin_id, a, b, c, d, sid_end_id]
                    node = trie
                    for token_id in path[:-1]:
                        node = node.setdefault(token_id, {})
                    node[path[-1]] = {}

    class TriePrefixAllowedTokens:
        def __init__(self, trie_dict, batch_size):
            self.trie = trie_dict
            self.batch_size = batch_size

        def __call__(self, batch_id, input_ids):
            node = self.trie
            for tid in input_ids.tolist():
                if tid in node:
                    node = node[tid]
                else:
                    return list(self.trie.keys())
            return list(node.keys())

    return TriePrefixAllowedTokens(trie, 1)


def extract_sid(token_ids, tokenizer):
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    match = re.search(r'<\|sid_begin\|>(.*?)<\|sid_end\|>', text)
    if match:
        return "<|sid_begin|>" + match.group(1) + "<|sid_end|>"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="yo_responses.parquet")
    parser.add_argument("--num_latent", type=int, default=8)
    parser.add_argument("--num_jacobi_steps", type=int, default=3)
    parser.add_argument("--use_prj", type=bool, default=True)
    parser.add_argument("--prj_dim", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LatentRecCoDi(
        model_name_or_path=args.model_path,
        num_latent=args.num_latent,
        num_jacobi_steps=args.num_jacobi_steps,
        use_prj=args.use_prj,
        prj_dim=args.prj_dim,
        use_lora=False,
    )

    prj_path = os.path.join(args.model_path, "prj.pt")
    if os.path.exists(prj_path) and args.use_prj:
        model.prj.load_state_dict(torch.load(prj_path, map_location="cpu", weights_only=True))

    model = model.to(device)
    model.eval()

    trie_fn = build_sid_trie(tokenizer)

    df = pd.read_parquet(args.data_path)
    if args.max_samples:
        df = df.head(args.max_samples)
    records = df.to_dict("records")
    print(f"Generating y_o for {len(records)} samples")

    results = []
    for idx in tqdm(range(0, len(records), args.batch_size), desc="Generating y_o"):
        batch = records[idx: idx + args.batch_size]
        descriptions = [str(r["description"]) for r in batch]

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
            padding="longest",
            truncation=True,
            max_length=2048,
            add_special_tokens=True,
        )
        encoder_input_ids = enc["input_ids"].to(device)
        encoder_attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            pred_ids = model.latent_generate(
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                num_return_sequences=1,
                prefix_allowed_tokens_fn=trie_fn,
            )

        for b_idx, row in enumerate(batch):
            seq = pred_ids[b_idx].cpu().tolist()
            yo_sid = extract_sid(seq, tokenizer) or ""
            results.append({
                "user_id": row.get("user_id", ""),
                "description": row["description"],
                "groundtruth": row["groundtruth"],
                "title": row.get("title", ""),
                "categories": row.get("categories", ""),
                "yo_sid": yo_sid,
            })

    out_df = pd.DataFrame(results)
    out_df.to_parquet(args.output_path, index=False)
    print(f"Saved {len(out_df)} rows to {args.output_path}")
    correct = (out_df["yo_sid"] == out_df["groundtruth"]).mean()
    print(f"y_o accuracy: {correct:.4f}")


if __name__ == "__main__":
    main()
