#!/usr/bin/env python3
import argparse
import re

import pandas as pd
import torch
from tqdm import tqdm

from model_codi import LatentRecCoDi
from dataset_codi import SYSTEM_MESSAGE, THINK_START, THINK_END


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to LatentRecCoDi checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to eval/yo parquet with description and groundtruth")
    parser.add_argument("--output_path", type=str, default="yr_responses.parquet")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading LatentRecCoDi model: {args.model_path}")
    model = LatentRecCoDi(
        model_name_or_path=args.model_path,
        use_lora=False,
    )
    model = model.to(device).eval()

    tokenizer = model.tokenizer

    df = pd.read_parquet(args.data_path)
    if args.max_samples:
        df = df.head(args.max_samples)
    records = df.to_dict("records")
    print(f"Generating y_r for {len(records)} samples")

    results = []
    for idx in tqdm(range(0, len(records), args.batch_size), desc="Generating y_r"):
        batch = records[idx: idx + args.batch_size]

        encoder_strs = []
        for r in batch:
            description = r["description"]
            encoder_str = (
                f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
                f"<|im_start|>user\n{description}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{THINK_START}\n"
            )
            encoder_strs.append(encoder_str)

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
            out_ids = model.latent_refine_generate(
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        for b_idx, row in enumerate(batch):
            new_ids = out_ids[b_idx].cpu().tolist()
            yr_full = tokenizer.decode(new_ids, skip_special_tokens=False)

            yr_sid_match = re.search(r'<\|sid_begin\|>(.*?)<\|sid_end\|>', yr_full)
            yr_sid = ""
            if yr_sid_match:
                yr_sid = "<|sid_begin|>" + yr_sid_match.group(1) + "<|sid_end|>"

            cot_pattern = re.escape(THINK_END) + r"\n"
            cot_match = re.search(r"(.*?)" + cot_pattern, yr_full, re.DOTALL)
            yr_cot = cot_match.group(1).strip() if cot_match else ""

            yr_response = THINK_START + "\n" + yr_full if not yr_full.startswith(THINK_START) else yr_full

            results.append({
                "user_id": row.get("user_id", ""),
                "description": row["description"],
                "groundtruth": row["groundtruth"],
                "title": row.get("title", ""),
                "categories": row.get("categories", ""),
                "yo_sid": row.get("yo_sid", ""),
                "yr_cot": yr_cot,
                "yr_sid": yr_sid,
                "yr_full_response": yr_response,
            })

    out_df = pd.DataFrame(results)
    out_df.to_parquet(args.output_path, index=False)
    print(f"Saved {len(out_df)} rows to {args.output_path}")
    yr_correct = (out_df["yr_sid"] == out_df["groundtruth"]).mean()
    yo_correct = (out_df["yo_sid"] == out_df["groundtruth"]).mean()
    print(f"y_o accuracy: {yo_correct:.4f}, y_r accuracy: {yr_correct:.4f}")


if __name__ == "__main__":
    main()
