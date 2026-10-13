#!/usr/bin/env python3
"""Quick sanity test for latent_generate + trie constraint.
Runs 5 samples and prints detailed debug info.
Usage: CUDA_VISIBLE_DEVICES=0 python test_eval_logic.py
"""
import json
import os
import re
import torch
from transformers import AutoTokenizer
from model_codi import LatentRecCoDi
from dataset_codi import SYSTEM_MESSAGE, THINK_START, THINK_END

WEIGHTS_PATH = "/code/checkpoints/latent_codi_v6/epoch_3/pytorch_model.bin"
TOKENIZER_PATH = "/code/checkpoints/latent_codi_v6"
BASE_MODEL = "/pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/checkpoints/beauty_sid_rec/checkpoint-8388"
DATA_PATH = "/pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/data/beauty/training_prediction_sid_data_test.parquet"
NUM_LATENT = 4
MAX_NEW_TOKENS = 6
NUM_BEAMS = 10

import pandas as pd
from collections import defaultdict

def build_trie(tokenizer, data_path):
    sid_pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
    df = pd.read_parquet(data_path)
    valid_sids = set()
    for _, row in df.iterrows():
        for m in re.findall(sid_pattern, str(row.get('description', ''))):
            valid_sids.add(m)
        gt = re.search(sid_pattern, str(row.get('groundtruth', '')))
        if gt:
            valid_sids.add(gt.group(0))
    print(f"[trie] Found {len(valid_sids)} unique SIDs")

    sid_token_sequences = []
    for sid in valid_sids:
        tokens = tokenizer.encode(sid, add_special_tokens=False)
        sid_token_sequences.append(tokens)

    exact_trie = defaultdict(lambda: defaultdict(set))
    for seq in sid_token_sequences:
        for pos in range(len(seq)):
            current_token = seq[pos]
            if pos + 1 < len(seq):
                exact_trie[pos][current_token].add(seq[pos + 1])
            else:
                eos_id = tokenizer.eos_token_id or 0
                exact_trie[pos][current_token].add(eos_id)

    allowed_tokens = {}
    for pos in exact_trie:
        allowed_tokens[pos] = {}
        for token_id in exact_trie[pos]:
            allowed_tokens[pos][token_id] = list(exact_trie[pos][token_id])

    print(f"[trie] {len(allowed_tokens)} positions")
    if 0 in allowed_tokens:
        keys = list(allowed_tokens[0].keys())
        print(f"[trie] pos=0: {len(keys)} possible first tokens, first 3: {keys[:3]}")

    return allowed_tokens

def make_prefix_fn(allowed_tokens, prompt_len, tokenizer):
    sid_begin_id = tokenizer.convert_tokens_to_ids("<|sid_begin|>")
    eos_id = tokenizer.eos_token_id or 0

    def prefix_fn(batch_id, sentence):
        ids = sentence.tolist()
        n_generated = len(ids) - prompt_len

        if n_generated <= 0:
            if 0 in allowed_tokens:
                allowed = list(allowed_tokens[0].keys())
                print(f"  [trie] batch={batch_id} n_gen={n_generated} -> first tokens: {len(allowed)} options")
                return allowed
            print(f"  [trie] batch={batch_id} n_gen={n_generated} -> sid_begin fallback")
            return [sid_begin_id]

        last_token = ids[-1]
        prev_pos = n_generated - 1

        if prev_pos in allowed_tokens and last_token in allowed_tokens[prev_pos]:
            allowed = allowed_tokens[prev_pos][last_token]
            print(f"  [trie] batch={batch_id} n_gen={n_generated} prev_pos={prev_pos} last={last_token} -> {len(allowed)} options: {allowed[:3]}")
            return allowed

        print(f"  [trie] batch={batch_id} n_gen={n_generated} prev_pos={prev_pos} last={last_token} -> EOS fallback")
        return [eos_id]

    return prefix_fn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading tokenizer from {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model architecture from {BASE_MODEL}")
    model = LatentRecCoDi(
        model_name_or_path=BASE_MODEL,
        num_latent=NUM_LATENT,
        num_jacobi_steps=3,
        use_prj=True,
        prj_dim=1024,
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

    print(f"Loading trained weights from {WEIGHTS_PATH}")
    state_dict = {}
    index_file = os.path.join(WEIGHTS_PATH, "pytorch_model.bin.index.json")
    single_file = os.path.join(WEIGHTS_PATH, "pytorch_model.bin")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index.get("weight_map", {}).values()))
        for shard in shard_files:
            shard_path = os.path.join(WEIGHTS_PATH, shard)
            print(f"  Loading {shard}...")
            state_dict.update(torch.load(shard_path, map_location="cpu", weights_only=True))
    elif os.path.exists(single_file):
        state_dict = torch.load(single_file, map_location="cpu", weights_only=True)
    else:
        print("WARNING: No weight files found")
        return

    state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    mapped = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            mapped[k[len("model."):]] = v
        else:
            mapped[k] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f"Loaded {len(mapped) - len(unexpected)}/{len(mapped)} params, missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(device)
    model.eval()

    allowed_tokens = build_trie(tokenizer, DATA_PATH)

    df = pd.read_parquet(DATA_PATH)
    records = df.head(5).to_dict("records")

    sid_pattern = r'<\|sid_begin\|>(.*?)<\|sid_end\|>'

    for i, row in enumerate(records):
        desc = str(row["description"])
        gt = str(row["groundtruth"])

        encoder_str = (
            f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n{desc}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{THINK_START}\n"
        )

        enc = tokenizer(encoder_str, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=2048)
        encoder_input_ids = enc["input_ids"].to(device)
        encoder_attention_mask = enc["attention_mask"].to(device)

        prompt_len = 0
        print(f"\n{'='*60}")
        print(f"Sample {i}: enc_len={encoder_input_ids.shape[1]}, num_latent={NUM_LATENT}, prompt_len={prompt_len} (inputs_embeds mode)")

        prefix_fn = make_prefix_fn(allowed_tokens, prompt_len, tokenizer)

        with torch.no_grad():
            pred_ids = model.latent_generate(
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                num_return_sequences=NUM_BEAMS,
                prefix_allowed_tokens_fn=prefix_fn,
            )

        pred_ids_list = pred_ids.cpu().tolist()

        print(f"\nGenerated {len(pred_ids_list)} sequences:")
        for j, seq in enumerate(pred_ids_list):
            text = tokenizer.decode(seq, skip_special_tokens=False)
            match = re.search(sid_pattern, text)
            sid = "<|sid_begin|>" + match.group(1) + "<|sid_end|>" if match else "NO_SID"
            gen_part = tokenizer.decode(seq[-MAX_NEW_TOKENS:], skip_special_tokens=False)
            print(f"  beam {j}: sid={sid[:60]}... | last {MAX_NEW_TOKENS} tokens: {gen_part[:80]}")

        gt_match = re.search(sid_pattern, gt)
        gt_sid = "<|sid_begin|>" + gt_match.group(1) + "<|sid_end|>" if gt_match else gt

        preds = []
        for seq in pred_ids_list:
            text = tokenizer.decode(seq, skip_special_tokens=False)
            match = re.search(sid_pattern, text)
            if match:
                sid = "<|sid_begin|>" + match.group(1) + "<|sid_end|>"
                if sid not in preds:
                    preds.append(sid)

        hit = int(gt_sid in preds[:5]) if gt_sid in preds else 0
        print(f"\n  GT: {gt_sid[:60]}...")
        print(f"  Unique preds: {len(preds)}")
        print(f"  Hit@5: {hit}")

    print(f"\n{'='*60}")
    print("Test complete!")

if __name__ == "__main__":
    main()
