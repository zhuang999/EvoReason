"""Convert simulated CoT (from induction pipeline) to training parquet.

This script is the bridge between the induction pipeline's output
(think_contents.jsonl from simulate_cot_all.py) and the Teacher
retraining data (RA parquet with think_content column).

The induction pipeline automatically discovers reasoning primitives
from QwQ-32B ReAct traces — we do NOT hardcode any template steps.

Full pipeline:
  1. react_vllm.py → react_traces_full.jsonl (QwQ-32B ReAct traces)
  2. react_traces_to_results.py → results.jsonl (format for induction.py)
  3. induction.py → report.json (auto-discovered reasoning primitives)
  4. generate_induced_configs.py → induced_ptools_*.py (ptool stubs)
  5. simulate_cot_all.py → think_contents.jsonl (LLM-simulated CoT per user)
  6. THIS SCRIPT: think_contents.jsonl → training_RA_cot_train.parquet

Usage:
    python convert_traces_to_cot_parquet.py \
        --think-contents out/think_contents_train.jsonl \
        --parquet /pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/data/beauty/training_RA_train.parquet \
        --output /pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/data/beauty/training_RA_cot_train.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_think_contents(path: str) -> dict[str, str]:
    contents: dict[str, str] = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(obj.get('user_id', ''))
            think = obj.get('think_content', '').strip()
            if uid and think:
                contents[uid] = think
    print(f'Loaded {len(contents)} think_content entries from {path}')
    return contents


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--think-contents', required=True,
                   help='think_contents.jsonl from simulate_cot_all.py')
    p.add_argument('--parquet', required=True,
                   help='Original RA training parquet')
    p.add_argument('--output', required=True,
                   help='Output parquet with think_content column')
    args = p.parse_args()

    think_map = load_think_contents(args.think_contents)

    print(f'Loading parquet: {args.parquet}')
    df = pd.read_parquet(args.parquet)
    print(f'Parquet shape: {df.shape}, columns: {list(df.columns)}')

    think_contents = []
    matched = 0
    unmatched = 0

    for idx, row in df.iterrows():
        uid = str(row.get('user_id', ''))
        if uid in think_map:
            think_contents.append(think_map[uid])
            matched += 1
        else:
            think_contents.append('')
            unmatched += 1

    df['think_content'] = think_contents

    valid = df[df['think_content'] != '']
    print(f'\n=== Conversion Summary ===')
    print(f'Total rows: {len(df)}')
    print(f'  Matched (has think_content): {matched}')
    print(f'  Unmatched (no think_content): {unmatched}')
    print(f'  Valid output rows: {len(valid)}')

    if len(valid) < len(df):
        df = valid
        print(f'  -> Outputting {len(df)} rows (dropped empty)')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f'\nSaved to {out_path}')

    if len(df) > 0:
        sample = df.iloc[0]
        print(f'\n--- Sample think_content ---')
        print(sample['think_content'][:500])


if __name__ == '__main__':
    main()
