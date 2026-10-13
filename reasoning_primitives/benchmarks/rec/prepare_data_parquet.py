"""Prepare recommendation cases for the primitives pipeline from parquet data.

Adaptation of prepare_data.py that reads Beauty parquet files directly
instead of RecThinker pkl/mem format.

Input:
  - training_prediction_sid_data_*.parquet (description, groundtruth, title, categories, user_id)
  - memory.jsonl (MemRec user long-term memory)

Output:
  data/rec_{split}.json with schema:
    {"examples": [
        {"user_id": "...",
         "context": "...",
         "question": "...",
         "candidates": [],
         "answer_item": "id"}
    ]}
"""

from __future__ import annotations

import argparse
import json
import re
import random
from pathlib import Path


SECTION_MEMORY = '=== USER MEMORY (MemRec) ==='
SECTION_HISTORY = '=== INTERACTION HISTORY ==='
SECTION_COLLAB = '=== COLLABORATIVE EVIDENCE ==='
SECTION_CANDIDATES = '=== CANDIDATE ITEMS ==='
SECTION_PROFILE = '=== USER PROFILE ==='


DEFAULT_QUESTION = (
    "Based on the user's profile, MemRec long-term memory, and interaction "
    "history, predict what product the user would most likely want to purchase next. "
    "Analyze their behavioral evidence, model their latent preferences, infer their "
    "intent, and provide a detailed recommendation rationale."
)

SID_PATTERN = re.compile(
    r'<\|sid_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)><s_d_(\d+)><\|sid_end\|>'
)


def load_pretrain_item_map(path: str | None) -> dict[str, dict]:
    sid_to_item: dict[str, dict] = {}
    if not path or not Path(path).exists():
        return sid_to_item
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    for item_id_str, info in data.items():
        sid = info.get('sid')
        if sid:
            sid_to_item[sid] = {
                'item_id': item_id_str,
                'title': info.get('title', ''),
                'categories': info.get('categories', ''),
                'description': info.get('description', ''),
            }
    print(f'  Loaded {len(sid_to_item)} items from pretrain (sid->item)')
    return sid_to_item


def sid_str_from_text(text: str) -> str | None:
    m = re.search(
        r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>',
        text,
    )
    return m.group(0) if m else None


def extract_sid_from_text(text: str) -> str | None:
    return sid_str_from_text(text)


def _item_str(item: dict, include_desc: bool = False, max_desc_len: int = 80) -> str:
    parts = [item['title']]
    if item['categories']:
        parts.append(f'[{item["categories"]}]')
    if include_desc and item['description']:
        desc_short = item['description'][:max_desc_len]
        parts.append(f'— {desc_short}')
    return ' | '.join(parts)


def resolve_sids_in_text(text: str, sid_map: dict[str, dict]) -> str:
    def replacer(m):
        sid_key = m.group(0)
        item = sid_map.get(sid_key)
        if item:
            return _item_str(item, include_desc=False)
        return sid_key
    return SID_PATTERN.sub(replacer, text)


def parse_description(desc: str, sid_map: dict[str, dict]) -> dict:
    readable = resolve_sids_in_text(desc, sid_map)
    sids = SID_PATTERN.findall(desc)
    sid_tokens = SID_PATTERN.finditer(desc)
    history_items = []
    for m in sid_tokens:
        sid_key = m.group(0)
        item = sid_map.get(sid_key)
        if item:
            history_items.append(_item_str(item, include_desc=True))
        else:
            history_items.append(sid_key)

    prefix_match = re.match(r'^[^<]+', readable)
    profile = prefix_match.group(0).strip() if prefix_match else ''

    return {
        'profile': profile if profile else '(no profile)',
        'history_items': history_items,
    }


def build_context_from_parquet(
    uid: str,
    description: str,
    groundtruth: str,
    memory: str,
    sid_map: dict[str, dict],
) -> str:
    parsed = parse_description(description, sid_map)

    profile = parsed['profile']

    hist_text = '\n'.join(f'- {item}' for item in parsed['history_items'])
    if not hist_text:
        hist_text = '(no interaction history)'

    gt_sid = extract_sid_from_text(groundtruth)
    answer_item = gt_sid or groundtruth
    gt_info = sid_map.get(answer_item) if gt_sid else None
    target_line = f'- target_item={answer_item}'
    if gt_info:
        target_line += f'\n  {_item_str(gt_info, include_desc=True)}'

    sections = [
        f'user_id={uid}',
        '',
        SECTION_PROFILE,
        profile if profile else '(no profile)',
        '',
        SECTION_MEMORY,
        memory.strip() if memory else '(no MemRec memory for this user)',
        '',
        SECTION_HISTORY,
        hist_text,
        '',
        SECTION_COLLAB,
        '(no collaborative evidence)',
        '',
        SECTION_CANDIDATES,
        target_line,
    ]
    return '\n'.join(sections)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parquet', required=True,
                   help='Beauty parquet file (training_prediction_sid_data_*.parquet)')
    p.add_argument('--pretrain', default=None,
                   help='Beauty.pretrain.json for SID->item mapping')
    p.add_argument('--memory', default=None,
                   help='MemRec memory.jsonl')
    p.add_argument('--output', required=True,
                   help='Output JSON path (e.g. data/rec_valid.json)')
    p.add_argument('--max_users', type=int, default=0,
                   help='If >0, subsample this many users')
    p.add_argument('--num_negatives', type=int, default=0,
                   help='Number of negative candidates to sample (0=no candidate list)')
    p.add_argument('--max_desc_len', type=int, default=80,
                   help='Max characters for item description in candidates')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    import pandas as pd

    rng = random.Random(args.seed)

    print(f'1. Loading parquet: {args.parquet}')
    df = pd.read_parquet(args.parquet)
    print(f'   {len(df)} rows, columns: {list(df.columns)}')

    sid_map = load_pretrain_item_map(args.pretrain)

    all_sids = list(sid_map.keys())

    memories = {}
    if args.memory:
        mem_path = Path(args.memory)
        if mem_path.exists():
            with open(mem_path, encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get('type') == 'user':
                        memories[str(entry['id'])] = entry['memory']
            print(f'  Loaded {len(memories)} MemRec user memories')

    print('2. Building cases...')
    examples = []
    uids = list(range(len(df)))
    if args.max_users and args.max_users < len(df):
        uids = rng.sample(uids, args.max_users)
        uids = sorted(uids)

    for idx in uids:
        row = df.iloc[idx]
        uid = str(row.get('user_id', idx))
        description = str(row.get('description', ''))
        groundtruth = str(row.get('groundtruth', ''))

        memory = memories.get(uid, '') or memories.get(str(int(uid) - 1), '')

        context = build_context_from_parquet(
            uid, description, groundtruth, memory, sid_map
        )

        gt_sid = extract_sid_from_text(groundtruth)
        answer_item = gt_sid or groundtruth

        candidates = [answer_item]
        cand_section = f'- {answer_item}'
        if args.num_negatives and all_sids:
            neg_sids = [s for s in all_sids if s != answer_item]
            neg_sample = rng.sample(neg_sids, min(args.num_negatives, len(neg_sids)))
            candidates = [answer_item] + neg_sample
            rng.shuffle(candidates)
            cand_lines = []
            for cid in candidates:
                cinfo = sid_map.get(cid)
                if cinfo:
                    cand_lines.append(f'- {cid}: {_item_str(cinfo, include_desc=True)}')
                else:
                    cand_lines.append(f'- {cid}')
            cand_section = '\n'.join(cand_lines)

        context = build_context_from_parquet(
            uid, description, groundtruth, memory, sid_map
        )

        context = context.replace(
            f'- target_item={answer_item}',
            cand_section,
        )

        examples.append({
            'user_id': uid,
            'context': context,
            'question': DEFAULT_QUESTION,
            'candidates': candidates,
            'answer_item': answer_item,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'examples': examples}, f, ensure_ascii=False, indent=2)
    print(f'3. Wrote {len(examples)} cases -> {out}')


if __name__ == '__main__':
    main()
