"""Prepare recommendation cases for the primitives pipeline.

Builds MuSR-style JSON cases from RecThinker-format tool files + MemRec
memory. Each case packs ALL textual evidence into a single `context`
string with labeled sections (so Phase-1 `search`/`lookup`/`search_memory`
work like MuSR search over a narrative).

Input (from pipeline/step2_prepare_recthinker_data.py output):
  user_profiles.pkl
  user_history_valid.mem   (or user_history_test.mem)
  item_info.mem
  sim_user.mem             (optional collaborative evidence)
  UIKG_train.mem           (optional; used only as text neighbors)
  *.valid.inter / labels
  candidate lists (optional ranked_lists / candidate file)

  + MemRec memory.jsonl

Output:
  data/rec_{split}.json  with schema:
    {"examples": [
        {"user_id": "...",
         "context": "...",
         "question": "...",
         "candidates": ["id", ...],
         "answer_item": "id"}   # leave-one-out label
    ]}

Critical design choice vs old RecThinker induction:
  Collaborative / KG neighbors are rendered as TEXT summaries WITHOUT
  scores or vote counts, so agent thoughts stay preference-reasoning
  rather than "rank by score".
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

from ptools.ptools_common import (
    SECTION_CANDIDATES,
    SECTION_COLLAB,
    SECTION_HISTORY,
    SECTION_MEMORY,
    SECTION_PROFILE,
)


DEFAULT_QUESTION = (
    "Based on the user's profile, MemRec long-term memory, interaction "
    "history, and collaborative cues, rank the candidate items from most "
    "to least preferred. Prefer qualitative preference matching over any "
    "numerical scoring."
)


def load_memory_jsonl(path: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path or not Path(path).exists():
        return result
    with open(path, encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('type') == 'user':
                result[str(entry['id'])] = entry['memory']
    print(f'  Loaded {len(result)} MemRec user memories')
    return result


def _load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_json(path: Path):
    if not path.exists():
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _load_labels(inter_path: Path) -> dict[str, str]:
    """Load uid -> label item id from a TSV .inter file (last col = item).

    Accepts either `user_id\\titem_id` or full inter with timestamp;
    if multiple rows per user, keeps the last.
    """
    labels: dict[str, str] = {}
    if not inter_path.exists():
        return labels
    with open(inter_path, encoding='utf-8') as f:
        header = f.readline()
        cols = header.strip().split('\t')
        # detect columns
        try:
            ui = cols.index('user_id')
            ii = cols.index('item_id')
        except ValueError:
            ui, ii = 0, 1
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) <= max(ui, ii):
                continue
            labels[str(parts[ui])] = str(parts[ii])
    print(f'  Loaded {len(labels)} labels from {inter_path.name}')
    return labels


def _item_line(iid: str, item_info: dict) -> str:
    info = item_info.get(str(iid), {}) or {}
    title = info.get('title') or f'Item-{iid}'
    cat = info.get('category') or info.get('categories') or ''
    desc = info.get('description') or info.get('knowledge') or ''
    if isinstance(cat, list):
        cat = ', '.join(str(c) for c in cat)
    desc = str(desc).replace('\n', ' ').strip()
    if len(desc) > 180:
        desc = desc[:180] + '...'
    parts = [f'item_id={iid}', f'title={title}']
    if cat:
        parts.append(f'category={cat}')
    if desc:
        parts.append(f'description={desc}')
    return ' | '.join(parts)


def _render_history(uid: str, history: dict, item_info: dict,
                    max_items: int = 30) -> str:
    items = history.get(str(uid), []) or []
    # history may be list of item ids or "iid:attitude"
    lines = []
    for raw in items[-max_items:]:
        if isinstance(raw, dict):
            iid = str(raw.get('item_id', raw))
            att = raw.get("user's attitude") or raw.get('attitude') or 'like'
        else:
            s = str(raw)
            if ':' in s:
                iid, att = s.split(':', 1)
            else:
                iid, att = s, 'like'
        lines.append(f'- [{att}] {_item_line(iid, item_info)}')
    return '\n'.join(lines) if lines else '(no history)'


def _render_collab(uid: str, sim_user: dict, profiles: dict,
                   item_info: dict, kg: dict | None,
                   limit: int = 5) -> str:
    """Render collaborative evidence as TEXT only (no scores)."""
    blocks = []

    # similar users (hybrid / flat list)
    sim_list = []
    if isinstance(sim_user, dict):
        mode = sim_user.get('hybrid') or sim_user.get('dense') or sim_user
        if isinstance(mode, dict):
            sim_list = mode.get(str(uid), []) or []
        elif isinstance(sim_user.get(str(uid)), list):
            sim_list = sim_user.get(str(uid), [])
    for sid in list(sim_list)[:limit]:
        prof = profiles.get(str(sid), '')
        if isinstance(prof, dict):
            prof = prof.get('profile', str(prof))
        prof = str(prof).replace('\n', ' ').strip()
        if len(prof) > 220:
            prof = prof[:220] + '...'
        blocks.append(f'- similar_user={sid}: {prof or "(no profile)"}')

    # optional KG neighbors as textual related-user profiles (no vote counts)
    if kg:
        target_items = set(kg.get('User_' + str(uid), []) or [])
        if target_items:
            vote = defaultdict(int)
            for iid in target_items:
                for u in kg.get('Item_' + str(iid), []) or []:
                    if str(u) != str(uid):
                        vote[str(u)] += 1
            # sort internally but DO NOT expose counts in the text
            related = [u for u, _ in sorted(
                vote.items(), key=lambda x: x[1], reverse=True
            )[:limit]]
            for rid in related:
                prof = profiles.get(str(rid), '')
                if isinstance(prof, dict):
                    prof = prof.get('profile', str(prof))
                prof = str(prof).replace('\n', ' ').strip()
                if len(prof) > 220:
                    prof = prof[:220] + '...'
                blocks.append(
                    f'- related_user={rid}: {prof or "(no profile)"}'
                )

    return '\n'.join(blocks) if blocks else '(no collaborative evidence)'


def build_context(uid: str, profile: str, memory: str, history_text: str,
                  collab_text: str, candidates: list[str],
                  item_info: dict) -> str:
    cand_lines = []
    for iid in candidates:
        cand_lines.append(f'- {_item_line(str(iid), item_info)}')
    parts = [
        f'user_id={uid}',
        '',
        SECTION_PROFILE,
        profile.strip() if profile else '(no profile)',
        '',
        SECTION_MEMORY,
        memory.strip() if memory else '(no MemRec memory for this user)',
        '',
        SECTION_HISTORY,
        history_text,
        '',
        SECTION_COLLAB,
        collab_text,
        '',
        SECTION_CANDIDATES,
        '\n'.join(cand_lines) if cand_lines else '(no candidates)',
    ]
    return '\n'.join(parts)


def _load_candidates(data_dir: Path, uid: str, label: str,
                     item_info: dict, neg_k: int,
                     rng: random.Random) -> list[str]:
    """Build a candidate list: label + random negatives from item pool."""
    # Prefer precomputed candidate file if present
    for name in ('candidates.json', 'yelpAll_recagent_candidates.json'):
        path = data_dir / name
        if path.exists():
            data = _load_json(path)
            cands = data.get(str(uid))
            if cands:
                return [str(c) for c in cands]

    pool = [str(i) for i in item_info.keys() if str(i) != str(label)]
    if not pool:
        return [str(label)]
    negs = rng.sample(pool, min(neg_k, len(pool)))
    cands = [str(label)] + negs
    rng.shuffle(cands)
    return cands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', required=True,
                   help='RecThinker dataset dir (from step2)')
    p.add_argument('--memory', default=None,
                   help='MemRec memory.jsonl')
    p.add_argument('--split', default='valid',
                   choices=['valid', 'test', 'train'])
    p.add_argument('--labels', default=None,
                   help='Optional .inter label file; auto-detected if omitted')
    p.add_argument('--output', required=True,
                   help='Output JSON path (e.g. data/rec_valid.json)')
    p.add_argument('--max_users', type=int, default=0,
                   help='If >0, subsample this many users')
    p.add_argument('--neg_k', type=int, default=19,
                   help='Random negatives when no candidate file (default 19 → 20-way)')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    rng = random.Random(args.seed)

    print('1. Loading tool / memory files...')
    profiles = _load_pickle(data_dir / 'user_profiles.pkl')
    if not isinstance(profiles, dict):
        profiles = {}
    # normalize keys to str
    profiles = {str(k): v for k, v in profiles.items()}

    hist_name = {
        'valid': 'user_history_valid.mem',
        'test': 'user_history_test.mem',
        'train': 'user_history_valid.mem',
    }[args.split]
    history = _load_json(data_dir / hist_name)
    history = {str(k): v for k, v in (history or {}).items()}

    item_info = _load_json(data_dir / 'item_info.mem')
    item_info = {str(k): v for k, v in (item_info or {}).items()}

    sim_user = _load_json(data_dir / 'sim_user.mem')
    kg = _load_json(data_dir / 'UIKG_train.mem') or None

    memories = load_memory_jsonl(args.memory)

    # labels
    if args.labels:
        label_path = Path(args.labels)
    else:
        label_path = None
        for prefix in (data_dir.name, f'instructrec-{data_dir.name}', 'yelp'):
            for suffix in (f'{args.split}.inter', f'.{args.split}.inter'):
                cand = data_dir / f'{prefix}{suffix}' if suffix.startswith('.') \
                    else data_dir / f'{prefix}.{args.split}.inter'
                # try a few patterns
                pass
        # explicit common patterns
        for cand in [
            data_dir / f'{data_dir.name}.{args.split}.inter',
            data_dir / f'instructrec-{data_dir.name}.{args.split}.inter',
            data_dir / f'yelp.{args.split}.inter',
            data_dir / f'instructrec-yelp.{args.split}.inter',
            data_dir / f'{args.split}.inter',
        ]:
            if cand.exists():
                label_path = cand
                break
    if label_path is None or not label_path.exists():
        raise SystemExit(
            f'Could not find label .inter under {data_dir}. '
            f'Pass --labels explicitly.'
        )
    labels = _load_labels(label_path)

    print('2. Building cases...')
    uids = sorted(labels.keys())
    if args.max_users and args.max_users < len(uids):
        uids = rng.sample(uids, args.max_users)
        uids = sorted(uids)

    examples = []
    for uid in uids:
        label = labels[uid]
        profile = profiles.get(uid, '')
        if isinstance(profile, dict):
            profile = profile.get('profile', str(profile))
        # If MemRec memory was already prepended into profile by step2/step3,
        # still also place the raw memory into the MEMORY section.
        memory = memories.get(uid, '')
        hist_text = _render_history(uid, history, item_info)
        collab_text = _render_collab(uid, sim_user, profiles, item_info, kg)
        candidates = _load_candidates(
            data_dir, uid, label, item_info, args.neg_k, rng
        )
        if str(label) not in {str(c) for c in candidates}:
            candidates = [str(label)] + [str(c) for c in candidates]
        context = build_context(
            uid, str(profile), memory, hist_text, collab_text,
            candidates, item_info,
        )
        examples.append({
            'user_id': uid,
            'context': context,
            'question': DEFAULT_QUESTION,
            'candidates': [str(c) for c in candidates],
            'answer_item': str(label),
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'examples': examples}, f, ensure_ascii=False, indent=2)
    print(f'3. Wrote {len(examples)} cases → {out}')


if __name__ == '__main__':
    main()
