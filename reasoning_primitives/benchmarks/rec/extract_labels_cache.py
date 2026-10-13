#!/usr/bin/env python3
"""Extract categorize results from cachier pickle directly (no cachier lock issues).

Reads the pickle file, parses all categorize responses, rebuilds
prompt->response mapping by hashing, then reconstructs labels.
"""
import hashlib
import json
import pathlib
import pickle
import re
import sys

_BENCHMARK_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCHMARK_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_BENCHMARK_DIR))

from induction import load_thoughts, CATEGORIZE_PROMPT, _format_batch, _parse_categorize

TASK_DESC = (
    'Item recommendation ranking: given a user recommendation context '
    '(profile, MemRec long-term memory, interaction history, textual '
    'collaborative cues, and candidate item attributes), rank the '
    'candidates from most to least preferred by qualitatively matching '
    'inferred preferences to item attributes. Do NOT rely on numerical '
    'scores, embeddings, or ranking algorithms — reason in natural language.'
)

BATCH_SIZE = 30
CACHE_FILE = _BENCHMARK_DIR / 'llm_cache' / '.secretagent.llm_util._llm_impl'
TRACE_PATH = _BENCHMARK_DIR / 'results' / 'results_full.jsonl'
OUTPUT_PATH = _BENCHMARK_DIR / 'induction_out' / 'original' / 'rec' / 'labels_cache.json'


def _hash_prompt(prompt: str, model: str) -> str:
    key = (prompt, model)
    return hashlib.sha256(pickle.dumps(key)).hexdigest()


def main():
    print(f'Loading pickle cache from {CACHE_FILE} ...')
    with open(CACHE_FILE, 'rb') as f:
        cache_dict = pickle.load(f)
    print(f'  {len(cache_dict)} cache entries')

    cache_by_hash = {}
    for key_bytes, entry in cache_dict.items():
        val = getattr(entry, 'value', None)
        if val is None:
            continue
        text, stats = val
        if not text or not isinstance(text, str):
            continue
        m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if not m:
            continue
        try:
            arr = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(arr, list):
            continue
        is_categorize = any(
            isinstance(e, dict) and 'index' in e and 'category' in e
            for e in arr
        )
        if is_categorize:
            cache_by_hash[key_bytes if isinstance(key_bytes, str) else key_bytes.hex()] = text

    print(f'  {len(cache_by_hash)} categorize responses found')

    items = load_thoughts(TRACE_PATH, only_correct=True)
    print(f'loaded {len(items)} thoughts')

    labels = [None] * len(items)
    matched_batches = 0
    unmatched_batches = 0

    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        prompt = CATEGORIZE_PROMPT.format(
            task_desc=TASK_DESC,
            items_block=_format_batch(batch, start),
        )
        h = _hash_prompt(prompt, 'openai/QwQ-32B')

        text = cache_by_hash.get(h)
        if text is None:
            unmatched_batches += 1
            if unmatched_batches <= 5:
                print(f'  no cache hit for batch {start}')
            continue

        for idx, cat in _parse_categorize(text):
            if 0 <= idx < len(items):
                labels[idx] = cat
        matched_batches += 1

        batch_idx = start // BATCH_SIZE
        if batch_idx % 100 == 0:
            n = sum(1 for l in labels if l)
            print(f'  batch {start}/{len(items)}: matched={matched_batches}, labeled={n}')

    n_labeled = sum(1 for l in labels if l)
    print(f'Final: matched={matched_batches}, unmatched={unmatched_batches}, labeled={n_labeled}/{len(items)}')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(labels, ensure_ascii=False), encoding='utf-8')
    print(f'Saved to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
