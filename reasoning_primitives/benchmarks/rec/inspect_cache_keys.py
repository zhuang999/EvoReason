#!/usr/bin/env python3
"""Inspect cachier pickle keys to understand key format."""
import pickle
import pathlib

CACHE_FILE = pathlib.Path(__file__).resolve().parent / 'llm_cache' / '.secretagent.llm_util._llm_impl'

with open(CACHE_FILE, 'rb') as f:
    d = pickle.load(f)

print(f'Total entries: {len(d)}')
for i, (key, entry) in enumerate(d.items()):
    print(f'\n--- Entry {i} ---')
    print(f'  Key type: {type(key)}')
    print(f'  Key repr: {repr(key)[:200]}')
    val = getattr(entry, 'value', None)
    if val is None:
        print(f'  Entry type: {type(entry)}')
        print(f'  Entry attrs: {[a for a in dir(entry) if not a.startswith("_")]}')
        continue
    text, stats = val
    print(f'  Value text len: {len(text)}')
    print(f'  Stats: {stats}')
    print(f'  Text preview: {text[:200]}')
    if i >= 2:
        break
