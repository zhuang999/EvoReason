#!/usr/bin/env python3
"""Fix cachier pickle cache: clear 'being computed' flags left by killed processes."""
import pickle
import pathlib

CACHE_FILE = pathlib.Path(__file__).resolve().parent / 'llm_cache' / '.secretagent.llm_util._llm_impl'

with open(CACHE_FILE, 'rb') as f:
    d = pickle.load(f)

print(f'Loaded {len(d)} entries')

fixed = 0
for key, entry in d.items():
    attrs = [a for a in dir(entry) if not a.startswith('__')]
    # Check for being_computed / _being_computed / processing flags
    for attr in attrs:
        if 'comput' in attr.lower() or 'process' in attr.lower() or 'lock' in attr.lower():
            val = getattr(entry, attr, None)
            if val:
                print(f'  Entry {key[:16]}...: {attr} = {val}')
                setattr(entry, attr, False if isinstance(val, bool) else None)
                fixed += 1

print(f'Fixed {fixed} entries')

with open(CACHE_FILE, 'wb') as f:
    pickle.dump(d, f)

print('Saved fixed cache')
