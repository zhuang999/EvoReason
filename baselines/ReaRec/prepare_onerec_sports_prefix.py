from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE = REPO_ROOT / 'data/Sports.sequential_data_processed.txt'
DEFAULT_METADATA = REPO_ROOT / 'data/Sports.pretrain.json'
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'data/onerec-sports-prefix'
EXPECTED_USERS = 35598
EXPECTED_ITEMS = 18357
EXPECTED_INTERACTIONS = 294488
MAX_HISTORY = 50

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-path', type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument('--metadata-path', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()

def read_sequences(path: Path) -> list[tuple[int, list[int]]]:
    rows: list[tuple[int, list[int]]] = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            user_id = int(parts[0])
            items = [int(value) for value in parts[1:]]
            if len(items) < 3:
                raise ValueError(f'{path}:{line_no} has fewer than 3 interactions')
            rows.append((user_id, items))
    rows.sort(key=lambda row: row[0])
    return rows

def validate(sequences: list[tuple[int, list[int]]], metadata_path: Path) -> None:
    with metadata_path.open('r', encoding='utf-8') as f:
        metadata = json.load(f)
    users = {user_id for user_id, _ in sequences}
    seq_items = {item for _, items in sequences for item in items}
    meta_items = {int(item_id) for item_id in metadata}
    interactions = sum((len(items) for _, items in sequences))
    expected = (EXPECTED_USERS, EXPECTED_ITEMS, EXPECTED_INTERACTIONS)
    actual = (len(users), len(seq_items), interactions)
    if actual != expected:
        raise ValueError(f'unexpected Sports shape: actual={actual}, expected={expected}')
    if seq_items != meta_items:
        missing_meta = sorted(seq_items - meta_items)
        unused_meta = sorted(meta_items - seq_items)
        raise ValueError(f'metadata mismatch: missing_meta={len(missing_meta)}, unused_meta={len(unused_meta)}')

def write_split(path: Path, rows: list[tuple[int, int, list[int]]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'item_id', 'history_id'])
        for user_id, target_item, history in rows:
            writer.writerow([user_id - 1, target_item - 1, ' '.join((str(item - 1) for item in history[-MAX_HISTORY:]))])

def main() -> None:
    args = parse_args()
    sequences = read_sequences(args.sequence_path)
    validate(sequences, args.metadata_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {'train': [], 'valid': [], 'test': []}
    for user_id, items in sequences:
        for end in range(1, len(items) - 2):
            split_rows['train'].append((user_id, items[end], items[:end]))
        split_rows['valid'].append((user_id, items[-2], items[:-2]))
        split_rows['test'].append((user_id, items[-1], items[:-1]))
    dataset_name = args.output_dir.name
    for split, rows in split_rows.items():
        write_split(args.output_dir / f'{dataset_name}.{split}.remap.csv', rows)
    manifest = {'dataset': dataset_name, 'users': EXPECTED_USERS, 'items': EXPECTED_ITEMS, 'interactions': EXPECTED_INTERACTIONS, 'rows_per_split': {split: len(rows) for split, rows in split_rows.items()}, 'max_history': MAX_HISTORY, 'item_id_range': [0, EXPECTED_ITEMS - 1], 'split': {'train_target': 'all prefixes before the validation/test holdouts', 'valid_target': 'second-last', 'test_target': 'last'}}
    with (args.output_dir / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
