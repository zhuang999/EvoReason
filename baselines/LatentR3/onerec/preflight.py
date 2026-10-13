from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from onerec.data import sha256_file
EXPECTED = {'beauty': {'rows': 22363, 'items': 12101, 'unique_sids': 12093}, 'sports': {'rows': 35598, 'items': 18357, 'unique_sids': 18357}, 'toys': {'rows': 19412, 'items': 11924, 'unique_sids': 11924}}

def main() -> None:
    parser = argparse.ArgumentParser(description='CPU preflight for LatentR3 OneRec matched runs')
    parser.add_argument('--dataset', choices=sorted(EXPECTED), required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--model')
    args = parser.parse_args()
    expected = EXPECTED[args.dataset]
    metadata_path = Path(args.metadata)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if len(metadata) != expected['items']:
        raise ValueError(f"Expected {expected['items']} metadata items, found {len(metadata)}")
    valid_sids = {str(item['sid']) for item in metadata.values() if item.get('sid')}
    if len(valid_sids) != expected['unique_sids']:
        raise ValueError(f"Expected {expected['unique_sids']} unique SIDs, found {len(valid_sids)}")
    user_orders = {}
    hashes = {}
    for split in ('train', 'val', 'test'):
        path = Path(args.data_dir) / f'training_prediction_sid_data_{split}.parquet'
        frame = pd.read_parquet(path, columns=['user_id', 'description', 'groundtruth'])
        if len(frame) != expected['rows']:
            raise ValueError(f"{path}: expected {expected['rows']} rows, found {len(frame)}")
        if frame.isnull().any().any() or frame['user_id'].duplicated().any():
            raise ValueError(f'{path}: null values or duplicate users found')
        unknown = set(frame['groundtruth'].astype(str)).difference(valid_sids)
        if unknown:
            raise ValueError(f'{path}: {len(unknown)} targets are absent from metadata')
        user_orders[split] = frame['user_id'].astype(str).tolist()
        hashes[split] = sha256_file(path)
    if user_orders['train'] != user_orders['val'] or user_orders['train'] != user_orders['test']:
        raise ValueError('train/val/test user ordering differs')
    if args.model and (not Path(args.model).exists()):
        raise FileNotFoundError(args.model)
    collisions = expected['items'] - expected['unique_sids']
    print(f"preflight ok: dataset={args.dataset} rows={expected['rows']} items={expected['items']} unique_sids={expected['unique_sids']} collisions={collisions}")
    for split, digest in hashes.items():
        print(f'{split}_sha256={digest}')
if __name__ == '__main__':
    main()
