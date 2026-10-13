from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
from lasar.data import sha256_file
EXPECTED_ROWS = {'beauty': 22363, 'sports': 35598, 'toys': 19412}

def parse_args():
    parser = argparse.ArgumentParser(description='Validate a paper-baseline LASAR run before using GPUs')
    parser.add_argument('--dataset', required=True, choices=sorted(EXPECTED_ROWS))
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--anchor-dir')
    parser.add_argument('--model')
    parser.add_argument('--require-anchors', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    expected_rows = EXPECTED_ROWS[args.dataset]
    user_orders = {}
    data_hashes = {}
    for split in ('train', 'val', 'test'):
        path = data_dir / f'training_prediction_sid_data_{split}.parquet'
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=['user_id', 'description', 'groundtruth'])
        if len(frame) != expected_rows:
            raise ValueError(f'{path}: expected {expected_rows} rows, found {len(frame)}')
        if frame.isnull().any().any():
            raise ValueError(f'{path}: null values found')
        if frame['user_id'].duplicated().any():
            raise ValueError(f'{path}: duplicate user_id values found')
        user_orders[split] = frame['user_id'].astype(str).tolist()
        data_hashes[split] = sha256_file(path)
    if user_orders['train'] != user_orders['val'] or user_orders['train'] != user_orders['test']:
        raise ValueError('train/val/test user ordering differs')
    metadata = Path(args.metadata)
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    if args.model and (not Path(args.model).exists()):
        raise FileNotFoundError(args.model)
    if args.anchor_dir:
        anchor_dir = Path(args.anchor_dir)
        for split in ('train', 'val'):
            path = anchor_dir / f'{split}.pt'
            if not path.is_file():
                if args.require_anchors:
                    raise FileNotFoundError(path)
                continue
            payload = torch.load(path, map_location='cpu', weights_only=False)
            anchors = payload['anchors'] if isinstance(payload, dict) else payload
            if len(anchors) != expected_rows:
                raise ValueError(f'{path}: expected {expected_rows} anchors, found {len(anchors)}')
            if args.require_anchors:
                if not isinstance(payload, dict) or not payload.get('data_sha256'):
                    raise ValueError(f'{path}: missing data fingerprint; regenerate formal-run anchors')
                if payload['data_sha256'] != data_hashes[split]:
                    raise ValueError(f'{path}: data fingerprint mismatch')
    print(f'preflight ok: dataset={args.dataset} rows={expected_rows}')
    for split, digest in data_hashes.items():
        print(f'{split}_sha256={digest}')
if __name__ == '__main__':
    main()
