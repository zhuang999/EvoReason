from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / 'data'
DATASET = 'onerec-sports-prefix'
OUTPUT_DIR = BASE_DIR / 'outputs/sports-prefix/prl-seed42'
PYTHON = Path(os.environ.get('PYTHON', sys.executable))
DEFAULT_METADATA = REPO_ROOT / 'data/Sports.pretrain.json'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='Run training.')
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES value.')
    parser.add_argument('--python', type=Path, default=PYTHON)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--early-stop', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--eval-batch-size', type=int, default=4096)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--emb-size', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--temp-scale', type=float, default=5.0)
    parser.add_argument('--reason-step', type=int, default=2)
    parser.add_argument('--filter-history', action='store_true', help='Filter previously visited item IDs during validation and test.')
    parser.add_argument('--sid-init-ckpt', type=Path, default=Path(os.environ.get('SID_INIT_CKPT', '')) if os.environ.get('SID_INIT_CKPT') else None, help='Optional OneRec SID checkpoint used to initialize ReaRec item embeddings.')
    parser.add_argument('--sid-init-file', type=Path, default=OUTPUT_DIR / 'sid_token_mean_chunks.pt', help='Cached ReaRec item embedding initialization generated from --sid-init-ckpt.')
    return parser.parse_args()

def parse_metric_block(text: str) -> dict[str, float]:
    pairs = re.findall('([A-Z]+@\\d+)\\s*:\\s*([0-9.]+)', text)
    return {key: float(value) for key, value in pairs}

def write_summary(log_path: Path, model_path: Path) -> None:
    text = log_path.read_text(encoding='utf-8', errors='replace')
    best_match = re.findall('Best Iter\\(valid\\)=\\s*(\\d+)\\s+valid=\\((.*?)\\)', text, re.S)
    test_match = re.findall('Test After Training:\\s+\\((.*?)\\)', text, re.S)
    if not best_match or not test_match:
        raise RuntimeError(f'could not parse final metrics from {log_path}')
    best_epoch, valid_block = best_match[-1]
    summary = {'model': 'ReaRec-PRL', 'dataset': DATASET, 'seed': 42, 'source': str(log_path), 'checkpoint': str(model_path), 'best_epoch': int(best_epoch), 'selection_metric': 'NDCG@10', 'valid': parse_metric_block(valid_block), 'test': parse_metric_block(test_match[-1])}
    summary_path = model_path.parent / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))

def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else BASE_DIR / args.output_dir
    data_manifest = DATA_DIR / DATASET / 'manifest.json'
    prepare_cmd = [str(args.python), str(BASE_DIR / 'prepare_onerec_sports_prefix.py'), '--output-dir', str(DATA_DIR / DATASET)]
    train_log = output_dir / 'train.log'
    model_path = output_dir / 'best.pt'
    rea_log = output_dir / 'rearec.log'
    sid_init_file = args.sid_init_file if args.sid_init_file.is_absolute() else BASE_DIR / args.sid_init_file
    item_emb_init = sid_init_file if args.sid_init_ckpt else None
    init_cmd = [str(args.python), str(BASE_DIR / 'build_sid_item_init.py'), '--metadata-path', str(DEFAULT_METADATA), '--checkpoint-path', str(args.sid_init_ckpt) if args.sid_init_ckpt else '', '--output', str(sid_init_file), '--emb-size', str(args.emb_size)]
    train_cmd = [str(args.python), 'main.py', '--model_name', 'PRL', '--path', str(DATA_DIR), '--dataset', DATASET, '--gpu', args.gpu, '--random_seed', '42', '--main_metric', 'NDCG@10', '--topk', '5,10,20,50', '--metric', 'NDCG,HR,MRR', '--epoch', str(args.epoch), '--early_stop', str(args.early_stop), '--batch_size', str(args.batch_size), '--eval_batch_size', str(args.eval_batch_size), '--num_workers', str(args.num_workers), '--lr', str(args.lr), '--l2', str(args.l2), '--emb_size', str(args.emb_size), '--dropout', str(args.dropout), '--temperature', str(args.temperature), '--temp_scale', str(args.temp_scale), '--reason_step', str(args.reason_step), '--log_file', str(rea_log), '--model_path', str(model_path), '--regenerate', '1']
    if args.filter_history:
        train_cmd.extend(['--filter_history', '1'])
    if item_emb_init:
        train_cmd.extend(['--item_emb_init', str(item_emb_init)])
    print('Data manifest:', data_manifest)
    print('Output dir:', output_dir)
    print('Prepare command:', ' '.join(prepare_cmd))
    if args.sid_init_ckpt:
        print('SID init command:', ' '.join(init_cmd))
    print('Train command:', ' '.join(train_cmd))
    if not args.execute:
        print('Dry run only. Add --execute to start GPU training.')
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(prepare_cmd, cwd=BASE_DIR, check=True)
    if args.sid_init_ckpt and (not sid_init_file.exists()):
        subprocess.run(init_cmd, cwd=BASE_DIR, check=True)
    with train_log.open('w', encoding='utf-8') as log_f:
        process = subprocess.Popen(train_cmd, cwd=BASE_DIR / 'upstream', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='')
            log_f.write(line)
        return_code = process.wait()
    if return_code != 0:
        return return_code
    write_summary(rea_log, model_path)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
