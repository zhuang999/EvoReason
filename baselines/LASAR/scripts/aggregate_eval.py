from __future__ import annotations
import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate LASAR evaluation shards')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    rows = []
    for path in args.inputs:
        row = json.loads(Path(path).read_text(encoding='utf-8'))
        if 'shard_index' in row:
            rows.append(row)
    if not rows:
        raise ValueError('No shard results supplied')
    expected = rows[0]['num_shards']
    shard_ids = sorted((row['shard_index'] for row in rows))
    if len(rows) != expected or shard_ids != list(range(expected)):
        raise ValueError(f'Expected shards 0..{expected - 1}, got {shard_ids}')
    if len({row['data_sha256'] for row in rows}) != 1:
        raise ValueError('Evaluation shards were produced from different datasets')
    if len({row['beam_width'] for row in rows}) != 1:
        raise ValueError('Evaluation shards use different beam widths')
    if len({row.get('fixed_latent_steps') for row in rows}) != 1:
        raise ValueError('Evaluation shards use different latent-step overrides')
    if len({row.get('direct_backbone', False) for row in rows}) != 1:
        raise ValueError('Evaluation shards mix direct and latent decoding')
    if len({row.get('hybrid_alpha') for row in rows}) != 1:
        raise ValueError('Evaluation shards use different hybrid weights')
    if len({row.get('hybrid_mode') for row in rows}) != 1:
        raise ValueError('Evaluation shards use different hybrid modes')
    if len({row.get('latent_adapter_scale') for row in rows}) != 1:
        raise ValueError('Evaluation shards use different latent adapter scales')
    if len({row.get('model_path') for row in rows}) != 1:
        raise ValueError('Evaluation shards were produced from different checkpoints')
    samples = sum((row['samples'] for row in rows))
    elapsed = sum((row['elapsed_seconds'] for row in rows))
    hit_counts = {key: sum((row['hit_counts'][key] for row in rows)) for key in rows[0]['hit_counts']}
    ndcg_sums = {key: sum((row['ndcg_sums'][key] for row in rows)) for key in rows[0]['ndcg_sums']}
    result = {**{f'HR@{key}': value / samples for key, value in hit_counts.items()}, **{f'NDCG@{key}': value / samples for key, value in ndcg_sums.items()}, 'mean_latent_steps': sum((row['latent_step_sum'] for row in rows)) / samples, 'latency_ms_per_sample_gpu_sum': elapsed * 1000.0 / samples, 'samples': samples, 'beam_width': rows[0]['beam_width'], 'fixed_latent_steps': rows[0].get('fixed_latent_steps'), 'direct_backbone': rows[0].get('direct_backbone', False), 'hybrid_alpha': rows[0].get('hybrid_alpha'), 'hybrid_mode': rows[0].get('hybrid_mode'), 'latent_adapter_scale': rows[0].get('latent_adapter_scale'), 'model_path': rows[0].get('model_path'), 'data_sha256': rows[0]['data_sha256'], 'shards': expected}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
