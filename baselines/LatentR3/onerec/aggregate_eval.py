from __future__ import annotations
import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate LatentR3 OneRec evaluation shards')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = [json.loads(Path(path).read_text(encoding='utf-8')) for path in args.inputs]
    expected = rows[0]['num_shards']
    shard_ids = sorted((row['shard_index'] for row in rows))
    if len(rows) != expected or shard_ids != list(range(expected)):
        raise ValueError(f'Expected shards 0..{expected - 1}, got {shard_ids}')
    matching_fields = ('data_sha256', 'metadata_sha256', 'beam_width', 'checkpoint', 'data_split', 'limit', 'limit_scope', 'mode', 'prompt_protocol', 'end_k', 'latent_component', 'latent_attention_enabled')
    for field in matching_fields:
        values = {json.dumps(row.get(field), sort_keys=True) for row in rows}
        if len(values) != 1:
            raise ValueError(f'Shard {field} values differ')
    samples = sum((row['samples'] for row in rows))
    if rows[0].get('limit') is not None and samples != rows[0]['limit']:
        raise ValueError(f"Global limit is {rows[0]['limit']}, but shards contain {samples} samples")
    hit_counts = {key: sum((row['hit_counts'][key] for row in rows)) for key in rows[0]['hit_counts']}
    ndcg_sums = {key: sum((row['ndcg_sums'][key] for row in rows)) for key in rows[0]['ndcg_sums']}
    result = {**{f'HR@{key}': value / samples for key, value in hit_counts.items()}, **{f'NDCG@{key}': value / samples for key, value in ndcg_sums.items()}, 'samples': samples, 'beam_width': rows[0]['beam_width'], 'data_sha256': rows[0]['data_sha256'], 'metadata_sha256': rows[0].get('metadata_sha256'), 'checkpoint': rows[0].get('checkpoint'), 'data_split': rows[0].get('data_split'), 'shards': expected, 'limit': rows[0].get('limit'), 'limit_scope': rows[0].get('limit_scope'), 'mode': rows[0].get('mode'), 'prompt_protocol': rows[0].get('prompt_protocol'), 'end_k': rows[0].get('end_k'), 'latent_component': rows[0].get('latent_component'), 'latent_attention_enabled': rows[0].get('latent_attention_enabled'), 'elapsed_seconds': sum((row.get('elapsed_seconds', 0.0) for row in rows))}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    records_output = output.with_name('records.jsonl')
    with records_output.open('w', encoding='utf-8') as target:
        for path in sorted((Path(value).with_name('records.jsonl') for value in args.inputs)):
            if not path.exists():
                raise ValueError(f'Missing shard records file: {path}')
            with path.open('r', encoding='utf-8') as source:
                for line in source:
                    if line.strip():
                        target.write(line)
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
