from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from lasar.cot import merge_semantic_boundaries, split_structured_cot
PAPER_SPORTS_DISTRIBUTION = {3: 0.834, 4: 0.163, 5: 0.002}

def parse_args():
    parser = argparse.ArgumentParser(description='Batch BGE semantic resegmentation for LASAR teacher CoT')
    parser.add_argument('--responses', required=True)
    parser.add_argument('--semantic-model', required=True)
    parser.add_argument('--output')
    parser.add_argument('--report', required=True)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--calibrate-sports', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device')
    parser.add_argument('--min-segments', type=int, default=3)
    parser.add_argument('--max-segments', type=int, default=8)
    return parser.parse_args()

def load_rows(path: str, limit: int | None) -> list[dict]:
    rows = []
    seen = set()
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = int(row['row_id'])
            if row_id in seen:
                raise ValueError(f'Duplicate row_id {row_id}')
            seen.add(row_id)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError('Teacher response file is empty')
    return rows

def distribution(counts: list[int]) -> dict[int, float]:
    frequencies = Counter(counts)
    return {int(key): float(frequencies[key] / len(counts)) for key in sorted(frequencies)}

def calibrate_threshold(similarities: list[list[float]]) -> tuple[float, dict]:
    candidates = np.linspace(0.35, 0.95, 601)
    target_total = sum(PAPER_SPORTS_DISTRIBUTION.values())
    target = {key: value / target_total for key, value in PAPER_SPORTS_DISTRIBUTION.items()}
    best = None
    for threshold in candidates:
        counts = [max(3, 1 + sum((score < threshold for score in row_scores))) for row_scores in similarities]
        observed = distribution(counts)
        error = sum((abs(observed.get(key, 0.0) - target.get(key, 0.0)) for key in range(3, 9)))
        candidate = (error, abs(float(threshold) - 0.82), float(threshold), observed)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    return (best[2], {'l1_error': best[0], 'observed': best[3], 'target': target})

def main() -> None:
    args = parse_args()
    if args.threshold is None and (not args.calibrate_sports):
        raise ValueError('Set --threshold or --calibrate-sports')
    rows = load_rows(args.responses, args.limit)
    structured = [split_structured_cot(row['cot'], max_segments=args.max_segments, min_segments=args.min_segments) for row in rows]
    flat_segments = [segment for segments in structured for segment in segments]
    model = SentenceTransformer(args.semantic_model, device=args.device)
    flat_embeddings = model.encode(flat_segments, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    offsets = np.cumsum([0] + [len(segments) for segments in structured])
    similarities = []
    for index, segments in enumerate(structured):
        vectors = flat_embeddings[offsets[index]:offsets[index + 1]]
        _, scores = merge_semantic_boundaries(segments, vectors, threshold=-1.0, min_segments=1, max_segments=args.max_segments)
        similarities.append(scores)
    calibration = None
    threshold = args.threshold
    if args.calibrate_sports:
        threshold, calibration = calibrate_threshold(similarities)
    output_rows = []
    counts = []
    for index, (row, segments) in enumerate(zip(rows, structured)):
        vectors = flat_embeddings[offsets[index]:offsets[index + 1]]
        merged, scores = merge_semantic_boundaries(segments, vectors, threshold=threshold, min_segments=args.min_segments, max_segments=args.max_segments)
        counts.append(len(merged))
        output_rows.append({'row_id': int(row['row_id']), 'data_sha256': row['data_sha256'], 'cot': row['cot'], 'segments': merged, 'semantic_boundary_scores': scores, 'semantic_boundary_threshold': threshold})
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf-8') as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + '\n')
    report = {'responses': str(Path(args.responses).resolve()), 'semantic_model': args.semantic_model, 'samples': len(rows), 'threshold': threshold, 'min_segments': args.min_segments, 'max_segments': args.max_segments, 'segment_distribution': distribution(counts), 'calibration': calibration}
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
