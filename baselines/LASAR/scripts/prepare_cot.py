from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from lasar.cot import TEACHER_PROMPT, split_structured_cot
from lasar.data import sha256_file

def parse_args():
    parser = argparse.ArgumentParser(description='Export teacher prompts or normalize teacher CoT responses')
    parser.add_argument('--data', required=True, help='OneRec-Think recommendation parquet')
    parser.add_argument('--ra-data', help='Aligned training_RA parquet providing semantic history, target title, and categories')
    parser.add_argument('--output', required=True, help='Output JSONL')
    parser.add_argument('--responses', help='JSONL containing row_id and cot fields')
    parser.add_argument('--max-segments', type=int, default=8)
    parser.add_argument('--semantic-model', help='Optional SentenceTransformer used to merge semantically redundant adjacent steps')
    parser.add_argument('--merge-threshold', type=float, default=0.82)
    return parser.parse_args()

def read_responses(path: str) -> tuple[dict[int, str], str]:
    responses = {}
    data_sha256 = None
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            item = json.loads(line)
            row_id = int(item['row_id'])
            if row_id in responses:
                raise ValueError(f'Duplicate teacher response for row {row_id}')
            row_hash = item.get('data_sha256')
            if not row_hash:
                raise ValueError(f'Teacher response {row_id} is missing data_sha256')
            if data_sha256 is None:
                data_sha256 = row_hash
            elif row_hash != data_sha256:
                raise ValueError('Teacher responses contain mixed data fingerprints')
            responses[row_id] = item['cot']
    if data_sha256 is None:
        raise ValueError('Teacher response file is empty')
    return (responses, data_sha256)

def semantic_segments(text, model, max_segments, threshold):
    segments = split_structured_cot(text, max_segments=max(64, max_segments))
    while len(segments) > 1:
        embeddings = model.encode(segments, normalize_embeddings=True)
        similarities = [float(embeddings[i] @ embeddings[i + 1]) for i in range(len(segments) - 1)]
        best_index = max(range(len(similarities)), key=similarities.__getitem__)
        if len(segments) <= max_segments and similarities[best_index] < threshold:
            break
        segments[best_index:best_index + 2] = [f'{segments[best_index]} {segments[best_index + 1]}']
    return segments

def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.data).reset_index(drop=True)
    data_sha256 = sha256_file(args.data)
    semantic_frame = frame
    if args.ra_data:
        semantic_frame = pd.read_parquet(args.ra_data).reset_index(drop=True)
        if len(semantic_frame) != len(frame):
            raise ValueError('Recommendation and RA parquet row counts differ')
        for column in ('user_id', 'groundtruth'):
            if not frame[column].astype(str).equals(semantic_frame[column].astype(str)):
                raise ValueError(f'Recommendation and RA parquet {column} ordering differs')
    responses = None
    if args.responses:
        responses, response_data_sha256 = read_responses(args.responses)
        if response_data_sha256 != data_sha256:
            raise ValueError('Teacher responses were generated for a different parquet file')
    if responses is not None:
        extra = sorted(set(responses).difference(range(len(frame))))
        if extra:
            raise ValueError(f'Teacher responses contain out-of-range row ids: {extra[:10]}')
    semantic_model = None
    if args.semantic_model:
        from sentence_transformers import SentenceTransformer
        semantic_model = SentenceTransformer(args.semantic_model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for row_id, row in frame.iterrows():
            semantic_row = semantic_frame.iloc[row_id]
            common = {'row_id': int(row_id), 'user_id': str(row.get('user_id', row_id)), 'groundtruth': str(row['groundtruth']), 'data_sha256': data_sha256}
            if responses is None:
                item = {**common, 'prompt': TEACHER_PROMPT.format(description=semantic_row['description'], groundtruth=row['groundtruth'], title=semantic_row.get('title', ''), categories=semantic_row.get('categories', ''))}
            else:
                if row_id not in responses:
                    raise ValueError(f'Missing teacher response for row {row_id}')
                cot = responses[row_id]
                segments = semantic_segments(cot, semantic_model, args.max_segments, args.merge_threshold) if semantic_model is not None else split_structured_cot(cot, args.max_segments)
                item = {**common, 'cot': cot, 'segments': segments}
            handle.write(json.dumps(item, ensure_ascii=True) + '\n')
    mode = 'segmented responses' if responses is not None else 'teacher prompts'
    print(f'Wrote {len(frame)} {mode} to {output_path}')
if __name__ == '__main__':
    main()
