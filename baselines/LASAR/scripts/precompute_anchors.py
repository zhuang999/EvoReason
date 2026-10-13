from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lasar.data import sha256_file

def parse_args():
    parser = argparse.ArgumentParser(description='Encode segmented CoT with the grounded OneRec backbone')
    parser.add_argument('--model', required=True, help='Stage-1 grounded checkpoint')
    parser.add_argument('--cot', required=True, help='Segmented JSONL from prepare_cot.py')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='float16')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    encoder = getattr(model, 'model', model)
    source_rows = []
    first_item = None
    with Path(args.cot).open('r', encoding='utf-8') as handle:
        for expected_row_id, line in enumerate(handle):
            item = json.loads(line)
            if first_item is None:
                first_item = item
            elif item.get('data_sha256') != first_item.get('data_sha256'):
                raise ValueError('CoT JSONL contains mixed data fingerprints')
            if int(item['row_id']) != expected_row_id:
                raise ValueError('CoT JSONL must preserve parquet row order')
            if not item['segments']:
                raise ValueError(f'row_id={expected_row_id} contains no CoT segments')
            source_rows.append(item)
    flat_segments = [segment for item in source_rows for segment in item['segments']]
    flat_states = []
    for start in range(0, len(flat_segments), args.batch_size):
        texts = flat_segments[start:start + args.batch_size]
        encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).to(args.device)
        with torch.inference_mode():
            output = encoder(**encoded, use_cache=False, output_hidden_states=False, return_dict=True)
        token_positions = torch.arange(encoded['attention_mask'].shape[1], device=args.device).unsqueeze(0)
        final_indices = (encoded['attention_mask'] * token_positions).max(dim=1).values
        batch_indices = torch.arange(len(texts), device=args.device)
        states = output.last_hidden_state[batch_indices, final_indices]
        flat_states.extend(states.detach().cpu().to(dtype).unbind(0))
        if len(flat_states) % 1000 < args.batch_size:
            print(f'Encoded {len(flat_states)}/{len(flat_segments)} segments')
    rows = []
    offset = 0
    for item in source_rows:
        count = len(item['segments'])
        rows.append({'states': torch.stack(flat_states[offset:offset + count]), 'count': count})
        offset += count
    if offset != len(flat_states):
        raise RuntimeError('Anchor reconstruction did not consume all encoded segments')
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'anchors': rows, 'model': args.model, 'source': args.cot, 'cot_sha256': sha256_file(args.cot), 'data_sha256': first_item.get('data_sha256') if first_item else None, 'batch_size': args.batch_size}, output_path)
    print(f'Saved {len(rows)} anchor sequences to {output_path}')
if __name__ == '__main__':
    main()
