from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
SID_PATTERN = re.compile('<s_[abcd]_\\d+>')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-path', type=Path, required=True)
    parser.add_argument('--checkpoint-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--emb-size', type=int, default=256)
    parser.add_argument('--target-std', type=float, default=0.02)
    return parser.parse_args()

def load_embed_tokens(checkpoint_path: Path) -> torch.Tensor:
    index_path = checkpoint_path / 'model.safetensors.index.json'
    if not index_path.exists():
        raise FileNotFoundError(f'Missing safetensors index: {index_path}')
    with index_path.open('r', encoding='utf-8') as handle:
        weight_map = json.load(handle)['weight_map']
    key = 'model.embed_tokens.weight'
    if key not in weight_map:
        candidates = [name for name in weight_map if name.endswith('embed_tokens.weight')]
        if not candidates:
            raise KeyError('Could not find embed_tokens.weight in safetensors index')
        key = candidates[0]
    shard = checkpoint_path / weight_map[key]
    weights = load_file(str(shard), device='cpu')
    return weights[key].float()

def project_embeddings(matrix: torch.Tensor, emb_size: int) -> torch.Tensor:
    hidden = matrix.shape[-1]
    if hidden == emb_size:
        return matrix
    if hidden % emb_size == 0:
        return matrix.reshape(matrix.shape[0], emb_size, hidden // emb_size).mean(dim=-1)
    if hidden > emb_size:
        return matrix[:, :emb_size]
    pad = torch.zeros(matrix.shape[0], emb_size - hidden, dtype=matrix.dtype)
    return torch.cat([matrix, pad], dim=-1)

def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    embed_tokens = load_embed_tokens(args.checkpoint_path)
    with args.metadata_path.open('r', encoding='utf-8') as handle:
        metadata = json.load(handle)
    max_item_id = max((int(item_id) for item_id in metadata))
    item_vectors = torch.zeros(max_item_id + 1, embed_tokens.shape[-1], dtype=torch.float32)
    seen = torch.zeros(max_item_id + 1, dtype=torch.bool)
    for item_id_text, item in metadata.items():
        item_id = int(item_id_text)
        sid_tokens = SID_PATTERN.findall(str(item['sid']))
        if not sid_tokens:
            raise ValueError(f"Item {item_id} has no SID tokens: {item.get('sid')}")
        token_ids = tokenizer.convert_tokens_to_ids(sid_tokens)
        if any((token_id is None or token_id < 0 for token_id in token_ids)):
            raise ValueError(f'Failed to tokenize SID for item {item_id}: {sid_tokens}')
        item_vectors[item_id] = embed_tokens[torch.tensor(token_ids, dtype=torch.long)].mean(dim=0)
        seen[item_id] = True
    missing = [index for index in range(1, max_item_id + 1) if not bool(seen[index])]
    if missing:
        raise ValueError(f'Metadata is missing item ids, first missing ids: {missing[:10]}')
    projected = project_embeddings(item_vectors[1:], args.emb_size)
    projected = projected - projected.mean(dim=1, keepdim=True)
    std = projected.std().clamp_min(1e-08)
    projected = projected / std * args.target_std
    padding = torch.zeros(1, args.emb_size, dtype=projected.dtype)
    item_emb = torch.cat([projected, padding], dim=0).contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'item_emb': item_emb, 'metadata_path': str(args.metadata_path), 'checkpoint_path': str(args.checkpoint_path), 'projection': 'mean_chunks', 'emb_size': args.emb_size, 'target_std': args.target_std, 'num_items': max_item_id}, args.output)
    print(json.dumps({'output': str(args.output), 'shape': list(item_emb.shape), 'num_items': max_item_id, 'std': float(item_emb[:-1].std()), 'projection': 'mean_chunks'}, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
