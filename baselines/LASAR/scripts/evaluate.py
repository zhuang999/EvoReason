from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from lasar import LasarCollator, LasarForCausalLM, OneRecLasarDataset, write_run_manifest
from lasar.trie import TrieLogitsProcessor, build_sid_trie

def parse_args():
    parser = argparse.ArgumentParser(description='Trie-constrained LASAR beam-search evaluation')
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-beams', type=int, default=50)
    parser.add_argument('--max-new-tokens', type=int, default=7)
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-sequence-length', type=int, default=512)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--fixed-latent-steps', type=int, help='Override the policy head with a fixed N for policy ablations.')
    parser.add_argument('--direct-backbone', action='store_true', help='Evaluate the shared Stage-1 backbone without LASAR latent tokens.')
    parser.add_argument('--hybrid-alpha', type=float)
    parser.add_argument('--hybrid-mode', choices=['log_linear', 'mixture'], default='log_linear')
    parser.add_argument('--latent-adapter-scale', type=float, help='Override the checkpoint residual-adapter scale for validation-selected sweeps.')
    parser.add_argument('--records-output', help='Optional JSONL with per-example ranks, beams, and confidence scores.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = LasarForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    if args.latent_adapter_scale is not None:
        model.set_latent_adapter_scale(args.latent_adapter_scale)
    dataset = OneRecLasarDataset(args.data, tokenizer, max_sequence_length=args.max_sequence_length, max_latent_steps=model.max_latent_steps, limit=args.limit, shard_index=args.shard_index, num_shards=args.num_shards)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=LasarCollator(tokenizer.pad_token_id, model.max_latent_steps))
    trie = build_sid_trie(args.metadata, tokenizer)
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    processor = TrieLogitsProcessor(trie, fallback_token_id=im_end_id)
    cutoffs = [5, 10, 20]
    hits = {cutoff: 0.0 for cutoff in cutoffs}
    ndcgs = {cutoff: 0.0 for cutoff in cutoffs}
    sample_count = 0
    step_total = 0
    records = [] if args.records_output else None
    started = time.perf_counter()
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        fixed_steps = None
        if args.fixed_latent_steps is not None:
            fixed_steps = torch.full((batch['prompt_ids'].shape[0],), args.fixed_latent_steps, dtype=torch.long, device=device)
        if args.direct_backbone:
            search = model.beam_search_direct
        elif args.hybrid_alpha is not None:
            search = model.beam_search_hybrid
        else:
            search = model.beam_search
        search_kwargs = {'max_new_tokens': args.max_new_tokens, 'num_beams': args.num_beams, 'logits_processor': processor}
        if not args.direct_backbone:
            search_kwargs['step_counts'] = fixed_steps
        if args.hybrid_alpha is not None:
            search_kwargs['latent_weight'] = args.hybrid_alpha
            search_kwargs['mode'] = args.hybrid_mode
        beams, beam_scores, steps = search(batch['prompt_ids'], batch['prompt_attention_mask'], **search_kwargs)
        for row in range(beams.shape[0]):
            target_length = int(batch['answer_attention_mask'][row].sum().item())
            target = batch['answer_ids'][row, :target_length]
            ranked = beams[row, :, :target_length]
            matches = ranked.eq(target.unsqueeze(0)).all(dim=1).nonzero(as_tuple=False)
            rank = int(matches[0].item()) + 1 if len(matches) else None
            if records is not None:
                row_scores = beam_scores[row].float().cpu()
                records.append({'source_index': (sample_count + row) * args.num_shards + args.shard_index, 'target_ids': target.cpu().tolist(), 'beam_ids': ranked.cpu().tolist(), 'beam_scores': row_scores.tolist(), 'score_margin_1_2': float(row_scores[0] - row_scores[1]) if row_scores.numel() > 1 else None, 'rank': rank, 'latent_steps': int(steps[row].item())})
            for cutoff in cutoffs:
                if rank is not None and rank <= cutoff:
                    hits[cutoff] += 1.0
                    ndcgs[cutoff] += 1.0 / math.log2(rank + 1)
        sample_count += beams.shape[0]
        step_total += int(steps.sum().item())
        print(f'Evaluated {sample_count}/{len(dataset)}')
    elapsed = time.perf_counter() - started
    result = {**{f'HR@{k}': hits[k] / sample_count for k in cutoffs}, **{f'NDCG@{k}': ndcgs[k] / sample_count for k in cutoffs}, 'mean_latent_steps': step_total / sample_count, 'latency_ms_per_sample': elapsed * 1000.0 / sample_count, 'samples': sample_count, 'beam_width': args.num_beams, 'fixed_latent_steps': args.fixed_latent_steps, 'direct_backbone': args.direct_backbone, 'hybrid_alpha': args.hybrid_alpha, 'hybrid_mode': args.hybrid_mode if args.hybrid_alpha is not None else None, 'latent_adapter_scale': model.latent_adapter_scale, 'model_path': str(Path(args.model).resolve()), 'hit_counts': {str(k): hits[k] for k in cutoffs}, 'ndcg_sums': {str(k): ndcgs[k] for k in cutoffs}, 'latent_step_sum': step_total, 'elapsed_seconds': elapsed, 'data_sha256': dataset.source_sha256, 'num_shards': args.num_shards, 'shard_index': args.shard_index}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if records is not None:
        records_path = Path(args.records_output)
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open('w', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(',', ':')) + '\n')
    write_run_manifest(output_path.parent, stage='evaluation', args=args, extra={'data_sha256': dataset.source_sha256, 'paper': 'arXiv:2605.10207v1'}, filename=f'{output_path.stem}.manifest.json')
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
