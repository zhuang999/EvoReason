from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src' / 'latent'))
from layers import LatentModel
from onerec.modeling import latent_model_class
from onerec import OneRecLatentDataset, TrieLogitsProcessor, build_sid_trie, ensure_thought_token, sha256_file, write_run_manifest

def parse_args():
    parser = argparse.ArgumentParser(description='LatentR3 trie-constrained OneRec evaluation')
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--records-output')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-beams', type=int, default=50)
    parser.add_argument('--max-new-tokens', type=int, default=8)
    parser.add_argument('--max-sequence-length', type=int, default=512)
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--mode', choices=['direct', 'latent', 'latent-off'], default='latent', help='direct omits <|Thought|>; latent-off keeps its embedding but bypasses LatentRATT.')
    parser.add_argument('--prompt-protocol', choices=['onerec', 'legacy-r3'], default='onerec')
    parser.add_argument('--end-k', type=int, default=-1)
    parser.add_argument('--latent-component', choices=['full', 'no-rope', 'no-query-proj', 'no-key-proj', 'no-value-proj', 'uniform'], default='full', help='Inference-only LatentRATT component ablation.')
    parser.add_argument('--disable-latent-attention', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args()

def collate_eval(features, pad_token_id):
    inputs = [torch.tensor(item['input_ids'], dtype=torch.long) for item in features]
    width = max((value.shape[0] for value in inputs))
    padded = []
    masks = []
    for value in inputs:
        padding = width - value.shape[0]
        padded.append(torch.cat([torch.full((padding,), pad_token_id), value]))
        masks.append(torch.cat([torch.zeros(padding, dtype=torch.long), torch.ones_like(value)]))
    targets = [torch.tensor(item['target_ids'], dtype=torch.long) for item in features]
    return {'input_ids': torch.stack(padded), 'attention_mask': torch.stack(masks), 'target_ids': pad_sequence(targets, batch_first=True, padding_value=-100), 'target_lengths': torch.tensor([len(value) for value in targets]), 'user_ids': [item['user_id'] for item in features], 'source_indices': [item['source_index'] for item in features], 'target_texts': [item['target_text'] for item in features]}

def infer_split(path: str) -> str:
    name = Path(path).name
    for split in ('train', 'val', 'test'):
        if f'_{split}.' in name:
            return split
    return 'unknown'

def canonical_path(path: str) -> str:
    candidate = Path(path)
    return str(candidate.resolve()) if candidate.exists() else path

def main() -> None:
    args = parse_args()
    if args.disable_latent_attention:
        if args.mode != 'latent':
            raise ValueError('Use either --mode or --disable-latent-attention, not both')
        args.mode = 'latent-off'
    if args.mode != 'latent' and args.latent_component != 'full':
        raise ValueError('--latent-component requires --mode latent')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    include_thought = args.mode != 'direct'
    thought_id = ensure_thought_token(tokenizer) if include_thought else None
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    if args.mode == 'direct':
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    else:
        model_class = latent_model_class(args.model, LatentModel)
        model = model_class.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
        model.config.thought_token_id = thought_id
        model.attention.end_k = args.end_k
        model.attention.ablation_mode = args.latent_component
        if args.mode == 'latent-off':

            def passthrough_embs(input_ids, attention_mask, **kwargs):
                del attention_mask, kwargs
                return model.model.embed_tokens(input_ids)
            model.generate_embs = passthrough_embs
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    dataset = OneRecLatentDataset(args.data, tokenizer, train=False, max_sequence_length=args.max_sequence_length, include_thought_token=include_thought, prompt_protocol=args.prompt_protocol, limit=args.limit, shard_index=args.shard_index, num_shards=args.num_shards)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda rows: collate_eval(rows, tokenizer.pad_token_id))
    trie = build_sid_trie(args.metadata, tokenizer)
    cutoffs = (5, 10, 20)
    hits = {cutoff: 0.0 for cutoff in cutoffs}
    ndcgs = {cutoff: 0.0 for cutoff in cutoffs}
    sample_count = 0
    records: list[dict] = []
    started = time.perf_counter()
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target_ids'].to(device)
        prompt_width = input_ids.shape[1]
        processor = TrieLogitsProcessor(trie, prompt_width=prompt_width, fallback_token_id=model.config.eos_token_id)
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=GenerationConfig(num_beams=args.num_beams, num_return_sequences=args.num_beams, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None, top_k=None, top_p=None, pad_token_id=tokenizer.pad_token_id, eos_token_id=model.config.eos_token_id, return_dict_in_generate=True, output_scores=True, early_stopping=True), logits_processor=LogitsProcessorList([processor]))
        generated = output.sequences[:, prompt_width:].view(input_ids.shape[0], args.num_beams, -1)
        sequence_scores = output.sequences_scores.view(input_ids.shape[0], args.num_beams)
        for row in range(input_ids.shape[0]):
            target_length = int(batch['target_lengths'][row])
            target = targets[row, :target_length]
            candidates = generated[row, :, :target_length]
            matches = candidates.eq(target.unsqueeze(0)).all(dim=1).nonzero(as_tuple=False)
            rank = int(matches[0].item()) + 1 if len(matches) else None
            for cutoff in cutoffs:
                if rank is not None and rank <= cutoff:
                    hits[cutoff] += 1.0
                    ndcgs[cutoff] += 1.0 / math.log2(rank + 1)
            records.append({'source_index': int(batch['source_indices'][row]), 'user_id': batch['user_ids'][row], 'target': batch['target_texts'][row], 'target_token_ids': target.tolist(), 'rank': rank, 'candidate_token_ids': generated[row].tolist(), 'candidates': tokenizer.batch_decode(generated[row], skip_special_tokens=False), 'sequence_scores': sequence_scores[row].float().cpu().tolist()})
        sample_count += input_ids.shape[0]
        print(f'Evaluated {sample_count}/{len(dataset)}')
    if sample_count == 0:
        raise ValueError('Evaluation dataset is empty')
    elapsed = time.perf_counter() - started
    metadata_sha256 = sha256_file(args.metadata)
    result = {**{f'HR@{k}': hits[k] / sample_count for k in cutoffs}, **{f'NDCG@{k}': ndcgs[k] / sample_count for k in cutoffs}, 'samples': sample_count, 'beam_width': args.num_beams, 'latency_ms_per_sample': elapsed * 1000.0 / sample_count, 'hit_counts': {str(k): hits[k] for k in cutoffs}, 'ndcg_sums': {str(k): ndcgs[k] for k in cutoffs}, 'elapsed_seconds': elapsed, 'checkpoint': canonical_path(args.model), 'data_split': infer_split(args.data), 'data_sha256': dataset.source_sha256, 'metadata_sha256': metadata_sha256, 'num_shards': args.num_shards, 'shard_index': args.shard_index, 'limit': args.limit, 'limit_scope': 'global_before_sharding', 'mode': args.mode, 'prompt_protocol': args.prompt_protocol, 'end_k': args.end_k, 'latent_component': args.latent_component, 'latent_attention_enabled': args.mode == 'latent'}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    records_output = Path(args.records_output) if args.records_output else output_path.with_name('records.jsonl')
    records_output.parent.mkdir(parents=True, exist_ok=True)
    with records_output.open('w', encoding='utf-8') as handle:
        for record in sorted(records, key=lambda value: value['source_index']):
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    write_run_manifest(output_path.parent, stage='latentr3_onerec_eval', args=args, extra={'protocol': 'onerec-sid-matched', 'paper': 'arXiv:2505.19092', 'checkpoint': canonical_path(args.model), 'data_split': infer_split(args.data), 'data_sha256': dataset.source_sha256, 'metadata_sha256': metadata_sha256, 'mode': args.mode, 'prompt_protocol': args.prompt_protocol, 'end_k': args.end_k, 'latent_component': args.latent_component})
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
