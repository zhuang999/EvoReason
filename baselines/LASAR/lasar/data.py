from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
SYSTEM_MESSAGE = "You are a professional recommendation expert who needs to recommend the next possible purchase for users based on their purchase history. Please predict the most likely next product that the user will purchase based on the user's historical purchase information."
QWEN3_EMPTY_THINK = '<think>\n\n</think>\n'

def build_onerec_prompt(description: str) -> str:
    return f'<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n<|im_start|>user\n{description}<|im_end|>\n<|im_start|>assistant\n{QWEN3_EMPTY_THINK}'

class OneRecLasarDataset(Dataset):

    def __init__(self, parquet_path: str | Path, tokenizer, anchors_path: Optional[str | Path]=None, max_prompt_length: int=480, max_answer_length: int=32, max_sequence_length: int=512, max_latent_steps: int=8, prompt_head_tokens: int=96, limit: Optional[int]=None, shard_index: int=0, num_shards: int=1) -> None:
        parquet_path = Path(parquet_path)
        if not 0 <= shard_index < num_shards:
            raise ValueError('shard_index must be in [0, num_shards)')
        frame = pd.read_parquet(parquet_path).reset_index(drop=True)
        required = {'description', 'groundtruth'}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f'Missing OneRec columns: {sorted(missing)}')
        frame = frame.iloc[shard_index::num_shards].reset_index(drop=True)
        self.frame = frame.head(limit).copy() if limit else frame
        self.tokenizer = tokenizer
        self.max_answer_length = max_answer_length
        prompt_budget = max_sequence_length - max_answer_length - max_latent_steps - 2
        if prompt_budget < 1:
            raise ValueError('max_sequence_length is too small for answer and latent budgets')
        self.max_prompt_length = min(max_prompt_length, prompt_budget)
        self.prompt_head_tokens = prompt_head_tokens
        self.source_path = parquet_path
        self.source_sha256 = sha256_file(parquet_path)
        self.anchors = None
        if anchors_path:
            payload = torch.load(anchors_path, map_location='cpu', weights_only=False)
            if isinstance(payload, dict):
                expected_hash = payload.get('data_sha256')
                if expected_hash and expected_hash != self.source_sha256:
                    raise ValueError(f'Anchor/data mismatch: {anchors_path} was built from {expected_hash}, but {parquet_path} is {self.source_sha256}')
                anchors = payload['anchors']
            else:
                anchors = payload
            anchors = anchors[shard_index::num_shards]
            self.anchors = anchors[:limit] if limit else anchors
            if len(self.anchors) != len(self.frame):
                raise ValueError('Anchor and parquet row counts differ')

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        prompt = build_onerec_prompt(str(row['description']))
        answer = str(row['groundtruth']) + '<|im_end|>'
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)['input_ids']
        if len(prompt_ids) > self.max_prompt_length:
            head = min(self.prompt_head_tokens, self.max_prompt_length // 2)
            prompt_ids = prompt_ids[:head] + prompt_ids[-(self.max_prompt_length - head):]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)['input_ids']
        if len(answer_ids) > self.max_answer_length:
            raise ValueError(f'Target at row {index} uses {len(answer_ids)} tokens, exceeding max_answer_length={self.max_answer_length}')
        item: dict[str, Any] = {'prompt_ids': torch.tensor(prompt_ids, dtype=torch.long), 'answer_ids': torch.tensor(answer_ids, dtype=torch.long)}
        if self.anchors is not None:
            anchor = self.anchors[index]
            if isinstance(anchor, dict):
                item['cot_anchors'] = anchor['states']
                item['policy_labels'] = int(anchor.get('count', len(anchor['states'])))
            else:
                item['cot_anchors'] = anchor
                item['policy_labels'] = int(anchor.shape[0])
        return item

class LasarCollator:

    def __init__(self, pad_token_id: int, max_latent_steps: int=8, teacher_force_steps: bool=False, fixed_steps: Optional[int]=None) -> None:
        self.pad_token_id = pad_token_id
        self.max_latent_steps = max_latent_steps
        self.teacher_force_steps = teacher_force_steps
        self.fixed_steps = fixed_steps

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_prompt = max((item['prompt_ids'].shape[0] for item in features))
        prompt_ids = []
        prompt_masks = []
        for item in features:
            ids = item['prompt_ids']
            pad_length = max_prompt - ids.shape[0]
            prompt_ids.append(F_pad(ids, pad_length, self.pad_token_id, left=True))
            prompt_masks.append(F_pad(torch.ones_like(ids), pad_length, 0, left=True))
        answers = [item['answer_ids'] for item in features]
        answer_ids = pad_sequence(answers, batch_first=True, padding_value=self.pad_token_id)
        answer_mask = pad_sequence([torch.ones_like(ids) for ids in answers], batch_first=True, padding_value=0)
        batch = {'prompt_ids': torch.stack(prompt_ids), 'prompt_attention_mask': torch.stack(prompt_masks), 'answer_ids': answer_ids, 'answer_attention_mask': answer_mask}
        if 'cot_anchors' in features[0]:
            hidden_size = features[0]['cot_anchors'].shape[-1]
            anchors = torch.zeros(len(features), self.max_latent_steps, hidden_size, dtype=features[0]['cot_anchors'].dtype)
            anchor_mask = torch.zeros(len(features), self.max_latent_steps, dtype=torch.bool)
            labels = []
            for index, item in enumerate(features):
                count = min(item['cot_anchors'].shape[0], self.max_latent_steps)
                anchors[index, :count] = item['cot_anchors'][:count]
                anchor_mask[index, :count] = True
                labels.append(min(int(item['policy_labels']), self.max_latent_steps))
            batch['cot_anchors'] = anchors
            batch['anchor_mask'] = anchor_mask
            batch['policy_labels'] = torch.tensor(labels, dtype=torch.long)
            if self.fixed_steps is not None:
                batch['step_counts'] = torch.full_like(batch['policy_labels'], self.fixed_steps)
            elif self.teacher_force_steps:
                batch['step_counts'] = batch['policy_labels'].clone()
        return batch

def F_pad(tensor: torch.Tensor, length: int, value: int, left: bool=False) -> torch.Tensor:
    if length <= 0:
        return tensor
    padding = torch.full((length,), value, dtype=tensor.dtype)
    return torch.cat([padding, tensor] if left else [tensor, padding])

def sha256_file(path: str | Path, chunk_size: int=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()
