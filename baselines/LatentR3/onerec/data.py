from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
import pandas as pd
from torch.utils.data import Dataset
SYSTEM_MESSAGE = "You are a professional recommendation expert who needs to recommend the next possible purchase for users based on their purchase history. Please predict the most likely next product that the user will purchase based on the user's historical purchase information."
THOUGHT_TOKEN = '<|Thought|>'
ONEREC_EMPTY_THINK = '<think>\n\n</think>\n'

def build_onerec_prompt(description: str, *, protocol: str='onerec') -> str:
    if protocol not in {'onerec', 'legacy-r3'}:
        raise ValueError(f'Unsupported prompt protocol: {protocol}')
    assistant_prefix = ONEREC_EMPTY_THINK if protocol == 'onerec' else ''
    return f'<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n<|im_start|>user\n{description}<|im_end|>\n<|im_start|>assistant\n{assistant_prefix}'

def ensure_thought_token(tokenizer) -> int:
    vocabulary = tokenizer.get_vocab()
    if THOUGHT_TOKEN not in vocabulary:
        tokenizer.add_special_tokens({'additional_special_tokens': [THOUGHT_TOKEN]}, replace_additional_special_tokens=False)
    thought_ids = tokenizer(THOUGHT_TOKEN, add_special_tokens=False)['input_ids']
    if len(thought_ids) != 1:
        raise ValueError(f'{THOUGHT_TOKEN} must map to exactly one token')
    thought_id = thought_ids[0]
    if thought_id != len(tokenizer) - 1:
        raise ValueError(f'{THOUGHT_TOKEN} must be the final vocabulary token')
    return thought_id

class OneRecLatentDataset(Dataset):

    def __init__(self, parquet_path: str | Path, tokenizer, *, train: bool, max_sequence_length: int=512, max_answer_length: int=32, prompt_head_tokens: int=96, include_thought_token: bool=True, prompt_protocol: str='onerec', limit: int | None=None, shard_index: int=0, num_shards: int=1) -> None:
        if not 0 <= shard_index < num_shards:
            raise ValueError('shard_index must be in [0, num_shards)')
        self.path = Path(parquet_path)
        frame = pd.read_parquet(self.path).reset_index(names='source_index')
        required = {'user_id', 'description', 'groundtruth'}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f'Missing OneRec columns: {sorted(missing)}')
        if limit is not None:
            if limit < 0:
                raise ValueError('limit must be non-negative')
            frame = frame.head(limit)
        self.frame = frame.iloc[shard_index::num_shards].reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.train = train
        self.include_thought_token = include_thought_token
        self.prompt_protocol = prompt_protocol
        self.max_answer_length = max_answer_length
        self.max_prompt_length = max_sequence_length - max_answer_length - int(include_thought_token)
        self.prompt_head_tokens = prompt_head_tokens
        self.thought_id = ensure_thought_token(tokenizer) if include_thought_token else None
        self.source_sha256 = sha256_file(self.path)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        prompt_ids = self.tokenizer(build_onerec_prompt(str(row['description']), protocol=self.prompt_protocol), add_special_tokens=True)['input_ids']
        if len(prompt_ids) > self.max_prompt_length:
            head = min(self.prompt_head_tokens, self.max_prompt_length // 2)
            prompt_ids = prompt_ids[:head] + prompt_ids[-(self.max_prompt_length - head):]
        if self.include_thought_token:
            prompt_ids = prompt_ids + [self.thought_id]
        answer_ids = self.tokenizer(str(row['groundtruth']) + '<|im_end|>', add_special_tokens=False)['input_ids']
        if len(answer_ids) > self.max_answer_length:
            raise ValueError(f'Target at row {index} has {len(answer_ids)} tokens; max_answer_length={self.max_answer_length}')
        item = {'input_ids': prompt_ids, 'attention_mask': [1] * len(prompt_ids)}
        if self.train:
            item['input_ids'] = prompt_ids + answer_ids
            item['attention_mask'] = [1] * len(item['input_ids'])
            item['labels'] = [-100] * len(prompt_ids) + answer_ids
        else:
            item['target_ids'] = answer_ids
            item['user_id'] = str(row['user_id'])
            item['source_index'] = int(row['source_index'])
            item['target_text'] = str(row['groundtruth'])
        return item

def sha256_file(path: str | Path, chunk_size: int=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()
