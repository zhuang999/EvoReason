from __future__ import annotations
import json
from pathlib import Path
import torch
from transformers import LogitsProcessor

class SidTrie:
    END = -1

    def __init__(self) -> None:
        self.root: dict[int, dict] = {}

    def insert(self, token_ids: list[int]) -> None:
        node = self.root
        for token_id in token_ids:
            node = node.setdefault(int(token_id), {})
        node[self.END] = {}

    def allowed(self, prefix: list[int]) -> list[int]:
        node = self.root
        for token_id in prefix:
            if token_id not in node:
                return []
            node = node[token_id]
        return [token_id for token_id in node if token_id != self.END]

class TrieLogitsProcessor(LogitsProcessor):

    def __init__(self, trie: SidTrie, prompt_width: int, fallback_token_id: int) -> None:
        self.trie = trie
        self.prompt_width = prompt_width
        self.fallback_token_id = fallback_token_id

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        masked = torch.full_like(scores, torch.finfo(scores.dtype).min)
        for row in range(scores.shape[0]):
            prefix = input_ids[row, self.prompt_width:].tolist()
            allowed = self.trie.allowed(prefix) or [self.fallback_token_id]
            masked[row, allowed] = scores[row, allowed]
        return masked

def build_sid_trie(metadata_path: str | Path, tokenizer) -> SidTrie:
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    trie = SidTrie()
    for item in metadata.values():
        sid = item.get('sid')
        if sid:
            token_ids = tokenizer(sid, add_special_tokens=False)['input_ids']
            trie.insert(token_ids + [im_end_id])
    return trie
