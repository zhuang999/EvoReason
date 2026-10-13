from __future__ import annotations
import json
from pathlib import Path
import torch

class TokenTrie:
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

class TrieLogitsProcessor:

    def __init__(self, trie: TokenTrie, fallback_token_id: int) -> None:
        self.trie = trie
        self.fallback_token_id = fallback_token_id

    def __call__(self, logits: torch.Tensor, prefixes: torch.Tensor) -> torch.Tensor:
        masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
        for row in range(logits.shape[0]):
            allowed = self.trie.allowed(prefixes[row].tolist())
            if not allowed:
                allowed = [self.fallback_token_id]
            masked[row, allowed] = logits[row, allowed]
        return masked

def build_sid_trie(metadata_path: str | Path, tokenizer) -> TokenTrie:
    with Path(metadata_path).open('r', encoding='utf-8') as handle:
        metadata = json.load(handle)
    trie = TokenTrie()
    for item in metadata.values():
        sid = item.get('sid')
        if sid:
            trie.insert(tokenizer(sid, add_special_tokens=False)['input_ids'])
    return trie
