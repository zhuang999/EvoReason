from __future__ import annotations
from typing import Optional, Tuple, Union
import torch
from transformers import AutoConfig, Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from layers import SelfAttentionLayer

class Qwen3LatentModel(Qwen3ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.attention = SelfAttentionLayer(config.hidden_size)

    def generate_embs(self, input_ids, attention_mask, position_ids=None, cache_position=None):
        output_mid = super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, cache_position=cache_position, output_hidden_states=True, return_dict=True)
        thought_id = getattr(self.config, 'thought_token_id', self.model.embed_tokens.num_embeddings - 1)
        if not 0 <= thought_id < self.model.embed_tokens.num_embeddings:
            raise ValueError(f'Invalid thought_token_id={thought_id}')
        thought_positions = torch.nonzero(input_ids == thought_id)
        if thought_positions.shape[0] != input_ids.shape[0]:
            raise ValueError('Each sample must contain exactly one <|Thought|> token')
        input_embs = self.model.embed_tokens(input_ids)
        input_embs[thought_positions[:, 0], thought_positions[:, 1]] = self.attention(hidden_states=output_mid.hidden_states[-1], attention_mask=attention_mask, thought_id_idx=thought_positions[:, 1]).to(input_embs.dtype)
        return input_embs

    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, cache_position: Optional[torch.LongTensor]=None, logits_to_keep: Union[int, torch.Tensor]=0, **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        if input_ids is not None and inputs_embeds is None and (attention_mask is not None) and (input_ids.size() == attention_mask.size()):
            inputs_embeds = self.generate_embs(input_ids, attention_mask, position_ids=position_ids, cache_position=cache_position)
            input_ids = None
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs)

def latent_model_class(model_path: str, qwen2_class):
    model_type = AutoConfig.from_pretrained(model_path).model_type
    if model_type == 'qwen3':
        return Qwen3LatentModel
    if model_type == 'qwen2':
        return qwen2_class
    raise ValueError(f'LatentR3 matched runner supports qwen2/qwen3, found {model_type}')
