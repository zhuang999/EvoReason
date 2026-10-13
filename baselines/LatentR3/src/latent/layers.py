import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

class VanillaRoPE(nn.Module):

    def __init__(self, hidden_size, device=None):
        super(VanillaRoPE, self).__init__()
        base = 1000
        dim = hidden_size
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))

class SelfAttentionLayer(torch.nn.Module):

    def __init__(self, hidden_size, end_k=-1):
        super().__init__()
        self.hidden_size = hidden_size
        self.end_k = end_k
        self.ablation_mode = 'full'
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale_score = 1.0 / self.hidden_size ** 0.5
        self.rope = VanillaRoPE(hidden_size)

    @staticmethod
    def mask_to_weights(attention_mask, thought_id_idx, end_k=-1):
        if thought_id_idx is not None and attention_mask.size(0) != thought_id_idx.size(0):
            raise ValueError('attention_mask must have the same size as thought_id_idx')
        if end_k != -1:
            for i in range(attention_mask.size(0)):
                idx = thought_id_idx[i].item()
                valid_indices = torch.nonzero(attention_mask[i, :idx], as_tuple=False).flatten()
                selected = valid_indices[-end_k:]
                attention_mask[i].zero_()
                attention_mask[i, selected] = 1
        else:
            for i in range(attention_mask.size(0)):
                idx = thought_id_idx[i].item()
                attention_mask[i, idx:] = 0
        attention_weight = torch.zeros_like(attention_mask, dtype=torch.float32)
        attention_weight = attention_weight.masked_fill(~attention_mask.bool(), float('-inf'))
        return attention_weight

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, thought_id_idx) -> torch.Tensor:
        attention_mask = SelfAttentionLayer.mask_to_weights(attention_mask.clone(), thought_id_idx, self.end_k)
        cos, sin = self.rope(hidden_states, position_ids=torch.arange(0, hidden_states.shape[1], device=hidden_states.device).unsqueeze(0))
        batch_size, seq_len, _ = hidden_states.shape
        valid_modes = {'full', 'no-rope', 'no-query-proj', 'no-key-proj', 'no-value-proj', 'uniform'}
        if self.ablation_mode not in valid_modes:
            raise ValueError(f'Unsupported LatentRATT ablation mode: {self.ablation_mode}')
        Q = hidden_states if self.ablation_mode == 'no-query-proj' else self.query(hidden_states)
        K = hidden_states if self.ablation_mode == 'no-key-proj' else self.key(hidden_states)
        V = hidden_states if self.ablation_mode == 'no-value-proj' else self.value(hidden_states)
        if self.ablation_mode not in {'no-rope', 'uniform'}:
            Q, K = apply_rotary_pos_emb(Q, K, cos.squeeze(0), sin.squeeze(0), unsqueeze_dim=0)
        if thought_id_idx is None:
            indices = torch.full((batch_size,), seq_len - 1, device=hidden_states.device)
        else:
            indices = thought_id_idx - 1
            if (indices < 0).any() or (indices >= seq_len).any():
                raise ValueError('thought_id_idx-1 exceeds valid sequence indices')
        Q_selected = Q[torch.arange(batch_size), indices].unsqueeze(1)
        if self.ablation_mode == 'uniform':
            attn_scores = torch.zeros(batch_size, 1, seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            attn_scores = torch.matmul(Q_selected, K.transpose(-2, -1)) * self.scale_score
        attn_scores += attention_mask.unsqueeze(1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        output_selected = torch.matmul(attn_weights, V)
        output = output_selected.squeeze(1)
        return output

class Noise(nn.Module):

    def __init__(self, hidden_size):
        super(Noise, self).__init__()
        self.mu = nn.Linear(hidden_size, hidden_size, bias=False)
        self.log_var = torch.nn.Parameter(torch.tensor([float('-10')] * hidden_size))

    def forward(self, x):
        x = self.mu(x)
        std = torch.exp(self.log_var / 2)
        return x + torch.randn_like(x).mul(std)

class LatentModel(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.attention = SelfAttentionLayer(config.hidden_size)

    def generate_embs(self, input_ids, attention_mask):
        output_mid = super().forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        thought_ids = getattr(self.config, 'thought_token_id', self.model.embed_tokens.num_embeddings - 1)
        where_thought_ids = torch.nonzero(input_ids == thought_ids)
        hidden_states = output_mid['hidden_states']
        input_embs = self.model.embed_tokens(input_ids)
        input_embs[where_thought_ids[:, 0], where_thought_ids[:, 1]] = self.attention(hidden_states=hidden_states[-1], attention_mask=attention_mask, thought_id_idx=where_thought_ids[:, 1]).to(input_embs.dtype)
        return input_embs

    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, cache_position: Optional[torch.LongTensor]=None, num_logits_to_keep: int=0, **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        if input_ids is not None and inputs_embeds is None and (input_ids.size() == attention_mask.size()):
            inputs_embeds = self.generate_embs(input_ids, attention_mask)
            input_ids = None
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, cache_position=cache_position, num_logits_to_keep=num_logits_to_keep, **kwargs)
