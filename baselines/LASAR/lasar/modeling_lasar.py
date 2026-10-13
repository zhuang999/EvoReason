from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.utils import ModelOutput

@dataclass
class LasarOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    alignment_loss: Optional[torch.Tensor] = None
    policy_loss: Optional[torch.Tensor] = None
    policy_logits: Optional[torch.Tensor] = None
    step_counts: Optional[torch.Tensor] = None
    latent_states: Optional[torch.Tensor] = None
    terminal_states: Optional[torch.Tensor] = None
    past_key_values: Optional[object] = None

def bidirectional_kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float()
    b = b.float()
    log_a = F.log_softmax(a, dim=-1)
    log_b = F.log_softmax(b, dim=-1)
    prob_a = log_a.exp()
    prob_b = log_b.exp()
    return 0.5 * ((prob_a - prob_b) * (log_a - log_b)).sum(dim=-1)

class LasarForCausalLM(nn.Module):

    def __init__(self, backbone: nn.Module, latent_start_id: int, latent_end_id: int, latent_thought_id: Optional[int]=None, max_latent_steps: int=8, align_weight: float=0.1, policy_weight: float=0.1, use_latent_adapter: bool=False, latent_adapter_scale: float=0.001, use_trainable_markers: bool=False) -> None:
        super().__init__()
        if max_latent_steps < 1:
            raise ValueError('max_latent_steps must be positive')
        self.backbone = backbone
        self.latent_start_id = latent_start_id
        self.latent_thought_id = latent_thought_id
        self.latent_end_id = latent_end_id
        self.max_latent_steps = max_latent_steps
        self.align_weight = align_weight
        self.policy_weight = policy_weight
        self.use_latent_adapter = use_latent_adapter
        self.latent_adapter_scale = latent_adapter_scale
        self.use_trainable_markers = use_trainable_markers
        hidden_size = getattr(backbone.config, 'hidden_size', None)
        if hidden_size is None:
            hidden_size = backbone.get_input_embeddings().embedding_dim
        self.policy_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, max_latent_steps))
        self.latent_adapter = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size, bias=False))
        nn.init.zeros_(self.latent_adapter[1].weight)
        embedding = backbone.get_input_embeddings().weight
        self.latent_start_embedding = nn.Parameter(embedding[latent_start_id].detach().clone())
        self.latent_end_embedding = nn.Parameter(embedding[latent_end_id].detach().clone())
        self.policy_head.to(device=embedding.device, dtype=embedding.dtype)
        self.latent_adapter.to(device=embedding.device, dtype=embedding.dtype)
        self.latent_start_embedding.data = self.latent_start_embedding.data.to(device=embedding.device, dtype=embedding.dtype)
        self.latent_end_embedding.data = self.latent_end_embedding.data.to(device=embedding.device, dtype=embedding.dtype)

    @property
    def config(self):
        return self.backbone.config

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def get_output_embeddings(self):
        return self.backbone.get_output_embeddings()

    def set_latent_adapter_scale(self, scale: float) -> None:
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError('latent adapter scale must be finite and non-negative')
        self.latent_adapter_scale = float(scale)

    def _position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        return attention_mask.long().sum(dim=-1, keepdim=True).clamp_min(1) - 1

    def _sequence_position_ids(self, attention_mask: torch.Tensor, sequence_length: Optional[int]=None) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)
        if sequence_length is not None:
            position_ids = position_ids[:, -sequence_length:]
        return position_ids

    def _one_token_forward(self, *, attention_mask: torch.Tensor, past_key_values, input_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None):
        return self.backbone(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=self._position_ids(attention_mask), past_key_values=past_key_values, use_cache=True, output_hidden_states=True, return_dict=True)

    def encode_prompt(self, prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor):
        outputs = self.backbone(input_ids=prompt_ids, attention_mask=prompt_attention_mask, position_ids=self._sequence_position_ids(prompt_attention_mask), use_cache=True, output_hidden_states=True, return_dict=True)
        prompt_state = outputs.hidden_states[-1][:, -1, :]
        return (prompt_state, outputs.past_key_values)

    def latent_reason(self, prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor, step_counts: Optional[torch.Tensor]=None, sample_steps: bool=False):
        prompt_state, past = self.encode_prompt(prompt_ids, prompt_attention_mask)
        policy_logits = self.policy_head(prompt_state)
        if step_counts is None:
            distribution = torch.distributions.Categorical(logits=policy_logits)
            step_counts = distribution.sample() + 1 if sample_steps else policy_logits.argmax(-1) + 1
        step_counts = step_counts.long().clamp(1, self.max_latent_steps)
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device
        start_ids = torch.full((batch_size, 1), self.latent_start_id, device=device, dtype=torch.long)
        attention_mask = torch.cat([prompt_attention_mask, torch.ones((batch_size, 1), device=device, dtype=prompt_attention_mask.dtype)], dim=1)
        start_kwargs = {'input_ids': start_ids}
        if self.use_trainable_markers:
            start_kwargs = {'inputs_embeds': self.latent_start_embedding.view(1, 1, -1).expand(batch_size, 1, -1)}
        start_output = self._one_token_forward(attention_mask=attention_mask, past_key_values=past, **start_kwargs)
        previous_state = start_output.hidden_states[-1][:, -1, :]
        past = start_output.past_key_values
        latent_states = []
        terminal_state = previous_state
        batch_max_steps = int(step_counts.max().item())
        for index in range(batch_max_steps):
            active = (step_counts > index).to(prompt_attention_mask.dtype).unsqueeze(1)
            attention_mask = torch.cat([attention_mask, active], dim=1)
            feedback_state = previous_state
            if self.use_latent_adapter:
                feedback_state = feedback_state + self.latent_adapter_scale * self.latent_adapter(feedback_state)
            step_output = self._one_token_forward(inputs_embeds=feedback_state.unsqueeze(1), attention_mask=attention_mask, past_key_values=past)
            current_state = step_output.hidden_states[-1][:, -1, :]
            terminal_state = torch.where(active.bool(), current_state, terminal_state)
            latent_states.append(current_state)
            previous_state = current_state
            past = step_output.past_key_values
        return {'prompt_state': prompt_state, 'policy_logits': policy_logits, 'step_counts': step_counts, 'latent_states': torch.stack(latent_states, dim=1), 'terminal_states': terminal_state, 'past_key_values': past, 'attention_mask': attention_mask}

    def forward(self, prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor, answer_ids: Optional[torch.Tensor]=None, answer_attention_mask: Optional[torch.Tensor]=None, step_counts: Optional[torch.Tensor]=None, policy_labels: Optional[torch.Tensor]=None, cot_anchors: Optional[torch.Tensor]=None, anchor_mask: Optional[torch.Tensor]=None, terminal_only_alignment: bool=False, sample_steps: bool=False) -> LasarOutput:
        reason = self.latent_reason(prompt_ids, prompt_attention_mask, step_counts=step_counts, sample_steps=sample_steps)
        policy_logits = reason['policy_logits']
        actual_steps = reason['step_counts']
        latent_states = reason['latent_states']
        terminal_states = reason['terminal_states']
        policy_loss = None
        if policy_labels is not None:
            policy_targets = policy_labels.long().clamp(1, self.max_latent_steps) - 1
            policy_loss = F.cross_entropy(policy_logits, policy_targets)
        alignment_loss = None
        if cot_anchors is not None:
            cot_anchors = cot_anchors.to(device=latent_states.device, dtype=latent_states.dtype)
            if terminal_only_alignment:
                if anchor_mask is None:
                    indices = torch.full_like(actual_steps, cot_anchors.shape[1] - 1)
                else:
                    indices = anchor_mask.to(actual_steps.device).long().sum(dim=1).sub(1).clamp_min(0)
                batch_indices = torch.arange(indices.shape[0], device=indices.device)
                final_anchor = cot_anchors[batch_indices, indices]
                alignment_loss = bidirectional_kl(terminal_states, final_anchor).mean()
            else:
                latent_width = latent_states.shape[1]
                cot_anchors = cot_anchors[:, :latent_width]
                per_step = bidirectional_kl(latent_states, cot_anchors)
                valid = torch.arange(latent_width, device=actual_steps.device)[None, :] < actual_steps[:, None]
                if anchor_mask is not None:
                    valid = valid & anchor_mask[:, :latent_width].to(device=valid.device, dtype=torch.bool)
                alignment_loss = (per_step * valid).sum() / valid.sum().clamp_min(1)
        logits = None
        lm_loss = None
        past = reason['past_key_values']
        if answer_ids is not None:
            if answer_attention_mask is None:
                answer_attention_mask = answer_ids.ne(self.config.pad_token_id).long()
            batch_size, answer_length = answer_ids.shape
            if answer_length < 1:
                raise ValueError('answer_ids must contain at least one token')
            end_ids = torch.full((batch_size, 1), self.latent_end_id, device=answer_ids.device, dtype=answer_ids.dtype)
            decoder_inputs = torch.cat([end_ids, answer_ids[:, :-1]], dim=1)
            decoder_mask = torch.cat([torch.ones_like(end_ids), answer_attention_mask[:, :-1]], dim=1)
            full_mask = torch.cat([reason['attention_mask'], decoder_mask], dim=1)
            decoder_kwargs = {'input_ids': decoder_inputs}
            if self.use_trainable_markers:
                token_embeds = self.get_input_embeddings()(decoder_inputs[:, 1:])
                end_embed = self.latent_end_embedding.view(1, 1, -1).expand(batch_size, 1, -1)
                decoder_kwargs = {'inputs_embeds': torch.cat([end_embed, token_embeds], dim=1)}
            decoder_output = self.backbone(attention_mask=full_mask, position_ids=self._sequence_position_ids(full_mask, decoder_inputs.shape[1]), past_key_values=past, use_cache=True, return_dict=True, **decoder_kwargs)
            logits = decoder_output.logits
            token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), answer_ids.reshape(-1), reduction='none').view_as(answer_ids)
            lm_loss = (token_loss * answer_attention_mask).sum() / answer_attention_mask.sum().clamp_min(1)
            past = decoder_output.past_key_values
        losses = [value for value in [lm_loss] if value is not None]
        if alignment_loss is not None and self.align_weight != 0.0:
            losses.append(self.align_weight * alignment_loss)
        if policy_loss is not None and self.policy_weight != 0.0:
            losses.append(self.policy_weight * policy_loss)
        loss = sum(losses) if losses else None
        return LasarOutput(loss=loss, logits=logits, lm_loss=lm_loss, alignment_loss=alignment_loss, policy_loss=policy_loss, policy_logits=policy_logits, step_counts=actual_steps, latent_states=latent_states, terminal_states=terminal_states, past_key_values=past)

    @torch.no_grad()
    def generate_sequences(self, prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor, max_new_tokens: int, step_counts: Optional[torch.Tensor]=None, do_sample: bool=False, temperature: float=1.0, logits_processor=None, eos_token_id: Optional[int]=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reason = self.latent_reason(prompt_ids, prompt_attention_mask, step_counts=step_counts, sample_steps=do_sample and step_counts is None)
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device
        current_ids = torch.full((batch_size, 1), self.latent_end_id, device=device, dtype=prompt_ids.dtype)
        current_mask = torch.ones_like(current_ids)
        attention_mask = torch.cat([reason['attention_mask'], current_mask], dim=1)
        initial_kwargs = {'input_ids': current_ids}
        if self.use_trainable_markers:
            initial_kwargs = {'inputs_embeds': self.latent_end_embedding.view(1, 1, -1).expand(batch_size, 1, -1)}
        output = self.backbone(attention_mask=attention_mask, position_ids=self._position_ids(attention_mask), past_key_values=reason['past_key_values'], use_cache=True, return_dict=True, **initial_kwargs)
        logits = output.logits[:, -1, :]
        past = output.past_key_values
        generated = []
        log_probs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for index in range(max_new_tokens):
            step_logits = logits / max(temperature, 1e-06)
            prefix = torch.stack(generated, dim=1) if generated else torch.empty(batch_size, 0, dtype=prompt_ids.dtype, device=device)
            if logits_processor is not None:
                step_logits = logits_processor(step_logits, prefix)
            distribution = torch.distributions.Categorical(logits=step_logits)
            next_ids = distribution.sample() if do_sample else step_logits.argmax(dim=-1)
            next_log_probs = F.log_softmax(step_logits, dim=-1).gather(1, next_ids[:, None]).squeeze(1)
            if eos_token_id is not None:
                next_ids = torch.where(finished, torch.full_like(next_ids, eos_token_id), next_ids)
                next_log_probs = torch.where(finished, torch.zeros_like(next_log_probs), next_log_probs)
                finished = finished | next_ids.eq(eos_token_id)
            generated.append(next_ids)
            log_probs.append(next_log_probs)
            if index + 1 == max_new_tokens or finished.all():
                break
            current_ids = next_ids[:, None]
            attention_mask = torch.cat([attention_mask, torch.ones_like(current_ids)], dim=1)
            output = self.backbone(input_ids=current_ids, attention_mask=attention_mask, position_ids=self._position_ids(attention_mask), past_key_values=past, use_cache=True, return_dict=True)
            logits = output.logits[:, -1, :]
            past = output.past_key_values
        return (torch.stack(generated, dim=1), torch.stack(log_probs, dim=1), reason['step_counts'])

    @staticmethod
    def _repeat_cache(past, repeats: int):
        if hasattr(past, 'batch_repeat_interleave'):
            past.batch_repeat_interleave(repeats)
            return past
        return tuple((tuple((value.repeat_interleave(repeats, dim=0) for value in layer)) for layer in past))

    @staticmethod
    def _select_cache(past, indices: torch.Tensor):
        if hasattr(past, 'batch_select_indices'):
            past.batch_select_indices(indices)
            return past
        return tuple((tuple((value.index_select(0, indices) for value in layer)) for layer in past))

    @torch.no_grad()
    def beam_search_hybrid(self, prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor, max_new_tokens: int, num_beams: int=50, step_counts: Optional[torch.Tensor]=None, logits_processor=None, latent_weight: float=0.1, mode: str='log_linear') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0.0 <= latent_weight <= 1.0:
            raise ValueError('latent_weight must be in [0, 1]')
        if mode not in {'log_linear', 'mixture'}:
            raise ValueError('mode must be log_linear or mixture')
        reason = self.latent_reason(prompt_ids, prompt_attention_mask, step_counts=step_counts)
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device
        direct = self.backbone(input_ids=prompt_ids, attention_mask=prompt_attention_mask, position_ids=self._sequence_position_ids(prompt_attention_mask), use_cache=True, return_dict=True)
        end_ids = torch.full((batch_size, 1), self.latent_end_id, device=device, dtype=prompt_ids.dtype)
        latent_mask = torch.cat([reason['attention_mask'], torch.ones_like(end_ids)], dim=1)
        latent_kwargs = {'input_ids': end_ids}
        if self.use_trainable_markers:
            latent_kwargs = {'inputs_embeds': self.latent_end_embedding.view(1, 1, -1).expand(batch_size, 1, -1)}
        latent = self.backbone(attention_mask=latent_mask, position_ids=self._position_ids(latent_mask), past_key_values=reason['past_key_values'], use_cache=True, return_dict=True, **latent_kwargs)

        def fuse(direct_logits, latent_logits, prefixes):
            if logits_processor is not None:
                direct_logits = logits_processor(direct_logits, prefixes)
                latent_logits = logits_processor(latent_logits, prefixes)
            direct_log_probs = F.log_softmax(direct_logits.float(), dim=-1)
            latent_log_probs = F.log_softmax(latent_logits.float(), dim=-1)
            if mode == 'log_linear':
                return (1.0 - latent_weight) * direct_log_probs + latent_weight * latent_log_probs
            stacked = torch.stack([direct_log_probs + math.log(max(1.0 - latent_weight, 1e-12)), latent_log_probs + math.log(max(latent_weight, 1e-12))], dim=0)
            return torch.logsumexp(stacked, dim=0)
        empty_prefix = torch.empty(batch_size, 0, dtype=prompt_ids.dtype, device=device)
        token_scores = fuse(direct.logits[:, -1, :], latent.logits[:, -1, :], empty_prefix)
        beam_scores, first_tokens = token_scores.topk(num_beams, dim=-1)
        prefixes = first_tokens.unsqueeze(-1)
        direct_past = self._repeat_cache(direct.past_key_values, num_beams)
        latent_past = self._repeat_cache(latent.past_key_values, num_beams)
        direct_mask = prompt_attention_mask.repeat_interleave(num_beams, dim=0)
        latent_mask = latent_mask.repeat_interleave(num_beams, dim=0)
        for _ in range(1, max_new_tokens):
            current_ids = prefixes[:, :, -1].reshape(batch_size * num_beams, 1)
            direct_mask = torch.cat([direct_mask, torch.ones_like(current_ids)], dim=1)
            latent_mask = torch.cat([latent_mask, torch.ones_like(current_ids)], dim=1)
            direct = self.backbone(input_ids=current_ids, attention_mask=direct_mask, position_ids=self._position_ids(direct_mask), past_key_values=direct_past, use_cache=True, return_dict=True)
            latent = self.backbone(input_ids=current_ids, attention_mask=latent_mask, position_ids=self._position_ids(latent_mask), past_key_values=latent_past, use_cache=True, return_dict=True)
            flat_prefixes = prefixes.reshape(batch_size * num_beams, -1)
            token_scores = fuse(direct.logits[:, -1, :], latent.logits[:, -1, :], flat_prefixes)
            vocab_size = token_scores.shape[-1]
            combined = token_scores.view(batch_size, num_beams, vocab_size)
            combined = combined + beam_scores.unsqueeze(-1)
            beam_scores, flat_indices = combined.view(batch_size, -1).topk(num_beams, dim=-1)
            source_beams = torch.div(flat_indices, vocab_size, rounding_mode='floor')
            next_tokens = flat_indices.remainder(vocab_size)
            gather_index = source_beams.unsqueeze(-1).expand(-1, -1, prefixes.shape[-1])
            prefixes = torch.cat([prefixes.gather(1, gather_index), next_tokens.unsqueeze(-1)], dim=-1)
            batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
            cache_indices = (source_beams + batch_offsets).reshape(-1)
            direct_past = self._select_cache(direct.past_key_values, cache_indices)
            latent_past = self._select_cache(latent.past_key_values, cache_indices)
            direct_mask = direct_mask.index_select(0, cache_indices)
            latent_mask = latent_mask.index_select(0, cache_indices)
        return (prefixes, beam_scores, reason['step_counts'])

    @torch.no_grad()
    def beam_search_direct(self, prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor, max_new_tokens: int, num_beams: int=50, logits_processor=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_beams < 1:
            raise ValueError('num_beams must be positive')
        output = self.backbone(input_ids=prompt_ids, attention_mask=prompt_attention_mask, position_ids=self._sequence_position_ids(prompt_attention_mask), use_cache=True, return_dict=True)
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device
        logits = output.logits[:, -1, :]
        empty_prefix = torch.empty(batch_size, 0, dtype=prompt_ids.dtype, device=device)
        if logits_processor is not None:
            logits = logits_processor(logits, empty_prefix)
        first_log_probs = F.log_softmax(logits.float(), dim=-1)
        beam_scores, first_tokens = first_log_probs.topk(num_beams, dim=-1)
        prefixes = first_tokens.unsqueeze(-1)
        past = self._repeat_cache(output.past_key_values, num_beams)
        attention_mask = prompt_attention_mask.repeat_interleave(num_beams, dim=0)
        for _ in range(1, max_new_tokens):
            current_ids = prefixes[:, :, -1].reshape(batch_size * num_beams, 1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(current_ids)], dim=1)
            output = self.backbone(input_ids=current_ids, attention_mask=attention_mask, position_ids=self._position_ids(attention_mask), past_key_values=past, use_cache=True, return_dict=True)
            logits = output.logits[:, -1, :]
            flat_prefixes = prefixes.reshape(batch_size * num_beams, -1)
            if logits_processor is not None:
                logits = logits_processor(logits, flat_prefixes)
            token_scores = F.log_softmax(logits.float(), dim=-1)
            vocab_size = token_scores.shape[-1]
            combined = token_scores.view(batch_size, num_beams, vocab_size)
            combined = combined + beam_scores.unsqueeze(-1)
            beam_scores, flat_indices = combined.view(batch_size, -1).topk(num_beams, dim=-1)
            source_beams = torch.div(flat_indices, vocab_size, rounding_mode='floor')
            next_tokens = flat_indices.remainder(vocab_size)
            gather_index = source_beams.unsqueeze(-1).expand(-1, -1, prefixes.shape[-1])
            prefixes = torch.cat([prefixes.gather(1, gather_index), next_tokens.unsqueeze(-1)], dim=-1)
            batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
            cache_indices = (source_beams + batch_offsets).reshape(-1)
            past = self._select_cache(output.past_key_values, cache_indices)
            attention_mask = attention_mask.index_select(0, cache_indices)
        steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        return (prefixes, beam_scores, steps)

    @torch.no_grad()
    def beam_search(self, prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor, max_new_tokens: int, num_beams: int=50, step_counts: Optional[torch.Tensor]=None, logits_processor=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_beams < 1:
            raise ValueError('num_beams must be positive')
        reason = self.latent_reason(prompt_ids, prompt_attention_mask, step_counts=step_counts)
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device
        end_ids = torch.full((batch_size, 1), self.latent_end_id, device=device, dtype=prompt_ids.dtype)
        attention_mask = torch.cat([reason['attention_mask'], torch.ones_like(end_ids)], dim=1)
        initial_kwargs = {'input_ids': end_ids}
        if self.use_trainable_markers:
            initial_kwargs = {'inputs_embeds': self.latent_end_embedding.view(1, 1, -1).expand(batch_size, 1, -1)}
        output = self.backbone(attention_mask=attention_mask, position_ids=self._position_ids(attention_mask), past_key_values=reason['past_key_values'], use_cache=True, return_dict=True, **initial_kwargs)
        logits = output.logits[:, -1, :]
        empty_prefix = torch.empty(batch_size, 0, dtype=prompt_ids.dtype, device=device)
        if logits_processor is not None:
            logits = logits_processor(logits, empty_prefix)
        first_log_probs = F.log_softmax(logits.float(), dim=-1)
        beam_scores, first_tokens = first_log_probs.topk(num_beams, dim=-1)
        prefixes = first_tokens.unsqueeze(-1)
        past = self._repeat_cache(output.past_key_values, num_beams)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
        for token_index in range(1, max_new_tokens):
            current_ids = prefixes[:, :, -1].reshape(batch_size * num_beams, 1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(current_ids)], dim=1)
            output = self.backbone(input_ids=current_ids, attention_mask=attention_mask, position_ids=self._position_ids(attention_mask), past_key_values=past, use_cache=True, return_dict=True)
            logits = output.logits[:, -1, :]
            flat_prefixes = prefixes.reshape(batch_size * num_beams, -1)
            if logits_processor is not None:
                logits = logits_processor(logits, flat_prefixes)
            token_scores = F.log_softmax(logits.float(), dim=-1)
            vocab_size = token_scores.shape[-1]
            combined = token_scores.view(batch_size, num_beams, vocab_size)
            combined = combined + beam_scores.unsqueeze(-1)
            beam_scores, flat_indices = combined.view(batch_size, -1).topk(num_beams, dim=-1)
            source_beams = torch.div(flat_indices, vocab_size, rounding_mode='floor')
            next_tokens = flat_indices.remainder(vocab_size)
            gather_index = source_beams.unsqueeze(-1).expand(-1, -1, prefixes.shape[-1])
            prefixes = torch.cat([prefixes.gather(1, gather_index), next_tokens.unsqueeze(-1)], dim=-1)
            batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
            cache_indices = (source_beams + batch_offsets).reshape(-1)
            past = self._select_cache(output.past_key_values, cache_indices)
            attention_mask = attention_mask.index_select(0, cache_indices)
        return (prefixes, beam_scores, reason['step_counts'])

    def save_pretrained(self, output_dir: str | Path, **kwargs) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output_dir / 'backbone', **kwargs)
        is_peft = hasattr(self.backbone, 'peft_config')
        base_model_name = None
        if is_peft:
            peft_configs = list(self.backbone.peft_config.values())
            if peft_configs:
                base_model_name = peft_configs[0].base_model_name_or_path
        torch.save(self.policy_head.state_dict(), output_dir / 'policy_head.pt')
        torch.save(self.latent_adapter.state_dict(), output_dir / 'latent_adapter.pt')
        torch.save({'start': self.latent_start_embedding.detach().cpu(), 'end': self.latent_end_embedding.detach().cpu()}, output_dir / 'latent_markers.pt')
        with (output_dir / 'lasar_config.json').open('w', encoding='utf-8') as handle:
            json.dump({'latent_start_id': self.latent_start_id, 'latent_thought_id': self.latent_thought_id, 'latent_end_id': self.latent_end_id, 'max_latent_steps': self.max_latent_steps, 'align_weight': self.align_weight, 'policy_weight': self.policy_weight, 'use_latent_adapter': self.use_latent_adapter, 'latent_adapter_scale': self.latent_adapter_scale, 'use_trainable_markers': self.use_trainable_markers, 'backbone_kind': 'peft' if is_peft else 'full', 'base_model_name_or_path': base_model_name, 'vocab_size': self.get_input_embeddings().num_embeddings}, handle, indent=2)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, **backbone_kwargs):
        model_dir = Path(model_dir)
        adapter_trainable = backbone_kwargs.pop('adapter_trainable', False)
        with (model_dir / 'lasar_config.json').open('r', encoding='utf-8') as handle:
            config = json.load(handle)
        backbone_kind = config.pop('backbone_kind', 'full')
        base_model_name = config.pop('base_model_name_or_path', None)
        vocab_size = config.pop('vocab_size', None)
        if backbone_kind == 'peft':
            from peft import PeftConfig, PeftModel
            adapter_dir = model_dir / 'backbone'
            peft_config = PeftConfig.from_pretrained(adapter_dir)
            base_model_name = base_model_name or peft_config.base_model_name_or_path
            backbone = AutoModelForCausalLM.from_pretrained(base_model_name, **backbone_kwargs)
            if vocab_size is not None and backbone.get_input_embeddings().num_embeddings != vocab_size:
                backbone.resize_token_embeddings(vocab_size)
            backbone = PeftModel.from_pretrained(backbone, adapter_dir, is_trainable=adapter_trainable)
        elif backbone_kind == 'full':
            backbone = AutoModelForCausalLM.from_pretrained(model_dir / 'backbone', **backbone_kwargs)
        else:
            raise ValueError(f'Unsupported LASAR backbone kind: {backbone_kind}')
        model = cls(backbone=backbone, **config)
        state = torch.load(model_dir / 'policy_head.pt', map_location='cpu', weights_only=True)
        model.policy_head.load_state_dict(state)
        adapter_path = model_dir / 'latent_adapter.pt'
        if adapter_path.exists():
            adapter_state = torch.load(adapter_path, map_location='cpu', weights_only=True)
            model.latent_adapter.load_state_dict(adapter_state)
        marker_path = model_dir / 'latent_markers.pt'
        if marker_path.exists():
            markers = torch.load(marker_path, map_location='cpu', weights_only=True)
            model.latent_start_embedding.data.copy_(markers['start'])
            model.latent_end_embedding.data.copy_(markers['end'])
        return model
