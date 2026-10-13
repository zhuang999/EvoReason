import torch
import torch.nn as nn
from typing import Any, Union
from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import selective_log_softmax
import transformers
from accelerate.utils import gather
from transformers import Qwen2ForCausalLM
from transformers import DataCollatorForSeq2Seq, LogitsProcessorList
from transformers.utils import is_sagemaker_mp_enabled
from latent_grpo_processor import CFEnhancedLogitsProcessor
from onerec.rl_math import normalize_advantages, reward_from_logps

def swap_adjacent_blocks(x, k):
    original_shape = x.shape
    x_2d = x.view(-1, k)
    n = x_2d.size(0)
    indices = torch.arange(n).view(-1, 2).flip(1).reshape(-1)
    return x_2d[indices].view(original_shape)

class NoiseGRPORecTrainer(GRPOTrainer):

    def __init__(self, prefix_allowed_tokens_fn, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        data_collator = DataCollatorForSeq2Seq(self.processing_class, pad_to_multiple_of=8, return_tensors='pt', padding=True)

        def data_collate_fn(batch):
            new_batch = data_collator(batch)
            return new_batch
        self.data_collator = data_collate_fn
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.generation_config = transformers.GenerationConfig(max_new_tokens=self.max_completion_length, do_sample=False, temperature=self.generation_config.temperature, pad_token_id=self.processing_class.pad_token_id)

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        logits = logits / self.temperature
        return selective_log_softmax(logits, input_ids)

    def my_get_per_token_logps(self, model, input_ids, inputs_embeds, attention_mask, logits_to_keep):
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'module'):
            model_to_call = model.module
        else:
            model_to_call = model
        outputs = model_to_call(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False, use_cache=False, logits_to_keep=logits_to_keep + 1)
        logits = outputs.logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        logits = logits / self.temperature
        return selective_log_softmax(logits, input_ids)

    @torch.no_grad()
    def _ppl_calculation(self, model, input_ids, attention_mask, input_embeds, logits_to_keep):
        per_token_logps = self.my_get_per_token_logps(model, input_ids, input_embeds, attention_mask, logits_to_keep)
        return (-per_token_logps.detach().clone().sum(dim=-1) / logits_to_keep).exp()

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(inputs)
        prompt_completion_ids, prompt_completion_mask = (prompt_inputs['input_ids'], prompt_inputs['attention_mask'])
        labels = prompt_inputs['labels']
        labels_length = (labels != -100).sum(dim=-1).clamp_min(1)
        if not labels_length.eq(labels_length[0]).all():
            raise ValueError('Released LR-GRPO batching requires a fixed target-token width')
        completion_width = int(labels_length[0].item())
        prompt_length = prompt_completion_ids.size(1) - completion_width
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completion_mask = prompt_completion_mask[:, prompt_length:]
        prompt_mask = prompt_completion_mask[:, :prompt_length]
        logits_to_keep = completion_ids.size(1)
        with torch.no_grad():
            batch_size = prompt_completion_ids.size(0)
            if batch_size % self.num_generations:
                raise ValueError('Each local LR-GRPO batch must contain complete noise groups')
            grouped_ids = prompt_completion_ids.view(-1, self.num_generations, prompt_completion_ids.size(1))
            if not grouped_ids.eq(grouped_ids[:, :1]).all():
                raise ValueError('LR-GRPO sampler must repeat the same (prompt, target) within each noise group')
            original_embeds = self.model.generate_embs(prompt_completion_ids, prompt_completion_mask)
            where_thought_ids = torch.nonzero(prompt_completion_ids == self.model.model.embed_tokens.num_embeddings - 1)
            noise = torch.randn((batch_size, original_embeds.size(-1)), device=self.model.device).mul(getattr(self, 'noise_std', 1.5))
            noise[::self.num_generations, :] = 0
            original_embeds[torch.arange(batch_size), where_thought_ids[:, 1]] += noise
        logits_to_keep = completion_ids.size(1)
        with torch.no_grad():
            old_per_token_logps = self.my_get_per_token_logps(self.model, prompt_completion_ids, original_embeds, prompt_completion_mask, logits_to_keep)
            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                ref_per_token_logps = self.my_get_per_token_logps(self.ref_model, prompt_completion_ids, original_embeds, prompt_completion_mask, logits_to_keep)
        prompts = [None for _ in range(len(prompt_ids))]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            output_reward_func_test = reward_from_logps(old_per_token_logps.clone(), labels_length, getattr(self, 'reward_transform', 'negative-ppl'))
            rewards_per_func[:, i] = output_reward_func_test
        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        temp_rewards = rewards.clone().view(-1, self.num_generations)
        clean_rewards = temp_rewards[:, 0]
        advantages = normalize_advantages(rewards, self.num_generations, getattr(self, 'advantage_mode', 'clean-batch'))
        mode = 'eval' if self.control.should_evaluate else 'train'
        self._metrics[mode]['reward'].append(rewards.mean().item())
        self._metrics[mode]['clean_reward'].append(clean_rewards.mean().item())
        self._metrics[mode]['reward_std'].append(rewards.std(unbiased=False).item())
        self._metrics[mode]['advantage_norm'].append(advantages.norm().item())
        process_slice = slice(self.accelerator.process_index * len(prompts), (self.accelerator.process_index + 1) * len(prompts))
        advantages = advantages[process_slice]
        return {'prompt_ids': prompt_ids, 'prompt_mask': prompt_mask, 'completion_ids': completion_ids, 'completion_mask': completion_mask, 'old_per_token_logps': old_per_token_logps, 'ref_per_token_logps': ref_per_token_logps, 'advantages': advantages, 'original_embeds': original_embeds, 'noise': noise}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError('The GRPOTrainer does not support returning outputs')
        prompt_ids, prompt_mask = (inputs['prompt_ids'], inputs['prompt_mask'])
        completion_ids, completion_mask = (inputs['completion_ids'], inputs['completion_mask'])
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        original_embeds = inputs['original_embeds']
        noise = inputs['noise']
        embeds = self.model.generate_embs(input_ids, attention_mask)
        where_thought_ids = torch.nonzero(input_ids == self.model.model.embed_tokens.num_embeddings - 1)
        batch_size = input_ids.size(0)
        embeds[torch.arange(batch_size), where_thought_ids[:, 1]] += noise
        per_token_logps = self.my_get_per_token_logps(model, input_ids, embeds, attention_mask, logits_to_keep)
        if self.beta != 0.0:
            ref_per_token_logps = inputs['ref_per_token_logps']
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        advantages = inputs['advantages']
        old_per_token_logps = inputs['old_per_token_logps'] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        mode = 'eval' if self.control.should_evaluate else 'train'
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]['kl'].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]['clip_ratio'].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss
