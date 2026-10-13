"""Access an LLM model, and monitor cost, latency, etc.

ADAPTATION (local vLLM only):
  - Reads optional `llm.api_base` / `llm.api_key` / `llm.api_bases` from config
    so openai/* models can hit a locally served OpenAI-compatible endpoint.
  - Local models may lack litellm pricing; cost falls back to 0.0.
  All other logic is identical to reasoning-primitives-main.
"""

import os
import sys
import time
import threading
from typing import Any

from secretagent import config
from secretagent.cache_util import cached
from litellm import completion, completion_cost, token_counter

# --- local vLLM: optional round-robin across multiple api bases ---
_RR_COUNTER = 0
_RR_LOCK = threading.Lock()


def _inject_local_endpoint(extra_kw: dict) -> None:
    """Attach api_base / api_key for local OpenAI-compatible servers."""
    bases = config.get('llm.api_bases', None)
    if bases:
        global _RR_COUNTER
        with _RR_LOCK:
            idx = _RR_COUNTER % len(bases)
            _RR_COUNTER += 1
        extra_kw['api_base'] = bases[idx]
    else:
        api_base = config.get('llm.api_base', None)
        if api_base:
            extra_kw['api_base'] = api_base
    api_key = config.get('llm.api_key', None) or os.environ.get('OPENAI_API_KEY', 'EMPTY')
    extra_kw['api_key'] = api_key

def echo_boxed(text: str, tag:str = ''):
    """Echo some text in a pretty box."""
    lines = text.split('\n')
    width = max(len(line) for line in lines)
    print('┌' + tag.center(width+2, '─') + '┐')
    for line in lines:
        print('│ ' + line.ljust(width) + ' │')
    print('└' + '─' * (width + 2) + '┘')

def _llm_impl(prompt: str, model: str) -> tuple[str, dict[str, Any]]:
  """Use an LLM model.

  Returns result as a string plus a dictionary of measurements,
  including # input_tokens, # output_tokens, latency in seconds, and cost.

  Set config 'llm.stream' to True to stream responses (visible with echo.stream).
  """
  if config.get('echo.model'):
    print(f'calling model {model}')

  if config.get('echo.llm_input'):
    echo_boxed(prompt, 'llm_input')

  messages = [dict(role='user', content=prompt)]
  stream = config.get('llm.stream', False)
  max_tokens = config.get('llm.max_tokens', None)
  temperature = config.get('llm.temperature', None)
  extra_kw = {}
  if max_tokens:
    extra_kw['max_tokens'] = int(max_tokens)
  if temperature is not None:
    extra_kw['temperature'] = float(temperature)
  extra_kw['timeout'] = config.get('llm.timeout', 600)
  extra_kw['num_retries'] = config.get('llm.num_retries', 1)
  _inject_local_endpoint(extra_kw)  # ADAPTATION: local vLLM
  start_time = time.time()

  if stream:
    chunks = []
    response_stream = completion(
        model=model, messages=messages, stream=True,
        stream_options={'include_usage': True},
        **extra_kw,
    )
    for chunk in response_stream:
      delta = ''
      if chunk.choices:
        delta = chunk.choices[0].delta.content or ''
      chunks.append(delta)
      if config.get('echo.stream') and delta:
        sys.stderr.write(delta)
        sys.stderr.flush()
    if config.get('echo.stream'):
      sys.stderr.write('\n')
    latency = time.time() - start_time
    model_output = ''.join(chunks)

    # Estimate tokens since streaming doesn't reliably return usage
    input_tokens = token_counter(model=model, messages=messages)
    output_tokens = token_counter(model=model, text=model_output)
    try:
      from litellm import model_cost
      cost_info = model_cost.get(model, {})
      cost = (input_tokens * cost_info.get('input_cost_per_token', 0) +
              output_tokens * cost_info.get('output_cost_per_token', 0))
    except Exception:
      cost = 0.0

    stats = dict(
      input_tokens=input_tokens,
      output_tokens=output_tokens,
      latency=latency,
      cost=cost,
    )
  else:
    response = completion(model=model, messages=messages, **extra_kw)
    latency = time.time() - start_time
    msg = response.choices[0].message
    content = msg.content or ''
    reasoning = getattr(msg, 'reasoning_content', None) or ''
    # Thinking models (e.g. Qwen 3.5) sometimes put <answer> tags in
    # reasoning_content instead of content. Prefer content; fall back
    # to reasoning_content only if content lacks the expected tags.
    if content and '<answer>' in content:
      model_output = content
    elif reasoning and '<answer>' in reasoning:
      # Extract only the LAST <answer>...</answer> block from reasoning,
      # since earlier ones are the model thinking about the format.
      import re
      matches = re.findall(r'<answer>(.*?)</answer>', reasoning, re.DOTALL)
      if matches:
        last_answer = matches[-1].strip()
        model_output = f'<answer>{last_answer}</answer>'
      else:
        model_output = reasoning
    else:
      model_output = content or reasoning

    try:
      cost = completion_cost(completion_response=response)
    except Exception:
      cost = 0.0  # ADAPTATION: local models often lack pricing metadata
    stats = dict(
      input_tokens=response.usage.prompt_tokens,
      output_tokens=response.usage.completion_tokens,
      latency=latency,
      cost=cost,
    )

  if config.get('echo.llm_output'):
    echo_boxed(model_output, 'llm_output')

  return model_output, stats

def llm(prompt: str, model: str) -> tuple[str, dict[str, Any]]:
  """Use an LLM model, with optional cachier caching via config.

  See cache_util.py for why this weird process is necessary.
  """
  return cached(_llm_impl)(prompt, model)
