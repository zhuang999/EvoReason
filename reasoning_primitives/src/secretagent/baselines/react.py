"""ReAct baseline: interleaved reasoning and acting.

Text-based Thought / Action / Observation loop via repeated LLM calls.

Reference: "ReAct: Synergizing Reasoning and Acting in Language Models"
(Yao et al., ICLR 2023)
"""

import inspect
import re
import time
from typing import Any, Callable

from secretagent import config
from secretagent.llm_util import llm


REACT_PROMPT = """\
Answer the following question as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}

Thought:"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_thought(text: str) -> str | None:
    m = re.search(
        r'Thought\s*:\s*(.*?)(?=\n(?:Action|Final Answer)\s*:|\Z)',
        text, re.DOTALL,
    )
    if m and m.group(1).strip():
        return m.group(1).strip()
    return None


def _parse_action(text: str) -> tuple[str | None, str | None]:
    action_m = re.search(r'Action\s*:\s*(.+?)(?:\n|$)', text)
    if not action_m:
        return None, None
    tool_name = action_m.group(1).strip()
    input_m = re.search(
        r'Action\s+Input\s*:\s*(.*?)(?=\n(?:Observation|Thought|Action|Final Answer)\s*:|\Z)',
        text[action_m.end():], re.DOTALL,
    )
    action_input = input_m.group(1).strip() if input_m else ''
    return tool_name, action_input


def _parse_final_answer(text: str) -> str | None:
    m = re.search(r'Final Answer\s*:\s*(.*)', text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return None


def _truncate_at_observation(text: str) -> str:
    m = re.search(r'\nObservation\s*:', text)
    return text[:m.start()] if m else text


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------

def _tool_description(fn: Callable) -> str:
    name = fn.__name__
    sig = inspect.signature(fn)
    params = ', '.join(f'{p.name}' for p in sig.parameters.values())
    doc = (fn.__doc__ or '').strip().split('\n')[0]
    return f'{name}({params}): {doc}'


def _coerce_arg(fn: Callable, arg_str: str) -> Any:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if not params:
        return fn()
    if len(params) == 1:
        ann = params[0].annotation
        if ann is inspect.Parameter.empty or ann is str:
            return fn(arg_str)
        try:
            return fn(ann(arg_str))
        except (ValueError, TypeError):
            return fn(arg_str)
    parts = [p.strip() for p in re.split(r'[|,]', arg_str)]
    coerced = []
    for part, param in zip(parts, params):
        ann = param.annotation
        if ann is inspect.Parameter.empty or ann is str:
            coerced.append(part)
        else:
            try:
                coerced.append(ann(part))
            except (ValueError, TypeError):
                coerced.append(part)
    return fn(*coerced)


def _resolve_tool(name: str, tools: dict[str, Callable]) -> str | None:
    if name in tools:
        return name
    for k in tools:
        if k.lower() == name.lower():
            return k
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def react(
    question: str,
    tools: dict[str, Callable],
    model: str | None = None,
    max_steps: int = 8,
) -> tuple[str, list[dict], dict]:
    """Run a ReAct loop on a question.

    Args:
        question: The question to answer.
        tools: Dict mapping tool name → callable.
        model: LLM model name (falls back to config).
        max_steps: Maximum Thought/Action rounds.

    Returns:
        (answer, trace, stats) where trace is a list of step dicts
        (thought/tool_call/tool_return) and stats has aggregated
        token counts, latency, and cost.
    """
    model = model or config.get('llm.model')

    tool_descs = '\n'.join(_tool_description(fn) for fn in tools.values())
    tool_names = ', '.join(tools.keys())

    prompt = REACT_PROMPT.format(
        tool_descriptions=tool_descs,
        tool_names=tool_names,
        question=question,
    )

    scratchpad = ''
    trace: list[dict] = []
    total_stats = dict(input_tokens=0, output_tokens=0, latency=0.0, cost=0.0)
    answer: str | None = None
    start_time = time.time()

    for step in range(1, max_steps + 1):
        response, stats = llm(prompt + scratchpad, model)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

        text = _truncate_at_observation(f'Thought: {response}')

        thought = _parse_thought(text)
        final = _parse_final_answer(text)
        tool_name, tool_input = _parse_action(text)

        if thought:
            trace.append({'thought': thought})

        # Final answer → done
        if final is not None:
            answer = final
            break

        # No action, no final answer → treat thought as answer
        if tool_name is None:
            answer = thought or response.strip()
            break

        # Build scratchpad
        scratchpad += (
            f' {thought or ""}\n'
            f'Action: {tool_name}\n'
            f'Action Input: {tool_input}\n'
        )
        trace.append({'tool_call': tool_name, 'args': tool_input})

        # Dispatch
        resolved = _resolve_tool(tool_name, tools)
        if resolved:
            try:
                observation = str(_coerce_arg(tools[resolved], tool_input))
            except Exception as e:
                observation = f'Error calling {resolved}: {e}'
        else:
            observation = (
                f'Unknown action "{tool_name}". '
                f'Available: [{tool_names}]'
            )

        trace.append({'tool_return': resolved or tool_name, 'output': observation})
        scratchpad += f'Observation: {observation}\nThought:'

    total_stats['latency'] = time.time() - start_time
    return answer or '', trace, total_stats
