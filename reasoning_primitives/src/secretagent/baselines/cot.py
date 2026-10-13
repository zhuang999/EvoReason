"""Chain-of-Thought prompting baseline.

Simple zero-shot CoT: append "Let's think step by step" to the question,
then extract the final answer.

Reference: "Chain-of-Thought Prompting Elicits Reasoning in Large
Language Models" (Wei et al., NeurIPS 2022)
"""

import re

from secretagent import config
from secretagent.llm_util import llm


COT_PROMPT = """\
Q: {question}

A: Let's think step by step.
"""

ANSWER_EXTRACTION_PROMPT = """\
{cot_response}

Therefore, the answer (as a single number or short phrase) is:"""


def cot(
    question: str,
    model: str | None = None,
    answer_pattern: str | None = None,
    extract_answer: bool = True,
) -> tuple[str, str, dict]:
    """Run zero-shot chain-of-thought on a question.

    Args:
        question: The question to answer.
        model: LLM model name (falls back to config).
        answer_pattern: Regex to extract answer from the CoT response.
            If None, uses a follow-up extraction prompt.
        extract_answer: If True, run a second LLM call to extract a
            concise answer from the reasoning trace.

    Returns:
        (answer, reasoning, stats) where reasoning is the full CoT
        trace and answer is the extracted final answer.
    """
    model = model or config.get('llm.model')

    prompt = COT_PROMPT.format(question=question)
    reasoning, stats = llm(prompt, model)
    reasoning = reasoning.strip()

    # Try regex extraction first
    if answer_pattern:
        m = re.search(answer_pattern, reasoning, re.DOTALL)
        if m:
            return m.group(1).strip(), reasoning, stats

    # Follow-up extraction call
    if extract_answer:
        extract_prompt = ANSWER_EXTRACTION_PROMPT.format(cot_response=reasoning)
        answer, extract_stats = llm(extract_prompt, model)
        for k in stats:
            if isinstance(stats[k], (int, float)):
                stats[k] += extract_stats.get(k, 0)
        return answer.strip(), reasoning, stats

    return reasoning, reasoning, stats
