"""AWM-style workflow induction from successful agent trajectories.

Given formatted rollout traces from successful runs, prompts an LLM to
extract a reusable step-by-step workflow.

Reference: "Agent Workflow Memory" (Wang et al., 2024)
"""

from secretagent import config
from secretagent.llm_util import llm


INDUCTION_PROMPT = """\
You are reading rollouts of an AI agent that solved {task_label} problems
using the ReAct paradigm. Each rollout shows the agent's tool calls
(search, lookup, finish) and its brief thoughts.

Below are {n_traces} *successful* rollouts. Your job: extract a single
step-by-step workflow that captures the strategy these successful runs
share. The workflow should be concrete and prescriptive — a future agent
reading it should know what to do.

Format requirements:
- 5–10 numbered steps.
- Each step is one or two sentences.
- Reference the tools by name (search, lookup, finish).
- Mention what kinds of queries to use, in what order, what to look for.
- The final step should be: call finish(answer_index) once you have the answer.

Do NOT include prose preamble. Output only the numbered list.

=== Rollouts ===

{rollouts}
"""


def induce_workflow(
    rollouts_text: str,
    task_label: str,
    n_traces: int,
    model: str | None = None,
) -> tuple[str, dict]:
    """Induce a workflow from formatted rollout text.

    Returns (workflow_text, llm_stats).
    """
    prompt = INDUCTION_PROMPT.format(
        task_label=task_label,
        n_traces=n_traces,
        rollouts=rollouts_text,
    )
    model = model or config.get('llm.model')
    text, stats = llm(prompt, model)
    return text.strip(), stats
