"""Self-Discover reasoning structure induction.

Three-step pipeline (SELECT → ADAPT → IMPLEMENT) that induces a
reusable reasoning plan from a task description and example instances.

Reference: "Self-Discover: Large Language Models Self-Compose Reasoning
Structures" (Zhou et al., NeurIPS 2024)
"""

from secretagent import config
from secretagent.llm_util import llm


SELECT_PROMPT = """\
You are selecting reasoning modules to solve a class of problems.

Task description:
{task_desc}

Here are two example instances of this task (narrative, question, choices):

--- Example 1 ---
{example_1}

--- Example 2 ---
{example_2}

Here are candidate reasoning modules:

{modules}

Select the 4-7 modules that are MOST relevant for solving this task. Output
only the selected modules, one per line, verbatim from the list above
(including the number). Do not add commentary.
"""


ADAPT_PROMPT = """\
You are adapting generic reasoning modules to a specific task.

Task description:
{task_desc}

Selected modules:
{selected}

Rephrase each selected module so that it is concretely tailored to the
task (reference the specific objects of reasoning — e.g. for murder
mystery, mention suspects / alibis / evidence). Keep it crisp: one
sentence per module. Output a numbered list and nothing else.
"""


IMPLEMENT_PROMPT = """\
You are composing adapted reasoning modules into a concrete reasoning
plan that a future solver will follow.

Task description:
{task_desc}

Adapted modules:
{adapted}

Write a 5-8 step reasoning plan that an LLM should execute for each new
instance of this task. Each step should be specific and actionable
(reference suspects, locations, characters, evidence, etc. — whichever
is relevant). The final step MUST be: "State the final answer as the
0-based index of the correct choice."

Output the plan as a numbered list and nothing else.
"""


def _call(prompt: str, model: str, tag: str, verbose: bool = True) -> str:
    text, stats = llm(prompt, model)
    if verbose:
        in_t = stats.get('input_tokens', 0)
        out_t = stats.get('output_tokens', 0)
        print(f'    [{tag}] tokens={in_t}+{out_t}')
    return text.strip()


def induce_structure(
    task_desc: str,
    example_1: str,
    example_2: str,
    modules: str,
    model: str | None = None,
    verbose: bool = True,
) -> str:
    """Run the 3-step Self-Discover pipeline.

    Args:
        task_desc: Human-readable description of the task class.
        example_1: Formatted first example instance.
        example_2: Formatted second example instance.
        modules: Newline-separated list of candidate reasoning modules.
        model: LLM model name (falls back to config).
        verbose: Print token counts per step.

    Returns:
        The induced reasoning plan as text.
    """
    model = model or config.get('llm.model')

    select_text = _call(
        SELECT_PROMPT.format(
            task_desc=task_desc,
            example_1=example_1,
            example_2=example_2,
            modules=modules,
        ),
        model, 'SELECT', verbose,
    )

    adapt_text = _call(
        ADAPT_PROMPT.format(task_desc=task_desc, selected=select_text),
        model, 'ADAPT', verbose,
    )

    implement_text = _call(
        IMPLEMENT_PROMPT.format(task_desc=task_desc, adapted=adapt_text),
        model, 'IMPLEMENT', verbose,
    )

    return implement_text
