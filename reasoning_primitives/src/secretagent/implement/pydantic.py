"""Pydantic-AI based implementation factories for secretagent.

Provides:
  * SimulatePydanticFactory — simulate-stub agent (stub framing prompt)
  * ReactPydanticFactory — clean task-preamble agent (no stub framing)
  * ReactPydanticWithWorkflowFactory — ReAct agent + AWM workflow prepended

The latter two are shared across all benchmarks; each benchmark registers
them via `register_factory('react_pydantic', ReactPydanticFactory(preamble=...))`
with its own task-specific preamble string.

ADAPTATION (local vLLM only):
  LiteLLMModel receives optional api_base / api_key from config so that
  openai/* model ids can talk to a local OpenAI-compatible server.
"""

import hashlib
import json
import os
import pathlib
import time
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai_litellm import LiteLLMModel
from litellm import cost_per_token

from secretagent import config, record
from secretagent.cache_util import cached
from secretagent.core import register_factory
from secretagent.implement.core import SimulateFactory, resolve_tools, _load_template
from secretagent.llm_util import echo_boxed


def _make_litellm_model(model_name: str) -> LiteLLMModel:
    """Construct LiteLLMModel; inject local endpoint settings when present."""
    kwargs = {'model_name': model_name}
    api_base = config.get('llm.api_base', None)
    bases = config.get('llm.api_bases', None)
    if bases and not api_base:
        api_base = bases[0]
    if api_base:
        kwargs['api_base'] = api_base
    api_key = config.get('llm.api_key', None) or os.environ.get('OPENAI_API_KEY', 'EMPTY')
    kwargs['api_key'] = api_key
    return LiteLLMModel(**kwargs)

def _run_agent_hashkey(_, kwds):
    """Create a hashkey for the arguments of _run_agent.
    """
    hashable = (
        (kwds['interface'].name,  # use the interface name
         kwds['model_name'],  # a string
         str(kwds['return_type']), # convert type to str
         kwds['prompt'], # a string
         tuple(tool.__name__ for tool in kwds['tools']))) # use function names
    # convert the tuple to a string and encode it to hash
    return hashlib.sha256(str(hashable).encode('utf-8')).hexdigest()

def _run_agent_impl(interface, model_name, return_type, prompt, tools):
    """Run a pydantic agent.
    """
    if config.get('echo.model'):
        print(f'calling model {model_name}')

    if config.get('echo.llm_input'):
        echo_boxed(prompt, 'llm_input')

    # create the Model the agent will use and the Agent
    model = _make_litellm_model(model_name)  # ADAPTATION: local vLLM
    agent = Agent(model, output_type=return_type, tools=tools)

    # run the agent and time that
    from pydantic_ai.usage import UsageLimits
    start_time = time.time()
    request_limit = int(config.get('pydantic.request_limit') or 50)
    result = agent.run_sync(prompt, usage_limits=UsageLimits(request_limit=request_limit))
    latency = time.time() - start_time

    # get the answer and maybe echo it
    answer = result.output
    if config.get('echo.llm_output'):
        echo_boxed(str(answer), 'llm_output')

    # compute the other stats
    usage = result.usage()
    try:
        input_cost, output_cost = cost_per_token(
            model=model_name,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
        )
        cost = input_cost + output_cost
    except Exception:
        cost = 0.0  # ADAPTATION: local models often lack pricing metadata
    stats = dict(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        latency=latency,
        cost=cost,
    )
    # also send back a summary of messages from the agent
    messages = _summarize_messages(result.all_messages())
    return answer, stats, messages

def _run_agent(interface, model_name, return_type, prompt, tools):
    """Run a pydantic agent, with optional cachier caching via config."""
    return cached(_run_agent_impl, hash_func=_run_agent_hashkey)(
        interface, model_name, return_type, prompt, tools)

class SimulatePydanticFactory(SimulateFactory):
    """Simulate a function call using a pydantic-ai Agent.

    Reuses SimulateFactory.create_prompt() for the prompt, but strips
    the <answer> scaffolding and delegates to a pydantic-ai Agent
    for execution and output parsing.

    Examples:
      foo.implement_via('simulate_pydantic')
      foo.implement_via('simulate_pydantic', tools='__all__')
      foo.implement_via('simulate_pydantic', tools=[bar, baz])

    The options tools can take on several values
      tools = None or missing means don't use tools
      tools = '__all__' means use all other registered interfaces
      tools = [f1,...,fk] where the f's are interfaces or functions
        means just those tools
    """

    tools: list = Field(default_factory=list)
    prompt_kw: dict = Field(default_factory=dict)

    def setup(self, tools=None, **prompt_kw):
        self.tools = resolve_tools(self.bound_interface, tools)
        self.prompt_kw = prompt_kw

    def __call__(self, *args, **kw):
        interface = self.bound_interface
        with config.configuration(**self.prompt_kw):
            prompt = self.create_prompt(interface, *args, **kw)
            try:
                answer, stats, messages = _run_agent(
                    interface=interface,
                    model_name=self.llm_model,
                    return_type=interface.annotations.get('return', str),
                    prompt=prompt,
                    tools=self.tools)
            except Exception as ex:
                record.record(
                    func=interface.name, args=args, kw=kw,
                    output=f'**exception**: {ex}', step_info=[],
                    stats=dict(input_tokens=0, output_tokens=0, latency=0, cost=0))
                raise
            record.record(
                func=interface.name,
                args=args,
                kw=kw,
                output=answer,
                step_info=messages,
                stats=stats)
            return answer

    def create_prompt(self, interface, *args, **kw):
        """Construct a prompt that calls an LLM to predict the output of the function.
        """
        template = _load_template('simulate_pydantic.txt')
        input_args = interface.format_args(*args, **kw)
        if config.get('llm.thinking'):
            thoughts = "<thought>\nANY THOUGHTS\n</thought>\n"
        else:
            thoughts = ""
        prompt = template.substitute(
            dict(stub_src=interface.src,
                 args=input_args,
                 thoughts=thoughts))
        return prompt

def _summarize_messages(messages):
    """Summarize pydantic-ai messages into a simple list of steps.

    Extracts model thoughts (text), tool calls (name + args),
    and tool returns (name + output).
    """
    steps = []
    for msg in messages:
        for part in msg.parts:
            match part.part_kind:
                case 'text':
                    if part.content.strip():
                        steps.append({'thought': part.content})
                case 'tool-call':
                    steps.append({'tool_call': part.tool_name, 'args': part.args})
                case 'tool-return':
                    steps.append({'tool_return': part.tool_name, 'output': part.content})
    return steps

register_factory('simulate_pydantic', SimulatePydanticFactory())


class ReactPydanticFactory(SimulatePydanticFactory):
    """Pydantic-ai agent with a clean task preamble (no simulate/stub framing).

    Uses pydantic-ai's native tool-calling loop. The prompt emitted is:
      <preamble>\\n\\n<formatted args>

    Each benchmark registers this factory with its own preamble:
      register_factory('react_pydantic',
                       ReactPydanticFactory(preamble=MY_PREAMBLE))
    """

    preamble: str = Field(
        default=(
            "Answer the following question as best you can. Use the "
            "available tools to gather evidence, then call finish() "
            "with your final answer."
        )
    )

    def create_prompt(self, interface, *args, **kw):
        input_args = interface.format_args(*args, **kw)
        return f"{self.preamble}\n\n{input_args}"


class ReactPydanticWithWorkflowFactory(ReactPydanticFactory):
    """ReactPydanticFactory + a learned workflow guide prepended to the prompt.

    Builder kwargs (supplied via YAML ptool config):
        workflow_file: path to a .json (keyed by workflow_key) or .txt
        workflow_key: key into workflow_file when JSON; omit to concat all
        tools: same as ReactPydanticFactory
    """

    workflow_text: str = Field(default='')

    def setup(self, workflow_file=None, workflow_key=None,
              tools=None, **prompt_kw):
        super().setup(tools=tools, **prompt_kw)
        if workflow_file:
            content = pathlib.Path(workflow_file).read_text()
            if str(workflow_file).endswith('.json'):
                d = json.loads(content)
                if workflow_key is not None:
                    self.workflow_text = d.get(workflow_key, '')
                else:
                    self.workflow_text = '\n\n'.join(
                        f"### {k}\n{v}" for k, v in d.items()
                    )
            else:
                self.workflow_text = content

    def create_prompt(self, interface, *args, **kw):
        prompt = super().create_prompt(interface, *args, **kw)
        if self.workflow_text:
            prompt = (
                "## Learned workflow guide\n"
                "The following workflow was induced from successful past "
                "trajectories on similar problems. Use it as guidance for "
                "the task below.\n\n"
                f"{self.workflow_text}\n\n"
                "---\n\n"
                f"{prompt}"
            )
        return prompt
