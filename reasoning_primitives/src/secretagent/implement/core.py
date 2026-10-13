"""Built-in Implementation.Factory classes for secretagent.

Provides DirectFactory, SimulateFactory, PromptLLMFactory, and PoTFactory.
"""

import ast
import importlib
import json
import pathlib
import re

from string import Template
from textwrap import dedent
from typing import Any, Callable

from pydantic import Field

from secretagent import config, llm_util, record
from secretagent.core import Interface, Implementation, register_factory, all_interfaces

PROMPT_TEMPLATE_DIR = pathlib.Path(__file__).parent / 'prompt_templates'

class DirectFactory(Implementation.Factory):
    """Use a Python function as the implementation.

    Defaults to the body of the interface functions.

    Example: foo.implement_via('direct', pyfn_implementing_foo)
    """
    direct_fn: Any = None

    def setup(self, fn: Callable | str | None = None, **_kw):
        if isinstance(fn, str):
            self.direct_fn = resolve_dotted(fn)
        elif fn is not None:
            self.direct_fn = fn
        else:
            self.direct_fn = self.bound_interface.func

    def __call__(self, *args, **kw):
        return self.direct_fn(*args, **kw)

class SimulateFactory(Implementation.Factory):
    """Simulate a function call with an LLM.

    Examples: 
      foo.implement_via('simulate')
      foo.implement_via('simulate', llm.model=..., example_file='foo_demonstrations.json')

    The json files look like this:
    {
      "sport_for": [
        {"input_args": ["Bam Adebayo"], "expected_output": "basketball"},
        {"input_args": ["scored a reverse layup"], "expected_output": "basketball"}
      ],
      "analyze_sentence": [
        {"input_args": ["Bam Adebayo scored a reverse layup."], "expected_output": ["Bam Adebayo", "scored a reverse layup", ""]}
      ]
    }

    If you were to configuring with a yaml file you would use:

    ptools:
      sport_for:
        method: simulate
        example_file: examples/examples.json
      analyze_sentence:
        method: simulate
        example_file: examples/examples.json

    """
    examples_cases: list | None = None
    prompt_kw: dict = Field(default_factory=dict)

    def setup(self, example_file=None, **prompt_kw):
        self.prompt_kw = prompt_kw
        if example_file:
            data = json.loads(pathlib.Path(example_file).read_text())
            self.examples_cases = data.get(self.bound_interface.name, [])

    def __call__(self, *args, **kw):
        interface = self.bound_interface
        with config.configuration(**self.prompt_kw):
            prompt = self.create_prompt(interface, *args, examples=self.examples_cases, **kw)
            llm_output, stats = llm_util.llm(prompt, self.llm_model)
            try:
                return_type = interface.annotations.get('return', str)
                answer = self.parse_output(return_type, llm_output)
            except Exception as ex:
                record.record(func=interface.name, args=args, kw=kw,
                              output=f'**exception**: {ex}', stats=stats)
                raise
            record.record(func=interface.name, args=args, kw=kw, output=answer, stats=stats)
            return answer

    def create_prompt(self, interface, *args, examples=None, **kw):
        """Construct a prompt that calls an LLM to predict the output of the function.
        """
        template = _load_template('simulate.txt')
        input_args = interface.format_args(*args, **kw)
        if (not input_args.strip()):
            raise ValueError(f'input_args null for {args=} {kw=}')
        if config.get('llm.thinking'):
            thoughts = "<thought>\nANY THOUGHTS\n</thought>\n"
        else:
            thoughts = ""
        examples_text = ""
        if examples:
            examples_text = _format_examples_as_doctests(interface.name, examples)
        prompt = template.substitute(
            dict(stub_src=interface.src,
                 input_args=input_args,
                 thoughts=thoughts,
                 examples=examples_text))
        return prompt

    def parse_output(self, return_type, text):
        """Take LLM output and return the final answer, in the correct type.

        Tries <answer>...</answer> tags first.  For dict/list return types,
        falls back to extracting JSON from a markdown code block or the first
        bare {...} / [...] object in the output — handles models that don't
        follow the tag format for complex types.
        """
        import json
        match_result = re.search(r'<answer>(.*)</answer>', text, re.DOTALL|re.MULTILINE)
        if match_result:
            final_answer = match_result.group(1).strip()
            if return_type in [int, str, float]:
                return return_type(final_answer)
            try:
                return json.loads(final_answer)
            except json.JSONDecodeError:
                return ast.literal_eval(final_answer)

        # Fallback for dict/list: extract JSON from the raw output
        if return_type in [dict, list] or (hasattr(return_type, '__origin__')
                                           and return_type.__origin__ in [dict, list]):
            open_ch, close_ch = ('{', '}') if return_type is dict else ('[', ']')
            # prefer a fenced code block
            code_match = re.search(
                r'```(?:json|python)?\s*(' + re.escape(open_ch) + r'.*?' + re.escape(close_ch) + r')\s*```',
                text, re.DOTALL)
            if code_match:
                candidate = code_match.group(1).strip()
            else:
                # find the outermost balanced bracket span
                start = text.find(open_ch)
                end = text.rfind(close_ch)
                candidate = text[start:end + 1].strip() if start != -1 and end != -1 else ''
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return ast.literal_eval(candidate)

        # For str return type without <answer> tags, return the raw text
        if return_type is str:
            return text.strip()

        raise ValueError('cannot find final answer')


class PromptLLMFactory(Implementation.Factory):
    """Prompt an LLM with a user-supplied template.

    Unlike SimulateFactory (which has a fixed prompt), this factory
    lets the caller supply their own prompt template and answer-extraction
    pattern.

    Examples:
      foo.implement_via('prompt_llm', prompt_template_str='Is $sentence true?')
      foo.implement_via('prompt_llm', prompt_template_file='prompts/foo.txt',
                        answer_pattern=r'<answer>(.*)</answer>')

    Builder kwargs (passed to implement_via):
        prompt_template_str: A string.Template with $args
            placeholders. Exactly one of prompt_template_str or
            prompt_template_file must be given.
        prompt_template_file: Path to a file containing the template.
        answer_pattern: Regex with one group to extract the answer.
            Defaults to r'<answer>(.*)</answer>'. If None and the
            return type is str, the full LLM output is returned.
    """
    template: Any = None
    answer_pattern: str | None = None
    prompt_kw: dict = Field(default_factory=dict)

    def setup(self, prompt_template_str=None, prompt_template_file=None,
              answer_pattern=r'<answer>(.*)</answer>', **prompt_kw):
        if (prompt_template_str is None) == (prompt_template_file is None):
            raise ValueError(
                'Exactly one of prompt_template_str or prompt_template_file must be given')
        if prompt_template_file is not None:
            prompt_template_str = pathlib.Path(prompt_template_file).read_text()
        self.template = Template(dedent(prompt_template_str))
        self.answer_pattern = answer_pattern
        self.prompt_kw = prompt_kw

    def __call__(self, *args, **kw):
        interface = self.bound_interface
        with config.configuration(**self.prompt_kw):
            arg_names = list(interface.annotations.keys())[:-1]
            arg_dict = dict(zip(arg_names, args))
            arg_dict.update(kw)
            prompt = self.template.substitute(arg_dict)
            llm_output, stats = llm_util.llm(
                prompt, self.llm_model)
            try:
                return_type = interface.annotations.get('return', str)
                answer = _extract_answer(return_type, llm_output, self.answer_pattern)
            except Exception as ex:
                record.record(func=interface.name, args=args, kw=kw,
                              output=f'**exception**: {ex}', stats=stats)
                raise
            record.record(func=interface.name, args=args, kw=kw,
                          output=answer, stats=stats)
            return answer


def _extract_answer(return_type, text, answer_pattern):
    """Extract and type-cast the answer from LLM output."""
    if answer_pattern is None and return_type is str:
        return text.strip()
    if answer_pattern is None:
        raise ValueError(
            'answer_pattern is required when return type is not str')
    match_result = re.search(answer_pattern, text, re.DOTALL | re.MULTILINE)
    if match_result is None:
        llm_util.echo_boxed(text, 'bad code_eval_output')
        raise ValueError(f'cannot find answer matching pattern {answer_pattern!r}')
    final_answer = match_result.group(1).strip()
    if return_type in [int, str, float]:
        return return_type(final_answer)
    return ast.literal_eval(final_answer)

class PoTFactory(Implementation.Factory):
    """Generate Python code with an LLM and execute it in a sandbox.

    Examples:
      foo.implement_via('program_of_thought')
      foo.implement_via('program_of_thought', tools='__all__')
      foo.implement_via('program_of_thought',
                        tools=[bar, baz],
                        additional_imports=['numpy'])
    """
    python_executor: Any = None
    tool_interfaces: list = Field(default_factory=list)
    additional_imports: Any = None
    inject_args: bool = False
    prompt_kw: dict = Field(default_factory=dict)

    def setup(self, tools='__all__', additional_imports=None, inject_args=False, **prompt_kw):
        from smolagents.local_python_executor import LocalPythonExecutor, BASE_PYTHON_TOOLS

        interface = self.bound_interface
        self.prompt_kw = prompt_kw
        self.additional_imports = additional_imports
        self.inject_args = inject_args
        resolved_tools = resolve_tools(interface, tools)
        tool_functions = {fn.__name__: fn for fn in resolved_tools}
        self.python_executor = LocalPythonExecutor(
            additional_authorized_imports=(additional_imports or []),
            )
        self.python_executor.custom_tools = tool_functions
        self.python_executor.static_tools = {
            **BASE_PYTHON_TOOLS,
            "final_answer": lambda x: x,
        }
        # collect interfaces for tool stubs in the prompt
        self.tool_interfaces = [
            iface for iface in all_interfaces()
            if iface is not interface
            and iface.implementation is not None
            and iface.implementation.implementing_fn in tool_functions.values()]

    def __call__(self, *args, **kw):
        interface = self.bound_interface
        with config.configuration(**self.prompt_kw):
            # Inject input arg values into sandbox so LLM can reference
            # them as variables without copying large strings into code.
            if self.inject_args:
                arg_names = list(interface.annotations.keys())[:-1]
                for name, val in zip(arg_names, args):
                    self.python_executor.custom_tools[name] = val
            prompt = self.create_prompt(
                interface, self.tool_interfaces, self.additional_imports,
                *args, inject_args=self.inject_args, **kw)
            llm_output, stats = llm_util.llm(
                prompt, self.llm_model)
            try:
                generated_code = _extract_answer(
                    str,
                    llm_output,
                    answer_pattern='```python\n([^`]*)\n```')
                if config.get('echo.code_eval_input'):
                    llm_util.echo_boxed(generated_code, 'code_eval_input')
                code_output = self.python_executor(generated_code)
                answer = code_output.output
            except Exception as ex:
                record.record(
                    func=interface.name, args=args, kw=kw,
                    output=f'**exception**: {ex}', stats=stats,
                    step_info=dict(generated_code=llm_output))
                raise
            if config.get('echo.code_eval_output'):
                llm_util.echo_boxed(str(answer), 'code_eval_output')
            record.record(
                func=interface.name, args=args, kw=kw, output=answer, stats=stats,
                step_info=dict(generated_code=generated_code))
            return answer

    def create_prompt(self, interface, tool_interfaces, additional_authorized_imports,
                      *args, inject_args=False, **kw):
        """Construct a prompt that calls an LLM to predict the output of the function.
        """
        if inject_args:
            arg_names = list(interface.annotations.keys())[:-1]
            var_list = ', '.join(f'`{n}`' for n in arg_names)
            input_args = interface.format_args(*args, **kw)
            input_args += f'\n\nThe input args are available in these variables: {var_list}'
        else:
            input_args = interface.format_args(*args, **kw)
        if (not input_args.strip()):
            raise ValueError(f'input_args null for {interface.name} on {args=} {kw=}')
        tool_stub_src_listing = '\n\n'.join([
            tool_interface.src
            for tool_interface in tool_interfaces
            ])
        if additional_authorized_imports:
            imports = '\n' + '\n'.join(
                ['You may use these packages:\n'] + 
                [f'import {package}'
                 for package in additional_authorized_imports]
            ) + '\n\n'
        else:
            imports = ''

        if config.get('llm.thinking'):
            thoughts = '<thought>\nANY THOUGHTS\n</thought>\n\n'
        else:
            thoughts = ''

        template_bindings=dict(
            main_stub_src=interface.src,
            func_name=interface.name,
            tool_stub_src_listing=tool_stub_src_listing,
            input_args=input_args,
            imports=imports,
            thoughts=thoughts
        )

        template = _load_template('program_of_thought.txt')
        prompt = template.substitute(**template_bindings)
        if inject_args:
            prompt = prompt.replace(
                'Start the\ncode block by initializing a variable for each input.  It is ESSENTIAL\n'
                'that the code is able to be run independently.',
                'The input variables are already loaded in the execution environment — '
                'do not write them out in the code. Use them in your code if needed.')
        return prompt


#
# helpers
# 

def _load_template(name: str) -> Template:
    """Load a prompt template from the prompt_templates directory."""
    return Template((PROMPT_TEMPLATE_DIR / name).read_text())

def _format_examples_as_doctests(interface_name: str, cases: list) -> str:
    """Format example cases as doctest-style examples for the prompt."""
    lines = ["Here are some examples:", ""]
    for case in cases:
        args = case.get('input_args', []) if isinstance(case, dict) else (case.input_args or [])
        out = case.get('expected_output') if isinstance(case, dict) else case.expected_output
        args_str = ', '.join(repr(a) for a in args)
        lines.append(f">>> {interface_name}({args_str})")
        lines.append(repr(out))
    return '\n'.join(lines) + '\n'


def resolve_dotted(name: str) -> Any:
    """Resolve a dotted name like 'module.func' or 'pkg.module.func' to the
    actual object. Tries the longest importable prefix as the module path,
    then walks the remainder as attributes.
    """
    parts = name.split('.')
    for n in range(len(parts), 0, -1):
        module_path = '.'.join(parts[:n])
        try:
            obj = importlib.import_module(module_path)
        except ImportError:
            continue
        for part in parts[n:]:
            obj = getattr(obj, part)
        return obj
    raise ImportError(f"Could not resolve any module prefix of {name!r}")


def resolve_tools(interface: Interface, tools) -> list[Callable]:
    """Resolve a tools specification into a list of callables.

    The tools parameter can be:
      - None or [] → no tools (returns [])
      - '__all__' → all implemented interfaces except the given one
      - a list where each element is:
          - a callable (used as-is)
          - an Interface (resolved to its implementing function)
          - a string (resolved via resolve_dotted)
    """
    if tools == '__all__':
        tools = [iface for iface in all_interfaces()
                 if iface is not interface and iface.implementation is not None]
    tools = tools or []
    resolved = []
    for tool in tools:
        if isinstance(tool, str):
            tool = resolve_dotted(tool)
        if isinstance(tool, Interface):
            if tool.implementation is None:
                raise ValueError(f'Interface {tool.name!r} has no implementation')
            resolved.append(tool.implementation.implementing_fn)
        else:
            resolved.append(tool)
    return resolved




register_factory('direct', DirectFactory())
register_factory('simulate', SimulateFactory())
register_factory('program_of_thought', PoTFactory())
register_factory('prompt_llm', PromptLLMFactory())
