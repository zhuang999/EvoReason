"""Generate induced ptool modules and eval configs from induction reports.

Faithful adaptation of benchmarks/musr/generate_induced_configs.py:

  narrative → context
  finish    → produce_ranking
  murder/object/team → rec
  + search_memory in variant A

Usage:
    python generate_induced_configs.py --seed 42 \\
        --report induction_out/original/rec/report.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from keyword import iskeyword

_BENCHMARK_DIR = pathlib.Path(__file__).resolve().parent


def _sanitize_identifier(name: str, used: set[str]) -> str:
    """Ensure `name` is a valid, unique Python identifier."""
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name.strip())
    if not name or not (name[0].isalpha() or name[0] == '_'):
        name = 'p_' + name
    if iskeyword(name):
        name = name + '_'
    base = name
    i = 2
    while name in used:
        name = f'{base}_{i}'
        i += 1
    used.add(name)
    return name


def _indent_docstring(doc: str, indent: str = '    ') -> str:
    """Format a docstring body with proper indentation inside triple quotes."""
    doc = doc.replace('\\n', '\n')
    doc = doc.replace('"""', '\\"\\"\\"')
    lines = doc.split('\n')
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(f'{indent}{ln}' if ln.strip() else '' for ln in lines)


def generate_ptool_module(report: dict, seed: int, suffix: str = '') -> tuple[str, list[str]]:
    """Return (python source, list of wrapper tool dotted paths).

    Handles both regular ptools and avoidance ptools from failed trajectories.
    Avoidance ptools get @implement_via('simulate') just like regular ones,
    but their stub name is prefixed with 'avoid_' and wrapper injects context.
    """
    tag = f'seed{seed}'
    if suffix:
        tag = f'{tag}_{suffix}'
    module_name = f'induced.induced_ptools_{tag}_rec'
    lines: list[str] = [
        f'"""Induced ptools for recommendation (seed={seed}).',
        '',
        f'Auto-generated from {report.get("trace_path", "?")}.',
        f'Model: {report.get("model", "?")}.',
        'Do not edit — regenerate via generate_induced_configs.py.',
        '"""',
        '',
        'from secretagent.core import implement_via',
        'from ptools.ptools_common import _REACT_STATE',
        '',
        '',
    ]

    used_stub_names: set[str] = set()
    used_tool_names: set[str] = set()
    tool_paths: list[str] = []

    all_ptools = (
        list(report.get('ptools', []))
        + list(report.get('success_ptools', []))
        + list(report.get('avoidance_ptools', []))
    )

    for p in all_ptools:
        raw_name = p.get('func_name') or ''
        if not raw_name:
            continue
        tool_name = _sanitize_identifier(raw_name, used_tool_names)
        stub_name = _sanitize_identifier(f'_{tool_name}_impl', used_stub_names)

        short_desc = (p.get('short_desc') or '').strip()
        docstring_body = _indent_docstring(p.get('docstring') or short_desc or tool_name)

        lines.append("@implement_via('simulate')")
        lines.append(f'def {stub_name}(context: str, focus: str) -> str:')
        lines.append('    """')
        if docstring_body:
            lines.append(docstring_body)
        lines.append('    """')
        lines.append('')
        lines.append('')

        wrapper_doc = short_desc if short_desc else f'Induced reasoning helper: {tool_name}.'
        wrapper_doc = wrapper_doc.replace('"""', '\\"""')
        lines.append(f'def {tool_name}(focus: str) -> str:')
        lines.append(f'    """{wrapper_doc}')
        lines.append('')
        lines.append('    Args:')
        lines.append('        focus: what preference / attribute / conflict aspect')
        lines.append('               to reason about (qualitative).')
        lines.append('    """')
        lines.append(f'    return {stub_name}(_REACT_STATE["context"], focus)')
        lines.append('')
        lines.append('')

        tool_paths.append(f'{module_name}.{tool_name}')

    source = '\n'.join(lines).rstrip() + '\n'
    return source, tool_paths


def generate_configs(seed: int, tool_paths: list[str], suffix: str = '') -> dict[str, str]:
    """Return dict {variant: yaml_string} for variants A and B."""
    tag = f'seed{seed}'
    if suffix:
        tag = f'{tag}_{suffix}'

    tools_a = (
        '      - ptools.ptools_common.search\n'
        '      - ptools.ptools_common.lookup\n'
        '      - ptools.ptools_common.search_memory\n'
        '      - ptools.ptools_common.produce_ranking\n'
        + ''.join(f'      - {p}\n' for p in tool_paths)
    ).rstrip()

    tools_b = (
        '      - ptools.ptools_common.produce_ranking\n'
        + ''.join(f'      - {p}\n' for p in tool_paths)
    ).rstrip()

    template = """# Induced ptools eval — variant {variant} | rec | seed={seed}
# Variant {variant}: {variant_desc}
# NOTE: dataset.split is overridden on the CLI to switch valid/test.
llm:
  model: openai/QwQ-32B
  api_base: http://localhost:12350/v1
  api_key: EMPTY
  max_tokens: 4096

cachier:
  cache_dir: llm_cache
  enable_caching: false

evaluate:
  expt_name: rec_induced_{variant}_{tag}
  result_dir: results
  entry_point: answer_question
  record_details: true

dataset:
  split: valid
  shuffle_seed: {seed}

ptools:
  react_solve:
    method: react_pydantic
    tools:
{tools}
  answer_question:
    method: direct
    fn: ptools.ptools_common.react_rank_impl
"""
    return {
        'a': template.format(
            variant='a',
            variant_desc='search+lookup+search_memory+produce_ranking + induced',
            seed=seed, tag=tag, tools=tools_a,
        ),
        'b': template.format(
            variant='b',
            variant_desc='produce_ranking + induced ptools only',
            seed=seed, tag=tag, tools=tools_b,
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--report', default=None,
                   help='Path to induction report.json')
    p.add_argument('--induction-dir', default='induction_out',
                   help='parent of original/rec/report.json if --report omitted')
    p.add_argument('--suffix', default='correct',
                   help='suffix for output files, e.g. correct')
    args = p.parse_args()

    if args.report:
        report_path = pathlib.Path(args.report)
    else:
        report_path = (
            _BENCHMARK_DIR / args.induction_dir / 'original' / 'rec' / 'report.json'
        )
    if not report_path.exists():
        raise SystemExit(f'report not found: {report_path}')

    report = json.loads(report_path.read_text(encoding='utf-8'))
    source, tool_paths = generate_ptool_module(report, args.seed, args.suffix)

    induced_dir = _BENCHMARK_DIR / 'induced'
    induced_dir.mkdir(parents=True, exist_ok=True)
    # ensure package
    init = induced_dir / '__init__.py'
    if not init.exists():
        init.write_text('# Induced ptool packages\n')

    tag = f'seed{args.seed}'
    if args.suffix:
        tag = f'{tag}_{args.suffix}'
    py_path = induced_dir / f'induced_ptools_{tag}_rec.py'
    py_path.write_text(source, encoding='utf-8')
    print(f'wrote {py_path} ({len(tool_paths)} tools)')

    conf_dir = _BENCHMARK_DIR / 'conf'
    conf_dir.mkdir(parents=True, exist_ok=True)
    for variant, yaml_text in generate_configs(args.seed, tool_paths, args.suffix).items():
        cfg_path = conf_dir / f'rec_induced_{variant}_{tag}.yaml'
        cfg_path.write_text(yaml_text, encoding='utf-8')
        print(f'wrote {cfg_path}')


if __name__ == '__main__':
    main()
