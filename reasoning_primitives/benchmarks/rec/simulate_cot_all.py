"""Simulate ALL induced ptools in a SINGLE LLM call to produce CoT segments.

Unlike simulate_cot.py which calls each ptool stub separately (N ptools = N
LLM calls), this script packs all ptool stubs into one prompt so the LLM
generates the entire CoT in one shot. Each reasoning segment is bracketed
with [primitive_name] ... [/primitive_name] tags for downstream parsing.

Usage:
    python simulate_cot_all.py \
        --ptools induced/induced_ptools_seed42_correct_rec.py \
        --data   data/rec_train.json \
        --output out/think_contents_train.jsonl \
        --model openai/QwQ-32B \
        --api-base http://localhost:12350/v1
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template

_BENCHMARK_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCHMARK_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_BENCHMARK_DIR))

from secretagent import config
from secretagent.llm_util import llm

ALL_PTOOLS_TEMPLATE = Template("""\
You are a recommendation expert. Given the user context below, perform ALL
of the following reasoning steps and output your thoughts for each step.

For each reasoning step, output your thoughts inside a section marked with
the step name in square brackets, like:

[step_name]
Your reasoning here...
[/step_name]

Do NOT skip any step. Perform every step in order. Write substantive,
detailed reasoning — not just restating the input. Focus on qualitative
preference analysis, attribute matching, and natural-language justification.
Do NOT invent numerical scores, embeddings, or ranking algorithms.

Reasoning steps to perform:

$stubs_block

User context:
$context

Now perform ALL reasoning steps above in order. Output each step's thoughts
between [step_name] and [/step_name] tags.
""")


def _load_stub_sources(ptools_path: Path) -> list[tuple[str, str, str]]:
    """Parse @implement_via('simulate') stubs -> (public_name, stub_src, docstring)."""
    src = ptools_path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    stubs: list[tuple[str, str, str]] = []
    lines = src.splitlines()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.endswith('_impl'):
            continue
        start = node.lineno - 1
        while start > 0 and lines[start - 1].lstrip().startswith('@'):
            start -= 1
        end = node.end_lineno or (start + 1)
        stub_src = '\n'.join(lines[start:end])
        public = node.name
        if public.startswith('_') and public.endswith('_impl'):
            public = public[1:-5]
        docstring = ast.get_docstring(node) or ''
        stubs.append((public, stub_src, docstring))
    return stubs


def _parse_segments(text: str, stub_names: list[str]) -> str:
    """Extract tagged segments from LLM output, return think_content string."""
    segments = {}
    for name in stub_names:
        pattern = re.compile(
            rf'\[{re.escape(name)}\](.*?)\[/{re.escape(name)}\]',
            re.DOTALL,
        )
        m = pattern.search(text)
        if m:
            segments[name] = m.group(1).strip()

    if not segments:
        for name in stub_names:
            pattern = re.compile(
                rf'\[{re.escape(name)}\]\s*(.*?)(?=\n\[|\Z)',
                re.DOTALL,
            )
            m = pattern.search(text)
            if m:
                segments[name] = m.group(1).strip()

    parts = []
    for name in stub_names:
        if name in segments:
            parts.append(f'[{name}] {segments[name]}')

    return '\n\n'.join(parts) if parts else text.strip()


def _simulate_all(
    stubs: list[tuple[str, str, str]],
    context: str,
    model: str,
) -> str:
    ctx = context
    max_ctx = int(config.get('simulate.max_context_chars', 6000) or 6000)
    if len(ctx) > max_ctx:
        ctx = ctx[:max_ctx] + '\n...[truncated]...'

    stubs_block_parts = []
    for name, stub_src, docstring in stubs:
        block = f"Step [{name}]:\n"
        if docstring:
            block += docstring + "\n"
        else:
            block += "```python\n" + stub_src + "\n```\n"
        stubs_block_parts.append(block)
    stubs_block = '\n'.join(stubs_block_parts)

    prompt = ALL_PTOOLS_TEMPLATE.substitute(
        stubs_block=stubs_block,
        context=ctx,
    )

    text, _ = llm(prompt, model)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ptools', required=True, help='induced_ptools_*.py')
    p.add_argument('--data', required=True, help='rec_*.json from prepare_data')
    p.add_argument('--output', required=True, help='think_contents.jsonl')
    p.add_argument('--model', default='openai/QwQ-32B')
    p.add_argument('--api-base', default='http://localhost:12350/v1')
    p.add_argument('--api-bases', nargs='*', default=None)
    p.add_argument('--max-users', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    llm_cfg: dict = {
        'model': args.model,
        'api_key': 'EMPTY',
        'max_tokens': 4096,
        'timeout': 300,
    }
    if args.api_bases:
        llm_cfg['api_bases'] = list(args.api_bases)
    else:
        llm_cfg['api_base'] = args.api_base

    config.configure(cfg={
        'llm': llm_cfg,
        'cachier': {'enable_caching': False},
        'simulate': {'max_context_chars': 6000},
    })

    stubs = _load_stub_sources(Path(args.ptools))
    if not stubs:
        raise SystemExit(f'No @implement_via simulate stubs found in {args.ptools}')
    stub_names = [s[0] for s in stubs]
    print(f'Loaded {len(stubs)} ptool stubs: {stub_names}')

    data = json.loads(Path(args.data).read_text(encoding='utf-8'))
    examples = data['examples']
    if args.max_users:
        examples = examples[:args.max_users]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def process(ex: dict) -> dict:
        raw_text = _simulate_all(stubs, ex['context'], args.model)
        think_content = _parse_segments(raw_text, stub_names)
        return {
            'user_id': ex.get('user_id'),
            'answer_item': ex.get('answer_item'),
            'think_content': think_content,
            'n_ptools': len(stubs),
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, ex): ex for ex in examples}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            results.append(row)
            if i % 10 == 0 or i == len(examples):
                print(f'  {i}/{len(examples)} done', flush=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Wrote {len(results)} rows -> {out_path}')


if __name__ == '__main__':
    main()
