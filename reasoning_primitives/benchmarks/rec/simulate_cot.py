"""Simulate induced ptools to produce CoT segments for OneRec-style training.

Faithful to reasoning-primitives SimulateFactory + simulate.txt:
  stub docstring + args → LLM free-form thoughts before <answer>.

Usage:
    python simulate_cot.py \\
        --report induction_out/original/rec/report.json \\
        --ptools induced/induced_ptools_seed42_correct_rec.py \\
        --data   data/rec_valid.json \\
        --output out/think_contents.jsonl \\
        --api-base http://localhost:12350/v1
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
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

SIMULATE_TEMPLATE = Template(
    (_PROJECT_ROOT / 'src' / 'secretagent' / 'implement' / 'prompt_templates' / 'simulate.txt')
    .read_text(encoding='utf-8')
)

_ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)


def _load_stub_sources(ptools_path: Path) -> list[tuple[str, str]]:
    """Parse @implement_via('simulate') stubs → (func_name, stub_src)."""
    src = ptools_path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    stubs: list[tuple[str, str]] = []
    lines = src.splitlines()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        # Only take _*_impl stubs (have context, focus)
        if not node.name.endswith('_impl'):
            continue
        # Reconstruct source from lineno
        start = node.lineno - 1
        # include decorator line(s)
        while start > 0 and lines[start - 1].lstrip().startswith('@'):
            start -= 1
        end = node.end_lineno or (start + 1)
        stub_src = '\n'.join(lines[start:end])
        # public name = strip leading _ and trailing _impl
        public = node.name
        if public.startswith('_') and public.endswith('_impl'):
            public = public[1:-5]
        stubs.append((public, stub_src))
    return stubs


def _thoughts_before_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    if not m:
        return text.strip()
    return text[:m.start()].strip()


def _simulate_one(stub_name: str, stub_src: str, context: str, focus: str,
                  model: str) -> str:
    # Truncate very long context to stay in window
    ctx = context
    max_ctx = int(config.get('simulate.max_context_chars', 6000) or 6000)
    if len(ctx) > max_ctx:
        ctx = ctx[:max_ctx] + '\n...[truncated]...'
    input_args = f'context={ctx!r}\nfocus={focus!r}'
    prompt = SIMULATE_TEMPLATE.substitute(
        stub_src=stub_src,
        examples='',
        input_args=input_args,
        thoughts='',
    )
    text, _ = llm(prompt, model)
    return _thoughts_before_answer(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ptools', required=True, help='induced_ptools_*.py')
    p.add_argument('--data', required=True, help='rec_*.json from prepare_data')
    p.add_argument('--output', required=True, help='think_contents.jsonl')
    p.add_argument('--model', default='openai/QwQ-32B')
    p.add_argument('--api-base', default='http://localhost:12350/v1')
    p.add_argument('--api-bases', nargs='*', default=None)
    p.add_argument('--focus', default='overall preference match',
                   help='Default focus passed to every ptool')
    p.add_argument('--max-users', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    llm_cfg: dict = {
        'model': args.model,
        'api_key': 'EMPTY',
        'max_tokens': 2048,
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
    print(f'Loaded {len(stubs)} ptool stubs')

    data = json.loads(Path(args.data).read_text(encoding='utf-8'))
    examples = data['examples']
    if args.max_users:
        examples = examples[:args.max_users]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def process(ex: dict) -> dict:
        segs = []
        for name, stub_src in stubs:
            thought = _simulate_one(
                name, stub_src, ex['context'], args.focus, args.model
            )
            if thought:
                segs.append(f'[{name}] {thought}')
        return {
            'user_id': ex.get('user_id'),
            'answer_item': ex.get('answer_item'),
            'think_content': '\n\n'.join(segs),
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
    print(f'Wrote {len(results)} rows → {out_path}')


if __name__ == '__main__':
    main()
