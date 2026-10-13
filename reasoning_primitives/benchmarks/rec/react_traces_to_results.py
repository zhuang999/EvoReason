"""Convert react_vllm.py traces to results.jsonl format for induction.py.

react_vllm.py outputs react_traces_full.jsonl with format:
    {user_id, steps: [...], recommendation, hit_at_1, answer_item, ...}

induction.py expects results.jsonl with format:
    {case_name, correct, rollout: [{step_info: [{thought: ...}, ...]}]}

This script maps the former to the latter so the full induction pipeline
(induction → generate_induced_configs → simulate_cot_all) works unchanged.

Usage:
    python react_traces_to_results.py \
        --traces results/react_traces_full.jsonl \
        --pretrain /pub/zhuang/distill/reasonrec/R3/pipeline/onerec_yelp_baseline/data/Beauty.pretrain.json \
        --output results/results_full.jsonl \
        --only-correct
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SID_PATTERN = re.compile(
    r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
)


def extract_sid(text: str) -> str | None:
    m = SID_PATTERN.search(text)
    return m.group(0) if m else None


def load_sid_map(pretrain_path: str | None) -> dict[str, dict]:
    if not pretrain_path:
        return {}
    with open(pretrain_path, encoding='utf-8') as f:
        data = json.load(f)
    sid_map = {}
    for item_id, info in data.items():
        sid = info.get('sid')
        if sid:
            sid_map[sid] = {
                'title': info.get('title', ''),
                'categories': info.get('categories', ''),
            }
    return sid_map


def _category_match(gt_categories: str, rec_text: str) -> bool:
    gt_parts = [p.strip() for p in gt_categories.split('>') if p.strip()]
    if len(gt_parts) < 2:
        return False
    prefix = ' > '.join(gt_parts[:2]).lower()
    return prefix in rec_text.lower()


def check_correctness(answer_item: str, recommendation: dict,
                      sid_map: dict[str, dict]) -> bool:
    gt_sid = extract_sid(answer_item)
    if not gt_sid:
        return False
    gt_info = sid_map.get(gt_sid, {})
    gt_title = gt_info.get('title', '').lower()
    gt_categories = gt_info.get('categories', '')

    rec_text = ' '.join(
        recommendation.get(f, '') for f in ['reasoning', 'recommendation']
    ).lower()

    if gt_title and gt_title[:20] in rec_text:
        return True

    if gt_categories and _category_match(gt_categories, rec_text):
        return True

    return False


def convert_traces_to_results(
    traces_path: str,
    output_path: str,
    sid_map: dict[str, str],
    only_correct: bool = False,
) -> dict:
    total = 0
    correct_count = 0
    skipped = 0

    with open(traces_path, encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            user_id = str(trace.get('user_id', ''))
            steps = trace.get('steps', [])
            recommendation = trace.get('recommendation', {})
            answer_item = trace.get('answer_item', '')

            step_info = []
            for step in steps:
                if 'reasoning' in step:
                    text = step['reasoning'].strip()
                    if text:
                        step_info.append({'thought': text})
                elif 'thought' in step:
                    text = step['thought'].strip()
                    if text:
                        step_info.append({'thought': text})
                elif 'tool_call' in step:
                    tc = {
                        'tool_call': step['tool_call'],
                        'args': step.get('args', {}),
                    }
                    if 'tool_return' in step:
                        tc['tool_return'] = step['tool_return']
                    step_info.append(tc)
                elif 'error' in step:
                    step_info.append({'thought': f'[Error: {step["error"]}]'})

            if recommendation:
                rec_text_parts = []
                for field in ['reasoning', 'recommendation']:
                    val = recommendation.get(field, '')
                    if val:
                        rec_text_parts.append(f'{field}: {val}')
                if rec_text_parts:
                    step_info.append({
                        'thought': 'Final recommendation reasoning:\n' +
                                   '\n'.join(rec_text_parts)
                    })

            is_correct = check_correctness(answer_item, recommendation, sid_map)
            if is_correct:
                correct_count += 1

            if only_correct and not is_correct:
                skipped += 1
                continue

            result_row = {
                'case_name': f'user_{user_id}',
                'correct': is_correct,
                'rollout': [{'step_info': step_info}],
            }
            fout.write(json.dumps(result_row, ensure_ascii=False) + '\n')

    return {
        'total': total,
        'correct': correct_count,
        'skipped': skipped,
        'output_rows': total - skipped,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces', required=True,
                   help='react_vllm traces JSONL')
    p.add_argument('--pretrain', default=None,
                   help='Beauty.pretrain.json for SID->title mapping')
    p.add_argument('--output', required=True,
                   help='Output results.jsonl for induction.py')
    p.add_argument('--only-correct', action='store_true',
                   help='Only include correct traces')
    args = p.parse_args()

    sid_map = load_sid_map(args.pretrain)
    if sid_map:
        print(f'Loaded {len(sid_map)} SID->title mappings')

    stats = convert_traces_to_results(
        args.traces, args.output, sid_map,
        only_correct=args.only_correct,
    )
    print(f'\n=== Conversion Summary ===')
    print(f'Total traces: {stats["total"]}')
    print(f'Correct: {stats["correct"]}')
    print(f'Skipped: {stats["skipped"]}')
    print(f'Output rows: {stats["output_rows"]}')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
