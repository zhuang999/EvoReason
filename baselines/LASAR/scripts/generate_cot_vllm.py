from __future__ import annotations
import argparse
import itertools
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
STEP_PATTERN = re.compile('(?im)^\\s*(?:step\\s+\\d+|\\d+)[.:)\\-]\\s*')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate LASAR teacher CoT with QwQ-32B vLLM')
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--endpoints', default='http://localhost:12350/v1', help='Comma-separated OpenAI-compatible vLLM endpoints')
    parser.add_argument('--model', default='QwQ-32B')
    parser.add_argument('--max-workers', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--limit', type=int)
    return parser.parse_args()

def normalise_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip('/')
    if not endpoint.endswith('/v1'):
        endpoint += '/v1'
    return endpoint

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    seen = set()
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = int(row['row_id'])
            if row_id in seen:
                raise ValueError(f'{path}:{line_number}: duplicate row_id {row_id}')
            seen.add(row_id)
            rows.append(row)
    return rows

def extract_structured_cot(text: str) -> str:
    text = (text or '').strip()
    if '</think>' in text.lower():
        text = re.split('</think>', text, flags=re.IGNORECASE)[-1].strip()
    text = re.sub('^```(?:text|markdown)?\\s*|\\s*```$', '', text, flags=re.IGNORECASE).strip()
    matches = list(STEP_PATTERN.finditer(text))
    if len(matches) < 3:
        raise ValueError(f'teacher returned {len(matches)} structured steps; expected at least 3')
    return text[matches[0].start():].strip()

class EndpointPool:

    def __init__(self, endpoints: list[str]) -> None:
        self.endpoints = endpoints
        self._cycle = itertools.cycle(endpoints)
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            return next(self._cycle)

def request_one(row: dict, args: argparse.Namespace, pool: EndpointPool) -> dict:
    last_error = None
    for attempt in range(args.retries):
        endpoint = pool.next()
        try:
            response = requests.post(f'{endpoint}/chat/completions', headers={'Authorization': 'Bearer EMPTY', 'Content-Type': 'application/json'}, json={'model': args.model, 'messages': [{'role': 'user', 'content': row['prompt']}], 'temperature': args.temperature, 'top_p': args.top_p, 'max_tokens': args.max_tokens}, timeout=args.timeout)
            response.raise_for_status()
            payload = response.json()
            cot = extract_structured_cot(payload['choices'][0]['message'].get('content', ''))
            return {'row_id': int(row['row_id']), 'data_sha256': row['data_sha256'], 'cot': cot, 'teacher_model': args.model, 'teacher_endpoint': endpoint, 'usage': payload.get('usage', {})}
        except Exception as error:
            last_error = error
            if attempt + 1 < args.retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"row_id={row['row_id']} failed after {args.retries} attempts: {last_error}")

def write_sorted(path: Path, records: dict[int, dict]) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row_id in sorted(records):
            handle.write(json.dumps(records[row_id], ensure_ascii=True) + '\n')
    temporary.replace(path)

def main() -> None:
    args = parse_args()
    endpoints = [normalise_endpoint(item) for item in args.endpoints.split(',') if item.strip()]
    if not endpoints:
        raise ValueError('At least one vLLM endpoint is required')
    prompt_rows = load_jsonl(Path(args.prompts))
    if args.limit is not None:
        prompt_rows = prompt_rows[:args.limit]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if output_path.exists():
        for row in load_jsonl(output_path):
            completed[int(row['row_id'])] = row
    prompts_by_id = {int(row['row_id']): row for row in prompt_rows}
    for row_id, record in completed.items():
        prompt = prompts_by_id.get(row_id)
        if prompt is not None and record.get('data_sha256') != prompt.get('data_sha256'):
            raise ValueError(f'Existing response {row_id} has a different data fingerprint')
    pending = [row for row in prompt_rows if int(row['row_id']) not in completed]
    print(f'teacher={args.model} endpoints={endpoints} total={len(prompt_rows)} completed={len(completed)} pending={len(pending)}')
    if not pending:
        write_sorted(output_path, completed)
        return
    pool = EndpointPool(endpoints)
    failures = []
    write_lock = threading.Lock()
    with output_path.open('a', encoding='utf-8') as append_handle:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(request_one, row, args, pool): row for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    record = future.result()
                    completed[int(record['row_id'])] = record
                    with write_lock:
                        append_handle.write(json.dumps(record, ensure_ascii=True) + '\n')
                        append_handle.flush()
                except Exception as error:
                    failures.append((int(row['row_id']), str(error)))
                if index % 100 == 0 or index == len(pending):
                    print(f'finished={index}/{len(pending)} failures={len(failures)}')
    write_sorted(output_path, completed)
    if failures:
        preview = '; '.join((f'{row_id}: {error}' for row_id, error in failures[:10]))
        raise RuntimeError(f'{len(failures)} teacher requests failed; rerun to resume. {preview}')
if __name__ == '__main__':
    main()
