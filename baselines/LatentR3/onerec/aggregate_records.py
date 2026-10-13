from __future__ import annotations
import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description='Merge R3 evaluation records by source row')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    records = []
    for input_path in args.inputs:
        with Path(input_path).open(encoding='utf-8') as handle:
            records.extend((json.loads(line) for line in handle if line.strip()))
    source_indices = [record['source_index'] for record in records]
    if len(source_indices) != len(set(source_indices)):
        raise ValueError('Duplicate source_index values found across record shards')
    records.sort(key=lambda record: record['source_index'])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + '\n')
    print(f'Merged {len(records)} records into {output}')
if __name__ == '__main__':
    main()
