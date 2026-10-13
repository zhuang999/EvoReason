"""Recommendation benchmark experiment — MuSR expt.py adapted for ranking.

Example:
    python expt.py run --config-file conf/rec_react.yaml dataset.n=10
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import typer

_BENCHMARK_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCHMARK_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_BENCHMARK_DIR))

from secretagent import config
from secretagent.core import implement_via_config
from secretagent.dataset import Dataset, Case
from secretagent.evaluate import Evaluator


class RecEvaluator(Evaluator):
    """Hit@1 correctness: predicted ranking[0] == answer_item."""

    def compare_predictions(self, predicted_output, expected_output) -> dict[str, Any]:
        pred = predicted_output
        if isinstance(pred, str):
            pred = [t for t in pred.replace(',', ' ').split() if t]
        if not isinstance(pred, (list, tuple)) or not pred:
            return dict(correct=False, hit_at_1=False)
        hit = str(pred[0]) == str(expected_output)
        return dict(correct=hit, hit_at_1=hit)


def load_dataset(split: str) -> Dataset:
    """Load data/rec_{split}.json produced by prepare_data.py."""
    name = split if split.startswith('rec_') else f'rec_{split}'
    json_file = _BENCHMARK_DIR / 'data' / f'{name}.json'
    if not json_file.exists():
        json_file = _BENCHMARK_DIR / 'data' / f'{split}.json'
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)

    cases = []
    for i, ex in enumerate(data['examples']):
        uid = ex.get('user_id', f'{i:04d}')
        cases.append(Case(
            name=f'u{uid}',
            input_args=(ex['context'], ex['question'], ex['candidates']),
            expected_output=ex['answer_item'],
        ))
    return Dataset(name='rec', split=split, cases=cases)


app = typer.Typer()


@app.callback()
def callback():
    """Recommendation primitives benchmark."""


@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": False})
def run(ctx: typer.Context,
        config_file: str = typer.Option(..., help="Config YAML file")):
    """Run recommendation evaluation. Extra args are config overrides."""
    cfg_path = Path(config_file)
    if not cfg_path.is_absolute():
        cfg_path = _BENCHMARK_DIR / cfg_path
    config.configure(yaml_file=str(cfg_path), dotlist=ctx.args)
    config.set_root(_BENCHMARK_DIR)

    # Import common first so register_factory('react_pydantic', ...) runs
    importlib.import_module('ptools.ptools_common')
    # Single-module bind (MuSR pattern): react_solve is re-exported here
    ptools = importlib.import_module('ptools.ptools_rec')
    implement_via_config(ptools, config.require('ptools'))

    split = config.require('dataset.split')
    dataset = load_dataset(split).configure(
        shuffle_seed=config.get('dataset.shuffle_seed'),
        n=config.get('dataset.n'),
    )
    print('dataset:', dataset.summary())

    entry_point = config.require('evaluate.entry_point')
    interface = getattr(ptools, entry_point)

    evaluator = RecEvaluator()
    csv_path = evaluator.evaluate(dataset, interface)
    df = pd.read_csv(csv_path)
    print(
        f'\nHit@1: {df["correct"].mean():.1%} '
        f'({df["correct"].sum()}/{len(df)})'
    )


if __name__ == '__main__':
    app()
