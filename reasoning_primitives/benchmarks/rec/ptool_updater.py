"""PtoolUpdater: dynamically update the primitive tool library from failed trajectories.

Inspired by SEED's SkillUpdater (agent_system/memory/skill_updater.py), adapted for
the rec-domain ptool induction pipeline.

Unlike SEED's SkillUpdater which uses Azure OpenAI o3, this version uses
the same secretagent LLM infrastructure as induction.py for consistency.

Usage in OPD training loop:
    from ptool_updater import PtoolUpdater

    updater = PtoolUpdater(model='openai/QwQ-32B', api_base='http://localhost:12350/v1')
    new_ptools = updater.analyze_failures(
        failed_trajectories=failed_items,
        current_ptool_names=['extract_user_profile', 'infer_user_intent', ...],
        task_key='rec',
    )
    # new_ptools is a list of avoidance ptool specs
    # Merge into existing ptool library and regenerate configs
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import Optional

_BENCHMARK_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCHMARK_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_BENCHMARK_DIR))

from secretagent import config
from secretagent.llm_util import llm

from induction import (
    TASK_DESCRIPTIONS,
    _extract_json,
    _dump_debug,
    synthesize_avoidance_ptools,
    synthesize_success_ptools,
)
from generate_induced_configs import generate_ptool_module, generate_configs


class PtoolUpdater:
    """Dynamically analyze failed trajectories and generate new avoidance ptools.

    Mirrors SEED's SkillUpdater.analyze_failures() flow:
    1. Collect failed trajectories from OPD training round
    2. Send to LLM for analysis (with existing ptool names for dedup)
    3. Parse response into avoidance ptool specs
    4. Merge into existing ptool library
    5. Regenerate Python module and YAML configs
    """

    def __init__(
        self,
        model: str = 'openai/QwQ-32B',
        api_base: Optional[str] = None,
        api_bases: Optional[list[str]] = None,
        max_new_ptools_per_update: int = 1,
        max_total_ptools: int = 7,
        max_completion_tokens: int = 4096,
    ):
        self.model = model
        self.api_base = api_base
        self.api_bases = api_bases
        self.max_new_ptools_per_update = max_new_ptools_per_update
        self.max_total_ptools = max_total_ptools
        self.max_completion_tokens = max_completion_tokens
        self.update_history: list[dict] = []
        self._configured = False

    def _ensure_config(self):
        if self._configured:
            return
        llm_cfg: dict = {
            'model': self.model,
            'max_tokens': self.max_completion_tokens,
            'api_key': 'EMPTY',
        }
        if self.api_bases:
            llm_cfg['api_bases'] = list(self.api_bases)
        elif self.api_base:
            llm_cfg['api_base'] = self.api_base

        config.configure(cfg={
            'llm': llm_cfg,
            'cachier': {
                'cache_dir': str(_BENCHMARK_DIR / 'llm_cache'),
                'enable_caching': True,
            },
        })
        self._configured = True

    def analyze_successes(
        self,
        correct_trajectories: list[dict],
        current_ptool_names: list[str],
        task_key: str = 'rec',
    ) -> list[dict]:
        """Analyze successful trajectories and generate new positive ptools.

        Mirrors SEED's SEEDEpisodeAnalyzer with episode_success=True:
        "extract the successful trajectory into workflow"

        Args:
            correct_trajectories: List of thought dicts with
                {case, pos, text, correct=True}
            current_ptool_names: List of func_name strings already in the bank
            task_key: Task key for TASK_DESCRIPTIONS

        Returns:
            List of new success ptool spec dicts
        """
        self._ensure_config()

        if not correct_trajectories:
            return []

        print(f'[PtoolUpdater] Analyzing {len(correct_trajectories)} successful thoughts')

        new_ptools = synthesize_success_ptools(
            correct_items=correct_trajectories,
            existing_ptool_names=current_ptool_names,
            task_key=task_key,
            model=self.model,
            max_new=self.max_new_ptools_per_update,
        )

        next_idx = self._next_success_dyn_index(current_ptool_names)
        reassigned = self._reassign_success_dyn_ids(new_ptools, next_idx)

        self.update_history.append({
            'num_successes_analyzed': len(correct_trajectories),
            'num_ptools_generated': len(reassigned),
            'ptool_ids': [p.get('func_name', '') for p in reassigned],
            'type': 'success',
        })

        return reassigned[:self.max_new_ptools_per_update]

    def analyze_failures(
        self,
        failed_trajectories: list[dict],
        current_ptool_names: list[str],
        task_key: str = 'rec',
    ) -> list[dict]:
        """Analyze failed trajectories and generate new avoidance ptools.

        Args:
            failed_trajectories: List of thought dicts with
                {case, pos, text, correct=False}
            current_ptool_names: List of func_name strings already in the bank,
                used for deduplication (mirrors SEED's existing_titles)
            task_key: Task key for TASK_DESCRIPTIONS

        Returns:
            List of new avoidance ptool spec dicts
        """
        self._ensure_config()

        if not failed_trajectories:
            return []

        print(f'[PtoolUpdater] Analyzing {len(failed_trajectories)} failed thoughts')

        new_ptools = synthesize_avoidance_ptools(
            failed_items=failed_trajectories,
            existing_ptool_names=current_ptool_names,
            task_key=task_key,
            model=self.model,
            max_avoidance=self.max_new_ptools_per_update,
        )

        # Reassign dyn_ IDs to avoid collisions (mirrors SEED's _reassign_dyn_ids)
        next_idx = self._next_dyn_index(current_ptool_names)
        reassigned = self._reassign_dyn_ids(new_ptools, next_idx)

        self.update_history.append({
            'num_failures_analyzed': len(failed_trajectories),
            'num_ptools_generated': len(reassigned),
            'ptool_ids': [p.get('func_name', '') for p in reassigned],
            'type': 'avoidance',
        })

        return reassigned[:self.max_new_ptools_per_update]

    def update_library(
        self,
        report_path: str,
        new_ptools: list[dict],
        ptool_type: str = 'avoidance',
        seed: int = 42,
        suffix: str = 'dynamic',
    ) -> dict:
        """Merge new ptools into existing report and regenerate configs.

        Mirrors SEED's SkillsOnlyMemory.add_skills() + save_skills() flow.

        Args:
            report_path: Path to existing report.json from induction
            new_ptools: List of ptool specs to add
            ptool_type: 'avoidance' or 'success'
            seed: Random seed for config generation
            suffix: Suffix for output files

        Returns:
            Dict with paths to generated files
        """
        report_path = pathlib.Path(report_path)
        if not report_path.exists():
            raise FileNotFoundError(f'Report not found: {report_path}')

        report = json.loads(report_path.read_text(encoding='utf-8'))

        existing_names = {
            p.get('func_name', '') for p in report.get('ptools', [])
        } | {
            p.get('func_name', '') for p in report.get('avoidance_ptools', [])
        } | {
            p.get('func_name', '') for p in report.get('success_ptools', [])
        }

        list_key = 'success_ptools' if ptool_type == 'success' else 'avoidance_ptools'
        is_flag = 'is_success' if ptool_type == 'success' else 'is_avoidance'
        count_source = 'n_correct' if ptool_type == 'success' else 'n_failed'
        source_prefix = 'success' if ptool_type == 'success' else 'avoidance'

        added = []
        skipped = 0
        round_num = len(self.update_history) + 1
        for p in new_ptools:
            name = p.get('func_name', '')
            if name in existing_names:
                print(f'[PtoolUpdater] Skipping duplicate: {name}')
                skipped += 1
                continue
            p['source_category'] = f"{source_prefix}:{name}"
            p['count'] = report.get(count_source, 0)
            p[is_flag] = True
            p['round_added'] = round_num
            report.setdefault(list_key, []).append(p)
            existing_names.add(name)
            added.append(p)
            print(f'[PtoolUpdater] Added {ptool_type} ptool: {name}')

        if not added:
            print('[PtoolUpdater] No new ptools to add')
            return {'added': 0, 'skipped': skipped}

        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        print(f'[PtoolUpdater] Updated {report_path}')

        evicted = self._evict_if_over_limit(report_path, report)

        source, tool_paths = generate_ptool_module(report, seed, suffix)

        induced_dir = _BENCHMARK_DIR / 'induced'
        induced_dir.mkdir(parents=True, exist_ok=True)
        init = induced_dir / '__init__.py'
        if not init.exists():
            init.write_text('# Induced ptool packages\n')

        tag = f'seed{seed}'
        if suffix:
            tag = f'{tag}_{suffix}'
        py_path = induced_dir / f'induced_ptools_{tag}_rec.py'
        py_path.write_text(source, encoding='utf-8')

        conf_dir = _BENCHMARK_DIR / 'conf'
        conf_dir.mkdir(parents=True, exist_ok=True)
        configs = generate_configs(seed, tool_paths, suffix)
        config_paths = {}
        for variant, yaml_text in configs.items():
            cfg_path = conf_dir / f'rec_induced_{variant}_{tag}.yaml'
            cfg_path.write_text(yaml_text, encoding='utf-8')
            config_paths[variant] = str(cfg_path)

        return {
            'added': len(added),
            'skipped': skipped,
            'evicted': evicted,
            'report_path': str(report_path),
            'module_path': str(py_path),
            'config_paths': config_paths,
        }

    def get_update_summary(self) -> dict:
        if not self.update_history:
            return {'total_updates': 0, 'total_ptools_generated': 0}
        return {
            'total_updates': len(self.update_history),
            'total_ptools_generated': sum(
                h['num_ptools_generated'] for h in self.update_history
            ),
            'all_ptool_ids': [
                pid for h in self.update_history for pid in h['ptool_ids']
            ],
        }

    @staticmethod
    def _next_dyn_index(current_names: list[str]) -> int:
        max_idx = 0
        pattern = re.compile(r'^avoid_dyn_(\d+)')
        for name in current_names:
            m = pattern.match(name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    @staticmethod
    def _next_success_dyn_index(current_names: list[str]) -> int:
        max_idx = 0
        pattern = re.compile(r'^success_dyn_(\d+)')
        for name in current_names:
            m = pattern.match(name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    @staticmethod
    def _reassign_dyn_ids(ptools: list[dict], start_idx: int) -> list[dict]:
        reassigned = []
        for i, p in enumerate(ptools):
            updated = dict(p)
            raw_name = updated.get('func_name', f'avoid_unknown_{i}')
            if not raw_name.startswith('avoid_'):
                raw_name = f'avoid_{raw_name}'
            updated['func_name'] = f'avoid_dyn_{start_idx + i:03d}_{raw_name}'
            updated['display_name'] = updated.get('display_name', '')
            reassigned.append(updated)
        return reassigned

    @staticmethod
    def _reassign_success_dyn_ids(ptools: list[dict], start_idx: int) -> list[dict]:
        reassigned = []
        for i, p in enumerate(ptools):
            updated = dict(p)
            raw_name = updated.get('func_name', f'success_unknown_{i}')
            if not raw_name.startswith('success_'):
                raw_name = f'success_{raw_name}'
            updated['func_name'] = f'success_dyn_{start_idx + i:03d}_{raw_name}'
            updated['display_name'] = updated.get('display_name', '')
            reassigned.append(updated)
        return reassigned

    def _evict_if_over_limit(
        self,
        report_path: pathlib.Path,
        report: dict,
    ) -> list[str]:
        all_dynamic = (
            list(report.get('success_ptools', []))
            + list(report.get('avoidance_ptools', []))
        )
        if len(all_dynamic) <= self.max_total_ptools:
            return []

        scores = []
        for p in all_dynamic:
            scores.append((self._score_ptool(p), p))

        scores.sort(key=lambda x: x[0])

        n_evict = len(all_dynamic) - self.max_total_ptools
        to_evict_names = set()
        for _, p in scores[:n_evict]:
            to_evict_names.add(p.get('func_name', ''))

        evicted = list(to_evict_names)
        for key in ('success_ptools', 'avoidance_ptools'):
            original = report.get(key, [])
            report[key] = [
                p for p in original
                if p.get('func_name', '') not in to_evict_names
            ]

        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        for name in evicted:
            print(f'[PtoolUpdater] Evicted low-score ptool: {name}')
        return evicted

    @staticmethod
    def _score_ptool(p: dict) -> float:
        if p.get('is_success'):
            base = 1.0
        elif p.get('is_avoidance'):
            base = 0.5
        else:
            base = 0.0

        count = p.get('count', 0)
        count_score = min(count / 100.0, 1.0)

        recency = p.get('round_added', 0) * 0.1

        return base + count_score + recency
