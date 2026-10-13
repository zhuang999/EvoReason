#!/usr/bin/env python3
"""C-4: Evolve the primitive library P* based on OPD training results.

Reads yr_responses.parquet from generate_yr, splits trajectories into
success/failure by yo_sid vs groundtruth, runs PtoolUpdater to analyze
and generate new ptools, updates report.json, and optionally re-simulates
CoT with the updated ptool library.

Usage:
    python update_ptools.py \
        --yr_path checkpoints/latent_opd/epoch_1/yr_responses.parquet \
        --report_path reasoning_primitives/benchmarks/rec/induction_out/original/rec/report.json \
        --data_json reasoning_primitives/benchmarks/rec/data/rec_train.json \
        --train_parquet /path/to/training_RA_train.parquet \
        --output_dir checkpoints/latent_opd/epoch_1/ptool_update \
        --model openai/QwQ-32B \
        --api-bases http://localhost:12350/v1 http://localhost:12351/v1 \
        --epoch 1

    # Skip CoT re-simulation (e.g. when vLLM instances are down):
    python update_ptools.py ... --skip-cot
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import pandas as pd

_BENCHMARK_DIR = pathlib.Path(__file__).resolve().parent / "reasoning_primitives" / "benchmarks" / "rec"
_PIPELINE_DIR = pathlib.Path(__file__).resolve().parent

sys.path.insert(0, str(_BENCHMARK_DIR.parent.parent / "src"))
sys.path.insert(0, str(_BENCHMARK_DIR))

from ptool_updater import PtoolUpdater  # noqa: E402


SIMULATE_COT_ALL = _BENCHMARK_DIR / "simulate_cot_all.py"
CONVERT_PARQUET = _BENCHMARK_DIR / "convert_traces_to_cot_parquet.py"


def _df_to_items(df: pd.DataFrame, correct: bool) -> list[dict]:
    items = []
    for _, row in df.iterrows():
        cot = str(row.get("yr_cot", ""))
        if cot and cot != "nan" and len(cot) > 10:
            paragraphs = [p.strip() for p in cot.split("\n") if p.strip()]
        else:
            desc = str(row.get("description", ""))
            paragraphs = [p.strip() for p in desc.split("\n") if p.strip()]
        for pos, para in enumerate(paragraphs):
            items.append({
                "case": str(row.get("user_id", "unknown")),
                "pos": pos,
                "text": para,
                "correct": correct,
            })
    return items


def _get_existing_ptool_names(report_path: pathlib.Path) -> list[str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    names = []
    for key in ("ptools", "success_ptools", "avoidance_ptools"):
        for p in report.get(key, []):
            name = p.get("func_name", "")
            if name:
                names.append(name)
    return names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yr_path", required=True, help="yr_responses.parquet from generate_yr")
    p.add_argument("--report_path", required=True, help="Path to induction report.json")
    p.add_argument("--data_json", default=None,
                   help="rec_*.json for simulate_cot_all (required if not --skip-cot)")
    p.add_argument("--train_parquet", default=None,
                   help="Original training parquet for convert (required if not --skip-cot)")
    p.add_argument("--output_dir", required=True, help="Output directory for C-4 artifacts")
    p.add_argument("--model", default="openai/QwQ-32B", help="LiteLLM model id")
    p.add_argument("--api-base", default=None, help="Single vLLM API base URL")
    p.add_argument("--api-bases", nargs="*", default=None, help="Multiple vLLM API base URLs")
    p.add_argument("--epoch", type=int, default=0, help="Current OPD epoch number")
    p.add_argument("--skip-cot", action="store_true",
                   help="Skip CoT re-simulation step")
    p.add_argument("--max-new-ptools", type=int, default=1,
                   help="Max new ptools per type per update")
    p.add_argument("--max-total-ptools", type=int, default=7,
                   help="Max total dynamic ptools (triggers eviction)")
    args = p.parse_args()

    if not args.skip_cot:
        if not args.data_json:
            p.error("--data_json is required when not using --skip-cot")
        if not args.train_parquet:
            p.error("--train_parquet is required when not using --skip-cot")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = pathlib.Path(args.report_path)
    if not report_path.exists():
        print(f"ERROR: report.json not found at {report_path}")
        sys.exit(1)

    # 1. Read yr_responses.parquet
    yr_path = pathlib.Path(args.yr_path)
    if not yr_path.exists():
        print(f"ERROR: yr_responses.parquet not found at {yr_path}")
        sys.exit(1)

    df = pd.read_parquet(yr_path)
    print(f"Loaded {len(df)} rows from {yr_path}")

    # 2. Split success/failure by yo_sid == groundtruth
    success_mask = df["yo_sid"] == df["groundtruth"]
    success_df = df[success_mask]
    failure_df = df[~success_mask]
    n_success = len(success_df)
    n_failure = len(failure_df)
    print(f"  success: {n_success}, failure: {n_failure}")

    # 3. Convert to PtoolUpdater format
    success_items = _df_to_items(success_df, correct=True)
    failure_items = _df_to_items(failure_df, correct=False)
    print(f"  success thoughts: {len(success_items)}, failure thoughts: {len(failure_items)}")

    # 4. Backup report.json
    backup_path = report_path.parent / f"report.json.bak.epoch_{args.epoch}"
    shutil.copy2(report_path, backup_path)
    print(f"Backed up report.json -> {backup_path}")

    # 5. Run PtoolUpdater
    updater = PtoolUpdater(
        model=args.model,
        api_base=args.api_base,
        api_bases=args.api_bases,
        max_new_ptools_per_update=args.max_new_ptools,
        max_total_ptools=args.max_total_ptools,
    )

    existing_names = _get_existing_ptool_names(report_path)
    print(f"Existing ptool names: {existing_names}")

    new_avoidance = []
    new_success = []

    if failure_items:
        print("\n--- Analyzing failures ---")
        new_avoidance = updater.analyze_failures(
            failed_trajectories=failure_items,
            current_ptool_names=existing_names,
            task_key="rec",
        )
        print(f"  Generated {len(new_avoidance)} avoidance ptools")
    else:
        print("No failure trajectories to analyze")

    if success_items:
        print("\n--- Analyzing successes ---")
        existing_after_avoidance = existing_names + [
            p.get("func_name", "") for p in new_avoidance
        ]
        new_success = updater.analyze_successes(
            correct_trajectories=success_items,
            current_ptool_names=existing_after_avoidance,
            task_key="rec",
        )
        print(f"  Generated {len(new_success)} success ptools")
    else:
        print("No success trajectories to analyze")

    # 6. Update report.json + regenerate ptool module
    last_result = {}

    if new_avoidance:
        print("\n--- Updating library with avoidance ptools ---")
        last_result = updater.update_library(
            report_path=str(report_path),
            new_ptools=new_avoidance,
            ptool_type="avoidance",
            seed=42,
            suffix=f"epoch{args.epoch}",
        )
        print(f"  Added: {last_result.get('added', 0)}, Skipped: {last_result.get('skipped', 0)}")
        if last_result.get("evicted"):
            print(f"  Evicted: {last_result['evicted']}")

    if new_success:
        print("\n--- Updating library with success ptools ---")
        last_result = updater.update_library(
            report_path=str(report_path),
            new_ptools=new_success,
            ptool_type="success",
            seed=42,
            suffix=f"epoch{args.epoch}",
        )
        print(f"  Added: {last_result.get('added', 0)}, Skipped: {last_result.get('skipped', 0)}")
        if last_result.get("evicted"):
            print(f"  Evicted: {last_result['evicted']}")

    if not new_avoidance and not new_success:
        print("\nNo new ptools generated, skipping library update")

    # 7. Optionally re-simulate CoT
    if not args.skip_cot and last_result.get("module_path"):
        ptool_module = last_result["module_path"]
        think_output = output_dir / "think_contents_train.jsonl"
        cot_parquet_output = output_dir / "training_RA_cot_train.parquet"

        print(f"\n--- Re-simulating CoT with updated ptools ---")
        print(f"  ptool module: {ptool_module}")
        print(f"  data_json: {args.data_json}")

        simulate_cmd = [
            sys.executable, str(SIMULATE_COT_ALL),
            "--ptools", ptool_module,
            "--data", args.data_json,
            "--output", str(think_output),
            "--model", args.model,
        ]
        if args.api_bases:
            simulate_cmd.extend(["--api-bases"] + list(args.api_bases))
        elif args.api_base:
            simulate_cmd.extend(["--api-base", args.api_base])

        print(f"  Running: {' '.join(simulate_cmd)}")
        result = subprocess.run(simulate_cmd, capture_output=False)
        if result.returncode != 0:
            print(f"WARNING: simulate_cot_all.py failed (exit {result.returncode})")
            print("Skipping CoT parquet conversion")
        else:
            convert_cmd = [
                sys.executable, str(CONVERT_PARQUET),
                "--think-contents", str(think_output),
                "--parquet", args.train_parquet,
                "--output", str(cot_parquet_output),
            ]
            print(f"  Running: {' '.join(convert_cmd)}")
            result = subprocess.run(convert_cmd, capture_output=False)
            if result.returncode != 0:
                print(f"WARNING: convert_traces_to_cot_parquet.py failed (exit {result.returncode})")
            else:
                print(f"  CoT training parquet: {cot_parquet_output}")

    # 8. Summary
    update_summary = updater.get_update_summary()
    print(f"\n=== C-4 Summary (epoch {args.epoch}) ===")
    print(f"  Success: {n_success}, Failure: {n_failure}")
    print(f"  New avoidance ptools: {len(new_avoidance)}")
    print(f"  New success ptools: {len(new_success)}")
    print(f"  Total updates: {update_summary.get('total_updates', 0)}")
    print(f"  Total ptools generated: {update_summary.get('total_ptools_generated', 0)}")

    summary = {
        "epoch": args.epoch,
        "n_success": n_success,
        "n_failure": n_failure,
        "n_new_avoidance": len(new_avoidance),
        "n_new_success": len(new_success),
        "update_summary": update_summary,
        "cot_resimulated": not args.skip_cot and last_result.get("module_path") is not None,
    }
    if last_result.get("module_path"):
        summary["module_path"] = last_result["module_path"]
    if not args.skip_cot and (output_dir / "training_RA_cot_train.parquet").exists():
        summary["cot_parquet_path"] = str(output_dir / "training_RA_cot_train.parquet")

    summary_path = output_dir / "update_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
