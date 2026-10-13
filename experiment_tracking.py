#!/usr/bin/env python3
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_COLORS = ["#2C6AA0", "#8A8F99", "#D17C00", "#4B9E6F", "#8D5DA7"]


def _to_builtin(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _normalize_row(row: Dict) -> Dict:
    out = {}
    for key, value in row.items():
        out[key] = _to_builtin(value)
    out.setdefault("timestamp", datetime.utcnow().isoformat(timespec="seconds"))
    return out


def ensure_metrics_dir(output_dir: str) -> Path:
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir


def append_jsonl(path: Path, row: Dict):
    row = _normalize_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: Dict):
    row = _normalize_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_table(path_or_dir: str, default_name: Optional[str] = None) -> pd.DataFrame:
    path = Path(path_or_dir)
    if path.is_dir():
        candidates = []
        if default_name:
            candidates.extend([
                path / default_name,
                path / "metrics" / default_name,
            ])
        candidates.extend([
            path / "metrics" / "train_metrics.csv",
            path / "metrics" / "train_metrics.jsonl",
            path / "metrics" / "sample_efficiency.csv",
            path / "metrics" / "sample_efficiency.jsonl",
        ])
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported table path: {path}")


def centered_moving_average(values: Iterable[float], window: int = 13):
    return pd.Series(list(values), dtype="float64").rolling(
        window=window, center=True, min_periods=1
    ).mean()


def plot_training_dynamics(
    run_specs: List[Dict],
    output_path: str,
    success_col: str = "rollout_success_rate",
    length_col: str = "episode_length_mean",
    x_col: str = "global_step",
    moving_avg_window: int = 13,
    title: Optional[str] = None,
):
    if not run_specs:
        raise ValueError("run_specs cannot be empty")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200)
    fig.patch.set_facecolor("white")

    panel_specs = [
        (axes[0], success_col, "Episode success rate", "Success rate"),
        (axes[1], length_col, "Episode length", "Turns per episode"),
    ]

    for idx, spec in enumerate(run_specs):
        df = read_table(spec["path"], default_name="train_metrics.csv")
        if df.empty or x_col not in df.columns:
            continue
        if success_col not in df.columns and length_col not in df.columns:
            continue

        df = df.sort_values(x_col)
        color = spec.get("color", DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
        label = spec.get("label", f"run_{idx}")

        for ax, col, panel_title, y_label in panel_specs:
            if col not in df.columns:
                continue
            raw = pd.to_numeric(df[col], errors="coerce")
            x = pd.to_numeric(df[x_col], errors="coerce")
            mask = raw.notna() & x.notna()
            x = x[mask]
            raw = raw[mask]
            if raw.empty:
                continue
            smooth = centered_moving_average(raw.tolist(), window=moving_avg_window)

            ax.plot(x, raw, color=color, alpha=0.16, linewidth=1.0)
            ax.plot(x, smooth, color=color, linewidth=2.3, label=label)
            ax.set_title(panel_title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Training Steps", fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.grid(True, alpha=0.25, linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(handles)), frameon=False)
    if title:
        fig.suptitle(title, y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_sample_efficiency(
    input_path: str,
    output_path: str,
    score_col: str = "score",
    fraction_col: str = "data_fraction",
    method_col: str = "method",
    title: str = "Sample efficiency",
    score_metric: Optional[str] = None,
):
    df = read_table(input_path, default_name="sample_efficiency.csv")
    if df.empty:
        raise ValueError("No data found for sample efficiency plot")

    for col in [score_col, fraction_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()
    if score_metric is not None and "score_metric" in df.columns:
        df = df[df["score_metric"] == score_metric]
    df[fraction_col] = pd.to_numeric(df[fraction_col], errors="coerce")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[fraction_col, score_col])

    if method_col not in df.columns:
        df[method_col] = "Method"

    agg = (
        df.groupby([method_col, fraction_col], as_index=False)[score_col]
        .mean()
        .sort_values([method_col, fraction_col])
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=200)
    fig.patch.set_facecolor("white")

    for idx, (method, sub_df) in enumerate(agg.groupby(method_col)):
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        ax.plot(
            sub_df[fraction_col] * 100.0,
            sub_df[score_col],
            marker="o",
            linewidth=2.3,
            markersize=5.5,
            color=color,
            label=method,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Training data fraction (%)", fontsize=11)
    ax.set_ylabel(score_col.replace("_", " ").title(), fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


class ExperimentTracker:
    def __init__(
        self,
        output_dir: str,
        run_name: str,
        method_name: str,
        moving_avg_window: int = 13,
        auto_plot: bool = True,
    ):
        self.output_dir = output_dir
        self.run_name = run_name
        self.method_name = method_name
        self.moving_avg_window = moving_avg_window
        self.auto_plot = auto_plot
        self.metrics_dir = ensure_metrics_dir(output_dir)
        self.train_jsonl_path = self.metrics_dir / "train_metrics.jsonl"
        self.train_csv_path = self.metrics_dir / "train_metrics.csv"
        self.training_plot_path = self.metrics_dir / "training_dynamics.png"

    def log_train_metrics(self, row: Dict):
        row = dict(row)
        row.setdefault("run_name", self.run_name)
        row.setdefault("method", self.method_name)
        append_jsonl(self.train_jsonl_path, row)
        append_csv(self.train_csv_path, row)

    def maybe_plot_training_dynamics(
        self,
        success_col: str = "rollout_success_rate",
        length_col: str = "episode_length_mean",
        x_col: str = "global_step",
    ):
        if not self.auto_plot:
            return
        if not self.train_csv_path.exists():
            return
        plot_training_dynamics(
            run_specs=[{
                "label": self.method_name,
                "path": str(self.train_csv_path),
            }],
            output_path=str(self.training_plot_path),
            success_col=success_col,
            length_col=length_col,
            x_col=x_col,
            moving_avg_window=self.moving_avg_window,
        )
