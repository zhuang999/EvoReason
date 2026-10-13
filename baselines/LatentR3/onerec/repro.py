from __future__ import annotations
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch

def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def write_run_manifest(output_dir: str | Path, *, stage: str, args: Any, extra: dict | None=None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        revision = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    payload = {'stage': stage, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'git_revision': revision, 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda, 'cuda_devices': torch.cuda.device_count(), 'arguments': vars(args)}
    if extra:
        payload.update(extra)
    (output_dir / 'run_manifest.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
