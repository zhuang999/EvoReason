"""Induction pipeline for recommendation — original method (MuSR-faithful),
extended with failure-trajectory analysis inspired by SEED's SkillUpdater.

Strict copy of benchmarks/musr/induction.py logic, adapted only for:
  - TASK_DESCRIPTIONS['rec']
  - context (not narrative) in synthesize signature
  - filter that drops computational / ranking-algorithm categories so
    induced ptools stay NL-simulatable
  - local-vLLM config hooks (api_base / api_bases)
  - NEW: failure-trajectory analysis (Stage 5) — generates avoidance
    ptools from failed traces, mirroring SEED's SkillUpdater pattern

Pipeline (extended):
  1. Load a ReAct trace's thoughts
  2. LLM categorize each thought into a short reasoning-action-type label
  3. Merge synonyms if too many unique labels
  4. For each top category (by frequency), LLM synthesize a ptool spec
  5. [NEW] Analyze failed trajectories → synthesize avoidance ptools

Usage:
    python induction.py --task rec --trace results/.../results.jsonl --only-correct
    python induction.py --task rec --trace ... --api-base http://localhost:12350/v1
    python induction.py --task rec --trace ... --analyze-failures

Output: induction_out/<variant>/rec/{report.json, report.md}
"""

import argparse
import json
import pathlib
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path setup — mirror induce_self_discover.py
_BENCHMARK_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCHMARK_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_BENCHMARK_DIR))

from secretagent import config  # noqa: E402
from secretagent.llm_util import llm as _llm  # noqa: E402


def llm_with_retry(prompt: str, model: str, max_retries: int = 5, base_delay: float = 10.0) -> tuple[str, dict]:
    for attempt in range(max_retries):
        try:
            return _llm(prompt, model)
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f'  [llm retry {attempt+1}/{max_retries}] {type(e).__name__}: {e}, waiting {delay:.0f}s...')
                time.sleep(delay)
            else:
                raise


# ---------------------------------------------------------------------------
# Task metadata — descriptions match induce_self_discover.py verbatim so
# the categorize/synthesize prompts stay consistent across baselines.
# ---------------------------------------------------------------------------

TASK_DESCRIPTIONS = {
    'rec': (
        'Item recommendation ranking: given a user recommendation context '
        '(profile, MemRec long-term memory, interaction history, textual '
        'collaborative cues, and candidate item attributes), rank the '
        'candidates from most to least preferred by qualitatively matching '
        'inferred preferences to item attributes. Do NOT rely on numerical '
        'scores, embeddings, or ranking algorithms — reason in natural language.'
    ),
}

# Default: set via --trace. Kept for CLI symmetry with MuSR.
DEFAULT_TRACE_PATHS = {
    'rec': 'results/rec_react_train/results.jsonl',
}

# Categories matching these keywords are dropped before synthesize —
# they produce non-NL-simulatable "compute/rank" primitives.
_COMPUTATIONAL_CATEGORY_RE = re.compile(
    r'\b(score|scoring|rank|ranking|compute|computation|embedding|'
    r'similarity\s*metric|dot[\s-]?product|cosine|vote\s*count|'
    r'weighted\s*sum|sort\s*by\s*score)\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Stage 1 — trace adapter: pydantic-ai ReAct trace → flat list of thoughts
# ---------------------------------------------------------------------------

def load_thoughts(trace_path: pathlib.Path,
                   only_correct: bool = False) -> list[dict]:
    """Extract thought steps from a results.jsonl file.

    Each row has rollout[].step_info which is a flat sequence of dicts
    shaped like {thought: ...}, {tool_call: ..., args: ...}, or
    {tool_return: ..., output: ...}. We pull every non-empty {thought}
    entry, tagged with case_name + its position in step_info.

    If only_correct is True, only include thoughts from correct examples.
    """
    items: list[dict] = []
    with open(trace_path) as f:
        for line in f:
            row = json.loads(line)
            case = row.get('case_name') or '???'
            correct = bool(row.get('correct'))
            if only_correct and not correct:
                continue
            for call in row.get('rollout', []):
                for pos, step in enumerate(call.get('step_info', [])):
                    text = step.get('thought', '') if isinstance(step, dict) else ''
                    if text and text.strip():
                        items.append({
                            'case': case,
                            'pos': pos,
                            'text': text.strip(),
                            'correct': correct,
                        })
    return items


# ---------------------------------------------------------------------------
# Stage 2 — LLM categorization (batched)
# ---------------------------------------------------------------------------

CATEGORIZE_PROMPT = """\
You are analyzing thoughts produced by an AI agent solving this task:

{task_desc}

Categorize each thought into a SHORT, REUSABLE reasoning-action-type
label (3 to 6 words).

Rules:
- Describe WHAT the agent is DOING (its reasoning move), not the specific
  content of this particular case. Do not use names, places, item ids, or
  case-specific details in your label.
- Use consistent canonical names so synonyms collapse to the same label.
- Labels must be functionally distinct from each other.
- Labels must be preference/reasoning moves that an LLM can simulate in
  natural language (e.g. extract preferences, match attributes, resolve
  conflicts, consult memory, weigh short vs long term, eliminate mismatches).
- Do NOT invent labels about numerical computation or ranking algorithms
  (avoid: score, rank by score, compute similarity, embedding, vote count).
- Do NOT label pure tool-orchestration ("call search tool"); label the
  underlying reasoning intent instead.
- Ensure category DIVERSITY by covering DIFFERENT reasoning dimensions,
  not just variants of "preference matching". The core dimensions are:
  (1) extracting user preferences and identifying unmet needs,
  (2) analyzing temporal patterns, behavioral data, and preference evolution,
  (3) leveraging collaborative/social signals (e.g. what similar users bought,
      brand loyalty patterns, group trends, peer preferences),
  (4) detecting and resolving conflicts/contradictions between signals.
  Do NOT collapse all thoughts into "preference extraction" or
  "match preferences" variants — assign distinct dimension labels.
  IMPORTANT: When a thought references what OTHER users do, brand loyalty,
  or group-level patterns, label it as collaborative/social reasoning,
  NOT as generic preference extraction.
- Output ONLY a JSON array inside <answer>...</answer> tags, no prose.

Thoughts to categorize:

{items_block}

<answer>
[{{"index": 0, "category": "..."}}, ...]
</answer>
"""


def _format_batch(batch: list[dict], start: int, char_limit: int = 400) -> str:
    lines = []
    for i, item in enumerate(batch):
        idx = start + i
        text = item['text'].replace('\n', ' ').strip()
        if len(text) > char_limit:
            text = text[:char_limit] + '...'
        lines.append(f'[{idx}] {text}')
    return '\n'.join(lines)


_ANSWER_BLOCK_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)


_DEBUG_DIR: pathlib.Path | None = None


def _dump_debug(tag: str, prompt: str, response: str) -> None:
    if _DEBUG_DIR is None:
        return
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    idx = len(list(_DEBUG_DIR.glob(f'{tag}_*.txt')))
    path = _DEBUG_DIR / f'{tag}_{idx:03d}.txt'
    path.write_text(
        f'=== PROMPT ===\n{prompt}\n\n=== RESPONSE ===\n{response}\n'
    )


def _extract_json(text: str, shape: str) -> str | None:
    """Extract a JSON array or object from an LLM response.

    Tries in order:
      1. Content inside <answer>...</answer> tags (stripped)
      2. The largest balanced JSON literal of the requested shape in the text
    `shape` is either 'array' or 'object'.
    """
    m = _ANSWER_BLOCK_RE.search(text)
    candidates: list[str] = []
    if m:
        candidates.append(m.group(1).strip())
    candidates.append(text)

    open_ch, close_ch = ('[', ']') if shape == 'array' else ('{', '}')

    for candidate in candidates:
        # Find the FIRST opening bracket, then walk to its matching close
        start = candidate.find(open_ch)
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            end = -1
            for i in range(start, len(candidate)):
                ch = candidate[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == '\\':
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                return candidate[start:end + 1]
            start = candidate.find(open_ch, start + 1)
    return None


def _parse_categorize(text: str) -> list[tuple[int, str]]:
    """Extract (index, category) pairs; tolerant to minor formatting drift."""
    block = _extract_json(text, 'array')
    if not block:
        return []
    try:
        arr = json.loads(block)
    except json.JSONDecodeError:
        return []
    out: list[tuple[int, str]] = []
    for entry in arr:
        if not isinstance(entry, dict):
            continue
        if 'index' not in entry or 'category' not in entry:
            continue
        try:
            out.append((int(entry['index']), str(entry['category']).strip()))
        except (ValueError, TypeError):
            continue
    return out


def categorize_items(items: list[dict], task_key: str, model: str,
                     batch_size: int = 30, num_workers: int = 8,
                     only_indices: list[int] | None = None) -> list[str | None]:
    task_desc = TASK_DESCRIPTIONS[task_key]
    if only_indices is not None:
        target_items = [(i, items[i]) for i in only_indices]
    else:
        target_items = [(i, items[i]) for i in range(len(items))]
    result: list[str | None] = [None] * len(target_items)

    batches = [(ti, target_items[ti:ti + batch_size])
               for ti in range(0, len(target_items), batch_size)]

    def _categorize_batch(batch_start, batch):
        actual_indices = [idx for idx, _ in batch]
        batch_items = [it for _, it in batch]
        prompt = CATEGORIZE_PROMPT.format(
            task_desc=task_desc,
            items_block=_format_batch(batch_items, actual_indices[0]),
        )
        text, stats = llm_with_retry(prompt, model)
        n_in = stats.get('input_tokens', 0)
        n_out = stats.get('output_tokens', 0)
        parsed = list(_parse_categorize(text))
        n_parsed = len(parsed)
        if n_parsed == 0:
            _dump_debug('categorize_fail', prompt, text)
        print(f'  [categorize {actual_indices[0]}-{actual_indices[-1]}] '
              f'tokens={n_in}+{n_out} parsed={n_parsed}/{len(batch)}')
        return batch_start, [(idx, cat) for idx, cat in parsed if idx in actual_indices]

    fail_batches = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_categorize_batch, s, b): s for s, b in batches}
        for future in as_completed(futures):
            batch_start, parsed = future.result()
            batch_len = len(batches[batch_start // batch_size][1]) if batch_start // batch_size < len(batches) else batch_size
            for idx, cat in parsed:
                pos = batch_start + next((j for j, (i, _) in enumerate(batches[batch_start // batch_size][1]) if i == idx), 0)
                if 0 <= pos < len(result):
                    result[pos] = cat
            if len(parsed) == 0:
                fail_batches.append((batch_start, batch_len))

    if fail_batches:
        print(f'  {len(fail_batches)} batches had parse failures, retrying in parallel...')
        def _retry_batch(batch_start, _):
            batch = batches[batch_start // batch_size][1]
            actual_indices = [idx for idx, _ in batch]
            batch_items = [it for _, it in batch]
            prompt = CATEGORIZE_PROMPT.format(
                task_desc=task_desc,
                items_block=_format_batch(batch_items, actual_indices[0]),
            )
            text, stats = llm_with_retry(prompt, model)
            parsed = list(_parse_categorize(text))
            n_in = stats.get('input_tokens', 0)
            n_out = stats.get('output_tokens', 0)
            print(f'  [categorize retry {actual_indices[0]}-{actual_indices[-1]}] '
                  f'tokens={n_in}+{n_out} parsed={len(parsed)}/{len(batch)}')
            return [(idx, cat) for idx, cat in parsed if idx in actual_indices]

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            retry_futures = {pool.submit(_retry_batch, s, b): s for s, b in fail_batches}
            for future in as_completed(retry_futures):
                parsed = future.result()
                for idx, cat in parsed:
                    for j, (i, _) in enumerate(target_items):
                        if i == idx:
                            result[j] = cat
                            break

    return result


# ---------------------------------------------------------------------------
# Stage 3 — merge synonyms if there are too many unique labels
# ---------------------------------------------------------------------------

MERGE_PROMPT = """\
You are consolidating reasoning-action-type labels extracted from an AI
agent solving this task:

{task_desc}

Here are the current labels with their counts:

{labels_block}

Merge synonyms and near-duplicates so we end up with {target_min} to
{target_max} canonical labels. Each canonical label must be:
- 3 to 6 words
- functionally distinct from the others
- describe WHAT the agent is doing (the reasoning move), not case specifics
- Ensure the canonical labels cover DISTINCT reasoning dimensions:
  (1) extracting user preferences and identifying unmet needs,
  (2) analyzing temporal patterns, behavioral data, and preference evolution,
  (3) leveraging collaborative/social signals (what similar users bought,
      brand loyalty patterns, group trends, peer preferences),
  (4) detecting and resolving conflicts/contradictions between signals.
  Do NOT merge everything into homogenous "preference" categories.
  IMPORTANT: Keep collaborative/social reasoning as its own dimension —
  do NOT absorb it into generic preference categories.

Output a JSON object mapping each CANONICAL label to the list of ORIGINAL
labels (verbatim, exactly as shown above) that should map to it. Every
original label must appear in exactly one canonical group. Wrap in
<answer>...</answer>.

<answer>
{{
  "canonical label 1": ["original a", "original b", "original c"],
  "canonical label 2": ["original d"]
}}
</answer>
"""

def _merge_one_chunk(counts_chunk: list[tuple[str, int]], task_key: str, model: str,
                     target_min: int, target_max: int) -> dict[str, str] | None:
    labels_block = '\n'.join(f'  "{k}" — {v}' for k, v in counts_chunk)
    prompt = MERGE_PROMPT.format(
        task_desc=TASK_DESCRIPTIONS[task_key],
        labels_block=labels_block,
        target_min=target_min,
        target_max=target_max,
    )
    text, stats = llm_with_retry(prompt, model)
    n_in = stats.get('input_tokens', 0)
    n_out = stats.get('output_tokens', 0)
    print(f'  [merge chunk {len(counts_chunk)} labels] tokens={n_in}+{n_out}')

    block = _extract_json(text, 'object')
    if not block:
        _dump_debug('merge_fail', prompt, text)
        return None
    try:
        grouped = json.loads(block)
    except json.JSONDecodeError:
        _dump_debug('merge_fail', prompt, text)
        return None

    mapping: dict[str, str] = {}
    for canonical, originals in grouped.items():
        if not isinstance(originals, list):
            continue
        for orig in originals:
            if isinstance(orig, str):
                mapping[orig] = canonical
    return mapping


def merge_categories(labels: list[str | None], task_key: str, model: str,
                     target_min: int = 5, target_max: int = 10,
                     chunk_size: int = 150,
                     ) -> list[str | None]:
    counts = Counter(l for l in labels if l)
    if len(counts) <= target_max:
        return labels

    sorted_counts = counts.most_common()

    if len(sorted_counts) <= chunk_size:
        mapping = _merge_one_chunk(sorted_counts, task_key, model, target_min, target_max)
        if mapping is None:
            print('  WARNING: merge failed, keeping original labels')
            return labels
    else:
        chunks = [sorted_counts[i:i + chunk_size]
                  for i in range(0, len(sorted_counts), chunk_size)]
        print(f'  [merge] {len(sorted_counts)} labels -> {len(chunks)} chunks')
        all_mappings: dict[str, str] = {}
        for ci, chunk in enumerate(chunks):
            chunk_target = max(target_min, len(chunk) // 3)
            mapping = _merge_one_chunk(chunk, task_key, model, chunk_target, chunk_target * 2)
            if mapping is None:
                print(f'  WARNING: chunk {ci} merge failed, keeping original')
                continue
            all_mappings.update(mapping)
            n_mapped = sum(1 for l in labels if l in all_mappings or (l and l.lower() in {k.lower(): v for k, v in all_mappings.items()}))
            print(f'  [merge chunk {ci}/{len(chunks)}] got {len(mapping)} groups, total mapped: {n_mapped}')

        if len(all_mappings) > target_max:
            canonical_counts = Counter(all_mappings.values())
            top_canonicals = [c for c, _ in canonical_counts.most_common(target_max * 2)]
            top_chunk = [(c, cnt) for c, cnt in canonical_counts.most_common() if c in top_canonicals]
            print(f'  [merge final] re-merging {len(canonical_counts)} canonicals -> {target_max}')
            mapping = _merge_one_chunk(top_chunk, task_key, model, target_min, target_max)
            if mapping:
                for orig_canonical, new_canonical in mapping.items():
                    for orig_label, old_canonical in list(all_mappings.items()):
                        if old_canonical == orig_canonical:
                            all_mappings[orig_label] = new_canonical

        mapping = all_mappings

    lc_mapping = {k.lower(): v for k, v in mapping.items()}
    n_mapped = sum(1 for l in labels if l in mapping or (l and l.lower() in lc_mapping))
    n_unmapped = sum(1 for l in labels if l and l not in mapping and l.lower() not in lc_mapping)
    print(f'  merged: {n_mapped} mapped, {n_unmapped} unmapped kept as-is')

    merged: list[str | None] = []
    for l in labels:
        if l is None:
            merged.append(None)
        elif l in mapping:
            merged.append(mapping[l])
        elif l.lower() in lc_mapping:
            merged.append(lc_mapping[l.lower()])
        else:
            merged.append(l)
    return merged


# ---------------------------------------------------------------------------
# Stage 4 — synthesize a ptool spec per selected category
# ---------------------------------------------------------------------------

SYNTHESIZE_PROMPT = """\
You are designing a reusable reasoning tool (Python function) for solving
this task:

{task_desc}

The tool should capture this frequently-used reasoning action:
  "{category}"

Below are example agent thoughts that exhibit this reasoning action:

{examples_block}

The tool will be a Python function with this signature:

    def {{func_name}}(context: str, focus: str) -> str

Where:
  - context: the full recommendation context (user profile, MemRec memory,
             interaction history, collaborative cues)
  - focus:   what specific aspect the agent wants to reason about (e.g. a
             preference dimension, an attribute, a conflict to resolve,
             a memory cue)
  - returns: a structured string (plain text or JSON-shaped) with the
             tool's output

CRITICAL RULES — the tool is a REASONING PRIMITIVE, not a recommender:
  - The tool must ONLY analyze, extract, or reason about user preferences
    and patterns. It must NOT rank, recommend, or select items.
  - Do NOT use words like "rank", "recommend", "candidate", "top-N",
    "best match", "selection" in func_name, short_desc, or docstring.
  - The Returns: example must describe analysis output (e.g. list of
    preferences, detected contradictions, extracted patterns), NEVER a
    ranked list of items or a recommendation.
  - explicitly forbid inventing numerical scores
  - This tool must capture a DISTINCT reasoning dimension, not a minor
    variant of preference matching. The 5 core dimensions are:
    (1) extracting user preferences from profile/history/memory,
    (2) analyzing temporal patterns and preference evolution over time,
    (3) leveraging collaborative/social signals from similar users,
    (4) matching candidate item attributes to inferred user needs,
    (5) detecting and resolving conflicts/contradictions between signals.
    Ensure this tool addresses a DIFFERENT dimension than a generic
    "match preferences to items" tool would.

Design the tool:
  1. func_name:    snake_case Python identifier (no rank/recommend/candidate)
  2. display_name: CamelCase version (no Rank/Recommend/Candidate)
  3. short_desc:   one-sentence description — describe the REASONING ACTION
                   (e.g. "Extracts preference patterns", "Detects conflicts
                   between stated and behavioral preferences")
  4. docstring:    multi-line docstring covering
     - what information to extract / reason about (qualitative only)
     - how the response should be structured
     - what the agent should pay attention to (hard preference mismatches,
       conflicting short- vs long-term tastes, subtle memory signals, etc.)
     - a concrete "Returns:" example showing the output shape (analysis
       output only, never item recommendations)

Output as JSON inside <answer>...</answer>. The docstring should use
literal "\\n" escape sequences for newlines so the JSON is valid.

<answer>
{{
  "func_name": "snake_case_name",
  "display_name": "CamelCaseName",
  "short_desc": "one sentence.",
  "docstring": "multi-line docstring with \\n escapes..."
}}
</answer>
"""


def synthesize_ptool(category: str, examples: list[dict], task_key: str,
                     model: str) -> dict | None:
    blocks = []
    for i, ex in enumerate(examples, 1):
        text = ex['text']
        if len(text) > 800:
            text = text[:800] + '...'
        blocks.append(
            f'--- Example {i} (case {ex["case"]}, pos {ex["pos"]}) ---\n{text}'
        )
    examples_block = '\n\n'.join(blocks)

    prompt = SYNTHESIZE_PROMPT.format(
        task_desc=TASK_DESCRIPTIONS[task_key],
        category=category,
        examples_block=examples_block,
    )
    text, stats = llm_with_retry(prompt, model)
    n_in = stats.get('input_tokens', 0)
    n_out = stats.get('output_tokens', 0)
    print(f'  [synthesize "{category}"] tokens={n_in}+{n_out}')

    block = _extract_json(text, 'object')
    if not block:
        _dump_debug('synth_fail', prompt, text)
        print(f'  WARNING: synthesis response unparseable for {category!r}')
        return None
    try:
        spec = json.loads(block)
    except json.JSONDecodeError:
        try:
            spec = json.loads(re.sub(r'(?<!\\)\n', r'\\n', block))
        except json.JSONDecodeError:
            _dump_debug('synth_fail', prompt, text)
            print(f'  WARNING: synthesis JSON invalid for {category!r}')
            return None
    return spec


# ---------------------------------------------------------------------------
# Stage 5 — Analyze failed trajectories → synthesize avoidance ptools
# Inspired by SEED's SkillUpdater: failed traces → avoidance rules
# ---------------------------------------------------------------------------

AVOIDANCE_SYNTHESIZE_PROMPT = """\
You are designing reusable avoidance tools for a recommendation agent.

Task:
{task_desc}

The agent made mistakes on the following FAILED trajectories. Each shows
the agent's reasoning steps and where it went wrong.

Failed trajectory examples:

{failures_block}

Current existing tool names (avoid duplicating these):
{existing_tools_block}

For each distinct failure pattern you identify, design an avoidance tool
that helps the agent AVOID this kind of mistake in the future.

The tool will be a Python function with this signature:

    def {{func_name}}(context: str, focus: str) -> str

Where:
  - context: the full recommendation context (user profile, MemRec memory,
             interaction history, collaborative cues, candidate attributes)
  - focus:   what specific pitfall or warning sign to check for — NOT a
             numerical ranking criterion
  - returns: a structured string describing what to avoid and what to do
             instead

Design the tool:
  1. func_name:    snake_case Python identifier, prefixed with "avoid_"
  2. display_name: CamelCase version, starting with "Avoid"
  3. short_desc:   one-sentence description of the pitfall to avoid
  4. docstring:    multi-line docstring covering
     - what the common mistake is
     - why it leads to wrong recommendations
     - what the agent should do instead
     - warning signs that indicate this mistake is about to happen
     - a concrete "Returns:" example showing the output shape
     - explicitly forbid inventing numerical scores

Output a JSON array inside <answer>...</answer> with 1-{max_avoidance} tools.
The docstring should use literal "\\n" escape sequences for newlines.

<answer>
[{{{{
  "func_name": "avoid_snake_case",
  "display_name": "AvoidCamelCase",
  "short_desc": "one sentence about the pitfall.",
  "docstring": "multi-line docstring with \\n escapes..."
}}}}]
</answer>
"""


def _format_failure_examples(failed_items: list[dict], max_per: int = 5,
                             char_limit: int = 600) -> str:
    blocks = []
    for i, item in enumerate(failed_items[:max_per], 1):
        text = item['text'].replace('\n', ' ').strip()
        if len(text) > char_limit:
            text = text[:char_limit] + '...'
        blocks.append(
            f'--- Failure {i} (case {item["case"]}, pos {item["pos"]}) ---\n'
            f'{text}'
        )
    return '\n\n'.join(blocks)


def synthesize_avoidance_ptools(
    failed_items: list[dict],
    existing_ptool_names: list[str],
    task_key: str,
    model: str,
    max_avoidance: int = 3,
) -> list[dict]:
    """Analyze failed trajectories and generate avoidance ptools.

    Mirrors SEED's SkillUpdater.analyze_failures(): takes failed thoughts,
    sends them to LLM with existing tool names for dedup, gets back
    avoidance ptool specs.

    Args:
        failed_items: list of thought dicts from incorrect traces,
                      each with {case, pos, text, correct=False}
        existing_ptool_names: list of func_name strings already in the bank
        task_key: task key for TASK_DESCRIPTIONS
        model: LLM model id
        max_avoidance: max number of avoidance ptools to generate

    Returns:
        list of ptool spec dicts (may be empty)
    """
    if not failed_items:
        return []

    failures_block = _format_failure_examples(failed_items)
    existing_tools_block = (
        '\n'.join(f'  - {n}' for n in existing_ptool_names)
        if existing_ptool_names
        else '(none yet)'
    )

    prompt = AVOIDANCE_SYNTHESIZE_PROMPT.format(
        task_desc=TASK_DESCRIPTIONS[task_key],
        failures_block=failures_block,
        existing_tools_block=existing_tools_block,
        max_avoidance=max_avoidance,
    )

    text, stats = llm_with_retry(prompt, model)
    n_in = stats.get('input_tokens', 0)
    n_out = stats.get('output_tokens', 0)
    print(f'  [avoidance synthesize] tokens={n_in}+{n_out}')

    block = _extract_json(text, 'array')
    if not block:
        _dump_debug('avoid_fail', prompt, text)
        print('  WARNING: avoidance synthesis response unparseable')
        return []

    try:
        specs = json.loads(block)
    except json.JSONDecodeError:
        try:
            specs = json.loads(re.sub(r'(?<!\\)\n', r'\\n', block))
        except json.JSONDecodeError:
            _dump_debug('avoid_fail', prompt, text)
            print('  WARNING: avoidance JSON invalid')
            return []

    if not isinstance(specs, list):
        specs = [specs]

    valid = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        if 'func_name' not in spec or 'docstring' not in spec:
            continue
        valid.append(spec)

    print(f'  generated {len(valid)} avoidance ptools')
    return valid


def synthesize_success_ptools(
    correct_items: list[dict],
    existing_ptool_names: list[str],
    task_key: str,
    model: str,
    max_new: int = 3,
) -> list[dict]:
    """Analyze successful trajectories and generate new positive ptools.

    Mirrors SEED's SEEDEpisodeAnalyzer with episode_success=True:
    "extract the successful trajectory into workflow: the core decision
    rule and action ordering that made this trajectory work."

    Unlike Stage 4 (categorize → synthesize), this directly sends
    successful thoughts to LLM for end-to-end ptool extraction,
    mirroring SEED's approach.

    Args:
        correct_items: list of thought dicts from correct traces,
                       each with {case, pos, text, correct=True}
        existing_ptool_names: list of func_name strings already in the bank
        task_key: task key for TASK_DESCRIPTIONS
        model: LLM model id
        max_new: max number of new ptools to generate

    Returns:
        list of ptool spec dicts (may be empty)
    """
    if not correct_items:
        return []

    success_block = _format_success_examples(correct_items)
    existing_tools_block = (
        '\n'.join(f'  - {n}' for n in existing_ptool_names)
        if existing_ptool_names
        else '(none yet)'
    )

    prompt = SUCCESS_SYNTHESIZE_PROMPT.format(
        task_desc=TASK_DESCRIPTIONS[task_key],
        success_block=success_block,
        existing_tools_block=existing_tools_block,
        max_new=max_new,
    )

    text, stats = llm_with_retry(prompt, model)
    n_in = stats.get('input_tokens', 0)
    n_out = stats.get('output_tokens', 0)
    print(f'  [success synthesize] tokens={n_in}+{n_out}')

    block = _extract_json(text, 'array')
    if not block:
        _dump_debug('success_fail', prompt, text)
        print('  WARNING: success synthesis response unparseable')
        return []

    try:
        specs = json.loads(block)
    except json.JSONDecodeError:
        try:
            specs = json.loads(re.sub(r'(?<!\\)\n', r'\\n', block))
        except json.JSONDecodeError:
            _dump_debug('success_fail', prompt, text)
            print('  WARNING: success JSON invalid')
            return []

    if not isinstance(specs, list):
        specs = [specs]

    valid = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        if 'func_name' not in spec or 'docstring' not in spec:
            continue
        valid.append(spec)

    print(f'  generated {len(valid)} success ptools')
    return valid


SUCCESS_SYNTHESIZE_PROMPT = """\
You are designing reusable reasoning tools for a recommendation agent.

Task:
{task_desc}

The agent produced CORRECT recommendations using the following reasoning
steps. Extract reusable workflow patterns from these successful trajectories.

Successful reasoning examples:

{success_block}

Current existing tool names (avoid duplicating these):
{existing_tools_block}

For each distinct successful reasoning pattern you identify, design a
reusable tool that captures the core decision rule and action ordering
that made the trajectory work.

IMPORTANT: The tools you generate must cover DIVERSE reasoning dimensions,
not just variants of preference matching. Ensure they span at least these
distinct categories:
  (1) extracting user preferences and identifying unmet needs,
  (2) analyzing temporal patterns, behavioral data, and preference evolution,
  (3) leveraging collaborative/social signals and brand loyalty patterns,
  (4) detecting and resolving conflicts/contradictions between signals.
Do NOT generate multiple tools that all do "match preferences to items"
with minor wording differences.

The tool will be a Python function with this signature:

    def {{func_name}}(context: str, focus: str) -> str

Where:
  - context: the full recommendation context (user profile, MemRec memory,
             interaction history, collaborative cues, candidate attributes)
  - focus:   what specific aspect the agent wants to reason about (e.g. a
             preference dimension, an item attribute, a pattern to extract)
             — NOT a numerical ranking criterion
  - returns: a structured string (plain text or JSON-shaped) with the
             tool's output

Design the tool:
  1. func_name:    snake_case Python identifier
  2. display_name: CamelCase version
  3. short_desc:   one-sentence description for the agent's system prompt
  4. docstring:    multi-line docstring covering
     - what information to extract / reason about (qualitative only)
     - how the response should be structured
     - what the agent should pay attention to (key preference signals,
       pattern indicators, decision checkpoints, etc.)
     - a concrete "Returns:" example showing the output shape
     - explicitly forbid inventing numerical scores

Output a JSON array inside <answer>...</answer> with 1-{max_new} tools.
The docstring should use literal "\\n" escape sequences for newlines.

<answer>
[{{{{
  "func_name": "snake_case_name",
  "display_name": "CamelCaseName",
  "short_desc": "one sentence.",
  "docstring": "multi-line docstring with \\n escapes..."
}}}}]
</answer>
"""


def _format_success_examples(success_items: list[dict], max_per: int = 5,
                             char_limit: int = 600) -> str:
    blocks = []
    for i, item in enumerate(success_items[:max_per], 1):
        text = item['text'].replace('\n', ' ').strip()
        if len(text) > char_limit:
            text = text[:char_limit] + '...'
        blocks.append(
            f'--- Success {i} (case {item["case"]}, pos {item["pos"]}) ---\n'
            f'{text}'
        )
    return '\n\n'.join(blocks)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one_task(task_key: str, trace_path: pathlib.Path, model: str,
                 output_dir: pathlib.Path, min_count: int, max_ptools: int,
                 only_correct: bool = False,
                 analyze_failures: bool = False,
                 analyze_successes: bool = False,
                 max_avoidance: int = 3,
                 max_success: int = 3,
                 ) -> dict:
    print(f'\n=== {task_key} ===')
    print(f'trace: {trace_path}')
    if only_correct:
        print('filtering: only correct traces')

    items = load_thoughts(trace_path, only_correct=only_correct)
    print(f'loaded {len(items)} thoughts')
    if not items:
        return {'task': task_key, 'error': 'no thoughts'}

    correct_items = [it for it in items if it['correct']]
    failed_items = [it for it in items if not it['correct']]
    print(f'  correct: {len(correct_items)}, failed: {len(failed_items)}')

    labels_cache = output_dir / task_key / 'labels_cache.json'
    if labels_cache.exists():
        print(f'Loading cached labels from {labels_cache}')
        labels = json.loads(labels_cache.read_text(encoding='utf-8'))
        n_labeled = sum(1 for l in labels if l)
        print(f'categorized {n_labeled}/{len(items)} thoughts (from cache)')
    else:
        labels = categorize_items(items, task_key, model, num_workers=8)
        n_labeled = sum(1 for l in labels if l)
        print(f'categorized {n_labeled}/{len(items)} thoughts')
        labels_cache.parent.mkdir(parents=True, exist_ok=True)
        labels_cache.write_text(json.dumps(labels, ensure_ascii=False), encoding='utf-8')
        print(f'Saved labels cache to {labels_cache}')

    merged_cache = output_dir / task_key / 'merged_labels_cache.json'
    if merged_cache.exists():
        print(f'Loading cached merged labels from {merged_cache}')
        labels = json.loads(merged_cache.read_text(encoding='utf-8'))
    else:
        labels = merge_categories(labels, task_key, model)
        merged_cache.parent.mkdir(parents=True, exist_ok=True)
        merged_cache.write_text(json.dumps(labels, ensure_ascii=False), encoding='utf-8')
        print(f'Saved merged labels cache to {merged_cache}')

    # ADAPTATION: drop computational categories before reporting / synthesize
    filtered_labels: list[str | None] = []
    n_dropped = 0
    for lab in labels:
        if lab and _COMPUTATIONAL_CATEGORY_RE.search(lab):
            filtered_labels.append(None)
            n_dropped += 1
        else:
            filtered_labels.append(lab)
    if n_dropped:
        print(f'filtered out {n_dropped} computational-category thoughts')
    labels = filtered_labels

    counts = Counter(l for l in labels if l).most_common()
    print(f'unique categories after merge: {len(counts)}')

    categories_report = []
    for cat, count in counts:
        examples_of_cat = [items[i] for i, l in enumerate(labels) if l == cat]
        categories_report.append({
            'category': cat,
            'count': count,
            'sample_thoughts': [
                {
                    'case': e['case'],
                    'pos': e['pos'],
                    'correct': e['correct'],
                    'text': e['text'][:500],
                }
                for e in examples_of_cat[:5]
            ],
        })

    # Stage 3b: LLM-group categories into max_ptools reasoning dimensions
    DIMENSION_GROUP_PROMPT = """\
You are grouping reasoning-action-type labels into {n_dims} distinct
reasoning DIMENSIONS for a recommendation agent.

Task:
{task_desc}

Here are the reasoning-action categories with their counts:

{categories_block}

Group these categories into exactly {n_dims} dimensions. Each dimension
must be functionally distinct — not a minor variant of another.
For each dimension provide:
  - dim_name: a short snake_case identifier (2-4 words)
  - keywords: a list of 5-10 lowercase keywords/phrases that identify
    categories belonging to this dimension
  - categories: the list of original category labels (verbatim) that
    belong to this dimension

Every category must appear in exactly one dimension.

Output JSON inside <answer>...</answer>:
<answer>
{{{{
  "dimensions": [
    {{{{
      "dim_name": "snake_case",
      "keywords": ["kw1", "kw2", ...],
      "categories": ["original label 1", "original label 2", ...]
    }}}},
    ...
  ]
}}}}
</answer>
"""

    top_k = max(max_ptools * 4, 20)
    top_counts = [(cat, count) for cat, count in counts if count >= min_count][:top_k]
    categories_block = '\n'.join(f'  "{cat}" — {count}' for cat, count in top_counts)

    dim_prompt = DIMENSION_GROUP_PROMPT.format(
        task_desc=TASK_DESCRIPTIONS[task_key],
        n_dims=max_ptools,
        categories_block=categories_block,
    )
    dim_text, dim_stats = llm_with_retry(dim_prompt, model)
    print(f'  [dimension grouping] tokens={dim_stats.get("input_tokens",0)}+{dim_stats.get("output_tokens",0)}')

    dim_block = _extract_json(dim_text, 'object')
    dimensions: list[dict] = []
    if dim_block:
        try:
            dim_obj = json.loads(dim_block)
            dimensions = dim_obj.get('dimensions', [])
        except json.JSONDecodeError:
            pass

    if not dimensions:
        print('  WARNING: dimension grouping failed, using keyword fallback')
        dimensions = [
            {'dim_name': 'preference_extraction', 'keywords': ['preference', 'extract', 'profile', 'unmet need', 'match attribute'], 'categories': []},
            {'dim_name': 'temporal_analysis', 'keywords': ['history', 'temporal', 'behavioral', 'recent', 'interaction', 'trend'], 'categories': []},
            {'dim_name': 'collaborative_signal', 'keywords': ['collaborative', 'social', 'brand loyalty', 'peer'], 'categories': []},
            {'dim_name': 'conflict_resolution', 'keywords': ['conflict', 'contradiction', 'mismatch', 'resolve', 'contrary'], 'categories': []},
            {'dim_name': 'recommendation_synthesis', 'keywords': ['recommend', 'synthesize', 'final'], 'categories': []},
        ]

    # Build category -> dimension mapping from LLM output
    cat_to_dim: dict[str, str] = {}
    for dim in dimensions:
        for cat in dim.get('categories', []):
            cat_to_dim[cat] = dim['dim_name']

    # Fallback: keyword matching for categories not covered by LLM grouping
    dim_keywords: dict[str, list[str]] = {d['dim_name']: d.get('keywords', []) for d in dimensions}

    def _get_dimension(cat_label: str) -> str | None:
        if cat_label in cat_to_dim:
            return cat_to_dim[cat_label]
        low = cat_label.lower()
        for dim_name, keywords in dim_keywords.items():
            if any(kw in low for kw in keywords):
                return dim_name
        return None

    candidates = [(cat, count) for cat, count in counts if count >= min_count][:top_k]

    # Pass 1: ensure every LLM-induced dimension gets at least 1 category
    # Pick the highest-frequency category per dimension from all candidates
    dim_best: dict[str, tuple[str, int]] = {}
    for cat, count in candidates:
        dim = _get_dimension(cat)
        if dim:
            if dim not in dim_best or count > dim_best[dim][1]:
                dim_best[dim] = (cat, count)

    for dim_name in [d['dim_name'] for d in dimensions]:
        if dim_name in dim_best and len(selected_cats) < max_ptools:
            cat, count = dim_best[dim_name]
            selected_cats.append((cat, count))
            covered_dims.add(dim_name)

    # Pass 2: fill remaining slots from candidates by frequency,
    # skipping already-selected categories
    selected_set = {cat for cat, _ in selected_cats}
    for cat, count in candidates:
        if len(selected_cats) >= max_ptools:
            break
        if cat not in selected_set:
            selected_cats.append((cat, count))
            selected_set.add(cat)

    print(f'selected {len(selected_cats)} categories across dimensions: {covered_dims}')
    for cat, count in selected_cats:
        print(f'  - {cat} ({count})')

    ptools = []
    for cat, count in selected_cats:
        correct_cat_examples = [
            items[i] for i, l in enumerate(labels)
            if l == cat and items[i]['correct']
        ][:5]
        if not correct_cat_examples:
            correct_cat_examples = [
                items[i] for i, l in enumerate(labels) if l == cat
            ][:5]
        spec = synthesize_ptool(cat, correct_cat_examples, task_key, model)
        if spec is None:
            continue
        spec['source_category'] = cat
        spec['count'] = count
        spec['example_cases'] = [
            f'{e["case"]}:pos{e["pos"]}' for e in correct_cat_examples
        ]
        ptools.append(spec)

    # Stage 4b: Analyze successful trajectories → new positive ptools
    success_ptools = []
    if analyze_successes and correct_items:
        print(f'\n--- Stage 4b: Analyzing {len(correct_items)} successful thoughts ---')
        existing_names = [p.get('func_name', '') for p in ptools]
        success_ptools = synthesize_success_ptools(
            correct_items=correct_items,
            existing_ptool_names=existing_names,
            task_key=task_key,
            model=model,
            max_new=max_success,
        )
        for sp in success_ptools:
            sp['source_category'] = f"success:{sp.get('func_name', '?')}"
            sp['count'] = len(correct_items)
            sp['is_success'] = True
        print(f'  generated {len(success_ptools)} success ptools')

    # Stage 5: Analyze failed trajectories → avoidance ptools
    avoidance_ptools = []
    if analyze_failures and failed_items:
        print(f'\n--- Stage 5: Analyzing {len(failed_items)} failed thoughts ---')
        existing_names = (
            [p.get('func_name', '') for p in ptools]
            + [p.get('func_name', '') for p in success_ptools]
        )
        avoidance_ptools = synthesize_avoidance_ptools(
            failed_items=failed_items,
            existing_ptool_names=existing_names,
            task_key=task_key,
            model=model,
            max_avoidance=max_avoidance,
        )
        for ap in avoidance_ptools:
            ap['source_category'] = f"avoidance:{ap.get('func_name', '?')}"
            ap['count'] = len(failed_items)
            ap['is_avoidance'] = True
        print(f'  generated {len(avoidance_ptools)} avoidance ptools')

    task_dir = output_dir / task_key
    task_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'task': task_key,
        'trace_path': str(trace_path),
        'model': model,
        'n_thoughts': len(items),
        'n_correct': len(correct_items),
        'n_failed': len(failed_items),
        'n_labeled': n_labeled,
        'n_unique_categories': len(counts),
        'min_count': min_count,
        'max_ptools': max_ptools,
        'categories': categories_report,
        'ptools': ptools,
        'success_ptools': success_ptools,
        'avoidance_ptools': avoidance_ptools,
    }
    (task_dir / 'report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    (task_dir / 'report.md').write_text(render_markdown(report))
    print(f'wrote {task_dir}/report.json and report.md')
    return report


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append(f'# Induction report — {report["task"]} (original method)')
    lines.append('')
    lines.append(f'- Trace: `{report["trace_path"]}`')
    lines.append(f'- Model: `{report["model"]}`')
    lines.append(f'- Thoughts: {report["n_thoughts"]} loaded '
                 f'({report.get("n_correct", "?")} correct, '
                 f'{report.get("n_failed", "?")} failed), '
                 f'{report["n_labeled"]} labeled')
    lines.append(f'- Unique categories: {report["n_unique_categories"]}')
    lines.append(f'- Synthesized ptools: {len(report["ptools"])} '
                 f'(min_count={report["min_count"]}, '
                 f'max_ptools={report["max_ptools"]})')
    avoidance = report.get('avoidance_ptools', [])
    if avoidance:
        lines.append(f'- Avoidance ptools: {len(avoidance)}')
    success = report.get('success_ptools', [])
    if success:
        lines.append(f'- Success ptools: {len(success)}')
    lines.append('')

    lines.append('## Categories (by frequency)')
    lines.append('')
    for c in report['categories']:
        lines.append(f'### {c["category"]} — {c["count"]}')
        for ex in c['sample_thoughts'][:3]:
            text = ex['text'].replace('\n', ' ').strip()
            if len(text) > 240:
                text = text[:240] + '...'
            mark = 'Y' if ex['correct'] else 'N'
            lines.append(
                f'- [{mark}] `{ex["case"]}:pos{ex["pos"]}` — {text}'
            )
        lines.append('')

    lines.append('## Synthesized ptools')
    lines.append('')
    for i, p in enumerate(report['ptools'], 1):
        name = p.get('display_name') or p.get('func_name') or '?'
        lines.append(f'### {i}. {name}')
        lines.append(
            f'- Source category: **{p["source_category"]}** '
            f'({p["count"]} occurrences)'
        )
        lines.append(f'- func_name: `{p.get("func_name", "?")}`')
        lines.append(f'- short_desc: {p.get("short_desc", "")}')
        lines.append('')
        lines.append('```python')
        lines.append(
            f'def {p.get("func_name", "unknown")}'
            '(context: str, focus: str) -> str:'
        )
        doc = p.get('docstring', '')
        lines.append('    """')
        for dl in doc.split('\n'):
            lines.append(f'    {dl}' if dl else '')
        lines.append('    """')
        lines.append('```')
        lines.append('')

    if avoidance:
        lines.append('## Avoidance ptools (from failed trajectories)')
        lines.append('')
        for i, p in enumerate(avoidance, 1):
            name = p.get('display_name') or p.get('func_name') or '?'
            lines.append(f'### {i}. {name}')
            lines.append(
                f'- Source: **{p.get("source_category", "?")}** '
                f'({p.get("count", "?")} failed thoughts)'
            )
            lines.append(f'- func_name: `{p.get("func_name", "?")}`')
            lines.append(f'- short_desc: {p.get("short_desc", "")}')
            lines.append('')
            lines.append('```python')
            lines.append(
                f'def {p.get("func_name", "unknown")}'
                '(context: str, focus: str) -> str:'
            )
            doc = p.get('docstring', '')
            lines.append('    """')
            for dl in doc.split('\n'):
                lines.append(f'    {dl}' if dl else '')
            lines.append('    """')
            lines.append('```')
            lines.append('')

    if success:
        lines.append('## Success ptools (from correct trajectories)')
        lines.append('')
        for i, p in enumerate(success, 1):
            name = p.get('display_name') or p.get('func_name') or '?'
            lines.append(f'### {i}. {name}')
            lines.append(
                f'- Source: **{p.get("source_category", "?")}** '
                f'({p.get("count", "?")} successful thoughts)'
            )
            lines.append(f'- func_name: `{p.get("func_name", "?")}`')
            lines.append(f'- short_desc: {p.get("short_desc", "")}')
            lines.append('')
            lines.append('```python')
            lines.append(
                f'def {p.get("func_name", "unknown")}'
                '(context: str, focus: str) -> str:'
            )
            doc = p.get('docstring', '')
            lines.append('    """')
            for dl in doc.split('\n'):
                lines.append(f'    {dl}' if dl else '')
            lines.append('    """')
            lines.append('```')
            lines.append('')

    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['rec'], default='rec')
    p.add_argument('--trace', default=None,
                   help='Override trace path (results.jsonl from expt.py)')
    p.add_argument('--variant', default='original',
                   help='Variant label for output subdirectory')
    p.add_argument('--model', default='openai/QwQ-32B',
                   help='LiteLLM model id (use openai/* for local vLLM)')
    p.add_argument('--api-base', default=None,
                   help='OpenAI-compatible base URL for local vLLM')
    p.add_argument('--api-bases', nargs='*', default=None,
                   help='Optional list of vLLM bases for round-robin')
    p.add_argument('--output-dir', default='induction_out')
    p.add_argument('--min-count', type=int, default=3)
    p.add_argument('--max-ptools', type=int, default=5)
    p.add_argument('--only-correct', action='store_true',
                   help='Only use thoughts from correct traces')
    p.add_argument('--analyze-failures', action='store_true',
                   help='Also analyze failed trajectories and generate '
                        'avoidance ptools (Stage 5)')
    p.add_argument('--analyze-successes', action='store_true',
                   help='Also analyze successful trajectories and generate '
                        'new positive ptools (Stage 4b)')
    p.add_argument('--max-avoidance', type=int, default=1,
                   help='Max avoidance ptools to generate from failures')
    p.add_argument('--max-success', type=int, default=1,
                   help='Max success ptools to generate from correct traces')
    args = p.parse_args()

    llm_cfg: dict = {'model': args.model, 'max_tokens': 8192, 'timeout': 1200}
    if args.api_bases:
        llm_cfg['api_bases'] = list(args.api_bases)
    elif args.api_base:
        llm_cfg['api_base'] = args.api_base
    llm_cfg['api_key'] = 'EMPTY'

    config.configure(cfg={
        'llm': {**llm_cfg, 'num_retries': 3},
        'cachier': {
            'enable_caching': False,
        },
    })

    out_dir = _BENCHMARK_DIR / args.output_dir / args.variant
    global _DEBUG_DIR
    _DEBUG_DIR = out_dir / '_debug'

    tasks = [args.task]

    summaries = []
    for task in tasks:
        if args.trace:
            trace_path = pathlib.Path(args.trace)
        else:
            trace_path = _BENCHMARK_DIR / DEFAULT_TRACE_PATHS[task]
        if not trace_path.exists():
            print(f'WARNING: {trace_path} not found, skipping')
            continue
        report = run_one_task(
            task, trace_path, args.model, out_dir,
            args.min_count, args.max_ptools,
            only_correct=args.only_correct,
            analyze_failures=args.analyze_failures,
            analyze_successes=args.analyze_successes,
            max_avoidance=args.max_avoidance,
            max_success=args.max_success,
        )
        summaries.append(
            {
                'task': task,
                'n_ptools': len(report.get('ptools', [])),
                'n_success': len(report.get('success_ptools', [])),
                'n_avoidance': len(report.get('avoidance_ptools', [])),
            }
        )

    print('\n=== summary ===')
    for s in summaries:
        parts = [f'{s["n_ptools"]} ptools']
        if s.get('n_success'):
            parts.append(f'{s["n_success"]} success')
        if s.get('n_avoidance'):
            parts.append(f'{s["n_avoidance"]} avoidance')
        print(f'  {s["task"]}: {", ".join(parts)}')


if __name__ == '__main__':
    main()
