"""Shared ReAct tools for recommendation — MuSR-aligned evidence tools.

Faithful adaptation of benchmarks/musr/ptools/ptools_common.py:

  MuSR                          →  Rec
  -----------------------------------------------
  narrative                     →  context (profile / MemRec memory /
                                   history / collab text / candidates)
  search / lookup               →  search / lookup  (same sentence match)
  (none)                        →  search_memory   (MemRec section only)
  finish(answer_index)          →  produce_ranking(ranking_list)
  react_answer_impl             →  react_rank_impl

Design intent (see README):
  Phase-1 tools gather TEXTUAL evidence only — no scores, votes, or
  ranking algorithms. Ranking is only expressed via produce_ranking.
  This keeps induced primitives NL-simulatable.
"""

from __future__ import annotations

import re

from secretagent.core import interface, register_factory
from secretagent.implement.pydantic import (
    ReactPydanticFactory,
    ReactPydanticWithWorkflowFactory,
)


# Section markers used by prepare_data.py when assembling context.
SECTION_MEMORY = '=== USER MEMORY (MemRec) ==='
SECTION_HISTORY = '=== INTERACTION HISTORY ==='
SECTION_COLLAB = '=== COLLABORATIVE EVIDENCE ==='
SECTION_CANDIDATES = '=== CANDIDATE ITEMS ==='
SECTION_PROFILE = '=== USER PROFILE ==='


from ptools.ptools_rec_tools import (
    search_section,
    search_history,
    search_candidates,
    search_collab,
    count_history,
)


_REC_REACT_PREAMBLE = (
    "Rank the candidate items for this user as best you can. "
    "Use the available tools to gather textual evidence from the "
    "recommendation context (user profile, MemRec long-term memory, "
    "interaction history, collaborative cues, and candidate attributes). "
    "Use section-level tools (search_section, search_history, search_candidates, "
    "search_collab) for targeted retrieval. Use extraction tools "
    "(extract_user_profile, extract_purchase_patterns, etc.) for structured "
    "reasoning steps. Reason QUALITATIVELY about preferences — do NOT invent "
    "numerical scores, similarity metrics, or ranking algorithms. "
    "When ready, call produce_ranking(ranking_list) with candidate "
    "item_ids ordered from most to least preferred."
)


@interface
def raw_ranking(context: str, question: str, candidates: list) -> str:
    """Read the recommendation context and produce a ranking of candidates."""


@interface
def extract_ranking(answer_text: str, candidates: list) -> list:
    """Given an answer and candidates, return a ranked list of item ids."""


# ============================================================
# ReAct support: search / lookup / search_memory / produce_ranking
# ============================================================
#
# Design notes (copied from MuSR ptools_common.py):
# These tools need read-access to the current context. We use
# module-level state (`_REACT_STATE`) instead of closures —
# evaluation is sequential, so a single mutable dict is safe.

_REACT_STATE: dict = {
    'context': '',
    'finish_ranking': None,
    'candidates': [],
    'lookup_last': None,
    'lookup_matches': [],
    'lookup_idx': 0,
}


def _reset_react_state(context: str, candidates: list | None = None) -> None:
    _REACT_STATE['context'] = context
    _REACT_STATE['finish_ranking'] = None
    _REACT_STATE['candidates'] = list(candidates or [])
    _REACT_STATE['lookup_last'] = None
    _REACT_STATE['lookup_matches'] = []
    _REACT_STATE['lookup_idx'] = 0


def _split_into_sentences(text: str) -> list[str]:
    """Naive sentence/line splitter — adequate for rec context blocks."""
    # Prefer line breaks (context is sectioned); fall back to sentence split.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 3:
        return lines
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _extract_section(context: str, header: str) -> str:
    """Return text from `header` until the next `=== ... ===` header."""
    if header not in context:
        return ''
    start = context.index(header) + len(header)
    rest = context[start:]
    m = re.search(r'\n=== .+ ===\n', rest)
    if m:
        return rest[:m.start()].strip()
    return rest.strip()


def search(query: str) -> str:
    """Search the recommendation context for lines/sentences containing the query.

    Returns up to 5 matching lines, each prefixed with '- '. If more
    than 5 lines match, indicates how many additional matches were
    truncated. Use this for broad keyword exploration over the full
    context (profile, memory, history, collab, candidates).
    """
    context = _REACT_STATE.get('context') or ''
    if not context:
        return "Error: no recommendation context loaded for this call."
    sentences = _split_into_sentences(context)
    q = query.lower()
    matches = [s for s in sentences if q in s.lower()]
    if not matches:
        return f"No lines contain {query!r}."
    head = matches[:5]
    body = '\n'.join(f"- {s}" for s in head)
    if len(matches) > 5:
        body += (
            f"\n(... {len(matches) - 5} more matches; "
            f"refine the query or use lookup to paginate)"
        )
    return body


def lookup(string: str) -> str:
    """Paginate through context lines containing the string.

    First call with a new string returns the first match; subsequent
    calls with the SAME string return the next match. Calling with a
    different string resets pagination. Returns 'No more matches.' when
    exhausted. Use this when search returned too many matches and you
    want to walk through them one at a time.
    """
    context = _REACT_STATE.get('context') or ''
    if not context:
        return "Error: no recommendation context loaded for this call."
    if _REACT_STATE['lookup_last'] != string:
        sentences = _split_into_sentences(context)
        s_lower = string.lower()
        _REACT_STATE['lookup_last'] = string
        _REACT_STATE['lookup_matches'] = [
            s for s in sentences if s_lower in s.lower()
        ]
        _REACT_STATE['lookup_idx'] = 0
    matches = _REACT_STATE['lookup_matches']
    if not matches:
        return f"No lines contain {string!r}."
    idx = _REACT_STATE['lookup_idx']
    if idx >= len(matches):
        return f"No more matches for {string!r} ({len(matches)} total)."
    result = matches[idx]
    _REACT_STATE['lookup_idx'] = idx + 1
    return f"({idx + 1}/{len(matches)}) {result}"


def search_memory(query: str) -> str:
    """Search ONLY the MemRec long-term memory section of the context.

    Use this when you specifically need the user's durable preference
    summary written by MemRec (habits, constraints, taste axes), as
    opposed to short-term interaction history. Returns up to 5 matching
    lines from the USER MEMORY section. If the section is empty, says so.
    """
    context = _REACT_STATE.get('context') or ''
    if not context:
        return "Error: no recommendation context loaded for this call."
    section = _extract_section(context, SECTION_MEMORY)
    if not section:
        return (
            "No MemRec memory section found in context. "
            "Fall back to search() over the full context."
        )
    sentences = _split_into_sentences(section)
    q = query.lower()
    matches = [s for s in sentences if q in s.lower()] if q else sentences[:5]
    if not matches:
        # If query misses, return a short preview so the agent still sees memory.
        preview = sentences[:5]
        body = '\n'.join(f"- {s}" for s in preview)
        return (
            f"No memory lines contain {query!r}. "
            f"Memory preview:\n{body}"
        )
    head = matches[:5]
    body = '\n'.join(f"- {s}" for s in head)
    if len(matches) > 5:
        body += f"\n(... {len(matches) - 5} more memory matches)"
    return body


def produce_ranking(ranking_list: str) -> str:
    """Submit the final ranking of candidate item_ids.

    Call this EXACTLY ONCE when you have determined the ranking.
    `ranking_list` is a space-separated or comma-separated string of
    item_ids, ordered from most preferred to least preferred.
    Example: "1234 5678 9012"

    After calling produce_ranking, stop calling other tools.
    """
    raw = (ranking_list or '').strip()
    items = [
        t.strip().strip(',')
        for t in raw.replace(',', ' ').split()
        if t.strip()
    ]
    _REACT_STATE['finish_ranking'] = items
    return (
        f"Recorded ranking {items}. "
        "Stop calling tools and respond with the same ranking."
    )


@interface
def react_solve(context: str, question: str, candidates: list) -> str:
    """Solve recommendation ranking using ReAct with evidence / induced tools.

    Interface stub — at runtime, bound to the `react_pydantic` factory
    via YAML config. The factory builds the prompt and supplies the
    tool set (search / lookup / search_memory / produce_ranking, or
    induced ptools, depending on the config).

    Return: space-separated item ids in ranked order.
    """


def react_rank_impl(context: str, question: str, candidates: list) -> list:
    """Direct-method entry point for ReAct configs (MuSR react_answer_impl).

    Recovery order:
      1. If the agent called produce_ranking(), use the recorded list.
      2. Else, parse item ids found in the agent's string output that
         appear in `candidates`.
      3. Else, return the original candidates order (treated as wrong
         by Hit@1 unless luck).
    """
    _reset_react_state(context, candidates)
    raw: str | None = None
    try:
        raw = react_solve(context, question, candidates)
    except Exception:
        pass
    final = _REACT_STATE['finish_ranking']
    if final is not None:
        return list(final)
    # Fallback: pull candidate ids mentioned in the raw string, in order.
    cand_set = {str(c) for c in candidates}
    if isinstance(raw, str):
        tokens = [
            t.strip().strip(',')
            for t in raw.replace(',', ' ').split()
            if t.strip()
        ]
        found = [t for t in tokens if t in cand_set]
        if found:
            # Deduplicate preserving order, then append missing candidates.
            seen = set()
            ordered = []
            for t in found:
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
            for c in candidates:
                cs = str(c)
                if cs not in seen:
                    ordered.append(cs)
            return ordered
    return [str(c) for c in candidates]


# Register ReAct factories with the recommendation preamble.
register_factory(
    'react_pydantic',
    ReactPydanticFactory(preamble=_REC_REACT_PREAMBLE),
)
register_factory(
    'react_pydantic_with_workflow',
    ReactPydanticWithWorkflowFactory(preamble=_REC_REACT_PREAMBLE),
)
