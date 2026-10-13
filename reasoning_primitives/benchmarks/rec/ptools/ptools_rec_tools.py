"""Deterministic recommendation domain tools — section-level search, filtering, counting.

These tools parse the flat-text recommendation context using Python (no LLM calls).
They provide more targeted access than the generic search/lookup evidence tools.

Context sections:
  === USER PROFILE ===
  === USER MEMORY (MemRec) ===
  === INTERACTION HISTORY ===
  === COLLABORATIVE EVIDENCE ===
  === CANDIDATE ITEMS ===
"""

from __future__ import annotations

import re
from typing import Optional

SECTION_MEMORY = '=== USER MEMORY (MemRec) ==='
SECTION_HISTORY = '=== INTERACTION HISTORY ==='
SECTION_COLLAB = '=== COLLABORATIVE EVIDENCE ==='
SECTION_CANDIDATES = '=== CANDIDATE ITEMS ==='
SECTION_PROFILE = '=== USER PROFILE ==='

_ALL_SECTIONS = [
    SECTION_PROFILE,
    SECTION_MEMORY,
    SECTION_HISTORY,
    SECTION_COLLAB,
    SECTION_CANDIDATES,
]

_ITEM_LINE_RE = re.compile(
    r'^-\s+\[(?P<att>like|dislike)\]\s+'
    r'item_id=(?P<id>\S+)\s*\|\s*'
    r'title=(?P<title>.*?)\s*\|\s*'
    r'category=(?P<cat>.*?)\s*\|\s*'
    r'description=(?P<desc>.*?)$',
    re.IGNORECASE,
)

_CAND_LINE_RE = re.compile(
    r'^-\s+item_id=(?P<id>\S+)\s*\|\s*'
    r'title=(?P<title>.*?)\s*\|\s*'
    r'category=(?P<cat>.*?)\s*\|\s*'
    r'description=(?P<desc>.*?)$',
    re.IGNORECASE,
)


def _extract_section(context: str, header: str) -> str:
    if header not in context:
        return ''
    start = context.index(header) + len(header)
    rest = context[start:]
    m = re.search(r'\n=== .+ ===\n', rest)
    if m:
        return rest[:m.start()].strip()
    return rest.strip()


def _split_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def search_section(context: str, query: str, max_chars: int = 3000) -> str:
    """Search the recommendation context by SECTION (header + full body).

    Returns up to 2 complete sections whose header or body contains the query
    (case-insensitive). Each section is returned with its header and full content.
    Use this instead of search() when you need the complete section content,
    not isolated line fragments.

    Example: search_section(context, 'INTERACTION HISTORY')
             search_section(context, 'CANDIDATE')
    """
    if not context:
        return "Error: no recommendation context provided."

    sections = []
    current_header = None
    current_body = []

    for line in context.split('\n'):
        if re.match(r'^=== .+ ===\s*$', line.strip()):
            if current_header is not None:
                sections.append((current_header, '\n'.join(current_body).strip()))
            current_header = line.strip()
            current_body = []
        else:
            if current_header is not None:
                current_body.append(line)

    if current_header is not None:
        sections.append((current_header, '\n'.join(current_body).strip()))

    q = query.lower()
    matches = []
    for header, body in sections:
        if q in header.lower() or q in body.lower():
            matches.append((header, body))

    if not matches:
        available = [h for h, _ in sections]
        return f"No sections contain {query!r}. Available sections: {available}"

    out = []
    for header, body in matches[:2]:
        if len(body) > max_chars:
            body = body[:max_chars] + '\n...(truncated)'
        out.append(f"{header}\n{body}")

    tail = ''
    if len(matches) > 2:
        tail = f'\n\n(... {len(matches) - 2} more sections match — refine the query)'
    return '\n\n---\n\n'.join(out) + tail


def search_history(context: str, query: str, attitude: str = "any") -> str:
    """Search ONLY the INTERACTION HISTORY section, optionally filtering by attitude.

    Args:
        context: The full recommendation context string.
        query: Keyword or phrase to search for in history items.
        attitude: Filter by attitude tag — "like", "dislike", or "any" (default).

    Returns up to 8 matching history lines. Each line includes the attitude tag,
    item_id, title, category, and description.
    """
    section = _extract_section(context, SECTION_HISTORY)
    if not section:
        return "No INTERACTION HISTORY section found in context."

    lines = _split_lines(section)
    q = query.lower()

    matches = []
    for line in lines:
        if q and q not in line.lower():
            continue
        if attitude.lower() != "any":
            m = _ITEM_LINE_RE.match(line)
            if m and m.group('att').lower() != attitude.lower():
                continue
        matches.append(line)

    if not matches:
        return f"No history items matching {query!r} with attitude={attitude!r}."

    head = matches[:8]
    body = '\n'.join(f"- {s}" for s in head)
    if len(matches) > 8:
        body += f"\n(... {len(matches) - 8} more matches)"
    return body


def search_collab(context: str, query: str) -> str:
    """Search ONLY the COLLABORATIVE EVIDENCE section.

    Returns up to 5 matching lines about similar/related users.
    """
    section = _extract_section(context, SECTION_COLLAB)
    if not section:
        return "No COLLABORATIVE EVIDENCE section found in context."

    lines = _split_lines(section)
    q = query.lower()

    matches = [ln for ln in lines if q in ln.lower()] if q else lines[:5]

    if not matches:
        preview = lines[:3]
        body = '\n'.join(f"- {s}" for s in preview)
        return f"No collaborative evidence matching {query!r}. Preview:\n{body}"

    head = matches[:5]
    body = '\n'.join(f"- {s}" for s in head)
    if len(matches) > 5:
        body += f"\n(... {len(matches) - 5} more matches)"
    return body


def count_history(context: str, query: str, attitude: str = "any") -> str:
    """Count how many history items match the query, optionally filtered by attitude.

    Returns a structured summary: total count, breakdown by attitude, and
    category distribution of matching items.

    Example: count_history(context, "skincare")
             count_history(context, "moisturizer", attitude="like")
    """
    section = _extract_section(context, SECTION_HISTORY)
    if not section:
        return "No INTERACTION HISTORY section found in context."

    lines = _split_lines(section)
    q = query.lower()

    matches = []
    for line in lines:
        if q and q not in line.lower():
            continue
        if attitude.lower() != "any":
            m = _ITEM_LINE_RE.match(line)
            if m and m.group('att').lower() != attitude.lower():
                continue
        matches.append(line)

    total = len(matches)
    like_count = 0
    dislike_count = 0
    cat_counts: dict[str, int] = {}

    for line in matches:
        m = _ITEM_LINE_RE.match(line)
        if m:
            att = m.group('att').lower()
            if att == 'like':
                like_count += 1
            elif att == 'dislike':
                dislike_count += 1
            cat = m.group('cat').strip()
            if cat:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

    parts = [f"Total matching items: {total}"]
    if total > 0:
        parts.append(f"  Liked: {like_count}, Disliked: {dislike_count}")
        if cat_counts:
            top_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:5]
            cat_str = ', '.join(f"{c}({n})" for c, n in top_cats)
            parts.append(f"  Top categories: {cat_str}")

    return '\n'.join(parts)
