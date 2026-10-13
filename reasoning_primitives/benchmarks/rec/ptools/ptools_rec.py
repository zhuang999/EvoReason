"""Recommendation task interfaces — MuSR ptools_murder.py analogue.

Re-exports react_solve so implement_via_config(ptools_rec, ...) can bind
both react_solve and answer_question from a single module (same pattern
as musr/ptools/ptools_murder.py).

Also provides @interface stubs for extraction tools (LLM-simulated).
These stubs are bound via YAML config with method: simulate.
ReAct wrappers inject _REACT_STATE['context'] automatically.
"""

from secretagent.core import interface
from ptools.ptools_common import (  # noqa: F401
    raw_ranking,
    extract_ranking,
    react_solve,
    _REACT_STATE,
)
from ptools.ptools_rec_tools import (
    SECTION_PROFILE,
    SECTION_HISTORY,
    SECTION_MEMORY,
    SECTION_COLLAB,
    _extract_section,
)


@interface
def answer_question(context: str, question: str, candidates: list) -> list:
    """Rank candidates for the user given the recommendation context.

    Returns a list of item_id strings ordered most → least preferred.
    """


@interface
def extract_user_profile(context: str, focus: str) -> str:
    """Extract structured user profile features related to the focus.

    Analyzes the USER PROFILE and USER MEMORY sections to identify:
    - Preference axes (price sensitivity, brand loyalty, ingredient preferences)
    - Constraints (allergies, budget limits, ethical considerations)
    - Demographic indicators that influence purchase decisions

    Args:
        focus: What aspect of the user profile to focus on
               (e.g., "skincare preferences", "budget sensitivity",
                "brand loyalty patterns").

    Returns: Structured summary of user profile features relevant to the focus.
    """


def solve_extract_user_profile(focus: str) -> str:
    ctx = _REACT_STATE.get('context', '')
    profile = _extract_section(ctx, SECTION_PROFILE)
    memory = _extract_section(ctx, SECTION_MEMORY)
    parts = []
    if profile:
        parts.append(f"=== USER PROFILE ===\n{profile}")
    if memory:
        parts.append(f"=== USER MEMORY (MemRec) ===\n{memory}")
    if not parts:
        return "No user profile or memory data available."
    body = '\n\n'.join(parts)
    return (
        f"[Extract user profile — focus: {focus}]\n"
        f"{body}\n\n"
        f"Analyze the above profile and memory to extract features related to: {focus}"
    )


@interface
def extract_purchase_patterns(context: str, focus: str) -> str:
    """Analyze purchase patterns from the user's interaction history.

    Identifies:
    - Category frequency and concentration
    - Repeat purchase behavior
    - Preference evolution over time
    - Attitude patterns (liked vs. disliked categories)

    Args:
        focus: What pattern to look for
               (e.g., "category preferences", "repeat purchases",
                "brand switching behavior", "seasonal patterns").

    Returns: Structured analysis of purchase patterns relevant to the focus.
    """


def solve_extract_purchase_patterns(focus: str) -> str:
    ctx = _REACT_STATE.get('context', '')
    history = _extract_section(ctx, SECTION_HISTORY)
    if not history:
        return "No interaction history available for pattern analysis."
    return (
        f"[Extract purchase patterns — focus: {focus}]\n"
        f"=== INTERACTION HISTORY ===\n{history}\n\n"
        f"Analyze the above history to identify purchase patterns related to: {focus}"
    )


@interface
def extract_collaborative_insights(context: str, focus: str) -> str:
    """Extract insights from similar users' behavior.

    Analyzes collaborative evidence to find:
    - What similar users prefer
    - Common purchase patterns among similar users
    - Products popular among users with similar profiles

    Args:
        focus: What aspect of collaborative signal to focus on
               (e.g., "similar user preferences", "popular categories",
                "complementary purchases").

    Returns: Structured summary of collaborative insights relevant to the focus.
    """


def solve_extract_collaborative_insights(focus: str) -> str:
    ctx = _REACT_STATE.get('context', '')
    collab = _extract_section(ctx, SECTION_COLLAB)
    if not collab:
        return "No collaborative evidence available."
    return (
        f"[Extract collaborative insights — focus: {focus}]\n"
        f"=== COLLABORATIVE EVIDENCE ===\n{collab}\n\n"
        f"Analyze the above collaborative evidence to extract insights related to: {focus}"
    )


@interface
def detect_preference_contradictions(context: str, focus: str) -> str:
    """Detect contradictions between stated preferences and actual behavior.

    Cross-references USER MEMORY (stated/deduced preferences) against
    INTERACTION HISTORY (actual purchase behavior) to find inconsistencies:
    - Stated preference vs. behavioral pattern mismatch
    - Long-term preference vs. recent behavior shift
    - Implicit vs. explicit signal conflicts

    Args:
        focus: What type of contradiction to look for
               (e.g., "stated vs actual", "preference shift",
                "category contradiction", "price behavior mismatch").

    Returns: Structured analysis of contradictions found, with evidence from both sources.
    """


def solve_detect_preference_contradictions(focus: str) -> str:
    ctx = _REACT_STATE.get('context', '')
    memory = _extract_section(ctx, SECTION_MEMORY)
    history = _extract_section(ctx, SECTION_HISTORY)
    if not memory and not history:
        return "No memory or history data available for contradiction analysis."
    parts = []
    if memory:
        parts.append(f"=== USER MEMORY (MemRec) ===\n{memory}")
    if history:
        parts.append(f"=== INTERACTION HISTORY ===\n{history}")
    body = '\n\n'.join(parts)
    return (
        f"[Detect preference contradictions — focus: {focus}]\n"
        f"{body}\n\n"
        f"Cross-reference the MEMORY (stated/deduced preferences) with the HISTORY "
        f"(actual behavior) to find contradictions related to: {focus}"
    )


@interface
def infer_user_intent(context: str, focus: str) -> str:
    """Infer the user's latent purchase intent from gathered evidence.

    Synthesizes all available signals (profile, memory, history, collaborative)
    to reason about what the user likely wants next:
    - Unmet needs suggested by purchase patterns
    - Routine gaps (products the user should be repurchasing)
    - Complementary products to recent purchases
    - Preference evolution direction

    Args:
        focus: What intent aspect to focus on
               (e.g., "next likely category", "unmet needs",
                "complementary products", "routine replenishment").

    Returns: Structured inference of user intent with supporting evidence.
    """


def solve_infer_user_intent(focus: str) -> str:
    ctx = _REACT_STATE.get('context', '')
    profile = _extract_section(ctx, SECTION_PROFILE)
    memory = _extract_section(ctx, SECTION_MEMORY)
    history = _extract_section(ctx, SECTION_HISTORY)
    collab = _extract_section(ctx, SECTION_COLLAB)
    parts = []
    if profile:
        parts.append(f"=== USER PROFILE ===\n{profile}")
    if memory:
        parts.append(f"=== USER MEMORY (MemRec) ===\n{memory}")
    if history:
        parts.append(f"=== INTERACTION HISTORY ===\n{history}")
    if collab:
        parts.append(f"=== COLLABORATIVE EVIDENCE ===\n{collab}")
    if not parts:
        return "No sufficient data available for intent inference."
    body = '\n\n'.join(parts)
    return (
        f"[Infer user intent — focus: {focus}]\n"
        f"{body}\n\n"
        f"Synthesize the above evidence to infer what the user likely wants next, "
        f"focusing on: {focus}"
    )


@interface
def summarize_reasoning_progress(context: str, focus: str) -> str:
    """Summarize what has been discovered so far in the reasoning process.

    Consolidates evidence gathered from previous tool calls into a structured
    summary:
    - Key findings about the user
    - Confirmed preferences and patterns
    - Open questions that still need investigation
    - Confidence level in current understanding

    Use this to organize your thinking before making a final recommendation,
    especially after multiple tool calls.

    Args:
        focus: What aspect of the reasoning to summarize
               (e.g., "all findings", "preference summary",
                "remaining unknowns", "confidence assessment").

    Returns: Structured summary of reasoning progress.
    """


def solve_summarize_reasoning_progress(focus: str) -> str:
    ctx = _REACT_STATE.get('context', '')
    profile = _extract_section(ctx, SECTION_PROFILE)
    memory = _extract_section(ctx, SECTION_MEMORY)
    history = _extract_section(ctx, SECTION_HISTORY)
    collab = _extract_section(ctx, SECTION_COLLAB)
    parts = []
    if profile:
        parts.append(f"=== USER PROFILE ===\n{profile}")
    if memory:
        parts.append(f"=== USER MEMORY (MemRec) ===\n{memory}")
    if history:
        parts.append(f"=== INTERACTION HISTORY ===\n{history}")
    if collab:
        parts.append(f"=== COLLABORATIVE EVIDENCE ===\n{collab}")
    if not parts:
        return "No data available for summarization."
    body = '\n\n'.join(parts)
    return (
        f"[Summarize reasoning progress — focus: {focus}]\n"
        f"{body}\n\n"
        f"Summarize what has been discovered so far, focusing on: {focus}"
    )
