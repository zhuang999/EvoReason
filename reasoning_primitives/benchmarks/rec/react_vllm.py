"""Phase-1 ReAct reasoning for recommendation using vLLM OpenAI API.

Bypasses secretagent/pydantic-ai compatibility issues.
Directly calls vLLM with tool definitions, iterates the tool-calling loop.

Usage:
    python react_vllm.py \
        --data data/rec_test.json \
        --api-bases http://localhost:12350/v1 http://localhost:12351/v1 \
        --model QwQ-32B \
        --max-samples 50 \
        --output results/react_traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
import random
from pathlib import Path
from openai import OpenAI

SECTION_MEMORY = '=== USER MEMORY (MemRec) ==='
SECTION_HISTORY = '=== INTERACTION HISTORY ==='
SECTION_COLLAB = '=== COLLABORATIVE EVIDENCE ==='
SECTION_CANDIDATES = '=== CANDIDATE ITEMS ==='
SECTION_PROFILE = '=== USER PROFILE ==='

_ITEM_LINE_RE = re.compile(
    r'^-\s+\[(?P<att>like|dislike)\]\s+'
    r'item_id=(?P<id>\S+)\s*\|\s*'
    r'title=(?P<title>.*?)\s*\|\s*'
    r'category=(?P<cat>.*?)\s*\|\s*'
    r'description=(?P<desc>.*?)$',
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are a recommendation reasoning agent. Use the provided tools to gather evidence from the user's profile, interaction history, and long-term memory before making a recommendation.

You decide WHICH tools to call, WHEN, and HOW MANY TIMES. There is no fixed workflow — reason freely based on what you discover.

Available tool categories:
1. EVIDENCE SEARCH: search, lookup, search_memory — broad keyword search over the full context
2. TARGETED RETRIEVAL: search_section, search_history, search_collab — search within a specific section only
3. STATISTICS: count_history — count matching history items with breakdowns
4. STRUCTURED ANALYSIS: extract_user_profile, extract_purchase_patterns, extract_collaborative_insights — extract structured reasoning from a specific data source
5. DEEP REASONING: detect_preference_contradictions, infer_user_intent, summarize_reasoning_progress — cross-reference, infer, and consolidate findings
6. FINAL: produce_recommendation — submit your recommendation when done

Guidelines:
- Use targeted retrieval tools (search_history, search_collab, etc.) for precise section-level searches
- Use extract_* tools when you need structured analysis of a specific aspect
- Use count_history to get quantitative summaries of purchase patterns
- You MUST make at least 3 tool calls before calling produce_recommendation
- Reason QUALITATIVELY — do NOT invent numerical scores or similarity metrics
- When you have gathered enough evidence, call produce_recommendation with your reasoning"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the FULL recommendation context for lines containing the query. Returns up to 5 matching lines across all sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword or phrase to search for in the context"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Paginate through context lines containing the string. First call returns first match, subsequent calls with same string return next match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "string": {
                        "type": "string",
                        "description": "String to search for in the context"
                    }
                },
                "required": ["string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search ONLY the MemRec long-term memory section. Use when you need the user's durable preference summary (habits, constraints, taste axes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword to search in the memory section"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_section",
            "description": "Search by SECTION — returns the full content of matching sections (e.g., 'INTERACTION HISTORY', 'CANDIDATE', 'MEMORY'). Use when you need complete section content, not isolated lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword to match section header or body (e.g., 'HISTORY', 'CANDIDATE', 'COLLABORATIVE')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": "Search ONLY the INTERACTION HISTORY section. Optionally filter by attitude (like/dislike). Returns up to 8 matching history items with full details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword to search for in history items"
                    },
                    "attitude": {
                        "type": "string",
                        "description": "Filter by attitude: 'like', 'dislike', or 'any' (default)",
                        "enum": ["like", "dislike", "any"],
                        "default": "any"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_collab",
            "description": "Search ONLY the COLLABORATIVE EVIDENCE section. Returns up to 5 matching lines about similar/related users.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword to search for in collaborative evidence"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_history",
            "description": "Count how many history items match the query. Returns total count, breakdown by attitude (liked/disliked), and category distribution. Deterministic — no LLM involved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword to count in history items"
                    },
                    "attitude": {
                        "type": "string",
                        "description": "Filter by attitude: 'like', 'dislike', or 'any' (default)",
                        "enum": ["like", "dislike", "any"],
                        "default": "any"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_user_profile",
            "description": "Extract structured user profile features (preference axes, constraints, demographics) from USER PROFILE and USER MEMORY sections. Returns the full profile/memory text with analysis prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What aspect to focus on (e.g., 'skincare preferences', 'budget sensitivity', 'brand loyalty')"
                    }
                },
                "required": ["focus"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_purchase_patterns",
            "description": "Analyze purchase patterns from interaction history — category frequency, repeat purchases, attitude patterns, preference evolution. Returns the full history text with analysis prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What pattern to look for (e.g., 'category preferences', 'repeat purchases', 'brand switching')"
                    }
                },
                "required": ["focus"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_collaborative_insights",
            "description": "Extract insights from similar users' behavior — what similar users prefer, common purchase patterns. Returns the full collaborative evidence text with analysis prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What aspect of collaborative signal to focus on (e.g., 'similar user preferences', 'popular categories')"
                    }
                },
                "required": ["focus"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_preference_contradictions",
            "description": "Cross-reference USER MEMORY (stated preferences) against INTERACTION HISTORY (actual behavior) to find contradictions — stated vs actual, preference shifts, implicit vs explicit conflicts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What type of contradiction to look for (e.g., 'stated vs actual', 'preference shift', 'category contradiction')"
                    }
                },
                "required": ["focus"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "infer_user_intent",
            "description": "Synthesize all signals (profile, memory, history, collaborative) to reason about the user's latent purchase intent — unmet needs, routine gaps, complementary products, preference evolution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What intent aspect to focus on (e.g., 'next likely category', 'unmet needs', 'complementary products', 'routine replenishment')"
                    }
                },
                "required": ["focus"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_reasoning_progress",
            "description": "Consolidate evidence gathered so far into a structured summary — key findings, confirmed preferences, open questions, confidence level. Use before making a final recommendation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What aspect to summarize (e.g., 'all findings', 'preference summary', 'remaining unknowns', 'confidence assessment')"
                    }
                },
                "required": ["focus"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "produce_recommendation",
            "description": "Submit your final recommendation. Call EXACTLY ONCE when you have gathered enough evidence. Explain your reasoning freely — describe what you recommend and why.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Your complete reasoning chain: evidence gathered, preferences inferred, and why you recommend this product"
                    },
                    "recommendation": {
                        "type": "string",
                        "description": "The product type or item you recommend and why it fits the user"
                    }
                },
                "required": ["reasoning", "recommendation"]
            }
        }
    }
]


def _split_sentences(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 3:
        return lines
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


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


def execute_tool(name: str, args: dict, context: str) -> str:
    if name == 'search':
        query = args.get('query', '').lower()
        sentences = _split_sentences(context)
        matches = [s for s in sentences if query in s.lower()]
        if not matches:
            return f"No lines contain {args['query']!r}."
        head = matches[:5]
        body = '\n'.join(f"- {s}" for s in head)
        if len(matches) > 5:
            body += f"\n(... {len(matches) - 5} more matches)"
        return body

    elif name == 'lookup':
        string = args.get('string', '').lower()
        sentences = _split_sentences(context)
        matches = [s for s in sentences if string in s.lower()]
        if not matches:
            return f"No lines contain {args['string']!r}."
        state = execute_tool._lookup_state
        key = args.get('string', '')
        if state.get('key') != key:
            state['key'] = key
            state['idx'] = 0
            state['matches'] = matches
        idx = state['idx']
        if idx >= len(state['matches']):
            return f"No more matches for {args['string']!r} ({len(state['matches'])} total)."
        result = state['matches'][idx]
        state['idx'] = idx + 1
        return f"({idx + 1}/{len(state['matches'])}) {result}"

    elif name == 'search_memory':
        query = args.get('query', '').lower()
        section = _extract_section(context, SECTION_MEMORY)
        if not section:
            return "No MemRec memory section found."
        sentences = _split_sentences(section)
        if query:
            matches = [s for s in sentences if query in s.lower()]
        else:
            matches = sentences[:5]
        if not matches:
            preview = sentences[:5]
            body = '\n'.join(f"- {s}" for s in preview)
            return f"No memory lines contain {args.get('query', '')!r}. Memory preview:\n{body}"
        head = matches[:5]
        body = '\n'.join(f"- {s}" for s in head)
        if len(matches) > 5:
            body += f"\n(... {len(matches) - 5} more memory matches)"
        return body

    elif name == 'search_section':
        query = args.get('query', '').lower()
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
        matches = [(h, b) for h, b in sections if query in h.lower() or query in b.lower()]
        if not matches:
            available = [h for h, _ in sections]
            return f"No sections contain {args['query']!r}. Available sections: {available}"
        out = []
        for header, body in matches[:2]:
            if len(body) > 3000:
                body = body[:3000] + '\n...(truncated)'
            out.append(f"{header}\n{body}")
        tail = ''
        if len(matches) > 2:
            tail = f'\n\n(... {len(matches) - 2} more sections match)'
        return '\n\n---\n\n'.join(out) + tail

    elif name == 'search_history':
        query = args.get('query', '').lower()
        attitude = args.get('attitude', 'any').lower()
        section = _extract_section(context, SECTION_HISTORY)
        if not section:
            return "No INTERACTION HISTORY section found."
        lines = _split_lines(section)
        matches = []
        for line in lines:
            if query and query not in line.lower():
                continue
            if attitude != 'any':
                m = _ITEM_LINE_RE.match(line)
                if m and m.group('att').lower() != attitude:
                    continue
            matches.append(line)
        if not matches:
            return f"No history items matching {args['query']!r} with attitude={attitude!r}."
        head = matches[:8]
        body = '\n'.join(f"- {s}" for s in head)
        if len(matches) > 8:
            body += f"\n(... {len(matches) - 8} more matches)"
        return body

    elif name == 'search_collab':
        query = args.get('query', '').lower()
        section = _extract_section(context, SECTION_COLLAB)
        if not section:
            return "No COLLABORATIVE EVIDENCE section found."
        lines = _split_lines(section)
        matches = [ln for ln in lines if query in ln.lower()] if query else lines[:5]
        if not matches:
            return f"No collaborative evidence matching {args['query']!r}."
        head = matches[:5]
        body = '\n'.join(f"- {s}" for s in head)
        if len(matches) > 5:
            body += f"\n(... {len(matches) - 5} more matches)"
        return body

    elif name == 'count_history':
        query = args.get('query', '').lower()
        attitude = args.get('attitude', 'any').lower()
        section = _extract_section(context, SECTION_HISTORY)
        if not section:
            return "No INTERACTION HISTORY section found."
        lines = _split_lines(section)
        matches = []
        for line in lines:
            if query and query not in line.lower():
                continue
            if attitude != 'any':
                m = _ITEM_LINE_RE.match(line)
                if m and m.group('att').lower() != attitude:
                    continue
            matches.append(line)
        total = len(matches)
        like_count = 0
        dislike_count = 0
        cat_counts = {}
        for line in matches:
            m = _ITEM_LINE_RE.match(line)
            if m:
                if m.group('att').lower() == 'like':
                    like_count += 1
                elif m.group('att').lower() == 'dislike':
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

    elif name == 'extract_user_profile':
        focus = args.get('focus', '')
        profile = _extract_section(context, SECTION_PROFILE)
        memory = _extract_section(context, SECTION_MEMORY)
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

    elif name == 'extract_purchase_patterns':
        focus = args.get('focus', '')
        history = _extract_section(context, SECTION_HISTORY)
        if not history:
            return "No interaction history available for pattern analysis."
        return (
            f"[Extract purchase patterns — focus: {focus}]\n"
            f"=== INTERACTION HISTORY ===\n{history}\n\n"
            f"Analyze the above history to identify purchase patterns related to: {focus}"
        )

    elif name == 'extract_collaborative_insights':
        focus = args.get('focus', '')
        collab = _extract_section(context, SECTION_COLLAB)
        if not collab:
            return "No collaborative evidence available."
        return (
            f"[Extract collaborative insights — focus: {focus}]\n"
            f"=== COLLABORATIVE EVIDENCE ===\n{collab}\n\n"
            f"Analyze the above collaborative evidence to extract insights related to: {focus}"
        )

    elif name == 'detect_preference_contradictions':
        focus = args.get('focus', '')
        memory = _extract_section(context, SECTION_MEMORY)
        history = _extract_section(context, SECTION_HISTORY)
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

    elif name == 'infer_user_intent':
        focus = args.get('focus', '')
        profile = _extract_section(context, SECTION_PROFILE)
        memory = _extract_section(context, SECTION_MEMORY)
        history = _extract_section(context, SECTION_HISTORY)
        collab = _extract_section(context, SECTION_COLLAB)
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

    elif name == 'summarize_reasoning_progress':
        focus = args.get('focus', '')
        profile = _extract_section(context, SECTION_PROFILE)
        memory = _extract_section(context, SECTION_MEMORY)
        history = _extract_section(context, SECTION_HISTORY)
        collab = _extract_section(context, SECTION_COLLAB)
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

    elif name == 'produce_recommendation':
        execute_tool._final_rec = {
            'reasoning': args.get('reasoning', ''),
            'recommendation': args.get('recommendation', ''),
        }
        return "Recommendation recorded. Stop calling tools and respond with your final recommendation."

    return f"Unknown tool: {name}"


execute_tool._lookup_state = {}
execute_tool._final_ranking = None
execute_tool._final_rec = None


def run_react(
    client: OpenAI,
    model: str,
    context: str,
    question: str,
    candidates: list,
    max_turns: int = 20,
    max_tokens: int = 4096,
    timeout: int = 300,
) -> dict:
    execute_tool._lookup_state = {}
    execute_tool._final_ranking = None
    execute_tool._final_rec = None

    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    all_steps = []
    total_input = 0
    total_output = 0
    start_time = time.time()
    final_ranking = None

    for turn in range(max_turns):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as e:
            all_steps.append({"error": str(e)})
            break

        choice = resp.choices[0]
        total_input += resp.usage.prompt_tokens if resp.usage else 0
        total_output += resp.usage.completion_tokens if resp.usage else 0

        msg = choice.message

        reasoning = getattr(msg, 'reasoning_content', None) or ''
        content = msg.content or ''

        if reasoning:
            all_steps.append({"reasoning": reasoning})
        if content:
            all_steps.append({"thought": content})

        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                tool_result = execute_tool(fn_name, fn_args, context)
                all_steps.append({
                    "tool_call": fn_name,
                    "args": fn_args,
                    "tool_return": tool_result[:500],
                })
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [{"id": tc.id, "type": "function",
                                                 "function": {"name": fn_name,
                                                              "arguments": tc.function.arguments}}]})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": tool_result})

                if fn_name == 'produce_recommendation':
                    final_ranking = candidates
                    break
            if final_ranking is not None:
                break
        else:
            break

        messages.append({"role": "assistant", "content": msg.content})

    latency = time.time() - start_time

    if final_ranking is None:
        final_ranking = candidates

    rec = execute_tool._final_rec or {}

    return {
        "ranking": final_ranking,
        "recommendation": rec,
        "steps": all_steps,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "latency": latency,
        "n_turns": len(all_steps),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='Input JSON from prepare_data_parquet.py')
    p.add_argument('--api-bases', nargs='+',
                   default=['http://localhost:12350/v1'],
                   help='vLLM API base URLs (round-robin)')
    p.add_argument('--model', default='QwQ-32B')
    p.add_argument('--max-samples', type=int, default=0, help='0 = all')
    p.add_argument('--max-turns', type=int, default=20)
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--output', required=True, help='Output JSONL for traces')
    p.add_argument('--num-workers', type=int, default=0,
                   help='Number of parallel workers (0=len(api_bases))')
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip users already in the output file')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    with open(args.data, encoding='utf-8') as f:
        data = json.load(f)
    examples = data['examples']

    if args.max_samples and args.max_samples < len(examples):
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(len(examples)), args.max_samples))
        examples = [examples[i] for i in indices]

    clients = [OpenAI(base_url=base, api_key='EMPTY') for base in args.api_bases]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_users = set()
    if args.skip_existing and out_path.exists():
        with open(out_path, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_users.add(str(r.get('user_id', '')))
                except Exception:
                    pass
        print(f"Skipping {len(done_users)} already completed users")

    examples = [ex for ex in examples if str(ex['user_id']) not in done_users]
    if not examples:
        print("All examples already completed!")
        return

    num_workers = args.num_workers if args.num_workers > 0 else len(clients)
    print(f"Running ReAct on {len(examples)} samples with {num_workers} parallel workers...")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    write_lock = threading.Lock()
    results_dict = {}
    counter = [0]
    counter_lock = threading.Lock()

    def process_one(idx, ex):
        client = clients[idx % num_workers]
        result = run_react(
            client=client,
            model=args.model,
            context=ex['context'],
            question=ex['question'],
            candidates=ex['candidates'],
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        result['user_id'] = ex['user_id']
        result['answer_item'] = ex['answer_item']
        hit = ex['answer_item'] in result['ranking'][:1] if result['ranking'] else False
        result['hit_at_1'] = hit

        with write_lock:
            with open(out_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        with counter_lock:
            counter[0] += 1
            n = counter[0]

        return idx, result, n

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_one, i, ex): i
            for i, ex in enumerate(examples)
        }
        for future in as_completed(futures):
            idx, result, n = future.result()
            results_dict[idx] = result
            if n % 10 == 0 or n == len(examples):
                hits = sum(r['hit_at_1'] for r in results_dict.values())
                avg_lat = sum(r['latency'] for r in results_dict.values()) / n
                print(f"  [{n}/{len(examples)}] hit@1={hits/n:.3f} avg_latency={avg_lat:.1f}s")

    results = [results_dict[i] for i in range(len(examples))]

    total = len(results)
    hits = sum(r['hit_at_1'] for r in results)
    print(f"\nFinal: {total} samples, hit@1={hits/total:.3f}")
    print(f"Traces saved to {out_path}")


if __name__ == '__main__':
    main()
