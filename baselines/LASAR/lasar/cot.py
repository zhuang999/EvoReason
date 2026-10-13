from __future__ import annotations
import re
import numpy as np
TEACHER_PROMPT = "You are generating latent-reasoning supervision for sequential recommendation.\nGiven the user's chronological item history and target item below, write 3-8 concise semantic steps:\n1. extract behavioral evidence;\n2. model latent preferences;\n3. infer current intent;\n4. formulate the recommendation;\n5. optionally rewrite a denoised preference sequence.\nDo not expose unsupported facts. Put each step on a new line starting with 'Step N:'.\n\nHistory:\n{description}\n\nTarget SID: {groundtruth}\nTarget title: {title}\nTarget categories: {categories}\n"
STEP_PATTERN = re.compile('(?im)^\\s*(?:step\\s+\\d+|\\d+)[.:)\\-]\\s*')

def split_structured_cot(text: str, max_segments: int=8, min_segments: int=1) -> list[str]:
    if not 1 <= min_segments <= max_segments:
        raise ValueError('Require 1 <= min_segments <= max_segments')
    matches = list(STEP_PATTERN.finditer(text))
    if matches:
        segments = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            segment = text[match.end():end].strip()
            if segment:
                segments.append(segment)
    else:
        segments = [part.strip() for part in re.split('\\n\\s*\\n|\\n', text) if part.strip()]
    if not segments:
        raise ValueError('CoT response contains no semantic segments')
    while len(segments) < min_segments:
        index = max(range(len(segments)), key=lambda item: len(segments[item]))
        parts = [part.strip() for part in re.split('(?<=[.!?])\\s+', segments[index]) if part.strip()]
        if len(parts) < 2:
            words = segments[index].split()
            if len(words) < 2:
                raise ValueError('CoT response cannot satisfy the minimum segment count')
            midpoint = len(words) // 2
            parts = [' '.join(words[:midpoint]), ' '.join(words[midpoint:])]
        split_at = max(1, len(parts) // 2)
        replacement = [' '.join(parts[:split_at]), ' '.join(parts[split_at:])]
        segments = segments[:index] + replacement + segments[index + 1:]
    if len(segments) > max_segments:
        segments = segments[:max_segments - 1] + [' '.join(segments[max_segments - 1:])]
    return segments

def merge_semantic_boundaries(segments: list[str], embeddings, threshold: float, min_segments: int=3, max_segments: int=8) -> tuple[list[str], list[float]]:
    if not segments:
        raise ValueError('segments must not be empty')
    if not 1 <= min_segments <= max_segments:
        raise ValueError('Require 1 <= min_segments <= max_segments')
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(segments):
        raise ValueError('embeddings must have one vector per segment')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    similarities = np.sum(vectors[:-1] * vectors[1:], axis=1).tolist()
    if not similarities:
        return (segments[:], [])
    boundaries = {index for index, score in enumerate(similarities) if score < threshold}
    ranked = sorted(range(len(similarities)), key=lambda index: (similarities[index], index))
    required = min(max(min_segments - 1, 0), len(similarities))
    boundaries.update(ranked[:required])
    allowed = min(max_segments - 1, len(similarities))
    if len(boundaries) > allowed:
        boundaries = set(ranked[:allowed])
    merged = []
    start = 0
    for boundary in sorted(boundaries):
        merged.append(' '.join(segments[start:boundary + 1]))
        start = boundary + 1
    merged.append(' '.join(segments[start:]))
    return (merged, similarities)
