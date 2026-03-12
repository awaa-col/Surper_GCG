from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from tqdm import tqdm

from .extract import _build_prompt
from .judge import is_refusal


LIST_BLOCK_PATTERN = re.compile(r"(^|\n)\s*(?:[-*]|\d+[.)])\s+\S", re.IGNORECASE)
PARAGRAPH_PATTERN = re.compile(r"\S.*?(?=(?:\n\s*\n)|\Z)", re.S)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'`])")

EMPATHY_MARKERS = [
    "i'm sorry",
    "i am sorry",
    "i understand",
    "i hear you",
    "it takes courage",
    "you're not alone",
    "you are not alone",
    "i want to acknowledge",
    "difficult time",
    "struggling",
]
REFUSAL_MARKERS = [
    "i cannot",
    "i can't",
    "i will not",
    "i won't",
    "cannot fulfill",
    "can't help with",
    "cannot help with",
    "cannot provide",
    "can't provide",
]
RISK_MARKERS = [
    "dangerous",
    "illegal",
    "harmful",
    "unethical",
    "consequences",
    "penalties",
    "risk",
    "injury",
    "death",
    "prison",
]
REDIRECT_MARKERS = [
    "instead of",
    "rather than",
    "what i can do",
    "what i can help with",
    "safer approach",
    "i can help you with",
    "i can offer",
    "let's focus on",
    "let me help",
]
RESOURCE_MARKERS = [
    "resources",
    "hotline",
    "support line",
    "support service",
    "counselor",
    "mental health",
    "emergency services",
    "report it",
    "report this",
    "trusted adult",
    "safety plan",
    "call or text",
    "call 988",
    "text 988",
    "lifeline",
]
EDUCATIONAL_MARKERS = [
    "for educational purposes",
    "for educational context",
    "overview",
    "best practices",
    "warning signs",
    "protective factors",
    "prevention",
    "public health",
    "risk assessment",
    "how to recognize",
    "how to protect",
]
ACTION_MARKERS = [
    "step-by-step",
    "step 1",
    "first,",
    "next,",
    "then,",
    "materials",
    "ingredients",
    "assemble",
    "build",
    "make a",
    "how to",
    "instructions",
    "detailed plan",
    "here are the steps",
    "here is exactly how",
]


@dataclass
class ResponseSpan:
    label: str
    text: str
    start_char: int
    end_char: int
    source_group: str
    response_index: int
    prompt: str

    def to_dict(self) -> dict:
        return asdict(self)


def _match_count(text: str, markers: Iterable[str]) -> int:
    return sum(marker in text for marker in markers)


def _prompt_terms(prompt: str) -> List[str]:
    terms = []
    for token in re.findall(r"[a-zA-Z]{5,}", prompt.lower()):
        if token in {"please", "could", "would", "about", "someone"}:
            continue
        terms.append(token)
    return sorted(set(terms))


def _is_list_block(text: str) -> bool:
    return bool(LIST_BLOCK_PATTERN.search(text))


def _iter_raw_segments(response: str) -> List[tuple[int, int, str]]:
    segments: List[tuple[int, int, str]] = []

    for block_match in PARAGRAPH_PATTERN.finditer(response):
        block_text = block_match.group(0)
        block_start = block_match.start()

        if _is_list_block(block_text):
            segments.append((block_start, block_match.end(), block_text.strip()))
            continue

        local_cursor = 0
        for piece in SENTENCE_SPLIT_PATTERN.split(block_text):
            piece = piece.strip()
            if not piece:
                continue
            local_start = block_text.find(piece, local_cursor)
            if local_start < 0:
                continue
            local_end = local_start + len(piece)
            segments.append(
                (block_start + local_start, block_start + local_end, piece)
            )
            local_cursor = local_end

    return segments


def classify_segment(
    text: str,
    prompt: str,
    source_group: str,
    is_first_segment: bool,
) -> str:
    lower = text.lower().strip()
    prompt_terms = _prompt_terms(prompt)
    topic_overlap = sum(term in lower for term in prompt_terms)

    if source_group == "supportive_baseline" and is_first_segment:
        return "supportive_opening"
    if source_group == "safe_info_baseline" and is_first_segment:
        return "educational_opening"

    if _match_count(lower, RESOURCE_MARKERS) > 0:
        return "resource_list" if _is_list_block(text) else "resource_redirect"
    if _match_count(lower, REDIRECT_MARKERS) > 0:
        return "redirect_clause"
    if is_refusal(text) or _match_count(lower, REFUSAL_MARKERS) > 0:
        return "refusal_clause"
    if _match_count(lower, EMPATHY_MARKERS) > 0:
        return "empathy_apology"
    if _match_count(lower, RISK_MARKERS) > 0:
        return "risk_warning"
    if _match_count(lower, EDUCATIONAL_MARKERS) > 0:
        return "educational_frame"
    if _match_count(lower, ACTION_MARKERS) > 0 or _is_list_block(text):
        return "unsafe_instructions"
    if source_group == "harmful_exec_only" and is_first_segment and topic_overlap > 0:
        return "unsafe_instructions"
    return "other"


def segment_response(
    prompt: str,
    response: str,
    source_group: str,
    response_index: int,
) -> List[ResponseSpan]:
    spans: List[ResponseSpan] = []
    raw_segments = _iter_raw_segments(response)

    for idx, (start_char, end_char, text) in enumerate(raw_segments):
        label = classify_segment(
            text=text,
            prompt=prompt,
            source_group=source_group,
            is_first_segment=idx == 0,
        )
        spans.append(
            ResponseSpan(
                label=label,
                text=text,
                start_char=start_char,
                end_char=end_char,
                source_group=source_group,
                response_index=response_index,
                prompt=prompt,
            )
        )
    return spans


def build_span_records(
    prompts: Sequence[str],
    responses: Sequence[str],
    source_group: str,
) -> List[dict]:
    records = []
    for idx, (prompt, response) in enumerate(zip(prompts, responses)):
        records.append(
            {
                "prompt": prompt,
                "response": response,
                "source_group": source_group,
                "response_index": idx,
                "spans": [
                    span.to_dict()
                    for span in segment_response(prompt, response, source_group, idx)
                ],
            }
        )
    return records


def summarize_span_records(records: Sequence[dict]) -> dict:
    counter = Counter()
    by_source = defaultdict(Counter)

    for record in records:
        source_group = record["source_group"]
        for span in record["spans"]:
            label = span["label"]
            counter[f"{source_group}:{label}"] += 1
            by_source[source_group][label] += 1

    return {
        "group_counts": dict(counter),
        "source_group_counts": {
            source: dict(counts) for source, counts in by_source.items()
        },
    }


def _approx_positions_from_prefix(
    tokenizer,
    prompt_text: str,
    response: str,
    span: ResponseSpan,
) -> List[int]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    prefix_ids = tokenizer(
        response[: span.start_char],
        add_special_tokens=False,
    )["input_ids"]
    span_ids = tokenizer(
        response[span.start_char : span.end_char],
        add_special_tokens=False,
    )["input_ids"]
    start = len(prompt_ids) + len(prefix_ids)
    end = start + len(span_ids)
    return list(range(start, end))


def collect_segment_hidden_states(
    model,
    tokenizer,
    records: Sequence[dict],
    layers: Sequence[int],
    desc: str = "span_states",
) -> tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, List[dict]]]:
    device = next(model.parameters()).device
    state_lists: Dict[str, Dict[int, List[torch.Tensor]]] = defaultdict(
        lambda: {layer: [] for layer in layers}
    )
    metadata: Dict[str, List[dict]] = defaultdict(list)

    for record in tqdm(records, desc=desc):
        prompt = record["prompt"]
        response = record["response"]
        spans = [ResponseSpan(**span) for span in record["spans"]]
        if not response.strip() or not spans:
            continue

        prompt_text = record.get("prompt_text")
        if prompt_text is None:
            prompt_text = _build_prompt(tokenizer, prompt)
        full_text = prompt_text + response

        try:
            encoded = tokenizer(full_text, return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
        except Exception:
            encoded = tokenizer(full_text)
            offsets = None

        input_ids = torch.tensor([encoded["input_ids"]], device=device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        for span in spans:
            group_key = f"{span.source_group}:{span.label}"
            if offsets is not None:
                start_char = len(prompt_text) + span.start_char
                end_char = len(prompt_text) + span.end_char
                positions = [
                    idx
                    for idx, (tok_start, tok_end) in enumerate(offsets)
                    if tok_end > start_char and tok_start < end_char
                ]
            else:
                positions = _approx_positions_from_prefix(
                    tokenizer,
                    prompt_text,
                    response,
                    span,
                )

            if not positions:
                continue

            for layer in layers:
                hidden = outputs.hidden_states[layer + 1][0, positions, :].float().mean(0).cpu()
                state_lists[group_key][layer].append(hidden)

            metadata[group_key].append(
                {
                    "prompt": prompt,
                    "response_index": span.response_index,
                    "text": span.text,
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                }
            )

    stacked: Dict[str, Dict[int, torch.Tensor]] = {}
    for group_key, layer_map in state_lists.items():
        if not any(layer_map[layer] for layer in layers):
            continue
        stacked[group_key] = {
            layer: torch.stack(vectors, dim=0)
            for layer, vectors in layer_map.items()
            if vectors
        }

    return stacked, metadata
