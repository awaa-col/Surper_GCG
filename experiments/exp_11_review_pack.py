"""
experiments/exp_11_review_pack.py
=================================
统一人工复核包导出脚本。

定位：
  - 这是新的 Exp11。
  - 先统一标签口径，再推进后续稳健性和机制实验。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from probes.review_labels import label_response


DEFAULT_INPUTS = [
    "results/exp03_system.json",
    "results/exp05_dbdi.json",
    "results/exp07_identity.json",
    "results/exp09_svd_identity.json",
    "results/exp09b_identity_compete.json",
    "results/exp10_suicide_contrast.json",
]


def iter_prompt_response_pairs(
    obj,
    path: str = "",
    current_prompt: str = "",
) -> Iterator[Tuple[str, str, str]]:
    if isinstance(obj, dict):
        prompt = current_prompt
        if isinstance(obj.get("prompt"), str):
            prompt = obj["prompt"]

        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, str) and "response" in key:
                yield prompt, value, new_path
            elif key == "samples" and isinstance(value, list):
                for idx, sample in enumerate(value):
                    if isinstance(sample, str):
                        yield prompt, sample, f"{new_path}[{idx}]"
                    else:
                        yield from iter_prompt_response_pairs(
                            sample,
                            path=f"{new_path}[{idx}]",
                            current_prompt=prompt,
                        )
            else:
                yield from iter_prompt_response_pairs(
                    value,
                    path=new_path,
                    current_prompt=prompt,
                )
        return

    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            yield from iter_prompt_response_pairs(
                item,
                path=f"{path}[{idx}]",
                current_prompt=current_prompt,
            )


def build_review_row(
    source_file: str,
    source_path: str,
    prompt: str,
    response: str,
) -> Dict[str, object]:
    label = label_response(response, prompt=prompt)
    feats = label["features"]
    return {
        "source_file": source_file,
        "source_path": source_path,
        "prompt": prompt,
        "response": response,
        "prompt_available": bool(prompt.strip()),
        "auto_primary_label": label["primary_label"],
        "auto_secondary_flags": "|".join(label["secondary_flags"]),
        "auto_actionable_unsafe": bool(label["actionable_unsafe"]),
        "auto_review_priority": bool(label["review_priority"]),
        "auto_action_score": feats["action_score"],
        "auto_redirect_score": feats["redirect_score"],
        "auto_disclaimer_score": feats["disclaimer_score"],
        "auto_topic_overlap": feats["topic_overlap"],
        "manual_primary_label": "",
        "manual_secondary_label": "",
        "manual_actionable_unsafe": "",
        "manual_notes": "",
    }


def stratified_select(
    rows: List[Dict[str, object]],
    max_per_bucket: int,
    max_total: int,
) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["source_file"]), str(row["auto_primary_label"]))].append(row)

    ordered = []
    for key in sorted(buckets.keys()):
        bucket_rows = sorted(
            buckets[key],
            key=lambda item: (
                not bool(item["prompt_available"]),
                not bool(item["auto_review_priority"]),
                -int(item["auto_action_score"]),
                -int(item["auto_redirect_score"]),
                -len(str(item["response"])),
            ),
        )
        ordered.extend(bucket_rows[:max_per_bucket])

    ordered.sort(
        key=lambda item: (
            not bool(item["auto_review_priority"]),
            item["source_file"],
            item["auto_primary_label"],
            item["source_path"],
        )
    )
    return ordered[:max_total]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--min_response_chars", type=int, default=20)
    parser.add_argument("--max_per_bucket", type=int, default=20)
    parser.add_argument("--max_total", type=int, default=240)
    parser.add_argument("--output_csv", default="results/exp11_review_pack.csv")
    parser.add_argument("--output_jsonl", default="results/exp11_review_pack.jsonl")
    args = parser.parse_args()

    all_rows: List[Dict[str, object]] = []
    for rel_path in tqdm(args.inputs, desc="exp11.files"):
        path = Path(rel_path)
        if not path.exists():
            print(f"[exp_11] Skip missing file: {path}")
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        file_rows = []
        for prompt, response, source_path in iter_prompt_response_pairs(data):
            if not response or len(response.strip()) < args.min_response_chars:
                continue
            file_rows.append(
                build_review_row(
                    source_file=path.name,
                    source_path=source_path,
                    prompt=prompt,
                    response=response,
                )
            )
        all_rows.extend(file_rows)

    rows = stratified_select(
        rows=all_rows,
        max_per_bucket=args.max_per_bucket,
        max_total=args.max_total,
    )

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "source_path",
        "prompt",
        "response",
        "prompt_available",
        "auto_primary_label",
        "auto_secondary_flags",
        "auto_actionable_unsafe",
        "auto_review_priority",
        "auto_action_score",
        "auto_redirect_score",
        "auto_disclaimer_score",
        "auto_topic_overlap",
        "manual_primary_label",
        "manual_secondary_label",
        "manual_actionable_unsafe",
        "manual_notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    jsonl_path = Path(args.output_jsonl)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[exp_11] exported_rows={len(rows)}")
    print(f"[exp_11] csv={csv_path}")
    print(f"[exp_11] jsonl={jsonl_path}")


if __name__ == "__main__":
    main()
