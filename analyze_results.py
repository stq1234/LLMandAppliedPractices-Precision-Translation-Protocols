"""Summarize experiment calls and prepare a deduplicated blind semantic-review queue."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import json
import os
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_experiment import load_dataset, load_jsonl


CONDITION_SPECS = (
    ("direct", "fixed_correct"),
    ("fragments", "fixed_correct"),
    ("pool_aggregate", "fixed_correct"),
    ("pool_select", "fixed_correct"),
    ("direct", "none"),
    ("direct", "retrieved"),
)


def normalize_translation(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def automatic_rules(source: str, translation: str) -> list[dict[str, Any]]:
    """Return lexical diagnostic signals only; these are not semantic labels."""
    source_lower = source.lower()
    rules: list[dict[str, Any]] = []

    numbers = re.findall(r"(?<![A-Za-z])\d+(?:,\d{3})*(?:\.\d+)?", source)
    missing_numbers = [number for number in numbers if number not in translation]
    if numbers:
        rules.append({
            "rule": "ascii_number_literal_retention",
            "signal": "all_present" if not missing_numbers else "some_missing",
            "source_literals": numbers,
            "missing_literals": missing_numbers,
        })

    modal_patterns = [
        (r"\b(?:must not|shall not)\b", "prohibition", ["不得", "禁止", "不能"]),
        (r"\b(?:should not|not recommended)\b", "negative_recommendation", ["不应", "不建议", "不宜"]),
        (r"(?<!not )\b(?:should|recommended)\b(?!\s+not)", "recommendation", ["应当", "通常应", "建议", "推荐"]),
        (r"\b(?:must|required|shall)\b(?!\s+not)", "obligation", ["必须", "应当"]),
        (r"\b(?:may|optional)\b", "permission", ["可以", "允许", "可", "可能"]),
    ]
    for pattern, name, signals in modal_patterns:
        if re.search(pattern, source_lower):
            found = [signal for signal in signals if signal in translation]
            rules.append({
                "rule": f"modal_lexical_signal:{name}",
                "signal": "present" if found else "missing",
                "observed_signals": found,
            })

    conditional_markers = [
        ("unless", ["除非", "除……外", "除...外"]),
        ("except", ["除", "例外"]),
        ("only if", ["仅当", "只有在", "只有"]),
        ("if and only if", ["当且仅当"]),
        ("provided that", ["前提是", "条件是", "仅当"]),
    ]
    for marker, signals in conditional_markers:
        if marker in source_lower:
            found = [signal for signal in signals if signal in translation]
            rules.append({
                "rule": f"conditional_lexical_signal:{marker.replace(' ', '_')}",
                "signal": "present" if found else "missing",
                "observed_signals": found,
            })

    if re.search(r"\bnot\b|\bno\s+\w+", source_lower):
        signals = ["不", "无", "未", "不得", "不能", "禁止", "没有", "并非"]
        found = [signal for signal in signals if signal in translation]
        rules.append({
            "rule": "negation_lexical_signal",
            "signal": "present" if found else "missing",
            "observed_signals": found,
        })

    return rules


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def _blind_salt(path: Path) -> bytes:
    if path.exists():
        value = path.read_bytes()
        if len(value) < 32:
            raise ValueError("Blind-review salt is invalid; inspect it before continuing")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    value = os.urandom(32)
    with path.open("xb") as stream:
        stream.write(value)
    return value


def _blind_id(salt: bytes, key: tuple[str, str, str]) -> str:
    packed = "\u241f".join(key).encode("utf-8")
    digest = hmac.new(salt, packed, hashlib.sha256).digest()[:10]
    return "B-" + base64.b32encode(digest).decode("ascii").rstrip("=")


def build_blind_material(records: list[dict[str, Any]], out_dir: Path) -> dict[str, int]:
    salt = _blind_salt(out_dir / ".blind_salt")
    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_candidate(record: dict[str, Any], translation: str, condition: dict[str, Any]) -> None:
        if not translation:
            return
        key = (
            normalize_translation(record["source_en"]),
            normalize_translation(record["anchor_zh"]),
            normalize_translation(translation),
        )
        entry = candidates.setdefault(key, {
            "source_en": record["source_en"],
            "anchor_zh": record["anchor_zh"],
            "translation_zh": translation,
            "split": record["split"],
            "blind_id": _blind_id(salt, key),
            "conditions": [],
        })
        if condition not in entry["conditions"]:
            entry["conditions"].append(condition)

    for record in records:
        if record.get("status") != "success" or not record.get("translation_zh"):
            continue
        condition = {"item_id": record["item_id"], "method": record["method"], "context": record["context"]}
        add_candidate(record, record["translation_zh"], condition)
        if record.get("method") == "pool_aggregate":
            for index, candidate in enumerate(record.get("stages", {}).get("candidate_translations", []), 1):
                add_candidate(record, candidate, {
                    "item_id": record["item_id"], "method": "shared_pool_candidate",
                    "context": record["context"], "candidate_index": index,
                })

    entries = sorted(candidates.values(), key=lambda entry: (entry["split"], entry["blind_id"]))
    blind_rows = [
        {"blind_id": entry["blind_id"], "source_en": entry["source_en"],
         "anchor_zh": entry["anchor_zh"], "translation_zh": entry["translation_zh"]}
        for entry in entries
    ]
    mapping_rows = [
        {"blind_id": entry["blind_id"], "conditions": entry["conditions"]}
        for entry in entries
    ]
    auto_rows = [
        {"blind_id": entry["blind_id"], "automatic_rules": automatic_rules(entry["source_en"], entry["translation_zh"])}
        for entry in entries
    ]
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, entry in candidates.items():
        groups[(key[0], key[1])].append(entry["blind_id"])
    disagreement_rows = []
    for (source_norm, anchor_norm), blind_ids in sorted(groups.items()):
        if len(blind_ids) > 1:
            sample = next(entry for entry in entries
                          if normalize_translation(entry["source_en"]) == source_norm
                          and normalize_translation(entry["anchor_zh"]) == anchor_norm)
            disagreement_rows.append({
                "source_en": sample["source_en"], "anchor_zh": sample["anchor_zh"],
                "distinct_blind_ids": sorted(blind_ids), "distinct_translation_count": len(blind_ids),
            })

    id_splits = {entry["blind_id"]: entry["split"] for entry in entries}
    split_blind = {
        split: [row for row in blind_rows if id_splits[row["blind_id"]] == split]
        for split in ("dev", "test")
    }
    split_auto = {
        split: [row for row in auto_rows if id_splits[row["blind_id"]] == split]
        for split in ("dev", "test")
    }
    split_disagreements = {
        split: [row for row in disagreement_rows if any(
            id_splits.get(blind_id) == split for blind_id in row["distinct_blind_ids"]
        )]
        for split in ("dev", "test")
    }
    _write_jsonl(out_dir / "blind_review.jsonl", split_blind["test"])
    _write_jsonl(out_dir / "blind_review_test.jsonl", split_blind["test"])
    _write_jsonl(out_dir / "blind_review_dev.jsonl", split_blind["dev"])
    _write_jsonl(out_dir / "review_mapping.jsonl", mapping_rows)
    _write_jsonl(out_dir / "auto_rule_checks.jsonl", split_auto["test"])
    _write_jsonl(out_dir / "auto_rule_checks_test.jsonl", split_auto["test"])
    _write_jsonl(out_dir / "auto_rule_checks_dev.jsonl", split_auto["dev"])
    _write_jsonl(out_dir / "disagreement_groups.jsonl", split_disagreements["test"])
    _write_jsonl(out_dir / "disagreement_groups_test.jsonl", split_disagreements["test"])
    _write_jsonl(out_dir / "disagreement_groups_dev.jsonl", split_disagreements["dev"])
    return {
        "unique_test_blind_translations": len(split_blind["test"]),
        "unique_dev_blind_translations": len(split_blind["dev"]),
        "test_source_groups_with_differences": len(split_disagreements["test"]),
        "dev_source_groups_with_differences": len(split_disagreements["dev"]),
    }


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def _actual_tokens(record: dict[str, Any], field: str) -> int | None:
    calls = [call for call in record.get("calls", [])
             if not call.get("shared_pool") and not call.get("cache_hit")]
    if not calls:
        return 0
    values = [call.get(field) for call in calls]
    if any(value is None for value in values):
        return None
    return sum(values)


def summarize(records: list[dict[str, Any]], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected: dict[str, int] = {
        split: sum(1 for item in items if item["split"] == split) for split in ("dev", "test")
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record.get("split", "unknown"), record.get("method", "unknown"), record.get("context", "unknown"))].append(record)
    rows = []
    for split in ("dev", "test"):
        for method, context in CONDITION_SPECS:
            group = grouped.get((split, method, context), [])
            successes = [row for row in group if row.get("status") == "success"]
            success_ids = {row["item_id"] for row in successes}
            measured = {}
            for field in ("logical_calls", "physical_calls", "prompt_tokens", "completion_tokens",
                          "actual_prompt_tokens", "actual_completion_tokens", "elapsed_seconds", "cache_hits",
                          "shared_pool_calls"):
                if field == "actual_prompt_tokens":
                    values = [_actual_tokens(row, "prompt_tokens") for row in successes]
                elif field == "actual_completion_tokens":
                    values = [_actual_tokens(row, "completion_tokens") for row in successes]
                else:
                    values = [row.get(field) for row in successes]
                values = [value for value in values if isinstance(value, (int, float))]
                measured[f"mean_{field}"] = _mean(values)
            retrieval_rows = [row for row in successes if context == "retrieved"]
            exact_hits = []
            recalls = []
            for row in retrieval_rows:
                trace = row.get("card_trace", {})
                selected = set(trace.get("selected_card_ids", []))
                target = set(trace.get("target_card_ids", []))
                exact_hits.append(float(selected == target))
                recalls.append(len(selected & target) / max(1, len(target)))
            rows.append({
                "split": split, "method": method, "context": context,
                "expected_items": expected[split], "attempted_items": len(group),
                "successful_items": len(successes), "failed_items": sum(1 for row in group if row.get("status") != "success"),
                "coverage": round(len(success_ids) / max(1, expected[split]), 4),
                **measured,
                "retrieval_exact_card_rate": _mean(exact_hits) if context == "retrieved" else None,
                "retrieval_card_recall": _mean(recalls) if context == "retrieved" else None,
                "semantic_review_status": "not_scored",
            })
    return rows


def build_request_inventory(records: list[dict[str, Any]]) -> dict[str, Any]:
    inventory: dict[str, dict[str, Any]] = {}
    for split in ("dev", "test"):
        split_records = [row for row in records if row.get("split") == split]
        actual_calls = [call for row in split_records for call in row.get("calls", [])
                        if not call.get("shared_pool") and not call.get("cache_hit")]
        cache_hits = [call for row in split_records for call in row.get("calls", [])
                      if not call.get("shared_pool") and call.get("cache_hit")]
        inventory[split] = {
            "condition_rows": len(split_records),
            "successful_condition_rows": sum(row.get("status") == "success" for row in split_records),
            "logical_method_calls": sum(row.get("logical_calls", 0) for row in split_records),
            "physical_request_attempts": sum(call.get("physical_calls", 0) for call in actual_calls),
            "successful_network_responses": sum(call.get("status") != "failed" for call in actual_calls),
            "cache_hits": len(cache_hits),
            "actual_prompt_tokens": sum(call["prompt_tokens"] for call in actual_calls
                                         if isinstance(call.get("prompt_tokens"), int))
            if all(isinstance(call.get("prompt_tokens"), int) for call in actual_calls) else None,
            "actual_completion_tokens": sum(call["completion_tokens"] for call in actual_calls
                                             if isinstance(call.get("completion_tokens"), int))
            if all(isinstance(call.get("completion_tokens"), int) for call in actual_calls) else None,
            "actual_models": dict(Counter(call["actual_model"] for call in actual_calls if call.get("actual_model"))),
            "finish_reasons": dict(Counter(call["finish_reason"] or "missing" for call in actual_calls)),
        }
    total_calls = [call for row in records for call in row.get("calls", [])
                   if not call.get("shared_pool") and not call.get("cache_hit")]
    inventory["all"] = {
        "condition_rows": len(records),
        "successful_condition_rows": sum(row.get("status") == "success" for row in records),
        "logical_method_calls": sum(row.get("logical_calls", 0) for row in records),
        "physical_request_attempts": sum(call.get("physical_calls", 0) for call in total_calls),
        "actual_prompt_tokens": sum(call["prompt_tokens"] for call in total_calls if isinstance(call.get("prompt_tokens"), int))
        if all(isinstance(call.get("prompt_tokens"), int) for call in total_calls) else None,
        "actual_completion_tokens": sum(call["completion_tokens"] for call in total_calls if isinstance(call.get("completion_tokens"), int))
        if all(isinstance(call.get("completion_tokens"), int) for call in total_calls) else None,
        "actual_models": dict(Counter(call["actual_model"] for call in total_calls if call.get("actual_model"))),
        "finish_reasons": dict(Counter(call["finish_reason"] or "missing" for call in total_calls)),
    }
    return inventory


def summarize_translation_reuse(mapping_rows: list[dict[str, Any]], id_splits: dict[str, str]) -> dict[str, Any]:
    result = {}
    for split in ("test", "dev"):
        entries = [row for row in mapping_rows if id_splits.get(row["blind_id"]) == split]
        counts = [len(row.get("conditions", [])) for row in entries]
        distinct_methods = [len({condition.get("method") for condition in row.get("conditions", [])}) for row in entries]
        result[split] = {
            "unique_blind_ids": len(entries),
            "condition_occurrences": sum(counts),
            "ids_reused_across_conditions": sum(count > 1 for count in counts),
            "mean_conditions_per_id": _mean(counts),
            "max_conditions_per_id": max(counts, default=0),
            "ids_shared_across_methods": sum(count > 1 for count in distinct_methods),
        }
    return result


def summarize_retrieval(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    items, _ = load_dataset()
    for split in ("dev", "test"):
        expected_items = sum(item["split"] == split for item in items)
        selected = [row for row in records if row.get("split") == split
                    and row.get("method") == "direct" and row.get("context") == "retrieved"
                    and row.get("status") == "success"]
        exact = partial = no_card = target_count = hit_count = 0
        for row in selected:
            trace = row.get("card_trace", {})
            ids = set(trace.get("selected_card_ids", []))
            targets = set(trace.get("target_card_ids", []))
            target_count += len(targets)
            hit_count += len(ids & targets)
            exact += int(ids == targets)
            partial += int(bool(ids & targets) and ids != targets)
            no_card += int(not ids)
        rows.append({
            "split": split, "expected_items": expected_items,
            "retrieval_evaluated": len(selected), "cards_used": len(selected) - no_card,
            "no_card_selected": no_card, "exact_card_set_hits": exact,
            "partial_card_set_hits": partial, "target_cards": target_count,
            "retrieved_target_card_hits": hit_count,
            "exact_rate": round(exact / len(selected), 4) if selected else None,
            "target_card_recall": round(hit_count / target_count, 4) if target_count else None,
            "card_use_rate": round((len(selected) - no_card) / len(selected), 4) if selected else None,
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def write_bar_svg(path: Path, rows: list[dict[str, Any]], metric: str, title: str, axis_label: str) -> None:
    valid = [row for row in rows if row["split"] == "test" and isinstance(row.get(metric), (int, float))]
    width, height = 1440, 620
    left, right, top, bottom = 340, 90, 82, 72
    plot_width = width - left - right
    max_value = max((float(row[metric]) for row in valid), default=1.0)
    max_value = max(max_value, 1e-6) * 1.12
    row_height = 31
    colors = {"none": "#4361ee", "fixed_correct": "#2a9d8f", "retrieved": "#e76f51"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="38" font-family="Arial,sans-serif" font-size="24" font-weight="700">{_escape(title)}</text>',
        f'<text x="{left}" y="63" font-family="Arial,sans-serif" font-size="14" fill="#444">{_escape(axis_label)}</text>',
    ]
    for index, row in enumerate(valid):
        y = top + index * row_height
        label = f"{row['method']} / {row['context']}"
        value = float(row[metric])
        bar_width = value / max_value * plot_width
        parts.extend([
            f'<text x="{left-12}" y="{y+18}" text-anchor="end" font-family="Arial,sans-serif" font-size="13">{_escape(label)}</text>',
            f'<rect x="{left}" y="{y}" width="{max(1,bar_width):.1f}" height="20" rx="3" fill="{colors[row["context"]]}"/>',
            f'<text x="{left+bar_width+8:.1f}" y="{y+16}" font-family="Arial,sans-serif" font-size="13">{value:.2f}</text>',
        ])
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#444"/>')
    parts.append('<text x="20" y="30" font-family="Arial,sans-serif" font-size="12" fill="#555">30 held-out test items; values are descriptive. Semantic quality was not automatically scored.</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def analyze(input_path: Path, out_dir: Path) -> dict[str, Any]:
    records = load_jsonl(input_path)
    items, _cards = load_dataset()
    summary_rows = summarize(records, items)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "summary.csv", summary_rows)
    blind_counts = build_blind_material(records, out_dir)
    mapping_rows = load_jsonl(out_dir / "review_mapping.jsonl")
    blind_ids_split = {row["blind_id"]: "test" for row in load_jsonl(out_dir / "blind_review_test.jsonl")}
    blind_ids_split.update({row["blind_id"]: "dev" for row in load_jsonl(out_dir / "blind_review_dev.jsonl")})
    inventory = build_request_inventory(records)
    reuse = summarize_translation_reuse(mapping_rows, blind_ids_split)
    retrieval = summarize_retrieval(records)
    payload = {
        "condition_count": len(records),
        "successful_conditions": sum(1 for record in records if record.get("status") == "success"),
        "failed_conditions": sum(1 for record in records if record.get("status") != "success"),
        "semantic_scores_present": False,
        "request_inventory": inventory,
        "translation_reuse": reuse,
        "retrieval": retrieval,
        "summary": summary_rows,
        **blind_counts,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "request_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "translation_reuse.json").write_text(json.dumps(reuse, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_dir / "retrieval_analysis.csv", retrieval)
    write_bar_svg(out_dir / "test_coverage.svg", summary_rows, "coverage", "Held-out condition coverage", "Share of 30 test clauses with a completed translation")
    write_bar_svg(out_dir / "test_mean_completion_tokens.svg", summary_rows, "mean_completion_tokens", "Attributed output token use per clause", "Provider-reported completion tokens; shared pool cost is attributed to each postprocessor")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/experiment.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()
    result = analyze(args.input, args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["failed_conditions"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
