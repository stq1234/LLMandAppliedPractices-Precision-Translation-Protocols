"""Map GPT-6.2 Astra's blinded ratings back to experiment conditions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_results import CONDITION_SPECS, _write_jsonl, load_jsonl, normalize_translation, write_bar_svg
from run_experiment import ROOT, load_dataset


ERROR_CODES = {"core", "modality", "scope", "condition", "numeric_time", "term"}


def load_ratings(path: Path) -> dict[str, dict[str, Any]]:
    ratings: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        blind_id = row.get("blind_id")
        acceptable = row.get("acceptable")
        errors = row.get("errors")
        note = row.get("note")
        if not isinstance(blind_id, str) or not blind_id:
            raise ValueError("Every semantic rating needs a blind_id")
        if blind_id in ratings:
            raise ValueError(f"Duplicate rating for {blind_id}")
        if acceptable not in (0, 1) or isinstance(acceptable, bool):
            raise ValueError(f"acceptable must be integer 0 or 1 for {blind_id}")
        if not isinstance(errors, list) or any(code not in ERROR_CODES for code in errors):
            raise ValueError(f"Invalid error short code for {blind_id}")
        if not isinstance(note, str):
            raise ValueError(f"note must be a string for {blind_id}")
        if not isinstance(row.get("extra_candidate_text"), bool):
            raise ValueError(f"extra_candidate_text must be boolean for {blind_id}")
        if not isinstance(row.get("doubled_separator"), bool):
            raise ValueError(f"doubled_separator must be boolean for {blind_id}")
        ratings[blind_id] = row
    return ratings


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rate(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def map_ratings(ratings: dict[str, dict[str, Any]], mapping_rows: list[dict[str, Any]],
                blind_rows: list[dict[str, Any]], experiment_rows: list[dict[str, Any]],
                expected_blind_ids: set[str]) -> dict[str, Any]:
    rated_ids = set(ratings)
    if rated_ids != expected_blind_ids:
        missing = len(expected_blind_ids - rated_ids)
        extra = len(rated_ids - expected_blind_ids)
        raise ValueError(f"Rating coverage mismatch: {missing} blind IDs missing, {extra} unexpected")

    items, _ = load_dataset()
    item_by_id = {item["id"]: item for item in items}
    condition_scores: dict[tuple[str, str, str], dict[str, Any]] = {}
    pool_candidate_scores: dict[tuple[str, int], dict[str, Any]] = {}

    for map_row in mapping_rows:
        blind_id = map_row["blind_id"]
        if blind_id not in ratings:
            continue
        rating = ratings[blind_id]
        for condition in map_row.get("conditions", []):
            item_id = condition["item_id"]
            method = condition["method"]
            context = condition["context"]
            key = (item_id, method, context)
            if method == "shared_pool_candidate":
                index = int(condition["candidate_index"])
                pool_candidate_scores[(item_id, index)] = {"blind_id": blind_id, "rating": rating}
            else:
                condition_scores[key] = {"blind_id": blind_id, "rating": rating}

    result_by_condition = {
        (row["item_id"], row["method"], row["context"]): row
        for row in experiment_rows
    }
    per_condition_rows = []
    condition_tables: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item_id in item_by_id:
        for method, context in CONDITION_SPECS:
            rated = condition_scores.get((item_id, method, context))
            result = result_by_condition.get((item_id, method, context))
            if not rated or not result:
                continue
            rating = rated["rating"]
            strict = rating["acceptable"]
            relaxed_term = int(strict == 1 or set(rating["errors"]) == {"term"})
            deliverable = int(strict == 1 and not rating["extra_candidate_text"] and not rating["doubled_separator"])
            row = {
                "item_id": item_id, "family": item_by_id[item_id]["family"],
                "split": item_by_id[item_id]["split"], "method": method, "context": context,
                "status": result.get("status"), "acceptable": strict,
                "semantic_acceptable": strict, "term_relaxed_acceptable": relaxed_term,
                "deliverable": deliverable, "errors": rating["errors"], "blind_id": rated["blind_id"],
                "extra_candidate_text": rating["extra_candidate_text"],
                "doubled_separator": rating["doubled_separator"],
            }
            per_condition_rows.append(row)
            condition_tables[(row["split"], method, context)].append(row)

    summary_rows = []
    paired_rows = []
    family_paired_rows = []
    for split in ("test", "dev"):
        for method, context in CONDITION_SPECS:
            scored = condition_tables.get((split, method, context), [])
            successes = [row for row in scored if row["status"] == "success"]
            summary_row = {
                "split": split, "method": method, "context": context,
                "expected_items": sum(1 for item in items if item["split"] == split),
                "rated_items": len(successes),
            }
            for label, field in (("semantic", "semantic_acceptable"),
                                 ("term_relaxed", "term_relaxed_acceptable"),
                                 ("deliverable", "deliverable")):
                values = [row[field] for row in successes]
                summary_row[f"{label}_acceptable"] = sum(values)
                summary_row[f"{label}_acceptable_rate"] = _rate(values)
            summary_rows.append(summary_row)
        baseline = {
            row["item_id"]: row
            for row in condition_tables.get((split, "direct", "fixed_correct"), [])
            if row["status"] == "success"
        }
        for method, context in CONDITION_SPECS:
            compared = condition_tables.get((split, method, context), [])
            for label, field in (("semantic", "semantic_acceptable"),
                                 ("term_relaxed", "term_relaxed_acceptable"),
                                 ("deliverable", "deliverable")):
                pairs = [(baseline[row["item_id"]][field], row[field]) for row in compared
                         if row["status"] == "success" and row["item_id"] in baseline]
                wins = sum(1 for base, current in pairs if current > base)
                losses = sum(1 for base, current in pairs if current < base)
                paired_rows.append({
                    "split": split, "criterion": label, "baseline": "direct/fixed_correct",
                    "method": method, "context": context,
                    "n_paired": len(pairs), "wins_vs_direct": wins, "losses_vs_direct": losses,
                    "ties": len(pairs) - wins - losses,
                    "mean_binary_gain": round((wins - losses) / len(pairs), 4) if pairs else None,
                })
                for family in sorted({item["family"] for item in items if item["split"] == split}):
                    family_pairs = [
                        (baseline[row["item_id"]][field], row[field]) for row in compared
                        if row["status"] == "success" and row["item_id"] in baseline
                        and item_by_id[row["item_id"]]["family"] == family
                    ]
                    family_wins = sum(1 for base, current in family_pairs if current > base)
                    family_losses = sum(1 for base, current in family_pairs if current < base)
                    family_paired_rows.append({
                        "split": split, "family": family, "criterion": label,
                        "method": method, "context": context, "n_paired": len(family_pairs),
                        "wins": family_wins, "losses": family_losses,
                        "ties": len(family_pairs) - family_wins - family_losses,
                        "net_gain": family_wins - family_losses,
                    })

    family_errors: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    family_denoms: Counter[tuple[str, str, str]] = Counter()
    family_error_counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in per_condition_rows:
        if row["status"] != "success":
            continue
        split, family, method = row["split"], row["family"], row["method"]
        family_denoms[(split, family, method)] += 1
        for code in set(row["errors"]):
            family_error_counts[(split, family, method, code)] += 1
    for item_id, row in item_by_id.items():
        family = row["family"]
        split = row["split"]
        for index in (1, 2, 3):
            rated = pool_candidate_scores.get((item_id, index))
            if not rated:
                continue
            family_denoms[(split, family, "pool_candidate")] += 1
            for code in set(rated["rating"]["errors"]):
                family_error_counts[(split, family, "pool_candidate", code)] += 1
    family_error_rows = []
    for (split, family, method), denominator in sorted(family_denoms.items()):
        codes = sorted(ERROR_CODES)
        for code in codes:
            count = family_error_counts[(split, family, method, code)]
            family_error_rows.append({
                "split": split, "family": family, "method": method,
                "n_rated": denominator, "error_code": code, "items_with_error": count,
                "error_rate": round(count / denominator, 4) if denominator else None,
            })

    unique_ids_by_item: dict[str, set[str]] = defaultdict(set)
    unique_ids_for_family: dict[tuple[str, str], set[str]] = defaultdict(set)
    for map_row in mapping_rows:
        blind_id = map_row["blind_id"]
        for condition in map_row.get("conditions", []):
            item_id = condition.get("item_id")
            if item_id not in item_by_id:
                continue
            unique_ids_by_item[item_id].add(blind_id)
            item = item_by_id[item_id]
            unique_ids_for_family[(item["split"], item["family"])].add(blind_id)
    unique_family_errors = []
    for (split, family), blind_ids in sorted(unique_ids_for_family.items()):
        for code in sorted(ERROR_CODES):
            count = sum(code in ratings[blind_id]["errors"] for blind_id in blind_ids if blind_id in ratings)
            unique_family_errors.append({
                "split": split, "family": family, "unique_translations": len(blind_ids),
                "error_code": code, "translations_with_error": count,
                "error_rate": round(count / len(blind_ids), 4) if blind_ids else None,
            })
    disagreement_rows = []
    for item_id, blind_ids in sorted(unique_ids_by_item.items()):
        values = [ratings[blind_id]["acceptable"] for blind_id in blind_ids if blind_id in ratings]
        if 0 in values and 1 in values:
            disagreement_rows.append({
                "split": item_by_id[item_id]["split"], "family": item_by_id[item_id]["family"],
                "item_id": item_id, "unique_translations": len(values),
                "acceptable_translations": sum(values), "unacceptable_translations": len(values) - sum(values),
            })

    pool_rows = []
    selector_distribution: Counter[str] = Counter()
    for split in ("test", "dev"):
        split_items = [item for item in items if item["split"] == split]
        complete_items = []
        aggregate_values: list[int] = []
        select_values: list[int] = []
        covered_count = 0
        raw_acceptable = raw_total = 0
        selected_acceptable = selected_matches_candidate = 0
        selected_total = 0
        selector_acceptable_when_covered = aggregate_recovers_uncovered = aggregate_drops_pool_acceptable = 0
        aggregate_rewritten = aggregate_rewritten_acceptable = 0
        for item in split_items:
            item_id = item["id"]
            candidate_infos = [pool_candidate_scores.get((item_id, index)) for index in (1, 2, 3)]
            if not all(candidate_infos):
                continue
            candidate_ratings = [entry["rating"] for entry in candidate_infos]
            complete_items.append(item_id)
            values = [rating["acceptable"] for rating in candidate_ratings]
            raw_acceptable += sum(values)
            raw_total += len(values)
            has_acceptable = any(values)
            covered_count += int(has_acceptable)
            aggregate_info = condition_scores.get((item_id, "pool_aggregate", "fixed_correct"))
            selected_info = condition_scores.get((item_id, "pool_select", "fixed_correct"))
            aggregate = aggregate_info["rating"] if aggregate_info else None
            selected = selected_info["rating"] if selected_info else None
            if aggregate:
                aggregate_values.append(aggregate["acceptable"])
            if selected:
                select_values.append(selected["acceptable"])
            if has_acceptable and aggregate and aggregate["acceptable"] == 0:
                aggregate_drops_pool_acceptable += 1
            if not has_acceptable and aggregate and aggregate["acceptable"] == 1:
                aggregate_recovers_uncovered += 1
            if has_acceptable and selected and selected["acceptable"] == 1:
                selector_acceptable_when_covered += 1
            output = result_by_condition.get((item_id, "pool_select", "fixed_correct"), {})
            aggregate_output = result_by_condition.get((item_id, "pool_aggregate", "fixed_correct"), {})
            aggregate_translation = normalize_translation(aggregate_output.get("translation_zh", ""))
            pool_translations = [normalize_translation(value) for value in aggregate_output.get("stages", {}).get("candidate_translations", [])]
            is_rewrite = bool(aggregate_translation and aggregate_translation not in pool_translations)
            if is_rewrite and aggregate:
                aggregate_rewritten += 1
                aggregate_rewritten_acceptable += aggregate["acceptable"]
            selected_index = output.get("stages", {}).get("selected_candidate")
            if selected_index in (1, 2, 3) and selected:
                selector_distribution[f"{split}:{selected_index}"] += 1
                selected_total += 1
                selected_candidate = candidate_ratings[selected_index - 1]
                selected_acceptable += selected_candidate["acceptable"]
                selected_matches_candidate += int(selected_candidate["acceptable"] == selected["acceptable"])
        pool_rows.append({
            "split": split, "expected_items": len(split_items), "complete_pool_items": len(complete_items),
            "raw_candidate_count": raw_total,
            "raw_candidate_acceptable_rate": _rate([
                pool_candidate_scores[(item_id, index)]["rating"]["acceptable"]
                for item_id in complete_items for index in (1, 2, 3)
            ]),
            "pool_coverage_items": covered_count,
            "pool_coverage_rate": round(covered_count / len(split_items), 4) if split_items else None,
            "aggregate_acceptable_rate": _rate(aggregate_values),
            "select_acceptable_rate": _rate(select_values),
            "selected_candidate_acceptable_rate": round(selected_acceptable / selected_total, 4) if selected_total else None,
            "selection_rating_consistency_count": selected_matches_candidate,
            "selector_acceptable_when_pool_covered": selector_acceptable_when_covered,
            "aggregate_recovers_when_pool_has_no_acceptable_candidate": aggregate_recovers_uncovered,
            "aggregate_rejects_despite_acceptable_pool": aggregate_drops_pool_acceptable,
            "aggregate_rewritten_items": aggregate_rewritten,
            "aggregate_rewritten_acceptable": aggregate_rewritten_acceptable,
            "aggregate_rewritten_acceptable_rate": round(aggregate_rewritten_acceptable / aggregate_rewritten, 4)
            if aggregate_rewritten else None,
            "selection_count": selected_total,
        })

    run_summary_path = ROOT / "results" / "analysis" / "summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8")) if run_summary_path.exists() else {}
    costs = {(cost["split"], cost["method"], cost["context"]): cost
             for cost in run_summary.get("summary", [])}
    cost_summary = []
    for row in summary_rows:
        matching_cost = costs.get((row["split"], row["method"], row["context"]), {})
        tokens = matching_cost.get("mean_completion_tokens")
        actual_tokens = matching_cost.get("mean_actual_completion_tokens")
        physical = matching_cost.get("mean_physical_calls")
        semantic_rate = row["semantic_acceptable_rate"]
        relaxed_rate = row["term_relaxed_acceptable_rate"]
        deliverable_rate = row["deliverable_acceptable_rate"]
        cost_summary.append({
            **row, "mean_logical_completion_tokens": tokens,
            "mean_actual_completion_tokens": actual_tokens, "mean_physical_calls": physical,
            "logical_completion_tokens_per_semantic_acceptable": round(tokens / semantic_rate, 4)
            if tokens is not None and semantic_rate else None,
            "actual_completion_tokens_per_semantic_acceptable": round(actual_tokens / semantic_rate, 4)
            if actual_tokens is not None and semantic_rate else None,
            "logical_completion_tokens_per_deliverable": round(tokens / deliverable_rate, 4)
            if tokens is not None and deliverable_rate else None,
            "term_relaxed_acceptable_rate": relaxed_rate,
        })

    rating_scope = {}
    for split in ("all", "test", "dev"):
        selected_ratings = list(ratings.values()) if split == "all" else [
            ratings[row["blind_id"]] for row in blind_rows if row.get("split") == split
        ]
        strict_values = [row["acceptable"] for row in selected_ratings]
        relaxed_values = [int(row["acceptable"] == 1 or set(row["errors"]) == {"term"}) for row in selected_ratings]
        delivery_values = [int(row["acceptable"] == 1 and not row["extra_candidate_text"] and not row["doubled_separator"])
                           for row in selected_ratings]
        rating_scope[split] = {
            "unique_pairs": len(selected_ratings),
            "semantic_acceptable": sum(strict_values), "semantic_acceptability_rate": _rate(strict_values),
            "term_relaxed_acceptable": sum(relaxed_values), "term_relaxed_rate": _rate(relaxed_values),
            "deliverable_acceptable": sum(delivery_values), "deliverable_rate": _rate(delivery_values),
            "extra_candidate_text": sum(row["extra_candidate_text"] for row in selected_ratings),
            "doubled_separator": sum(row["doubled_separator"] for row in selected_ratings),
        }

    return {
        "expected_blind_ids": len(expected_blind_ids), "rated_blind_ids": len(rated_ids),
        "rating_scope": rating_scope,
        "condition_ratings": per_condition_rows, "condition_summary": summary_rows,
        "paired_vs_direct": paired_rows, "family_paired_vs_direct": family_paired_rows,
        "family_errors": family_error_rows, "unique_family_errors": unique_family_errors,
        "semantic_disagreement_groups": disagreement_rows,
        "pool_analysis": pool_rows, "selection_distribution": dict(selector_distribution),
        "acceptability_cost": cost_summary,
    }


def analyze_semantics(ratings_path: Path, result_path: Path, analysis_dir: Path, out_dir: Path) -> dict[str, Any]:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rating_rows = load_ratings(ratings_path)
    mapping_rows = load_jsonl(analysis_dir / "review_mapping.jsonl")
    test_blind = [{**row, "split": "test"} for row in load_jsonl(analysis_dir / "blind_review.jsonl")]
    dev_blind = [{**row, "split": "dev"} for row in load_jsonl(analysis_dir / "blind_review_dev.jsonl")]
    blind_rows = test_blind + dev_blind
    expected_ids = {row["blind_id"] for row in blind_rows}
    output = map_ratings(rating_rows, mapping_rows, blind_rows, load_jsonl(result_path), expected_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "condition_acceptability.csv", output["condition_summary"])
    _write_csv(out_dir / "paired_vs_direct.csv", output["paired_vs_direct"])
    _write_csv(out_dir / "family_paired_vs_direct.csv", output["family_paired_vs_direct"])
    _write_csv(out_dir / "family_errors.csv", output["family_errors"])
    _write_csv(out_dir / "unique_family_errors.csv", output["unique_family_errors"])
    _write_csv(out_dir / "pool_analysis.csv", output["pool_analysis"])
    _write_csv(out_dir / "acceptability_cost.csv", output["acceptability_cost"])
    _write_csv(out_dir / "semantic_disagreement_groups.csv", output["semantic_disagreement_groups"])
    _write_jsonl(out_dir / "condition_ratings.jsonl", output["condition_ratings"])
    (out_dir / "semantic_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    test_rows = [row for row in output["condition_summary"] if row["split"] == "test"]
    write_bar_svg(out_dir / "test_acceptability.svg", [
        {**row, "acceptable_rate": row["semantic_acceptable_rate"]} for row in test_rows
    ], "acceptable_rate", "GPT-6.2 Astra acceptability by test condition",
        "Share of reviewed test translations marked semantically acceptable")
    pool_plot_rows = [
        {"split": row["split"], "method": name, "context": "fixed_correct", "metric": value}
        for row in output["pool_analysis"] if row["split"] == "test"
        for name, value in (("pool raw candidates", row["raw_candidate_acceptable_rate"]),
                            ("pool covers item", row["pool_coverage_rate"]),
                            ("aggregate", row["aggregate_acceptable_rate"]),
                            ("select", row["select_acceptable_rate"])) if value is not None
    ]
    write_bar_svg(out_dir / "test_candidate_pool.svg", pool_plot_rows, "metric",
                  "Candidate-pool coverage and postprocessing", "Test-set item / candidate share marked acceptable")
    write_pngs(output, out_dir)
    return {
        "rated_blind_ids": output["rated_blind_ids"],
        "condition_rows": len(output["condition_ratings"]),
        "test_semantic_acceptability": output["rating_scope"]["test"],
        "semantic_analysis_dir": str(out_dir.resolve()),
    }


def write_pngs(output: dict[str, Any], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    test_rows = [row for row in output["condition_summary"] if row["split"] == "test"]
    labels = [f"{row['method']} / {row['context']}" for row in test_rows]
    y = np.arange(len(test_rows))
    height = 0.23
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    for offset, field, label, color in (
        (-height, "semantic_acceptable_rate", "Semantic judgement", "#355C9A"),
        (0, "term_relaxed_acceptable_rate", "Term only relaxed", "#F2A541"),
        (height, "deliverable_acceptable_rate", "Semantic + clean format", "#398B6A"),
    ):
        vals = [row[field] or 0 for row in test_rows]
        bars = ax.barh(y + offset, vals, height, label=label, color=color)
        for bar, row, value in zip(bars, test_rows, vals):
            ax.text(value + 0.012, bar.get_y() + bar.get_height()/2,
                    f"{value:.0%} ({row[field.replace('_rate', '')]}/{row['rated_items']})",
                    va="center", fontsize=8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.22)
    ax.set_xlabel("Share of 30 held-out source clauses")
    ax.set_title("Translation acceptability under three review criteria")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.11), ncol=3, frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=.2)
    fig.savefig(out_dir / "test_acceptability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    test_pool = next(row for row in output["pool_analysis"] if row["split"] == "test")
    pool_labels = ["Raw candidate", "At least one acceptable", "Aggregate", "Select"]
    pool_values = [test_pool["raw_candidate_acceptable_rate"], test_pool["pool_coverage_rate"],
                   test_pool["aggregate_acceptable_rate"], test_pool["select_acceptable_rate"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    bars = ax.bar(pool_labels, pool_values, color=["#8798B8", "#46A381", "#D98C42", "#5C70A5"])
    for bar, value in zip(bars, pool_values):
        ax.text(bar.get_x() + bar.get_width()/2, value + .025, f"{value:.0%}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Share of candidates / test items")
    ax.set_title("Shared three-candidate pool: coverage and postprocessing")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(out_dir / "test_candidate_pool.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    costs = [row for row in output["acceptability_cost"] if row["split"] == "test"]
    cost_labels = [f"{row['method']} / {row['context']}" for row in costs]
    y = np.arange(len(costs))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True, sharey=True)
    logical = [row["mean_logical_completion_tokens"] or 0 for row in costs]
    actual = [row["mean_actual_completion_tokens"] or 0 for row in costs]
    axes[0].barh(y - .18, logical, height=.34, color="#355C9A", label="Logical protocol attribution")
    axes[0].barh(y + .18, actual, height=.34, color="#E6A23C", alpha=.9, label="Physical run allocation")
    axes[0].set_yticks(y, cost_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Completion tokens per output")
    axes[0].set_title("Provider-reported output tokens")
    axes[0].legend(fontsize=8)
    calls = [row["mean_physical_calls"] or 0 for row in costs]
    axes[1].barh(y, calls, color="#398B6A")
    axes[1].set_xlabel("Physical request attempts per output")
    axes[1].set_title("Observed request allocation")
    axes[1].text(.02, -.14, "Candidate generation is counted once physically and attributed to both pool protocols logically.",
                 transform=axes[1].transAxes, fontsize=8)
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
    fig.savefig(out_dir / "test_method_cost.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    unique_errors = [row for row in output["unique_family_errors"] if row["split"] == "test"]
    families = sorted({row["family"] for row in unique_errors})
    error_codes = sorted(ERROR_CODES)
    matrix = np.array([
        [next((row["translations_with_error"] for row in unique_errors
               if row["family"] == family and row["error_code"] == code), 0)
         for code in error_codes]
        for family in families
    ])
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(error_codes)), error_codes, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(families)), families)
    ax.set_title("Unique reviewed translations with each error type")
    for i in range(len(families)):
        for j in range(len(error_codes)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max()/2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Unique blind IDs")
    fig.savefig(out_dir / "test_family_errors.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=Path("results/analysis/astra_ratings.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("results/experiment.jsonl"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis/semantic"))
    args = parser.parse_args()
    result = analyze_semantics(args.ratings, args.results, args.analysis_dir, args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
