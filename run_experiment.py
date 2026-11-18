"""Run the frozen precision-translation experiment using the CASDAO chat API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ENDPOINT = "https://maasapi.casdao.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v4.1-flash"
MAX_TOKENS = 320
PROMPT_VERSION = "precision-translation-v2"
CONDITION_SPECS = (
    ("direct", "fixed_correct"),
    ("fragments", "fixed_correct"),
    ("pool_aggregate", "fixed_correct"),
    ("pool_select", "fixed_correct"),
    ("direct", "none"),
    ("direct", "retrieved"),
)


class ExperimentError(RuntimeError):
    """Sanitized error that never contains request headers or response bodies."""

    def __init__(self, code: str, attempts: int = 0, elapsed_seconds: float = 0.0,
                 actual_model: str | None = None, finish_reason: str | None = None):
        super().__init__(code)
        self.code = code
        self.attempts = attempts
        self.elapsed_seconds = elapsed_seconds
        self.actual_model = actual_model
        self.finish_reason = finish_reason


@dataclass
class CallResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    elapsed_seconds: float
    cache_hit: bool
    physical_calls: int
    attempts: int
    cache_key: str
    actual_model: str | None
    finish_reason: str | None

    def trace(self, step: str) -> dict[str, Any]:
        return {
            "step": step,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "cache_hit": self.cache_hit,
            "physical_calls": self.physical_calls,
            "attempts": self.attempts,
            "actual_model": self.actual_model,
            "finish_reason": self.finish_reason,
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path.name}:{line_no}: {exc.msg}") from exc
    return rows


def load_dataset() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    items = load_jsonl(DATA_DIR / "items.jsonl")
    cards_list = json.loads((DATA_DIR / "cards.json").read_text(encoding="utf-8"))
    cards = {card["id"]: card for card in cards_list}
    ids: set[str] = set()
    family_splits: dict[str, set[str]] = {}
    for item in items:
        if item["id"] in ids:
            raise ValueError(f"Duplicate item id: {item['id']}")
        ids.add(item["id"])
        if not item.get("source_en") or not item.get("anchor_zh"):
            raise ValueError(f"Missing source or frozen anchor for {item['id']}")
        family_splits.setdefault(item["family"], set()).add(item["split"])
        if not set(item["target_card_ids"]).issubset(cards):
            raise ValueError(f"Unknown target card on {item['id']}")
    if any(len(splits) != 1 for splits in family_splits.values()):
        raise ValueError("A semantic family appears in more than one split")
    split_families = {
        split: {family for family, splits in family_splits.items() if split in splits}
        for split in ("dev", "test")
    }
    if split_families["dev"] & split_families["test"]:
        raise ValueError("Development and test families overlap")
    if len(items) != 40 or len(split_families["dev"]) != 2 or len(split_families["test"]) != 6:
        raise ValueError("Frozen dataset shape changed; expected 40 items, 2 dev and 6 test families")
    return items, cards


def select_cards(item: dict[str, Any], cards: dict[str, dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_ids = list(item["target_card_ids"])
    if mode == "none":
        return [], {"selected_card_ids": [], "target_card_ids": target_ids}
    if mode == "fixed_correct":
        selected = [cards[card_id] for card_id in target_ids]
        return selected, {"selected_card_ids": target_ids, "target_card_ids": target_ids, "retrieval_score": None}

    source = item["source_en"].lower()
    rankings = []
    for order, card in enumerate(cards.values()):
        phrase_hits = sum(1 for phrase in card["trigger_phrases"] if phrase.lower() in source)
        terms = {term for term in re.findall(r"[a-z0-9]+", " ".join(card["trigger_phrases"]).lower()) if len(term) > 2}
        source_terms = set(re.findall(r"[a-z0-9]+", source))
        token_hits = len(terms & source_terms)
        score = phrase_hits * 3.0 + token_hits / max(1, len(terms))
        rankings.append((score, -order, card))
    rankings.sort(key=lambda row: (row[0], row[1]), reverse=True)
    k = len(target_ids)
    top = [row for row in rankings[:k] if row[0] > 0]
    selected = [row[2] for row in top]
    if not selected:
        return [], {"selected_card_ids": [], "target_card_ids": target_ids, "retrieval_score": 0.0}
    return selected, {
        "selected_card_ids": [card["id"] for card in selected],
        "target_card_ids": target_ids,
        "retrieval_score": round(top[0][0], 4),
    }


def split_fragments(source: str) -> list[str]:
    """Split only at frozen semicolon boundaries; preserve the boundary mark."""
    fragments = [part.strip() for part in re.split(r"(?<=;)", source) if part.strip()]
    return fragments if len(fragments) > 1 else [source]


def _usage_int(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "cache_key TEXT PRIMARY KEY, text TEXT NOT NULL, prompt_tokens INTEGER, "
            "completion_tokens INTEGER, elapsed REAL NOT NULL, attempts INTEGER NOT NULL, "
            "actual_model TEXT, finish_reason TEXT, created REAL NOT NULL)"
        )
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(responses)").fetchall()}
        if "actual_model" not in columns:
            self.conn.execute("ALTER TABLE responses ADD COLUMN actual_model TEXT")
        if "finish_reason" not in columns:
            self.conn.execute("ALTER TABLE responses ADD COLUMN finish_reason TEXT")
        self.conn.commit()
        self.lock = threading.Lock()

    def get(self, key: str) -> tuple[str, int | None, int | None, float, int, str | None, str | None] | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT text,prompt_tokens,completion_tokens,elapsed,attempts,actual_model,finish_reason "
                "FROM responses WHERE cache_key=?",
                (key,),
            ).fetchone()
        return tuple(row) if row else None

    def put(self, key: str, result: CallResult) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO responses "
                "(cache_key,text,prompt_tokens,completion_tokens,elapsed,attempts,actual_model,finish_reason,created) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (key, result.text, result.prompt_tokens, result.completion_tokens,
                 result.elapsed_seconds, result.attempts, result.actual_model, result.finish_reason, time.time()),
            )
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()


class Checkpoints:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS conditions ("
            "run_key TEXT NOT NULL, condition_key TEXT NOT NULL, status TEXT NOT NULL, "
            "record TEXT NOT NULL, updated REAL NOT NULL, PRIMARY KEY(run_key,condition_key))"
        )
        self.conn.commit()
        self.lock = threading.Lock()

    def completed(self, run_key: str) -> set[str]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT condition_key FROM conditions WHERE run_key=? AND status='success'", (run_key,)
            ).fetchall()
        return {row[0] for row in rows}

    def save(self, run_key: str, key: str, record: dict[str, Any]) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO conditions VALUES (?,?,?,?,?)",
                (run_key, key, record.get("status", "failed"), canonical_json(record), time.time()),
            )
            self.conn.commit()

    def export(self, run_key: str, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            rows = self.conn.execute(
                "SELECT record FROM conditions WHERE run_key=? ORDER BY condition_key", (run_key,)
            ).fetchall()
        temp = output.with_suffix(output.suffix + ".tmp")
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            for (record,) in rows:
                decoded = json.loads(record)
                child_records = decoded.get("result_records")
                for child in child_records if isinstance(child_records, list) else [decoded]:
                    stream.write(canonical_json(with_network_usage(child)) + "\n")
        temp.replace(output)

    def close(self) -> None:
        with self.lock:
            self.conn.close()


def with_network_usage(record: dict) -> dict:
    """Report observed usage once, excluding cached responses and shared pool copies."""
    actual = [c for c in record["calls"] if c.get("physical_calls", 0) > 0]
    for field in ("prompt_tokens", "completion_tokens"):
        values = [c.get(field) for c in actual]
        record["actual_" + field] = sum(values) if all(isinstance(x, int) for x in values) else None
    return record


class ChatClient:
    def __init__(self, model: str, cache: ResponseCache, retries: int = 4):
        key = os.environ.get("CASDAO_API_KEY")
        if not key:
            raise ExperimentError("CASDAO_API_KEY is missing from the process environment")
        self._api_key = key
        self.model = model
        self.cache = cache
        self.retries = retries
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            NoRedirectHandler(),
        )

    def chat(self, system: str, user: str, temperature: float = 0.0) -> CallResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "thinking": {"type": "disabled"},
            "temperature": temperature,
            "max_tokens": MAX_TOKENS,
        }
        key = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        cached = self.cache.get(key)
        if cached:
            text, prompt_tokens, completion_tokens, elapsed, attempts, actual_model, finish_reason = cached
            return CallResult(text, prompt_tokens, completion_tokens, elapsed, True, 0, attempts, key,
                              actual_model, finish_reason)

        body = canonical_json(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}
        last_code = "network_error"
        total_elapsed = 0.0
        last_model = None
        last_finish_reason = None
        for attempt in range(1, self.retries + 1):
            started = time.perf_counter()
            request = urllib.request.Request(ENDPOINT, data=body, headers=headers, method="POST")
            try:
                with self.opener.open(request, timeout=180) as response:
                    raw = response.read()
                elapsed = time.perf_counter() - started
                total_elapsed += elapsed
                try:
                    obj = json.loads(raw.decode("utf-8"))
                    choice = obj["choices"][0]
                    message = choice["message"]
                    text = message.get("content")
                    last_model = obj.get("model") if isinstance(obj.get("model"), str) else None
                    last_finish_reason = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError):
                    raise ExperimentError("invalid_api_response", actual_model=last_model,
                                          finish_reason=last_finish_reason) from None
                if not isinstance(text, str) or not text.strip():
                    raise ExperimentError("empty_api_response", actual_model=last_model,
                                          finish_reason=last_finish_reason)
                if last_finish_reason == "length":
                    raise ExperimentError("truncated_api_response", actual_model=last_model,
                                          finish_reason=last_finish_reason)
                usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
                result = CallResult(
                    text.strip(), _usage_int(usage, "prompt_tokens"), _usage_int(usage, "completion_tokens"),
                    elapsed, False, attempt, attempt, key, last_model, last_finish_reason,
                )
                self.cache.put(key, result)
                return result
            except urllib.error.HTTPError as exc:
                total_elapsed += time.perf_counter() - started
                last_code = f"http_{exc.code}"
                retryable = exc.code in (408, 429) or 500 <= exc.code <= 599
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                total_elapsed += time.perf_counter() - started
                last_code = "network_error"
                retryable = True
            except ExperimentError as exc:
                total_elapsed += time.perf_counter() - started
                last_code = str(exc)
                retryable = False
            if not retryable or attempt == self.retries:
                raise ExperimentError(f"{last_code}_after_{attempt}_attempts", attempt, total_elapsed,
                                      last_model, last_finish_reason) from None
            time.sleep(min(8.0, 2 ** (attempt - 1)) + random.random() * 0.2)
        raise ExperimentError(f"{last_code}_after_{self.retries}_attempts", self.retries, total_elapsed)


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so the authorization header cannot cross to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


SYSTEM_PROMPT = (
    "Translate the supplied English technical standard clause into precise Simplified Chinese. "
    "Preserve every actor, scope, normative strength, negation, condition, exception, logical relation, "
    "quantity, unit, time boundary, and reference. Do not add, omit, explain, or summarize requirements. "
    "Return only the Chinese translation."
)


def card_context(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return ""
    lines = [f"- {card['guidance']}" for card in cards]
    return "\nTerminology guidance (preserve the clause's full meaning):\n" + "\n".join(lines) + "\n"


def candidate_user(source: str, cards: list[dict[str, Any]], variant: int | None = None) -> str:
    variant_text = ""
    if variant is not None:
        variant_text = f"\nProduce candidate {variant} with a distinct but equally faithful Chinese phrasing."
    return f"{card_context(cards)}English clause:\n{source}{variant_text}"


def clean_translation(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'“”":
        value = value[1:-1].strip()
    return value


def parse_candidate_index(value: str) -> int:
    match = re.fullmatch(r"\s*(?:candidate\s*)?([1-3])\s*[.)]?\s*", value, flags=re.IGNORECASE)
    if not match:
        raise ExperimentError("invalid_candidate_selection")
    return int(match.group(1)) - 1


def _run_condition(item: dict[str, Any], cards: dict[str, dict[str, Any]],
                   method: str, context: str, client: ChatClient) -> dict[str, Any]:
    selected_cards, card_trace = select_cards(item, cards, context)
    calls: list[dict[str, Any]] = []
    stages: dict[str, Any] = {}

    def invoke(step: str, system: str, user: str, temperature: float = 0.0) -> CallResult:
        try:
            result = client.chat(system, user, temperature)
        except ExperimentError as exc:
            calls.append({
                "step": step, "prompt_tokens": None, "completion_tokens": None,
                "elapsed_seconds": round(exc.elapsed_seconds, 4), "cache_hit": False,
                "physical_calls": exc.attempts, "attempts": exc.attempts,
                "status": "failed", "error_code": exc.code,
                "actual_model": exc.actual_model, "finish_reason": exc.finish_reason,
            })
            raise
        calls.append(result.trace(step))
        return result

    try:
        if method == "direct":
            result = invoke("direct", SYSTEM_PROMPT, candidate_user(item["source_en"], selected_cards))
            translation = clean_translation(result.text)
        elif method == "fragments":
            fragment_translations = []
            for index, fragment in enumerate(split_fragments(item["source_en"]), 1):
                result = invoke(
                    f"fragment_{index}", SYSTEM_PROMPT,
                    "Translate this fragment independently. Do not use or infer any other fragment.\n"
                    + candidate_user(fragment, selected_cards),
                )
                fragment_translations.append(clean_translation(result.text))
            stages["fragment_count"] = len(fragment_translations)
            translation = "；".join(fragment_translations)
        elif method in ("pool_aggregate", "pool_select"):
            raise ExperimentError("candidate_pool_must_use_shared_runner")
        else:
            raise ExperimentError("unknown_method")
        if not translation:
            raise ExperimentError("empty_translation")
        usage_fields = ("prompt_tokens", "completion_tokens")
        totals = {
            name: (sum(call[name] for call in calls) if all(call[name] is not None for call in calls) else None)
            for name in usage_fields
        }
        return {
            "item_id": item["id"], "family": item["family"], "split": item["split"],
            "source_en": item["source_en"], "anchor_zh": item["anchor_zh"],
            "method": method, "context": context, "status": "success",
            "translation_zh": translation, "card_trace": card_trace, "stages": stages,
            "calls": calls, "logical_calls": len(calls),
            "physical_calls": sum(call["physical_calls"] for call in calls),
            **totals,
            "actual_prompt_tokens": totals["prompt_tokens"],
            "actual_completion_tokens": totals["completion_tokens"],
            "elapsed_seconds": round(sum(call["elapsed_seconds"] for call in calls), 4),
            "logical_elapsed_seconds": round(sum(call["elapsed_seconds"] for call in calls), 4),
            "cache_hits": sum(1 for call in calls if call["cache_hit"]),
            "shared_pool_calls": 0,
        }
    except ExperimentError as exc:
        return {
            "item_id": item["id"], "family": item["family"], "split": item["split"],
            "source_en": item["source_en"], "anchor_zh": item["anchor_zh"],
            "method": method, "context": context, "status": "failed",
            "error_code": exc.code, "card_trace": card_trace, "stages": stages,
            "calls": calls, "logical_calls": len(calls),
            "physical_calls": sum(call["physical_calls"] for call in calls),
            "prompt_tokens": None, "completion_tokens": None,
            "actual_prompt_tokens": None, "actual_completion_tokens": None,
            "elapsed_seconds": round(sum(call["elapsed_seconds"] for call in calls), 4),
            "logical_elapsed_seconds": round(sum(call["elapsed_seconds"] for call in calls), 4),
            "cache_hits": sum(1 for call in calls if call["cache_hit"]),
            "shared_pool_calls": 0,
        }


def _condition_record(item: dict[str, Any], method: str, context: str,
                      card_trace: dict[str, Any], stages: dict[str, Any],
                      calls: list[dict[str, Any]], translation: str | None,
                      error_code: str | None = None) -> dict[str, Any]:
    def total(field: str, ignore_shared: bool = False) -> int | None:
        counted = [call.get(field) for call in calls if not (ignore_shared and call.get("shared_pool"))]
        if not counted or any(value is None for value in counted):
            return None
        return sum(counted)

    return {
        "item_id": item["id"], "family": item["family"], "split": item["split"],
        "source_en": item["source_en"], "anchor_zh": item["anchor_zh"],
        "method": method, "context": context,
        "status": "success" if translation else "failed",
        "translation_zh": translation,
        **({"error_code": error_code} if error_code else {}),
        "card_trace": card_trace, "stages": stages, "calls": calls,
        "logical_calls": len(calls),
        "physical_calls": sum(call.get("physical_calls", 0) for call in calls if not call.get("shared_pool")),
        "prompt_tokens": total("prompt_tokens"), "completion_tokens": total("completion_tokens"),
        "actual_prompt_tokens": total("prompt_tokens", ignore_shared=True),
        "actual_completion_tokens": total("completion_tokens", ignore_shared=True),
        "elapsed_seconds": round(sum(call.get("elapsed_seconds", 0.0) for call in calls if not call.get("shared_pool")), 4),
        "logical_elapsed_seconds": round(sum(call.get("elapsed_seconds", 0.0) for call in calls), 4),
        "cache_hits": sum(1 for call in calls if call.get("cache_hit") and not call.get("shared_pool")),
        "shared_pool_calls": sum(1 for call in calls if call.get("shared_pool")),
    }


def _run_pool_pair(item: dict[str, Any], cards: dict[str, dict[str, Any]],
                    client: ChatClient) -> dict[str, Any]:
    """Generate one candidate pool, then share it between aggregation and selection."""
    method_context = "fixed_correct"
    selected_cards, card_trace = select_cards(item, cards, method_context)
    candidate_calls: list[dict[str, Any]] = []
    candidates: list[str] = []
    aggregate_calls: list[dict[str, Any]] = []
    select_calls: list[dict[str, Any]] = []
    stages = {"pool_size": 3}

    def invoke(step: str, user: str, target: list[dict[str, Any]]) -> tuple[CallResult | None, ExperimentError | None]:
        try:
            result = client.chat(SYSTEM_PROMPT, user, 0.6 if step.startswith("pool_candidate_") else 0.0)
        except ExperimentError as exc:
            trace = {
                "step": step, "prompt_tokens": None, "completion_tokens": None,
                "elapsed_seconds": round(exc.elapsed_seconds, 4), "cache_hit": False,
                "physical_calls": exc.attempts, "attempts": exc.attempts,
                "status": "failed", "error_code": exc.code,
                "actual_model": exc.actual_model, "finish_reason": exc.finish_reason,
            }
            target.append(trace)
            return None, exc
        target.append(result.trace(step))
        return result, None

    failure: ExperimentError | None = None
    for index in range(1, 4):
        result, error = invoke(
            f"pool_candidate_{index}",
            candidate_user(item["source_en"], selected_cards, variant=index), candidate_calls,
        )
        if error:
            failure = error
            break
        candidate = clean_translation(result.text)
        if not candidate:
            failure = ExperimentError("empty_candidate_translation")
            break
        candidates.append(candidate)

    if failure is None:
        stages["candidate_translations"] = candidates
        candidate_block = "\n".join(f"{index}. {candidate}" for index, candidate in enumerate(candidates, 1))
        aggregate_user = (
            f"{card_context(selected_cards)}English source clause:\n{item['source_en']}\n\n"
            f"Three independently generated Chinese translations:\n{candidate_block}\n\n"
            "Synthesize one translation that preserves all meaning in the English source and uses "
            "the candidates only as alternatives. Return only the final Chinese translation."
        )
        aggregate_result, failure = invoke("aggregate", aggregate_user, aggregate_calls)
        aggregate_translation = clean_translation(aggregate_result.text) if aggregate_result else None
        if aggregate_result and not aggregate_translation:
            failure = ExperimentError("empty_aggregate_translation")

        candidate_block = "\n".join(f"{index}. {candidate}" for index, candidate in enumerate(candidates, 1))
        select_user = (
            f"{card_context(selected_cards)}English source clause:\n{item['source_en']}\n\n"
            f"Chinese candidates:\n{candidate_block}\n\n"
            "Choose the candidate that most faithfully preserves all source meaning. "
            "Return only its number: 1, 2, or 3."
        )
        selected_result = None
        if failure is None:
            selected_result, failure = invoke("select", select_user, select_calls)
        selected_translation = None
        if selected_result:
            try:
                selected_index = parse_candidate_index(selected_result.text)
                stages["selected_candidate"] = selected_index + 1
                selected_translation = candidates[selected_index]
            except ExperimentError as exc:
                failure = exc
    else:
        aggregate_translation = None
        selected_translation = None

    shared_traces = []
    for call in candidate_calls:
        trace = dict(call)
        trace["shared_pool"] = True
        trace["physical_calls"] = 0
        shared_traces.append(trace)

    aggregate_record = _condition_record(
        item, "pool_aggregate", method_context, card_trace, dict(stages),
        candidate_calls + aggregate_calls, aggregate_translation,
        failure.code if failure and not aggregate_translation else None,
    )
    select_record = _condition_record(
        item, "pool_select", method_context, card_trace, dict(stages),
        shared_traces + select_calls, selected_translation,
        failure.code if failure and not selected_translation else None,
    )
    group_status = "success" if aggregate_record["status"] == select_record["status"] == "success" else "failed"
    return {
        "item_id": item["id"], "split": item["split"], "status": group_status,
        "result_records": [aggregate_record, select_record],
    }


def _condition_key(item_id: str, method: str, context: str) -> str:
    return f"{item_id}|{method}|{context}"


def run(args: argparse.Namespace) -> int:
    items, cards = load_dataset()
    if args.split != "all":
        items = [item for item in items if item["split"] == args.split]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_material = (DATA_DIR / "items.jsonl").read_bytes() + (DATA_DIR / "cards.json").read_bytes()
    run_material = dataset_material + f"|{PROMPT_VERSION}|{args.model}|{MAX_TOKENS}".encode()
    run_key = hashlib.sha256(run_material).hexdigest()
    cache = ResponseCache(RESULTS_DIR / "api_cache.sqlite3")
    checkpoints = Checkpoints(RESULTS_DIR / "run_checkpoints.sqlite3")
    try:
        client = ChatClient(args.model, cache)
        tasks = []
        for item in items:
            for method, context in CONDITION_SPECS:
                if method == "pool_aggregate":
                    continue
                group_method = "pool_pair" if method == "pool_select" else method
                tasks.append((item, group_method, context,
                              _condition_key(item["id"], group_method, context)))
        completed = checkpoints.completed(run_key)
        pending = [task for task in tasks if task[3] not in completed]
        if not pending:
            print(f"No unfinished groups; exporting {len(items) * len(CONDITION_SPECS)} frozen condition rows.")
        else:
            print(f"Running {len(pending)} unfinished conditions with {args.workers} workers.")
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for item, method, context, key in pending:
                if method == "pool_pair":
                    future = pool.submit(_run_pool_pair, item, cards, client)
                else:
                    future = pool.submit(_run_condition, item, cards, method, context, client)
                futures[future] = (item, method, context, key)
            for future in as_completed(futures):
                item, method, context, key = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # keep checkpoint failure diagnostics sanitized
                    error_code = "internal_error"
                    if isinstance(exc, ExperimentError):
                        error_code = str(exc)
                    if method == "pool_pair":
                        children = [
                            {"item_id": item["id"], "family": item["family"], "split": item["split"],
                             "source_en": item["source_en"], "anchor_zh": item["anchor_zh"],
                             "method": pool_method, "context": context, "status": "failed",
                             "error_code": error_code, "calls": [], "logical_calls": 0, "physical_calls": 0,
                             "prompt_tokens": None, "completion_tokens": None, "elapsed_seconds": 0.0,
                             "cache_hits": 0}
                            for pool_method in ("pool_aggregate", "pool_select")
                        ]
                        record = {"item_id": item["id"], "split": item["split"], "status": "failed", "result_records": children}
                    else:
                        record = {
                            "item_id": item["id"], "family": item["family"], "split": item["split"],
                            "source_en": item["source_en"], "anchor_zh": item["anchor_zh"],
                            "method": method, "context": context, "status": "failed",
                            "error_code": error_code, "calls": [], "logical_calls": 0, "physical_calls": 0,
                            "prompt_tokens": None, "completion_tokens": None, "elapsed_seconds": 0.0,
                            "cache_hits": 0,
                        }
                checkpoints.save(run_key, key, record)
                done += 1
                if done == 1 or done % 10 == 0 or done == len(pending):
                    print(f"Completed {done}/{len(pending)} pending conditions; latest status={record['status']}.")
        checkpoints.export(run_key, Path(args.output).resolve())
        print(f"Wrote condition results to {Path(args.output).resolve()}")
        failures = sum(1 for record in load_jsonl(Path(args.output).resolve()) if record.get("status") != "success")
        print(f"Failed conditions: {failures}")
        return 1 if failures else 0
    finally:
        checkpoints.close()
        cache.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=str(RESULTS_DIR / "experiment.jsonl"))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    try:
        return run(args)
    except (ExperimentError, ValueError) as exc:
        print(f"Experiment cannot start: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
