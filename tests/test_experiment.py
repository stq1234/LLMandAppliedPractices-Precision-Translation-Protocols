import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from analyze_results import automatic_rules, build_blind_material, normalize_translation
from run_experiment import (
    CONDITION_SPECS,
    ChatClient,
    CallResult,
    ExperimentError,
    NoRedirectHandler,
    ResponseCache,
    _run_condition,
    _run_pool_pair,
    load_dataset,
    parse_candidate_index,
    select_cards,
    split_fragments,
)


class DatasetAndProtocolTests(unittest.TestCase):
    def test_frozen_dataset_has_family_level_split(self):
        items, cards = load_dataset()
        self.assertEqual(len(items), 40)
        self.assertEqual(len({item["family"] for item in items if item["split"] == "dev"}), 2)
        self.assertEqual(len({item["family"] for item in items if item["split"] == "test"}), 6)
        dev = {item["family"] for item in items if item["split"] == "dev"}
        test = {item["family"] for item in items if item["split"] == "test"}
        self.assertFalse(dev & test)
        self.assertEqual(len(CONDITION_SPECS), 6)
        self.assertEqual(CONDITION_SPECS.count(("direct", "fixed_correct")), 1)
        self.assertEqual(CONDITION_SPECS.count(("direct", "none")), 1)
        self.assertEqual(CONDITION_SPECS.count(("direct", "retrieved")), 1)
        self.assertEqual(CONDITION_SPECS.count(("fragments", "fixed_correct")), 1)
        self.assertEqual(CONDITION_SPECS.count(("pool_aggregate", "fixed_correct")), 1)
        self.assertEqual(CONDITION_SPECS.count(("pool_select", "fixed_correct")), 1)
        self.assertTrue(all(set(item["target_card_ids"]) <= set(cards) for item in items))

    def test_fragment_split_uses_semicolon_without_cross_fragment_context(self):
        parts = split_fragments("First rule; second rule; third rule.")
        self.assertEqual(parts, ["First rule;", "second rule;", "third rule."])

    def test_card_modes_include_fixed_or_retrieved_cards(self):
        items, cards = load_dataset()
        item = next(row for row in items if row["id"] == "TEST-NB-01")
        fixed, fixed_trace = select_cards(item, cards, "fixed_correct")
        self.assertEqual({card["id"] for card in fixed}, set(item["target_card_ids"]))
        self.assertEqual(fixed_trace["selected_card_ids"], item["target_card_ids"])
        retrieved, trace = select_cards(item, cards, "retrieved")
        self.assertLessEqual(len(retrieved), len(item["target_card_ids"]))
        self.assertTrue(set(trace["selected_card_ids"]) <= set(cards))

    def test_candidate_index_parser_rejects_non_single_selection(self):
        self.assertEqual(parse_candidate_index("2"), 1)
        self.assertEqual(parse_candidate_index("candidate 3."), 2)
        with self.assertRaises(ExperimentError):
            parse_candidate_index("candidate 2 is best")


class ReviewMaterialTests(unittest.TestCase):
    def test_blind_review_deduplicates_exact_source_translation_pairs(self):
        records = [
            {"status": "success", "item_id": "T01", "family": "x", "split": "test",
             "source_en": "The client MUST wait 5 seconds.", "anchor_zh": "客户端必须等待 5 秒。",
             "translation_zh": "客户端必须等待 5 秒。", "method": "direct", "context": "none"},
            {"status": "success", "item_id": "T01", "family": "x", "split": "test",
             "source_en": "The client MUST wait 5 seconds.", "anchor_zh": "客户端必须等待 5 秒。",
             "translation_zh": " 客户端必须等待 5 秒。 ", "method": "pool_select", "context": "retrieved"},
            {"status": "success", "item_id": "T01", "family": "x", "split": "test",
             "source_en": "The client MUST wait 5 seconds.", "anchor_zh": "客户端必须等待 5 秒。",
             "translation_zh": "客户端不得等待 5 秒。", "method": "fragments", "context": "none"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            counts = build_blind_material(records, out)
            blind = [json.loads(line) for line in (out / "blind_review.jsonl").read_text(encoding="utf-8").splitlines()]
            mapping = [json.loads(line) for line in (out / "review_mapping.jsonl").read_text(encoding="utf-8").splitlines()]
            checks = [json.loads(line) for line in (out / "auto_rule_checks.jsonl").read_text(encoding="utf-8").splitlines()]
            groups = [json.loads(line) for line in (out / "disagreement_groups.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(counts["unique_test_blind_translations"], 2)
        self.assertEqual(len(blind), 2)
        self.assertTrue(all(set(row) == {"blind_id", "source_en", "anchor_zh", "translation_zh"} for row in blind))
        self.assertTrue(all("method" not in row and "context" not in row for row in blind))
        self.assertEqual(len(mapping[0]["conditions"]) + len(mapping[1]["conditions"]), 3)
        self.assertEqual(len(checks), 2)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["distinct_translation_count"], 2)
        self.assertEqual(counts["unique_dev_blind_translations"], 0)

    def test_automatic_rules_are_explicitly_lexical_signals(self):
        checks = automatic_rules("The client MUST NOT wait more than 5 seconds.", "客户端不得等待超过 5 秒。")
        by_name = {row["rule"]: row for row in checks}
        self.assertEqual(by_name["ascii_number_literal_retention"]["signal"], "all_present")
        self.assertEqual(by_name["modal_lexical_signal:prohibition"]["signal"], "present")
        self.assertNotIn("semantic_correctness", by_name)

    def test_translation_key_normalizes_whitespace_and_unicode(self):
        self.assertEqual(normalize_translation("A   B"), normalize_translation(" A B "))

    def test_three_candidate_pool_is_called_once_and_shared(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def chat(self, system, user, temperature=0.0):
                self.calls.append((system, user, temperature))
                if "Produce candidate 1" in user:
                    text = "候选译文一"
                elif "Produce candidate 2" in user:
                    text = "候选译文二"
                elif "Produce candidate 3" in user:
                    text = "候选译文三"
                elif "Synthesize one translation" in user:
                    text = "聚合后的译文"
                elif "Choose the candidate" in user:
                    text = "2"
                else:
                    text = "翻译结果"
                return CallResult(text, 20, 10, 0.01, False, 1, 1, f"k{len(self.calls)}", "fake-model", "stop")

        items, cards = load_dataset()
        item = next(row for row in items if row["id"] == "TEST-EX-01")
        client = FakeClient()
        group = _run_pool_pair(item, cards, client)
        aggregate, select = group["result_records"]
        self.assertEqual(len(client.calls), 5)
        self.assertEqual(group["status"], "success")
        self.assertEqual(aggregate["stages"]["candidate_translations"], select["stages"]["candidate_translations"])
        self.assertEqual(aggregate["logical_calls"], 4)
        self.assertEqual(select["logical_calls"], 4)
        self.assertEqual(aggregate["physical_calls"], 4)
        self.assertEqual(select["physical_calls"], 1)
        self.assertEqual(select["shared_pool_calls"], 3)
        self.assertEqual(select["translation_zh"], "候选译文二")

    def test_api_request_disables_thinking_proxy_and_redirects_and_caches_provider_metadata(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return json.dumps({
                    "model": "deepseek-v4.1-flash", "usage": {"prompt_tokens": 7, "completion_tokens": 5},
                    "choices": [{"message": {"content": "译文"}, "finish_reason": "stop"}],
                }).encode("utf-8")

        class FakeOpener:
            def __init__(self):
                self.calls = 0
                self.payload = None
                self.authorization = None

            def open(self, request, timeout):
                self.calls += 1
                self.payload = json.loads(request.data.decode("utf-8"))
                self.authorization = request.get_header("Authorization")
                return FakeResponse()

        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"CASDAO_API_KEY": "test-only-secret"}):
            cache_path = Path(temp) / "api_cache.sqlite3"
            cache = ResponseCache(cache_path)
            try:
                client = ChatClient("deepseek-v4.1-flash", cache)
                self.assertFalse(any(handler.__class__.__name__ == "ProxyHandler" for handler in client.opener.handlers))
                self.assertTrue(any(isinstance(handler, NoRedirectHandler) for handler in client.opener.handlers))
                client.opener = FakeOpener()
                first = client.chat("system", "source")
                second = client.chat("system", "source")
                self.assertEqual(client.opener.calls, 1)
                self.assertEqual(client.opener.authorization, "Bearer test-only-secret")
                self.assertEqual(client.opener.payload["thinking"], {"type": "disabled"})
                self.assertEqual(client.opener.payload["model"], "deepseek-v4.1-flash")
                self.assertEqual(first.actual_model, "deepseek-v4.1-flash")
                self.assertEqual(first.finish_reason, "stop")
                self.assertFalse(first.cache_hit)
                self.assertTrue(second.cache_hit)
                self.assertEqual(second.actual_model, "deepseek-v4.1-flash")
                self.assertEqual(second.finish_reason, "stop")
            finally:
                cache.close()
            self.assertNotIn(b"test-only-secret", cache_path.read_bytes())
            self.assertIsNone(NoRedirectHandler().redirect_request(None, None, 302, "Found", {}, "https://other.example/"))


if __name__ == "__main__":
    unittest.main()
