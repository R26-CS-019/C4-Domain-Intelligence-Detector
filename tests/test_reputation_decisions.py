"""Known-threat decisions stay separate from the fitted ML model and labels."""
import contextlib
import hashlib
import io
import json
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import requests

import test_batch_domains as batch_test_support
from test_rdap_alignment import live
batch = batch_test_support.batch


def snapshot(*urls, age_hours=0):
    canonical = sorted({live.canonical_reputation_url(url) for url in urls})
    payload = {"source": live.REPUTATION_SOURCE,
               "retrieved_at_utc": (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat(),
               "feed_sha256": "a" * 64, "urls": canonical, "url_count": len(canonical),
               "normalized_urls_sha256": hashlib.sha256("\n".join(canonical).encode()).hexdigest()}
    return payload


class FakeResponse:
    def __init__(self, body, content_type="text/plain", status=200):
        self.body = body
        self.headers = {"Content-Type": content_type}
        self.url = live.REPUTATION_SOURCE
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("Feed response failed")

    def iter_content(self, chunk_size):
        yield self.body


class ReputationDecisionTests(unittest.TestCase):
    def test_known_exact_host_changes_final_verdict_but_preserves_ml_result(self):
        decision = live.resolve_prediction("listed.example", 0.01,
                                           live.validate_reputation_snapshot(snapshot("https://listed.example/login")))
        self.assertEqual(decision["model_prediction"], "LEGITIMATE")
        self.assertEqual(decision["prediction"], "PHISHING")
        self.assertEqual(decision["decision_source"], "OPENPHISH_EXACT_HOST")

    def test_shared_parent_and_other_subdomains_are_not_flagged(self):
        data = live.validate_reputation_snapshot(snapshot("https://bad.provider.example/login"))
        for value in ("provider.example", "good.provider.example", "provider.example.evil.test"):
            with self.subTest(value=value):
                decision = live.resolve_prediction(value, 0.1, data)
                self.assertEqual(decision["prediction"], "LEGITIMATE")
                self.assertFalse(decision["reputation"]["matched"])

    def test_exact_url_normalizes_authority_but_preserves_path_and_query(self):
        data = live.validate_reputation_snapshot(snapshot("https://listed.example/login?code=Ab"))
        match = live.resolve_prediction("HTTPS://LISTED.EXAMPLE:443/login?code=Ab#section", 0.1, data)
        self.assertEqual(match["decision_source"], "OPENPHISH_EXACT_URL")
        for value in ("https://listed.example/other?code=Ab", "https://listed.example/Login?code=Ab",
                      "https://listed.example/login?code=ab", "http://listed.example/login?code=Ab"):
            with self.subTest(value=value):
                decision = live.resolve_prediction(value, 0.1, data)
                self.assertEqual(decision["prediction"], "LEGITIMATE")
                self.assertTrue(decision["reputation"]["host_has_reported_urls"])

    def test_hostname_case_trailing_dot_and_idna_are_normalized(self):
        data = live.validate_reputation_snapshot(snapshot("https://bücher.example/login"))
        decision = live.resolve_prediction("BÜCHER.EXAMPLE.", 0.1, data)
        self.assertEqual(decision["prediction"], "PHISHING")

    def test_no_match_keeps_ml_phishing_verdict(self):
        data = live.validate_reputation_snapshot(snapshot("https://listed.example/login"))
        decision = live.resolve_prediction("unlisted.example", 0.9, data)
        self.assertEqual(decision["prediction"], "PHISHING")
        self.assertEqual(decision["decision_source"], "TUNED_80")

    def test_stale_snapshot_cannot_override_model(self):
        data = {**snapshot("https://listed.example/login", age_hours=13), "available": True}
        decision = live.resolve_prediction("listed.example", 0.1, data)
        self.assertEqual(decision["prediction"], "LEGITIMATE")
        self.assertEqual(decision["reputation"]["status"], "UNAVAILABLE")

    def test_offline_or_disabled_reputation_retains_binary_model_output(self):
        for data in (None, {"available": False, "error": "ConnectionError"}):
            decision = live.resolve_prediction("listed.example", 0.9, data)
            self.assertEqual(decision["prediction"], "PHISHING")
            self.assertEqual(decision["decision_source"], "TUNED_80")


class ReputationCollectionTests(unittest.TestCase):
    def setUp(self):
        self.folder = live.ROOT / "work" / ("reputation_tests_" + uuid.uuid4().hex)
        self.folder.mkdir()
        self.cache = self.folder / "snapshot.json"

    def test_download_parses_and_caches_only_the_official_feed(self):
        response = FakeResponse(b"https://listed.example/login\nhttps://other.example/path\n")
        with patch.object(live.requests, "get", return_value=response) as fetch:
            data = live.load_reputation_snapshot(self.cache)
        self.assertTrue(data["available"])
        self.assertEqual(data["url_count"], 2)
        self.assertTrue(self.cache.exists())
        self.assertEqual(fetch.call_args.args, (live.REPUTATION_SOURCE,))
        self.assertFalse(fetch.call_args.kwargs["allow_redirects"])
        with patch.object(live.requests, "get", side_effect=AssertionError("Unexpected network lookup")):
            cached = live.load_reputation_snapshot(self.cache)
        self.assertTrue(cached["cache_used"])

    def test_expired_cache_and_network_failure_are_unavailable(self):
        self.cache.write_text(json.dumps(snapshot("https://listed.example/login", age_hours=13)), encoding="utf-8")
        with patch.object(live.requests, "get", side_effect=requests.ConnectionError("Offline")):
            data = live.load_reputation_snapshot(self.cache)
        self.assertFalse(data["available"])

    def test_fresh_cache_can_survive_a_failed_refresh(self):
        self.cache.write_text(json.dumps(snapshot("https://listed.example/login")), encoding="utf-8")
        with patch.object(live.requests, "get", side_effect=requests.ConnectionError("Offline")):
            data = live.load_reputation_snapshot(self.cache, force_refresh=True)
        self.assertTrue(data["available"])
        self.assertTrue(data["cache_used"])
        self.assertEqual(data["refresh_error"], "ConnectionError")

    def test_empty_html_invalid_or_redirected_feeds_are_unavailable(self):
        responses = (FakeResponse(b""), FakeResponse(b"<html>error</html>", "text/html"),
                     FakeResponse(b"not a feed\n"), FakeResponse(b"", status=302))
        for response in responses:
            with self.subTest(body=response.body), patch.object(live.requests, "get", return_value=response):
                data = live.load_reputation_snapshot(self.cache)
                self.assertFalse(data["available"])
                self.assertFalse(self.cache.exists())

    def test_oversized_response_is_not_used_as_a_partial_feed(self):
        with patch.object(live, "REPUTATION_MAX_BYTES", 10), \
                patch.object(live.requests, "get", return_value=FakeResponse(b"https://listed.example/login\n")):
            data = live.load_reputation_snapshot(self.cache)
        self.assertFalse(data["available"])
        self.assertFalse(self.cache.exists())

    def test_corrupt_cache_is_not_trusted_when_network_is_offline(self):
        data = snapshot("https://listed.example/login")
        data["urls"] = ["https://changed.example/login"]
        self.cache.write_text(json.dumps(data), encoding="utf-8")
        with patch.object(live.requests, "get", side_effect=requests.ConnectionError("Offline")):
            result = live.load_reputation_snapshot(self.cache)
        self.assertFalse(result["available"])


class ReputationBatchTests(unittest.TestCase):
    def test_final_feed_verdict_is_not_taken_from_the_csv_label(self):
        helper = batch_test_support.BatchDomainTests()
        helper.setUp()
        fake = helper.fake()
        fake.predict.return_value = (0.01, None)
        item = {**helper.item, "expected_label": "LEGITIMATE"}
        data = live.validate_reputation_snapshot(snapshot("https://example.test/login"))
        record = batch.test_domain(item, fake, helper.bundles, helper.output, reputation_snapshot=data)
        model = record["models"][0]
        self.assertEqual(model["model_prediction"], "LEGITIMATE")
        self.assertEqual(model["prediction"], "PHISHING")
        self.assertEqual(model["raw_phishing_score"], 0.01)
        self.assertFalse(model["matches_reference_label"])

    def test_single_command_reports_feed_decision_separately_from_model_result(self):
        data = live.validate_reputation_snapshot(snapshot("https://example.test/login"))
        bundle = ("TUNED 80", object(), object(), ["rdap_abuse_count"])
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(live, "load_selected_model", return_value=bundle))
            stack.enter_context(patch.object(live, "load_reputation_snapshot", return_value=data))
            stack.enter_context(patch.object(live, "collect_dns", return_value={"available": False}))
            stack.enter_context(patch.object(live, "collect_rdap", return_value={"available": False}))
            stack.enter_context(patch.object(live, "collect_tls", return_value={"available": False}))
            stack.enter_context(patch.object(live, "collect_ip", return_value={"available": False}))
            stack.enter_context(patch.object(live, "predict", return_value=(0.01, None)))
            stack.enter_context(contextlib.redirect_stdout(output))
            live.run("example.test", False, False, False, True, False, False)
        text = output.getvalue()
        self.assertTrue(any("0.0100" in line and line.endswith("LEGITIMATE") for line in text.splitlines()))
        self.assertIn("Decision source: OPENPHISH_EXACT_HOST", text)
        self.assertIn("FINAL PREDICTION: PHISHING", text)


if __name__ == "__main__":
    unittest.main()
