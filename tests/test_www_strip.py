"""C4 API-only www normalization: collection, reputation, evidence and safety."""
import contextlib
import hashlib
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import api_server as api
from fastapi import HTTPException


class StripWWWTests(unittest.TestCase):
    def test_required_examples(self):
        examples = {
            "https://www.hackerone.com/": "https://hackerone.com/",
            "https://www.paypal.com/signin?x=1": "https://paypal.com/signin?x=1",
            "www.microsoft.com": "microsoft.com",
            "HTTPS://WWW.Example.com/A": "HTTPS://Example.com/A",
            "https://www.example.com:8443/x": "https://example.com:8443/x",
            "https://login.paypal.com/": "https://login.paypal.com/",
            "https://www2.example.com/": "https://www2.example.com/",
            "https://wwwexample.com/": "https://wwwexample.com/",
            "https://www.com/": "https://www.com/",
        }
        for original, expected in examples.items():
            with self.subTest(original=original):
                self.assertEqual(api.strip_www(original), expected)

    def test_only_one_host_prefix_is_removed_and_other_url_parts_are_preserved(self):
        examples = {
            "https://www.www.example.com/a": "https://www.example.com/a",
            "//WWW.Example.com:8443/A?x=www.example.com#WWW": "//Example.com:8443/A?x=www.example.com#WWW",
            "www.Example.com:8443/A?x=1&x=2#top": "Example.com:8443/A?x=1&x=2#top",
            "https://user:pass@WWW.Example.com:8443/A?x=%2F#Top": "https://user:pass@Example.com:8443/A?x=%2F#Top",
            "https://www.example.com?": "https://example.com?",
            "https://www.example.com#": "https://example.com#",
            "https://example.com/www.phishing.com?next=www.other.com": "https://example.com/www.phishing.com?next=www.other.com",
            "https://www.example.com@other.example/path": "https://www.example.com@other.example/path",
            "https://[::1]/": "https://[::1]/",
            "www.com": "www.com", "localhost": "localhost", "": "",
        }
        for original, expected in examples.items():
            with self.subTest(original=original):
                self.assertEqual(api.strip_www(original), expected)


class WWWAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(api, "API_KEY", ""))
        self.stack.enter_context(patch.object(api, "MODEL_ONLY", False))
        self.real_resolves_only_public = api.resolves_only_public
        self.public = self.stack.enter_context(patch.object(api, "resolves_only_public", return_value=True))
        self.collectors = {}
        for name in ("dns", "rdap", "tls", "ip"):
            self.collectors[name] = self.stack.enter_context(patch.object(
                api.live, "collect_" + name, return_value={"available": False}))
        self.stack.enter_context(patch.object(api.live, "predict", return_value=(0.014, None)))
        self.feed = self.stack.enter_context(patch.object(api.live, "load_reputation_snapshot", return_value=None))
        self.decide = self.stack.enter_context(patch.object(api.live, "resolve_prediction", wraps=api.live.resolve_prediction))

    def analyze(self, url):
        return api.analyze(api.AnalyzeRequest(url=url), x_api_key=None)

    def test_collectors_safety_and_reputation_receive_the_stripped_target(self):
        original = "https://WWW.Example.com:8443/A?x=1#Top"
        result = self.analyze(original)
        self.assertEqual(result["status"], "ok")
        for name in ("dns", "rdap", "tls"):
            self.assertEqual(self.collectors[name].call_args.args[0], "example.com")
        self.collectors["ip"].assert_called_once_with(
            self.collectors["dns"].return_value, self.collectors["dns"].call_args.args[1], do_rdap=False)
        self.public.assert_called_once_with("example.com")
        self.decide.assert_called_once_with("https://Example.com:8443/A?x=1#Top", 0.014, None)
        self.assertEqual(result["evidence"]["input_url"], original)
        self.assertEqual(result["evidence"]["analysed_url"], "https://Example.com:8443/A?x=1#Top")
        self.assertTrue(result["evidence"]["www_removed"])

    def test_equivalent_inputs_keep_the_same_model_score_and_threshold(self):
        www = self.analyze("https://www.hackerone.com/")
        bare = self.analyze("https://hackerone.com/")
        self.assertEqual(www["score"], bare["score"])
        self.assertEqual(www["threshold"], bare["threshold"])
        self.assertEqual(www["evidence"]["analysed_url"], bare["evidence"]["analysed_url"])
        self.assertTrue(www["evidence"]["www_removed"])
        self.assertFalse(bare["evidence"]["www_removed"])
        self.assertEqual(www["evidence"]["model_prediction"], "LEGITIMATE")

    def test_exact_url_feed_match_uses_stripped_url_and_preserves_path_and_query(self):
        urls = ["https://listed.example/A?x=1"]
        snapshot = api.live.validate_reputation_snapshot({
            "source": api.live.REPUTATION_SOURCE, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "feed_sha256": "a" * 64, "urls": urls, "url_count": 1,
            "normalized_urls_sha256": hashlib.sha256("\n".join(urls).encode()).hexdigest(),
        })
        self.feed.return_value = snapshot
        match = self.analyze("https://www.listed.example/A?x=1")
        self.assertEqual(match["label"], "phishing")
        self.assertEqual(match["evidence"]["decision_source"], "OPENPHISH_EXACT_URL")
        self.assertEqual(match["evidence"]["model_prediction"], "LEGITIMATE")
        self.assertEqual(match["score"], 0.014)
        other_path = self.analyze("https://www.listed.example/other?x=1")
        self.assertEqual(other_path["label"], "legitimate")
        self.assertFalse(other_path["evidence"]["reputation"]["matched"])
        bare_domain = self.analyze("www.listed.example")
        self.assertEqual(bare_domain["evidence"]["decision_source"], "OPENPHISH_EXACT_HOST")

    def test_original_input_is_retained_and_whitespace_alone_is_not_www_removal(self):
        for original, expected, removed in [
            ("  https://www.example.com/A  ", "https://example.com/A", True),
            ("  https://example.com/A  ", "https://example.com/A", False),
        ]:
            with self.subTest(original=original):
                evidence = self.analyze(original)["evidence"]
                self.assertEqual(evidence["input_url"], original)
                self.assertEqual(evidence["analysed_url"], expected)
                self.assertEqual(evidence["www_removed"], removed)

    def test_invalid_hostname_still_rejected_before_collection(self):
        result = self.analyze("https://www..example.com/")
        self.assertEqual(result["status"], "invalid_input")
        self.assertIsNone(result["label"])
        self.public.assert_not_called()
        self.collectors["dns"].assert_not_called()

    def test_private_and_local_hosts_still_rejected_after_normalization(self):
        answers = [(None, None, None, None, ("127.0.0.1", 443))]
        with patch.object(api, "resolves_only_public", side_effect=self.real_resolves_only_public), \
                patch.object(api.socket, "getaddrinfo", return_value=answers):
            for value in ("localhost", "https://www.localhost.localdomain/", "https://www.127.0.0.1/"):
                with self.subTest(value=value):
                    result = self.analyze(value)
                    self.assertEqual(result["status"], "invalid_input")
                    self.assertIsNone(result["label"])
        self.collectors["dns"].assert_not_called()

    def test_busy_gate_and_api_key_are_preserved(self):
        with patch.object(api, "API_KEY", "test-key"):
            with self.assertRaises(HTTPException) as raised:
                self.analyze("https://www.example.com/")
            self.assertEqual(raised.exception.status_code, 401)
        api.gate.acquire()
        try:
            with self.assertRaises(HTTPException) as raised:
                self.analyze("https://www.example.com/")
            self.assertEqual(raised.exception.status_code, 429)
        finally:
            api.gate.release()
        self.collectors["dns"].assert_not_called()

    def test_collection_failure_keeps_generic_error_and_releases_gate(self):
        self.collectors["dns"].side_effect = RuntimeError("internal details must not leak")
        result = self.analyze("https://www.example.com/")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Analysis failed.")
        self.assertIsNone(result["label"])
        self.assertNotIn("internal details", json.dumps(result))
        self.collectors["dns"].side_effect = None
        self.assertEqual(self.analyze("https://www.example.com/")["status"], "ok")


if __name__ == "__main__":
    unittest.main()
