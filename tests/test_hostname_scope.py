"""Prevent input hostnames from being replaced by registered parents."""
import contextlib
import csv
import io
import unittest
from unittest.mock import Mock, patch

import test_batch_domains as support
from test_rdap_alignment import live


class HostnameScopeTests(unittest.TestCase):
    def test_command_collectors_keep_www_subdomains_and_actual_url_authority(self):
        examples = {
            "google.com": "google.com",
            "www.google.com": "www.google.com",
            "https://0365-microsoft.gitbook.io/login?x=1": "0365-microsoft.gitbook.io",
            "https://www.bbc.co.uk/news": "www.bbc.co.uk",
            "https://google.com@attacker.net/login": "attacker.net",
        }
        bundle = ("TUNED 80", object(), object(), ["rdap_abuse_count"])
        for raw, expected in examples.items():
            with self.subTest(raw=raw), contextlib.ExitStack() as stack:
                output = io.StringIO()
                stack.enter_context(patch.object(live, "load_selected_model", return_value=bundle))
                collectors = [stack.enter_context(patch.object(live, name, return_value={"available": False}))
                              for name in ("collect_dns", "collect_rdap", "collect_tls")]
                stack.enter_context(patch.object(live, "collect_ip", return_value={"available": False}))
                stack.enter_context(patch.object(live, "predict", return_value=(0.1, None)))
                feed = stack.enter_context(patch.object(live, "load_reputation_snapshot"))
                stack.enter_context(contextlib.redirect_stdout(output))
                live.run(raw, False, False, False, True, False, False, True)
                for collector in collectors:
                    self.assertEqual(collector.call_args.args[0], expected)
                feed.assert_not_called()
                self.assertIn("Domain: " + expected, output.getvalue())
                self.assertIn("FINAL PREDICTION: LEGITIMATE", output.getvalue())

    def test_batch_collects_aliases_and_platform_children_as_distinct_hosts(self):
        helper = support.BatchDomainTests()
        helper.setUp()
        values = ("https://www.example.com/login?x=1", "example.com", "bad.gitbook.io")
        targets = ("www.example.com", "example.com", "bad.gitbook.io")
        path = helper.output / "inputs.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["domain", "expected_label"])
            writer.writeheader()
            writer.writerows({"domain": value, "expected_label": ""} for value in values)
        inputs = support.batch.load_inputs(path, support.real)
        self.assertEqual([item["domain"] for item in inputs], list(targets))
        self.assertEqual([item["input_value"] for item in inputs], list(values))
        fake = helper.fake(available=True)
        for name in ("collect_dns", "collect_rdap", "collect_tls"):
            setattr(fake, name, Mock(wraps=getattr(fake, name)))
        for item in inputs:
            record = support.batch.test_domain(item, fake, helper.bundles, helper.output)
            self.assertEqual(record["domain"], item["domain"])
            self.assertTrue((helper.output / item["domain"] / "features.csv").is_file())
        for name in ("collect_dns", "collect_rdap", "collect_tls"):
            collector = getattr(fake, name)
            self.assertEqual([call.args[0] for call in collector.call_args_list], list(targets))


if __name__ == "__main__":
    unittest.main()
