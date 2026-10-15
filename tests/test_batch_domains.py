"""Verify binary batch output and separate data/error handling offline."""
from pathlib import Path
import importlib.util
import sys
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("batch_under_test", ROOT / "src/14_batch_test_domains.py")
batch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = batch
spec.loader.exec_module(batch)
real = batch.load_live()


class BatchDomainTests(unittest.TestCase):
    def setUp(self):
        self.output = ROOT / "work" / ("batch_tests_" + uuid.uuid4().hex)
        self.output.mkdir()
        self.item = {"domain": "example.test", "expected_label": "PHISHING", "label_context": "historical"}
        self.bundles = [("model", object(), object(), ["rdap_abuse_count"])]

    def fake(self, *, available=False, partial=False):
        def dns(host, features):
            return {"available": available}
        def rdap(host, features, debug):
            if available:
                features["rdap_abuse_count"] = 1
            return {"available": available, "registry_only_fallback": partial}
        return SimpleNamespace(
            collect_dns=dns, collect_rdap=rdap,
            collect_tls=lambda host, features: {"available": available, "certificate_verified": True},
            collect_ip=lambda dns, features, do_rdap: {"available": available},
            feature_quality=real.feature_quality, nonmissing=real.nonmissing, THRESHOLD=0.5,
            binary_prediction=real.binary_prediction, prediction_data_note=real.prediction_data_note,
            resolve_prediction=real.resolve_prediction,
            predict=Mock(return_value=(0.9, None)), explain_prediction=Mock(),
        )

    def test_no_data_produces_a_binary_prediction_with_an_availability_note(self):
        live = self.fake()
        result = batch.test_domain(self.item, live, self.bundles, self.output)
        self.assertEqual(result["models"][0]["prediction"], "PHISHING")
        self.assertEqual(result["models"][0]["raw_phishing_score"], 0.9)
        self.assertEqual(result["models"][0]["prediction_status"], "CLASSIFIED")
        self.assertIn("No live features", result["models"][0]["data_availability_note"])
        live.predict.assert_called_once()
        self.assertTrue((self.output / "example.test" / "collection_log.txt").exists())

    def test_registry_only_result_keeps_binary_label_and_separate_note(self):
        result = batch.test_domain(self.item, self.fake(available=True, partial=True), self.bundles, self.output)
        self.assertEqual(result["models"][0]["prediction"], "PHISHING")
        self.assertEqual(result["models"][0]["prediction_status"], "CLASSIFIED")
        self.assertIn("registrar data", result["models"][0]["data_availability_note"])
        self.assertTrue(result["tls_certificate_verified"])

    def test_collection_error_is_saved_without_a_fabricated_prediction(self):
        live = self.fake()
        live.collect_dns = Mock(side_effect=RuntimeError("collection failed"))
        result = batch.test_domain(self.item, live, self.bundles, self.output)
        self.assertEqual(result["data_quality"], "ERROR")
        self.assertEqual(result["models"], [])
        live.predict.assert_not_called()
        self.assertIn("collection failed", (self.output / "example.test" / "collection_log.txt").read_text())

    def test_no_data_legitimate_output_is_not_overridden_by_phishing_reference_label(self):
        live = self.fake()
        live.predict.return_value = (0.1, None)
        result = batch.test_domain(self.item, live, self.bundles, self.output)
        model = result["models"][0]
        self.assertEqual(model["prediction"], "LEGITIMATE")
        self.assertFalse(model["matches_reference_label"])

    def test_nonfinite_model_output_is_an_error_and_not_a_legitimate_prediction(self):
        live = self.fake()
        live.predict.return_value = (float("nan"), None)
        result = batch.test_domain(self.item, live, self.bundles, self.output)
        self.assertEqual(result["data_quality"], "ERROR")
        self.assertEqual(result["models"], [])
        self.assertIn("non-finite", result["error"])


if __name__ == "__main__":
    unittest.main()
