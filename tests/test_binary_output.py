"""Check the single-domain command's binary output under missing live data."""
import contextlib
import io
import unittest
from unittest.mock import Mock, patch

from test_rdap_alignment import live


class BinaryCommandTests(unittest.TestCase):
    def run_command(self, available, score):
        output = io.StringIO()
        bundle = ("TUNED 80", object(), object(), ["rdap_abuse_count"])

        def rdap(host, features, debug):
            if available:
                features["rdap_abuse_count"] = 1
            return {"available": available}

        predict = Mock(return_value=(score, None))
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(live, "load_selected_model", return_value=bundle))
            stack.enter_context(patch.object(live, "load_three_models", side_effect=AssertionError("Comparison loader invoked")))
            stack.enter_context(patch.object(live, "collect_dns", return_value={"available": False}))
            stack.enter_context(patch.object(live, "collect_rdap", side_effect=rdap))
            stack.enter_context(patch.object(live, "collect_tls", return_value={"available": False}))
            stack.enter_context(patch.object(live, "collect_ip", return_value={"available": False}))
            stack.enter_context(patch.object(live, "predict", predict))
            stack.enter_context(patch.object(live, "load_reputation_snapshot", return_value={"available": False}))
            stack.enter_context(contextlib.redirect_stdout(output))
            live.run("example.test", False, False, False, True, False, False)
        self.assertEqual(predict.call_count, 1)
        self.assertNotIn("UNKNOWN", output.getvalue())
        self.assertNotIn("TENTATIVE", output.getvalue().upper())
        self.assertNotIn("ORIGINAL 80", output.getvalue())
        self.assertNotIn("TUNED 40", output.getvalue())
        return output.getvalue()

    def test_no_features_runs_only_selected_model_and_reports_defaults_separately(self):
        output = self.run_command(False, 0.2)
        model_rows = [line for line in output.splitlines() if line.strip().startswith("TUNED 80") and "0.2000" in line]
        self.assertEqual(len(model_rows), 1)
        self.assertTrue(all(line.rstrip().endswith("LEGITIMATE") for line in model_rows))
        self.assertIn("No live features were collected", output)

    def test_limited_data_keeps_phishing_label_and_separate_availability_note(self):
        output = self.run_command(True, 0.9)
        model_rows = [line for line in output.splitlines() if "0.9000" in line]
        self.assertEqual(len(model_rows), 1)
        self.assertTrue(all(line.rstrip().endswith("PHISHING") for line in model_rows))
        self.assertIn("Limited live information", output)


if __name__ == "__main__":
    unittest.main()
