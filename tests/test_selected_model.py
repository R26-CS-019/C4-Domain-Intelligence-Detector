"""Selected model independence and fitted-preprocessor compatibility checks."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_rdap_alignment import live


class SelectedModelTests(unittest.TestCase):
    def objects(self):
        names = [feature for group in live.GROUPS.values() for feature in group]
        self.assertEqual(len(names), 80)
        prep = SimpleNamespace(feature_names_in_=names, get_feature_names_out=lambda: list(range(80)))
        model = SimpleNamespace(n_features_in_=80)
        return model, prep

    def test_loads_only_tuned80_pair_with_comparison_files_unavailable(self):
        model, prep = self.objects()

        def load(path):
            if path == live.TUNED_MODEL:
                return model
            if path == live.TUNED_PREPROCESSOR:
                return prep
            raise AssertionError(f"Unexpected artifact loaded: {path}")

        unavailable = live.ROOT / "work" / "nonexistent_comparison_model.joblib"
        with patch.object(live, "ORIGINAL_MODEL", unavailable), \
                patch.object(live, "ORIGINAL_PREPROCESSOR", unavailable), \
                patch.object(live, "FEATURE40_MODEL", unavailable), \
                patch.object(live, "FEATURE40_PREPROCESSOR", unavailable), \
                patch.object(live.joblib, "load", side_effect=load) as loader:
            label, loaded_model, loaded_prep, names = live.load_selected_model()
        self.assertEqual(label, "TUNED 80")
        self.assertIs(loaded_model, model)
        self.assertIs(loaded_prep, prep)
        self.assertEqual(len(names), 80)
        self.assertEqual([call.args[0] for call in loader.call_args_list],
                         [live.TUNED_PREPROCESSOR, live.TUNED_MODEL])

    def test_rejects_incompatible_model_and_preprocessor(self):
        model, prep = self.objects()
        model.n_features_in_ = 81
        with patch.object(live, "load_pair", return_value=(model, prep)):
            with self.assertRaisesRegex(RuntimeError, "processed features"):
                live.load_selected_model()

    def test_rejects_features_without_a_matching_collector(self):
        model, prep = self.objects()
        prep.feature_names_in_[0] = "unsupported_feature"
        with patch.object(live, "load_pair", return_value=(model, prep)):
            with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
                live.load_selected_model()


if __name__ == "__main__":
    unittest.main()
