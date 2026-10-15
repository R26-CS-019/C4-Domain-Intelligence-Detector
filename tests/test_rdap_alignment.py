"""Offline regressions for collection and training/live RDAP feature parity."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


live = load_module("rdap_live_test", ROOT / "src/08_test_domain.py")
training = load_module("rdap_training_test", ROOT / "src/01_preprocess_dataset.py")
NOW = datetime(2026, 10, 14, 0, 0, tzinfo=timezone.utc)


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is not None else NOW.replace(tzinfo=None)


def domain(entities=None):
    return {"objectClassName": "domain", "handle": "TEST", "ldhName": "example.test",
            "entities": entities or []}


def extract(raw):
    features = {key: np.nan for key in training.extract_rdap_features({})}
    with patch.object(live, "find_domain_rdap", return_value=(raw, "example.test", "https://rdap.example.test", [])), \
            patch.object(live, "find_domain_whois", return_value=(None, None, None)), \
            patch.object(live, "datetime", FrozenDatetime), contextlib.redirect_stdout(io.StringIO()):
        status = live.collect_rdap("example.test", features)
    return features, status


class RDAPAlignmentTests(unittest.TestCase):
    def test_nested_contact_is_retained(self):
        raw = domain([{"handle": "R", "roles": ["registrar"], "entities": [
            {"objectClassName": "entity", "roles": ["abuse"]}]}])
        self.assertEqual(live.role_counts(raw)["abuse"], 1)

    def test_role_without_any_parseable_fields_is_not_counted(self):
        self.assertEqual(live.role_counts(domain([{"roles": ["abuse"]}]))["abuse"], 0)

    def test_empty_email_with_entity_type_is_counted(self):
        raw = domain([{"objectClassName": "entity", "roles": ["abuse"],
                       "vcardArray": ["vcard", [["email", {}, "text", ""]]]}])
        self.assertEqual(live.role_counts(raw)["abuse"], 1)

    def test_same_handle_different_contact_is_not_collapsed(self):
        raw = domain([{"handle": "A", "roles": ["abuse"],
                       "vcardArray": ["vcard", [["email", {}, "text", address]]]} 
                      for address in ("one@example.test", "two@example.test")])
        self.assertEqual(live.role_counts(raw)["abuse"], 2)

    def test_case_duplicate_handles_are_collapsed(self):
        raw = domain([{"handle": handle, "roles": ["abuse"]} for handle in ("a", "A")])
        self.assertEqual(live.role_counts(raw)["abuse"], 1)

    def test_all_rdap_features_match_training_with_same_observation_time(self):
        raw = domain([{"handle": "R", "roles": ["registrar", "admin"]},
                      {"objectClassName": "entity", "roles": ["abuse", "technical"]}])
        raw.update({"events": [
            {"eventAction": "registration", "eventDate": "2020-01-01T00:00:00Z"},
            {"eventAction": "expiration", "eventDate": "2027-01-01T00:00:00Z"},
            {"eventAction": "last changed", "eventDate": "2025-01-01T00:00:00Z"}],
            "nameservers": [{"objectClassName": "nameserver", "ldhName": "ns.example.test"}, {}],
            "status": [" active "], "port43": " whois.example.test ",
            "remarks": [{"description": ["one", "two"]}], "secureDNS": {"delegationSigned": False}})
        normalized = live.normalize_domain_rdap(raw, "example.test")
        normalized = json.loads(json.dumps(normalized, default=lambda value: value.isoformat()))
        expected = training.extract_rdap_features({"rdap": normalized, "evaluated_on": NOW.isoformat()})
        actual, status = extract(raw)
        self.assertTrue(status["available"])
        self.assertEqual(actual, expected)

    def test_dates_without_timezone_match_mongodb_utc_semantics(self):
        raw = domain()
        raw["events"] = [{"eventAction": "registration", "eventDate": "2026-10-13T00:00:00"}]
        features, _ = extract(raw)
        self.assertEqual(features["domain_age_days"], 1)

    def test_absent_secure_dns_is_present_false_in_normalized_data(self):
        features, _ = extract(domain())
        self.assertEqual(features["rdap_dnssec_information_present"], 1)

    def test_registrar_overlay_preserves_registry_input_and_empty_fields(self):
        raw = domain([{"handle": "R", "roles": ["registrar"]}])
        raw["status"] = ["active"]
        raw["links"] = [{"rel": "related", "href": "https://rdap.example.test/domain/example.test"}]
        original = copy.deepcopy(raw)
        registrar = domain([{"handle": "T", "roles": ["technical"]}])
        registrar["status"] = []
        messages = []
        with patch.object(live, "request_json", return_value=registrar):
            merged = live.follow_related_domain_rdap(None, raw, "example.test", messages)
        self.assertEqual(raw, original)
        self.assertEqual(live.role_counts(merged)["technical"], 1)
        self.assertEqual(merged["status"], ["active"])
        self.assertTrue(any("Successful related" in entry for entry in messages))

    def test_failed_or_mismatched_referral_returns_registry_with_diagnostic(self):
        raw = domain()
        raw["links"] = [{"rel": "registration", "href": "https://rdap.example.test/domain/example.test"}]
        for response in (None, {"objectClassName": "domain", "ldhName": "different.test"}):
            with self.subTest(response=response), patch.object(live, "request_json", return_value=response):
                messages = []
                self.assertEqual(live.follow_related_domain_rdap(None, raw, "example.test", messages), raw)
                self.assertTrue(any("registry data only" in entry for entry in messages))

    def test_invalid_rdap_remains_missing_instead_of_fabricating_counts(self):
        raw = domain()
        raw.pop("handle")
        features, status = extract(raw)
        self.assertFalse(status["available"])
        self.assertTrue(all(np.isnan(value) for value in features.values()))


if __name__ == "__main__":
    unittest.main()
