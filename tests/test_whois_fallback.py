"""Offline tests for WHOIS fallback and parity with training feature definitions."""
from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock

import numpy as np

from test_rdap_alignment import FrozenDatetime, NOW, domain, live, training


# Synthetic registration record: no real contact details are retained here.
WHOIS_RECORD = """Domain Name: example.us
Registry Domain ID: EXAMPLE
Registrar: Example Registrar
Registrar WHOIS Server: whois.example.test
Registrar Abuse Contact Email: abuse@example.test
Updated Date: 2026-10-08T13:13:21Z
Creation Date: 2026-08-09T13:13:18Z
Registry Expiry Date: 2027-08-09T13:13:18Z
Domain Status: clientTransferProhibited https://example.test/status
Domain Status: clientTransferProhibited
Name Server: ns1.example.test
Name Server: ns2.example.test
DNSSEC: unsigned
"""

CHILE_RECORD = """Registrant name: Example Organisation
Registrar name: NIC Chile
Registrar URL: https://www.nic.cl
Creation date: 2008-12-01 23:44:40 CLST
Expiration date: 2031-12-28 20:44:40 CLST
Name server: ns1.example.test
Name server: ns2.example.test

%% For communication with domain contacts please use website.
%% See https://www.nic.cl/registry/Whois.do?d=example.cl
"""

DENIC_RECORD = """% Copyright DENIC
Domain: example.de
Nserver: ns1.example.test
Nserver: ns2.example.test
Status: connect
Changed: 2026-06-11T09:26:02+02:00
"""


class WhoisFallbackTests(unittest.TestCase):
    def features(self):
        return {key: np.nan for key in training.extract_rdap_features({})}

    def collect(self, rdap=None, reply=WHOIS_RECORD, transport_error=None):
        features = self.features()
        with patch.object(live, "find_domain_rdap", return_value=(rdap, "example.us", "https://rdap.example.test", [])), \
                patch.object(live, "discover_whois_server", return_value="whois.example.test"), \
                patch.object(live, "query_whois_server", return_value=reply, side_effect=transport_error), \
                patch.object(live, "datetime", FrozenDatetime), contextlib.redirect_stdout(io.StringIO()):
            status = live.collect_rdap("example.us", features)
        return features, status

    def test_fallback_features_match_training_at_same_time(self):
        parsed = live.parse_collector_whois(WHOIS_RECORD, "example.us")
        self.assertIsNotNone(parsed)
        normalized = live.whois_to_collector_record(parsed)
        normalized = json.loads(json.dumps(normalized, default=lambda value: value.isoformat()))
        expected = training.extract_rdap_features({"rdap": normalized, "evaluated_on": NOW.isoformat()})
        features, status = self.collect()
        self.assertTrue(status["available"])
        self.assertEqual(status["method"], "WHOIS")
        self.assertEqual(status["source"], "whois://whois.example.test/example.us")
        self.assertEqual(features, expected)
        self.assertAlmostEqual(features["domain_age_days"], 65.4491)
        self.assertEqual(features["rdap_nameserver_count"], 2)
        self.assertEqual(features["rdap_status_count"], 1)

    def test_original_contact_placeholders_and_absent_fields_are_preserved(self):
        parsed = live.parse_collector_whois(WHOIS_RECORD, "example.us")
        parsed.registrar = None
        record = live.whois_to_collector_record(parsed)
        self.assertEqual(record["entities"]["registrar"], [{"name": ""}])
        self.assertEqual(record["entities"]["registrant"], [{"name": ""}])
        self.assertEqual(record["entities"]["admin"], [{"name": ""}])
        self.assertEqual(live.role_counts(record), {
            "registrar": 1, "registrant": 1, "abuse": 1,
            "admin": 1, "administrative": 0, "technical": 0})
        self.assertEqual(record["handle"], "")
        self.assertEqual(record["whois_server"], "")
        self.assertNotIn("dnssec", record)
        features, _ = self.collect()
        self.assertEqual(features["rdap_dnssec_information_present"], 0)
        self.assertEqual(features["rdap_has_handle"], 0)
        self.assertEqual(features["rdap_has_whois_server"], 0)

    def test_unparseable_rdap_uses_whois(self):
        raw = domain()
        raw.pop("handle")
        features, status = self.collect(rdap=raw)
        self.assertTrue(status["available"])
        self.assertEqual(status["method"], "WHOIS")
        self.assertEqual(features["rdap_available"], 1)

    def test_valid_rdap_does_not_query_whois(self):
        raw = domain()
        raw["ldhName"] = "example.us"
        with patch.object(live, "find_domain_rdap", return_value=(raw, "example.us", "https://rdap.example.test", [])), \
                patch.object(live, "find_domain_whois", side_effect=AssertionError("Unexpected fallback")), \
                contextlib.redirect_stdout(io.StringIO()):
            status = live.collect_rdap("example.us", self.features())
        self.assertEqual(status["method"], "RDAP")

    def test_missing_mismatched_and_quota_replies_leave_features_missing(self):
        replies = ("No match for example.us", WHOIS_RECORD.replace("example.us", "different.us"),
                   "Quota exceeded")
        for reply in replies:
            with self.subTest(reply=reply[:40]):
                features, status = self.collect(reply=reply)
                self.assertFalse(status["available"])
                self.assertTrue(all(np.isnan(value) for value in features.values()))

    def test_timeout_leaves_features_missing(self):
        features, status = self.collect(transport_error=TimeoutError("Timed out"))
        self.assertFalse(status["available"])
        self.assertTrue(all(np.isnan(value) for value in features.values()))
        self.assertTrue(any("TimeoutError" in message for message in status["diagnostics"]))

    def test_chile_invalid_hostname_reply_cannot_supply_registration(self):
        reply = "%% This is the NIC Chile Whois server (whois.nic.cl).\nInvalid domain name: www.example.cl\n"
        self.assertIsNone(live.parse_collector_whois(reply, "www.example.cl"))

    def test_chile_current_format_uses_exact_registry_identity(self):
        parsed = live.parse_collector_whois(CHILE_RECORD, "example.cl")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.name, "example.cl")
        self.assertEqual(parsed.creation_date.isoformat(), "2008-12-01T23:44:40")
        self.assertEqual(parsed.expiration_date.isoformat(), "2031-12-28T20:44:40")
        self.assertEqual(len(parsed.name_servers), 2)
        record = live.whois_to_collector_record(parsed)
        normalized = json.loads(json.dumps(record, default=lambda value: value.isoformat()))
        expected = training.extract_rdap_features({"rdap": normalized, "evaluated_on": NOW.isoformat()})
        features = self.features()
        with patch.object(live, "find_domain_rdap", return_value=(None, None, None, [])), \
                patch.object(live, "discover_whois_server", return_value="whois.nic.cl"), \
                patch.object(live, "query_whois_server", return_value=CHILE_RECORD), \
                patch.object(live, "datetime", FrozenDatetime), contextlib.redirect_stdout(io.StringIO()):
            status = live.collect_rdap("example.cl", features)
        self.assertEqual(status["source"], "whois://whois.nic.cl/example.cl")
        for name, value in expected.items():
            if value is None:
                self.assertTrue(np.isnan(features[name]), name)
            else:
                self.assertEqual(features[name], value, name)

    def test_chile_reply_for_another_domain_or_without_identity_is_rejected(self):
        self.assertIsNone(live.parse_collector_whois(CHILE_RECORD, "www.example.cl"))
        self.assertIsNone(live.parse_collector_whois(CHILE_RECORD.replace("d=example.cl", "d=other.cl"), "example.cl"))
        self.assertIsNone(live.parse_collector_whois(CHILE_RECORD.split("%%")[0], "example.cl"))

    def test_chile_subdomain_falls_back_to_usable_parent_registration(self):
        invalid = "%% whois.nic.cl\nInvalid domain name: www.example.cl\n"
        with patch.object(live, "discover_whois_server", return_value="whois.nic.cl"), \
                patch.object(live, "query_whois_server", side_effect=[invalid, CHILE_RECORD]) as query:
            record, domain_name, source = live.find_domain_whois("www.example.cl", [])
        self.assertEqual(domain_name, "example.cl")
        self.assertEqual(record["name"], "example.cl")
        self.assertEqual(source, "whois://whois.nic.cl/example.cl")
        self.assertEqual([call.args[1] for call in query.call_args_list], ["www.example.cl", "example.cl"])

    def test_error_reply_with_a_matching_domain_and_registrar_is_rejected(self):
        for message in ("No match for example.us", "Quota exceeded", "Too many queries", "Not registered"):
            with self.subTest(message=message):
                self.assertIsNone(live.parse_collector_whois(message + "\n" + WHOIS_RECORD, "example.us"))

    def test_denic_known_referral_does_not_depend_on_iana_connection(self):
        with patch.object(live, "query_whois_server", side_effect=ConnectionResetError("IANA unavailable")) as query:
            self.assertEqual(live.discover_whois_server("de"), "whois.denic.de")
        query.assert_not_called()

    def test_denic_transport_requests_technical_records_only_for_denic(self):
        for server, payload in (("whois.denic.de", b"-T dn example.de\r\n"),
                                ("whois.nic.cl", b"example.de\r\n")):
            with self.subTest(server=server):
                connection = MagicMock()
                connection.__enter__.return_value = connection
                connection.recv.side_effect = [b"reply", b""]
                with patch.object(live.socket, "create_connection", return_value=connection):
                    self.assertEqual(live.query_whois_server(server, "example.de"), "reply")
                connection.sendall.assert_called_once_with(payload)

    def test_denic_caller_cannot_supply_custom_query_options(self):
        with patch.object(live.socket, "create_connection") as connect:
            with self.assertRaisesRegex(ValueError, "Invalid WHOIS"):
                live.query_whois_server("whois.denic.de", "-T dn example.de")
        connect.assert_not_called()

    def test_denic_features_match_training_and_absent_dates_stay_missing(self):
        parsed = live.parse_collector_whois(DENIC_RECORD, "example.de")
        self.assertIsNotNone(parsed)
        record = live.whois_to_collector_record(parsed)
        normalized = json.loads(json.dumps(record, default=lambda value: value.isoformat()))
        expected = training.extract_rdap_features({"rdap": normalized, "evaluated_on": NOW.isoformat()})
        features = self.features()
        with patch.object(live, "find_domain_rdap", return_value=(None, None, None, [])), \
                patch.object(live, "query_whois_server", return_value=DENIC_RECORD), \
                patch.object(live, "datetime", FrozenDatetime), contextlib.redirect_stdout(io.StringIO()):
            status = live.collect_rdap("example.de", features)
        self.assertEqual(status["source"], "whois://whois.denic.de/example.de")
        self.assertEqual(features["rdap_nameserver_count"], 2)
        self.assertEqual(features["rdap_status_count"], 1)
        self.assertEqual(features["rdap_has_registration_date"], 0)
        self.assertEqual(features["rdap_has_expiration_date"], 0)
        self.assertTrue(np.isnan(features["domain_age_days"]))
        self.assertTrue(np.isnan(features["days_until_expiration"]))
        for name, value in expected.items():
            if value is None:
                self.assertTrue(np.isnan(features[name]), name)
            else:
                self.assertEqual(features[name], value, name)


if __name__ == "__main__":
    unittest.main()
