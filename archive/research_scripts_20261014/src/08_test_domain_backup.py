
r"""
08 - Compare ORIGINAL 80, TUNED 80, and TUNED 40 LightGBM models on a live domain.

Run from the project root:
    python .\src\08_test_domain.py google.com --details
    python .\src\08_test_domain.py wikipedia.org --explain
    python .\src\08_test_domain.py microsoft.com --details --explain
    python .\src\08_test_domain.py example.com --save-features

Install dependencies if required:
    pip install dnspython requests cryptography pandas numpy joblib lightgbm scikit-learn

IMPORTANT RESEARCH LIMITATION:
The training dataset contains historical enrichment features which a live lookup
cannot reproduce perfectly. These are left missing (NaN), NOT fabricated. All three
models receive the SAME live intelligence collection, through their own saved
preprocessors (80, 80, and 40 raw features respectively). The new 40-feature
model is NOT a modification of the existing 80-feature models. These outputs
are preliminary until validated on independent, verified live-domain samples.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import dns.resolver
import joblib
import numpy as np
import pandas as pd
import requests
from cryptography import x509
from cryptography.x509.oid import NameOID

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

ORIGINAL_MODEL = MODELS / "lightgbm_phishing_model.joblib"
ORIGINAL_PREPROCESSOR = MODELS / "training_preprocessor.joblib"
ORIGINAL_CALIBRATOR = MODELS / "lightgbm_probability_calibrator.joblib"

TUNED_MODEL = MODELS / "lightgbm_80_tuned_model.joblib"
TUNED_PREPROCESSOR = MODELS / "training_preprocessor_80_tuned.joblib"

FEATURE40_MODEL = MODELS / "lightgbm_40_tuned_model.joblib"
FEATURE40_PREPROCESSOR = MODELS / "training_preprocessor_40_tuned.joblib"
FEATURE40_LIST = RESULTS / "lightgbm_40_selected_features.json"

THRESHOLD = 0.50
DNS_TIMEOUT = 8.0
RDAP_TIMEOUT = 5.0
TLS_TIMEOUT = 5.0
MAX_IP_RDAP_LOOKUPS = 3

# Feature names were checked against 01_preprocess_dataset.py.
GROUPS = {
    "DNS": (
        "dns_a_count",
        "dns_aaaa_count",
        "dns_has_a",
        "dns_has_aaaa",
        "dns_mx_count",
        "dns_has_mx",
        "dns_mx_related_ip_count",
        "dns_mx_min_priority",
        "dns_mx_max_priority",
        "dns_mx_mean_priority",
        "dns_ns_count",
        "dns_has_ns",
        "dns_ns_related_ip_count",
        "dns_txt_count",
        "dns_has_txt",
        "dns_has_cname",
        "dns_ipv4_count",
        "dns_ipv6_count",
        "dns_unique_ip_count",
        "dnssec_information_present",
        "dns_has_soa",
        "dns_has_zone_soa",
        "dns_source_count",
        "dns_ttl_information_count",
    ),
    "RDAP": (
        "rdap_available",
        "domain_age_days",
        "registration_period_days",
        "days_until_expiration",
        "days_since_last_change",
        "rdap_has_registration_date",
        "rdap_has_expiration_date",
        "rdap_has_last_changed_date",
        "rdap_has_whois_server",
        "rdap_has_name",
        "rdap_has_handle",
        "rdap_registrar_count",
        "rdap_registrant_count",
        "rdap_abuse_count",
        "rdap_admin_count",
        "rdap_technical_count",
        "rdap_has_registrar",
        "rdap_has_registrant",
        "rdap_has_abuse_contact",
        "rdap_nameserver_count",
        "rdap_status_count",
        "rdap_dnssec_information_present",
        "rdap_description_count",
    ),
    "TLS": (
        "tls_available",
        "tls_certificate_count",
        "tls_non_root_certificate_count",
        "tls_root_certificate_count",
        "tls_has_leaf_certificate",
        "tls_has_common_name",
        "tls_has_organization",
        "tls_has_country",
        "tls_certificate_age_days",
        "tls_certificate_validity_days",
        "tls_certificate_days_remaining",
        "tls_certificate_expired",
        "tls_extension_count",
        "tls_critical_extension_count",
        "tls_protocol",
        "tls_cipher",
    ),
    "IP": (
        "ip_data_available",
        "ip_record_count",
        "unique_ip_count",
        "ip_ipv4_count",
        "ip_ipv6_count",
        "ip_from_a_count",
        "ip_from_aaaa_count",
        "ip_from_cname_count",
        "ip_alive_count",
        "ip_alive_ratio",
        "ip_average_rtt",
        "ip_rdap_available_count",
        "asn_count",
        "asn_organization_count",
        "country_count",
        "country_code_count",
        "rir_count",
    ),
}


# ============================================================
# 1. GENERAL HELPERS
# ============================================================

def heading(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def normalize_host(raw: str) -> str:
    parsed = urlparse(
        raw.strip()
        if "://" in raw
        else "//" + raw.strip()
    )

    host = (
        parsed.hostname or ""
    ).strip().rstrip(".").lower()

    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError(
            "Invalid internationalized domain"
        ) from exc

    if (
        not host
        or len(host) > 253
        or not re.fullmatch(r"[a-z0-9.-]+", host)
    ):
        raise ValueError(
            "Enter a valid hostname or URL"
        )

    if any(
        not label or len(label) > 63
        for label in host.split(".")
    ):
        raise ValueError(
            "Invalid DNS hostname labels"
        )

    return host


def put(features: dict, **values) -> None:
    for key, value in values.items():
        if key in features:
            features[key] = value


def parsed_utc_date(value) -> datetime | None:
    if not value:
        return None

    try:
        dt = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )

        return (
            dt
            if dt.tzinfo
            else dt.replace(tzinfo=timezone.utc)
        )

    except (TypeError, ValueError):
        return None


def days_between(
    start: datetime | None,
    end: datetime | None,
):
    if start is None or end is None:
        return np.nan

    return round(
        (end - start).total_seconds() / 86400,
        4,
    )


# ============================================================
# 2. DNS COLLECTION
# ============================================================

@dataclass
class DNSResult:
    items: list
    status: str  # ok, noanswer, nxdomain, error


def dns_lookup(
    host: str,
    kind: str,
) -> DNSResult:

    resolver = dns.resolver.Resolver()
    resolver.timeout = DNS_TIMEOUT
    resolver.lifetime = DNS_TIMEOUT

    try:
        return DNSResult(
            list(resolver.resolve(host, kind)),
            "ok",
        )

    except dns.resolver.NoAnswer:
        return DNSResult([], "noanswer")

    except dns.resolver.NXDOMAIN:
        return DNSResult([], "nxdomain")

    except Exception as error:
        print(
            f"  DNS lookup failed for {kind} ({host}): "
            f"{type(error).__name__}: {error}"
        )
        return DNSResult([], "error")


def dns_count(result: DNSResult):
    return (
        len(result.items)
        if result.status in ("ok", "noanswer")
        else np.nan
    )


def dns_has(result: DNSResult):
    count = dns_count(result)

    return (
        int(count > 0)
        if pd.notna(count)
        else np.nan
    )


def resolved_related_ips(
    names: list[str],
):
    ips = set()
    attempted = False

    for name in names:
        for kind in ("A", "AAAA"):

            result = dns_lookup(name, kind)

            if result.status == "error":
                continue

            attempted = True

            ips.update(
                str(item)
                for item in result.items
            )

    return (
        len(ips)
        if attempted or not names
        else np.nan
    )


def collect_dns(
    host: str,
    features: dict,
) -> dict:

    heading("DNS INTELLIGENCE")

    kinds = (
        "A",
        "AAAA",
        "MX",
        "NS",
        "TXT",
        "CNAME",
        "SOA",
    )

    records = {
        kind: dns_lookup(host, kind)
        for kind in kinds
    }

    ipv4 = sorted({
        str(item)
        for item in records["A"].items
    })

    ipv6 = sorted({
        str(item)
        for item in records["AAAA"].items
    })

    ips = ipv4 + ipv6

    mx_items = records["MX"].items
    ns_items = records["NS"].items

    priorities = [
        float(mx.preference)
        for mx in mx_items
    ]

    mx_hosts = [
        str(mx.exchange).rstrip(".")
        for mx in mx_items
    ]

    ns_hosts = [
        str(ns).rstrip(".")
        for ns in ns_items
    ]

    # Historical DNS sources, TTL metadata,
    # zone SOA and DNSSEC structures are not
    # reproduced from ordinary live DNS queries.

    put(
        features,

        dns_a_count=dns_count(
            records["A"]
        ),

        dns_aaaa_count=dns_count(
            records["AAAA"]
        ),

        dns_has_a=dns_has(
            records["A"]
        ),

        dns_has_aaaa=dns_has(
            records["AAAA"]
        ),

        dns_mx_count=dns_count(
            records["MX"]
        ),

        dns_has_mx=dns_has(
            records["MX"]
        ),

        dns_mx_related_ip_count=(
            resolved_related_ips(mx_hosts)
            if records["MX"].status in (
                "ok",
                "noanswer",
            )
            else np.nan
        ),

        dns_mx_min_priority=(
            min(priorities)
            if priorities
            else np.nan
        ),

        dns_mx_max_priority=(
            max(priorities)
            if priorities
            else np.nan
        ),

        dns_mx_mean_priority=(
            sum(priorities) / len(priorities)
            if priorities
            else np.nan
        ),

        dns_ns_count=dns_count(
            records["NS"]
        ),

        dns_has_ns=dns_has(
            records["NS"]
        ),

        dns_ns_related_ip_count=(
            resolved_related_ips(ns_hosts)
            if records["NS"].status in (
                "ok",
                "noanswer",
            )
            else np.nan
        ),

        dns_txt_count=dns_count(
            records["TXT"]
        ),

        dns_has_txt=dns_has(
            records["TXT"]
        ),

        dns_has_cname=dns_has(
            records["CNAME"]
        ),

        dns_ipv4_count=(
            len(ipv4)
            if records["A"].status in (
                "ok",
                "noanswer",
            )
            else np.nan
        ),

        dns_ipv6_count=(
            len(ipv6)
            if records["AAAA"].status in (
                "ok",
                "noanswer",
            )
            else np.nan
        ),

        dns_unique_ip_count=(
            len(ips)
            if (
                records["A"].status in (
                    "ok",
                    "noanswer",
                )
                and records["AAAA"].status in (
                    "ok",
                    "noanswer",
                )
            )
            else np.nan
        ),

        dns_has_soa=dns_has(
            records["SOA"]
        ),
    )

    for kind in (
        "A",
        "AAAA",
        "MX",
        "NS",
        "TXT",
        "CNAME",
    ):

        result = records[kind]

        shown = (
            str(len(result.items))
            if result.status in (
                "ok",
                "noanswer",
            )
            else result.status.upper()
        )

        print(
            f"  {kind:6s}: {shown}"
        )

    available = any(
        records[k].items
        for k in kinds
    )

    return {
        "available": available,
        "ips": ips,
        "a_ips": ipv4,
        "aaaa_ips": ipv6,
        "cname_hosts": [
            str(item.target).rstrip(".")
            for item in records["CNAME"].items
        ],
    }


# ============================================================
# 3. RDAP / WHOIS COLLECTION
# ============================================================

def fetch_rdap(url: str):
    try:
        response = requests.get(
            url,
            timeout=RDAP_TIMEOUT,
            headers={
                "User-Agent":
                    "AcademicDomainIntelligencePrototype/1.0"
            },
        )

        if response.ok:
            value = response.json()

            return (
                value
                if isinstance(value, dict)
                else None
            )

    except (
        requests.RequestException,
        ValueError,
    ):
        pass

    return None


def find_domain_rdap(host: str):
    labels = host.split(".")

    # Try hostname and progressively shorter
    # parent domains for subdomain support.

    for index in range(
        max(1, len(labels) - 1)
    ):

        name = ".".join(
            labels[index:]
        )

        if name.count(".") < 1:
            continue

        data = fetch_rdap(
            "https://rdap.org/domain/" + name
        )

        if data and (
            data.get("objectClassName") == "domain"
            or data.get("events")
            or data.get("nameservers")
        ):
            return data, name

    return None, None


def collect_rdap(
    host: str,
    features: dict,
) -> dict:

    heading(
        "RDAP / WHOIS INTELLIGENCE"
    )

    data, queried_domain = find_domain_rdap(
        host
    )

    if data is None:

        print(
            "  RDAP unavailable "
            "(lookup failed or registry did not respond)"
        )

        return {
            "available": False
        }

    today = datetime.now(
        timezone.utc
    )

    events = data.get("events") or []

    event_dates = {}

    for event in events:

        if not isinstance(event, dict):
            continue

        action = str(
            event.get("eventAction", "")
        ).lower()

        value = parsed_utc_date(
            event.get("eventDate")
        )

        if value is not None:
            event_dates.setdefault(
                action,
                value,
            )

    created = event_dates.get(
        "registration"
    )

    expired = event_dates.get(
        "expiration"
    )

    changed = (
        event_dates.get("last changed")
        or event_dates.get("last update")
    )

    roles = {
        r: 0
        for r in (
            "registrar",
            "registrant",
            "abuse",
            "admin",
            "administrative",
            "technical",
        )
    }

    for entity in data.get(
        "entities"
    ) or []:

        if not isinstance(
            entity,
            dict,
        ):
            continue

        for role in entity.get(
            "roles"
        ) or []:

            if role in roles:
                roles[role] += 1

    name_servers = data.get(
        "nameservers"
    )

    status = data.get(
        "status"
    )

    remarks = data.get(
        "remarks"
    )

    put(
        features,

        rdap_available=1,

        domain_age_days=days_between(
            created,
            today,
        ),

        registration_period_days=days_between(
            created,
            expired,
        ),

        days_until_expiration=days_between(
            today,
            expired,
        ),

        days_since_last_change=days_between(
            changed,
            today,
        ),

        rdap_has_registration_date=int(
            created is not None
        ),

        rdap_has_expiration_date=int(
            expired is not None
        ),

        rdap_has_last_changed_date=int(
            changed is not None
        ),

        rdap_has_whois_server=int(
            bool(data.get("port43"))
        ),

        rdap_has_name=int(
            bool(
                data.get("ldhName")
                or data.get("unicodeName")
            )
        ),

        rdap_has_handle=int(
            bool(data.get("handle"))
        ),

        rdap_registrar_count=roles[
            "registrar"
        ],

        rdap_registrant_count=roles[
            "registrant"
        ],

        rdap_abuse_count=roles[
            "abuse"
        ],

        rdap_admin_count=(
            roles["admin"]
            + roles["administrative"]
        ),

        rdap_technical_count=roles[
            "technical"
        ],

        rdap_has_registrar=int(
            roles["registrar"] > 0
        ),

        rdap_has_registrant=int(
            roles["registrant"] > 0
        ),

        rdap_has_abuse_contact=int(
            roles["abuse"] > 0
        ),

        rdap_nameserver_count=(
            len(name_servers)
            if name_servers is not None
            else 0
        ),

        rdap_status_count=(
            len(status)
            if status is not None
            else 0
        ),

        rdap_dnssec_information_present=int(
            data.get("secureDNS") is not None
        ),

        rdap_description_count=(
            len(remarks)
            if remarks is not None
            else 0
        ),
    )

    print(
        f"  RDAP domain: {queried_domain}"
    )

    print(
        "  Domain age:",
        days_between(created, today),
        "days",
    )

    print(
        f"  RDAP statuses: {len(status or [])}"
    )

    return {
        "available": True
    }


# ============================================================
# 4. TLS CERTIFICATE COLLECTION
# ============================================================

def tls_connect(
    host: str,
    verify: bool,
):
    context = (
        ssl.create_default_context()
        if verify
        else ssl._create_unverified_context()
    )

    with socket.create_connection(
        (host, 443),
        timeout=TLS_TIMEOUT,
    ) as sock:

        sock.settimeout(
            TLS_TIMEOUT
        )

        with context.wrap_socket(
            sock,
            server_hostname=host,
        ) as secure:

            return (
                secure.getpeercert(
                    binary_form=True
                ),
                secure.version(),
                secure.cipher(),
            )


def cert_has_name(
    cert: x509.Certificate,
    oid,
) -> int:

    return int(
        bool(
            cert.subject.get_attributes_for_oid(
                oid
            )
        )
    )


def collect_tls(
    host: str,
    features: dict,
) -> dict:

    heading(
        "TLS CERTIFICATE INTELLIGENCE"
    )

    validation_error = None

    try:

        der, protocol, cipher = tls_connect(
            host,
            verify=True,
        )

    except ssl.SSLCertVerificationError as error:

        validation_error = str(error)

        print(
            "  Certificate validation FAILED. "
            "Retrying only to inspect metadata."
        )

        try:

            der, protocol, cipher = tls_connect(
                host,
                verify=False,
            )

        except (
            OSError,
            ssl.SSLError,
        ) as second:

            print(
                "  TLS unavailable:",
                second,
            )

            return {
                "available": False
            }

    except (
        OSError,
        ssl.SSLError,
    ) as error:

        print(
            "  TLS unavailable:",
            error,
        )

        return {
            "available": False
        }

    cert = x509.load_der_x509_certificate(
        der
    )

    start = getattr(
        cert,
        "not_valid_before_utc",
        None,
    )

    end = getattr(
        cert,
        "not_valid_after_utc",
        None,
    )

    if start is None:
        start = cert.not_valid_before.replace(
            tzinfo=timezone.utc
        )

    if end is None:
        end = cert.not_valid_after.replace(
            tzinfo=timezone.utc
        )

    now = datetime.now(
        timezone.utc
    )

    extensions = list(
        cert.extensions
    )

    put(
        features,

        tls_available=1,

        # Full certificate-chain counts are
        # deliberately left missing.

        tls_has_leaf_certificate=1,

        tls_has_common_name=cert_has_name(
            cert,
            NameOID.COMMON_NAME,
        ),

        tls_has_organization=cert_has_name(
            cert,
            NameOID.ORGANIZATION_NAME,
        ),

        tls_has_country=cert_has_name(
            cert,
            NameOID.COUNTRY_NAME,
        ),

        tls_certificate_age_days=days_between(
            start,
            now,
        ),

        tls_certificate_validity_days=days_between(
            start,
            end,
        ),

        tls_certificate_days_remaining=days_between(
            now,
            end,
        ),

        tls_certificate_expired=int(
            end < now
        ),

        tls_extension_count=len(
            extensions
        ),

        tls_critical_extension_count=sum(
            bool(extension.critical)
            for extension in extensions
        ),

        tls_protocol=protocol,

        tls_cipher=(
            cipher[0]
            if cipher
            else np.nan
        ),
    )

    print(
        f"  Protocol: {protocol}; "
        f"validity: {days_between(start, end)} days"
    )

    print(
        "  Certificate verified:",
        validation_error is None,
    )

    if validation_error:

        print(
            "  Verification warning:",
            validation_error[:180],
        )

    return {
        "available": True,
        "certificate_verified": (
            validation_error is None
        ),
    }


# ============================================================
# 5. IP / HOSTING COLLECTION
# ============================================================

def collect_ip(
    dns_data: dict,
    features: dict,
    do_rdap: bool = True,
) -> dict:

    heading(
        "IP / HOSTING INTELLIGENCE"
    )

    a_ips = dns_data[
        "a_ips"
    ]

    aaaa_ips = dns_data[
        "aaaa_ips"
    ]

    ips = sorted(
        set(a_ips + aaaa_ips)
    )

    # Historical RTT, alive, ASN and
    # geolocation values are not invented.

    if ips:

        put(
            features,

            ip_data_available=1,

            ip_record_count=(
                len(a_ips)
                + len(aaaa_ips)
            ),

            unique_ip_count=len(
                ips
            ),

            ip_ipv4_count=len(
                a_ips
            ),

            ip_ipv6_count=len(
                aaaa_ips
            ),

            ip_from_a_count=len(
                a_ips
            ),

            ip_from_aaaa_count=len(
                aaaa_ips
            ),
        )

    else:

        print(
            "  No A/AAAA IP addresses resolved"
        )

    print(
        f"  Unique IPs: {len(ips)} "
        f"(IPv4={len(a_ips)}, IPv6={len(aaaa_ips)})"
    )

    print(
        "  Historical alive/RTT/ASN/geolocation: "
        "not reproducible; left missing"
    )

    if (
        do_rdap
        and ips
        and len(ips) <= MAX_IP_RDAP_LOOKUPS
    ):

        found = 0

        for ip in ips:

            if fetch_rdap(
                "https://rdap.org/ip/" + ip
            ):
                found += 1

        put(
            features,
            ip_rdap_available_count=found,
        )

        print(
            f"  IP RDAP lookups: "
            f"{found}/{len(ips)} successful"
        )

    elif (
        len(ips)
        > MAX_IP_RDAP_LOOKUPS
    ):

        print(
            "  IP RDAP not collected: "
            f"more than {MAX_IP_RDAP_LOOKUPS} IPs"
        )

    return {
        "available": bool(ips)
    }


# ============================================================
# 6. FEATURE QUALITY
# ============================================================

def nonmissing(value) -> bool:
    try:
        return bool(
            pd.notna(value)
        )

    except (
        TypeError,
        ValueError,
    ):
        return False


def feature_quality(
    features: dict,
    group_status: dict,
    details: bool,
):

    heading(
        "FEATURE QUALITY / AVAILABILITY"
    )

    for name, keys in GROUPS.items():

        selected = [
            f
            for f in keys
            if f in features
        ]

        observed = [
            f
            for f in selected
            if nonmissing(features[f])
        ]

        status = (
            "AVAILABLE"
            if group_status[name]["available"]
            else "UNAVAILABLE"
        )

        print(
            f"  {name:5s}: {status:11s} "
            f"| observed "
            f"{len(observed):2d}/{len(selected):2d} "
            "raw features"
        )

        if details:

            missing = [
                k
                for k in selected
                if k not in observed
            ]

            print(
                "         Missing:",
                ", ".join(missing)
                if missing
                else "none",
            )

    count = sum(
        nonmissing(v)
        for v in features.values()
    )

    groups = sum(
        bool(
            group_status[key]["available"]
        )
        for key in GROUPS
    )

    print(
        f"  Total observed: "
        f"{count}/{len(features)} raw features; "
        f"groups: {groups}/4"
    )

    print(
        "  Note: Zero is an observed value; "
        "missing/NaN is not a zero."
    )

    if groups == 0:

        print(
            "  DATA QUALITY: INSUFFICIENT DATA "
            "(do not trust a binary ML output)"
        )

        return "INSUFFICIENT DATA"

    if groups < 3:

        print(
            "  DATA QUALITY: LIMITED "
            "(ML predictions are tentative)"
        )

        return "LIMITED"

    print(
        "  DATA QUALITY: PARTIAL/AVAILABLE "
        "(still subject to training-live mismatch)"
    )

    return "PARTIAL"


# ============================================================
# 7. LOAD MODEL + PREPROCESSOR
# ============================================================

def load_pair(
    model_path: Path,
    preprocess_path: Path,
    label: str,
):

    missing = [
        str(p)
        for p in (
            model_path,
            preprocess_path,
        )
        if not p.exists()
    ]

    if missing:

        raise FileNotFoundError(
            f"{label} files missing: "
            + ", ".join(missing)
        )

    prep = joblib.load(
        preprocess_path
    )

    model = joblib.load(
        model_path
    )

    if not hasattr(
        prep,
        "feature_names_in_",
    ):

        raise RuntimeError(
            f"{label} preprocessor has "
            "no feature_names_in_"
        )

    return model, prep


# ============================================================
# 8. PREDICT USING CORRECT FEATURES
# ============================================================

def predict(
    model,
    preprocess,
    features: dict,
):

    names = list(
        preprocess.feature_names_in_
    )

    data = pd.DataFrame(
        [
            {
                k: features[k]
                for k in names
            }
        ],
        columns=names,
    )

    for name in (
        "tls_protocol",
        "tls_cipher",
    ):

        if name in data.columns:

            data[name] = data[name].astype(
                object
            )

    transformed = preprocess.transform(
        data
    )

    probability = float(
        model.predict_proba(
            transformed
        )[0, 1]
    )

    return probability, transformed


# ============================================================
# 9. LIGHTGBM FEATURE EXPLANATIONS
# ============================================================

def explain_prediction(
    model,
    preprocessor,
    matrix,
    label: str,
):

    try:

        contrib = model.booster_.predict(
            matrix,
            pred_contrib=True,
        )

        if hasattr(
            contrib,
            "toarray",
        ):
            contrib = contrib.toarray()

        values = np.asarray(
            contrib
        ).ravel()[:-1]

        names = preprocessor.get_feature_names_out()

        if len(values) != len(names):

            print(
                f"  {label}: explanation "
                "feature-name mismatch; skipped"
            )

            return

        indices = np.argsort(
            np.abs(values)
        )[-8:][::-1]

        print(
            f"\n  {label} — highest LightGBM "
            "contributions (log-odds):"
        )

        for index in indices:

            direction = (
                "phishing"
                if values[index] > 0
                else "legitimate"
            )

            print(
                f"    {str(names[index])[:48]:48s} "
                f"{values[index]:+8.4f} "
                f"-> {direction}"
            )

    except Exception as error:

        print(
            f"  Could not compute "
            f"{label} explanation: {error}"
        )


# ============================================================
# 10. LOAD ALL THREE TRAINED MODELS
# ============================================================

def load_three_models():
    """
    Load each model with its OWN
    fitted preprocessing pipeline.
    """

    configurations = (

        (
            "ORIGINAL 80",
            ORIGINAL_MODEL,
            ORIGINAL_PREPROCESSOR,
            80,
        ),

        (
            "TUNED 80",
            TUNED_MODEL,
            TUNED_PREPROCESSOR,
            80,
        ),

        (
            "TUNED 40",
            FEATURE40_MODEL,
            FEATURE40_PREPROCESSOR,
            40,
        ),
    )

    bundles = []

    for (
        label,
        model_path,
        prep_path,
        expected_count,
    ) in configurations:

        model, prep = load_pair(
            model_path,
            prep_path,
            label,
        )

        names = list(
            prep.feature_names_in_
        )

        if (
            len(names) != expected_count
            or len(set(names)) != expected_count
        ):

            raise RuntimeError(
                f"{label} must have "
                f"{expected_count} unique raw features; "
                f"its preprocessor has {len(names)}"
            )

        processed_count = len(
            prep.get_feature_names_out()
        )

        trained_count = getattr(
            model,
            "n_features_in_",
            None,
        )

        if (
            trained_count is not None
            and int(trained_count) != processed_count
        ):

            raise RuntimeError(
                f"{label} model expects "
                f"{trained_count} processed features "
                f"but its preprocessor outputs "
                f"{processed_count}"
            )

        bundles.append(
            (
                label,
                model,
                prep,
                names,
            )
        )

    original_names = bundles[0][3]

    if (
        set(bundles[1][3])
        != set(original_names)
    ):

        raise RuntimeError(
            "Both 80-feature models must "
            "use the same raw features"
        )

    known_features = {
        feature
        for keys in GROUPS.values()
        for feature in keys
    }

    unrecognized = (
        set(original_names)
        - known_features
    )

    if unrecognized:

        raise RuntimeError(
            "Unrecognized original feature names: "
            + ", ".join(
                sorted(unrecognized)
            )
        )

    missing_from_80 = (
        set(bundles[2][3])
        - set(original_names)
    )

    if missing_from_80:

        raise RuntimeError(
            "40-feature model has features "
            "missing from original set: "
            + ", ".join(
                sorted(missing_from_80)
            )
        )

    # Cross-check the JSON saved by the
    # 40-feature training script.

    if FEATURE40_LIST.exists():

        try:

            with FEATURE40_LIST.open(
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(
                    file
                )

            expected = metadata.get(
                "feature_names"
            )

            if (
                expected is None
                or list(expected) != bundles[2][3]
            ):

                raise RuntimeError(
                    "40-feature JSON does not match "
                    "fitted preprocessor raw-column "
                    "order. Use files saved by the "
                    "same training run."
                )

        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
        ) as error:

            raise RuntimeError(
                "Cannot read 40-feature "
                f"selection JSON: {error}"
            ) from error

    return bundles


# ============================================================
# 11. MODEL-SPECIFIC FEATURE AVAILABILITY
# ============================================================

def model_coverage(
    features: dict,
    bundles,
    details: bool,
) -> None:

    heading(
        "MODEL-SPECIFIC FEATURE AVAILABILITY"
    )

    for (
        label,
        _model,
        _prep,
        names,
    ) in bundles:

        observed = [
            name
            for name in names
            if nonmissing(features[name])
        ]

        missing = [
            name
            for name in names
            if not nonmissing(features[name])
        ]

        print(
            f"  {label:12s}: "
            f"observed {len(observed):2}/{len(names):2} "
            "raw features"
        )

        if details:

            print(
                "      Missing:",
                ", ".join(missing)
                if missing
                else "none",
            )


# ============================================================
# 12. RUN LIVE DOMAIN TEST
# ============================================================

def run(
    domain: str,
    details: bool,
    explain: bool,
    save_features: bool,
    skip_ip_rdap: bool,
):

    host = normalize_host(
        domain
    )

    heading(
        "08 - LIVE DOMAIN INTELLIGENCE: "
        "THREE-MODEL COMPARISON"
    )

    print(
        "  Domain:",
        host,
    )

    bundles = load_three_models()

    print(
        "  Loaded: ORIGINAL 80, "
        "TUNED 80, TUNED 40"
    )

    # Collect live intelligence ONCE,
    # so differences come from models
    # rather than separate live lookups.

    original_cols = bundles[0][3]

    features = {
        name: np.nan
        for name in original_cols
    }

    group_status = {}

    group_status["DNS"] = collect_dns(
        host,
        features,
    )

    group_status["RDAP"] = collect_rdap(
        host,
        features,
    )

    group_status["TLS"] = collect_tls(
        host,
        features,
    )

    group_status["IP"] = collect_ip(
        group_status["DNS"],
        features,
        do_rdap=not skip_ip_rdap,
    )

    quality = feature_quality(
        features,
        group_status,
        details,
    )

    model_coverage(
        features,
        bundles,
        details,
    )

    # Optional feature export.

    if save_features:

        RESULTS.mkdir(
            exist_ok=True,
            parents=True,
        )

        safe_name = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            host,
        )

        dest = (
            RESULTS
            / f"live_features_{safe_name}.csv"
        )

        pd.DataFrame(
            [features],
            columns=original_cols,
        ).to_csv(
            dest,
            index=False,
        )

        print(
            "  Features saved:",
            dest,
        )

    # Predict using each model's
    # own fitted preprocessor.

    predictions = []

    for (
        label,
        model,
        prep,
        _names,
    ) in bundles:

        probability, transformed = predict(
            model,
            prep,
            features,
        )

        predictions.append(
            (
                label,
                probability,
                model,
                prep,
                transformed,
            )
        )

    # --------------------------------------------------------
    # DISPLAY THREE MODEL RESULTS
    # --------------------------------------------------------

    heading(
        "SIDE-BY-SIDE MODEL RESULTS"
    )

    print(
        "  Model          "
        "Raw phishing score    "
        "ML prediction"
    )

    print(
        "  " + "-" * 62
    )

    for (
        label,
        probability,
        _model,
        _prep,
        _transformed,
    ) in predictions:

        prediction = (
            "PHISHING"
            if probability >= THRESHOLD
            else "LEGITIMATE"
        )

        print(
            f"  {label:13s}     "
            f"{probability:8.4f}             "
            f"{prediction}"
        )

    # --------------------------------------------------------
    # ORIGINAL MODEL CALIBRATION (REFERENCE ONLY)
    # --------------------------------------------------------

    if ORIGINAL_CALIBRATOR.exists():

        try:

            calibrator = joblib.load(
                ORIGINAL_CALIBRATOR
            )

            calibrated = float(
                calibrator.predict(
                    np.array(
                        [predictions[0][1]]
                    )
                )[0]
            )

            print(
                "\n  Original 80 calibrated score "
                f"(reference only): {calibrated:.4f}"
            )

        except Exception as error:

            print(
                "  Original calibrator unavailable:",
                error,
            )

    print(
        "  Both tuned scores are RAW; "
        "neither has a fitted probability calibrator."
    )

    print(
        "  Raw scores use the same 0.50 "
        "decision threshold; do not treat them as"
    )

    print(
        "  automatically calibrated or as "
        "verified real-world risk percentages."
    )

    print(
        "  Data quality:",
        quality,
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    if quality == "INSUFFICIENT DATA":

        print(
            "\n  FINAL STATUS: "
            "UNKNOWN / INSUFFICIENT DATA"
        )

        print(
            "  All three binary outputs are "
            "exploratory and must not be trusted."
        )

    elif quality == "LIMITED":

        print(
            "\n  FINAL STATUS: LIMITED DATA — "
            "THREE MODEL PREDICTIONS ARE TENTATIVE"
        )

        print(
            "  Do not treat a binary output "
            "as a reliable safety decision."
        )

    else:

        print(
            "\n  FINAL STATUS: "
            "COMPARE ALL THREE; "
            "NO AUTOMATIC MODEL SWITCH"
        )

        print(
            "  Models require evaluation on "
            "independently verified live domains."
        )

    # --------------------------------------------------------
    # EXPLAIN ALL THREE MODEL PREDICTIONS
    # --------------------------------------------------------

    if explain:

        for (
            label,
            _probability,
            model,
            prep,
            transformed,
        ) in predictions:

            explain_prediction(
                model,
                prep,
                transformed,
                label,
            )

    print(
        "\n  IMPORTANT: Feature availability "
        "does not prove a domain is legitimate."
    )

    print(
        "  Live feature mismatches remain: "
        "investigate before deployment."
    )

    print(
        "=" * 72
    )


# ============================================================
# 13. COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare original 80, tuned 80, "
            "and tuned 40 LightGBM "
            "live domain predictions"
        )
    )

    parser.add_argument(
        "domain",
        nargs="?",
        help="Domain name or URL",
    )

    parser.add_argument(
        "--details",
        action="store_true",
        help=(
            "Show missing raw feature "
            "names per model"
        ),
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help=(
            "Show top LightGBM feature "
            "contributions for all three"
        ),
    )

    parser.add_argument(
        "--save-features",
        action="store_true",
        help=(
            "Save shared live feature row "
            "to results CSV"
        ),
    )

    parser.add_argument(
        "--skip-ip-rdap",
        action="store_true",
        help=(
            "Skip slow IP registration lookups"
        ),
    )

    args = parser.parse_args()

    domain = (
        args.domain
        or input(
            "Domain or URL: "
        ).strip()
    )

    run(
        domain,
        args.details,
        args.explain,
        args.save_features,
        args.skip_ip_rdap,
    )


if __name__ == "__main__":
    main()
