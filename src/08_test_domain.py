
r"""
08 - Classify a domain with the selected TUNED 80 LightGBM model.

Run:
    python .\src\08_test_domain.py google.com --details
    python .\src\08_test_domain.py microsoft.com --explain
    python .\src\08_test_domain.py albertsonstore.us --rdap-debug
    python .\src\08_test_domain.py google.com --rdap-ablation

Dependencies:
    pip install dnspython requests cryptography pandas numpy joblib lightgbm scikit-learn whoisit==2.7.7 whoisdomain==1.20240129.1

Research limitation:
Historical training enrichment features may not be reproducible
during live collection. Such values are left missing (NaN).

The live command loads only Tuned 80 and its matching preprocessor.
Other saved models remain available for research comparison.
An exact-host/URL match in a fresh OpenPhish snapshot supplies a separate
known-phishing decision. The original ML score is retained for research.
No training files or models are modified.
Predictions are always PHISHING or LEGITIMATE after successful inference.
Missing inputs use the saved preprocessors; data availability is separate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
import socket
import ssl

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, quote, urlsplit, urlunsplit

import dns.resolver
import joblib
import numpy as np
import pandas as pd
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cryptography import x509
from cryptography.x509.oid import NameOID

from whoisit.bootstrap import Bootstrap as RDAPBootstrap
from whoisit.errors import WhoisItError
from whoisit.parser import ParseDomain as RDAPDomainParser
from whoisit.utils import recursive_merge as merge_collector_rdap
from whoisdomain import Domain as WhoisDomain, WhoisParser
from whoisdomain import exceptions as whois_errors
from whoisdomain.context.dataContext import DataContext as WhoisDataContext
from whoisdomain.context.parameterContext import ParameterContext as WhoisParameters
from whoisdomain.helpers import filterTldToSupportedPattern


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODELS = ROOT / "models"
RESULTS = ROOT / "results"

ORIGINAL_MODEL = (
    MODELS / "lightgbm_phishing_model.joblib"
)

ORIGINAL_PREPROCESSOR = (
    MODELS / "training_preprocessor.joblib"
)

TUNED_MODEL = (
    MODELS / "lightgbm_80_tuned_model.joblib"
)

TUNED_PREPROCESSOR = (
    MODELS / "training_preprocessor_80_tuned.joblib"
)

FEATURE40_MODEL = (
    MODELS / "lightgbm_40_tuned_model.joblib"
)

FEATURE40_PREPROCESSOR = (
    MODELS / "training_preprocessor_40_tuned.joblib"
)

FEATURE40_LIST = (
    RESULTS / "lightgbm_40_selected_features.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

THRESHOLD = 0.50
ACTIVE_MODEL_LABEL = "TUNED 80"
REPUTATION_SOURCE = "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
REPUTATION_CACHE = RESULTS / "openphish_reputation_cache.json"
REPUTATION_MAX_AGE_SECONDS = 12 * 3600
REPUTATION_MAX_BYTES = 2 * 1024 * 1024

DNS_TIMEOUT = 8.0
RDAP_TIMEOUT = 12.0
TLS_TIMEOUT = 5.0

IANA_BOOTSTRAP_URL = (
    "https://data.iana.org/rdap/dns.json"
)

IANA_BOOTSTRAP_CACHE = (
    RESULTS / "rdap_dns_bootstrap_cache.json"
)

BOOTSTRAP_CACHE_SECONDS = 24 * 3600

RDAP_REQUEST_TIMEOUT = (
    5,
    RDAP_TIMEOUT
)

MAX_IP_RDAP_LOOKUPS = 3


# ============================================================
# 3. FEATURE GROUPS
# ============================================================

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
# 4. GENERAL HELPERS
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
        host = host.encode(
            "idna"
        ).decode("ascii")

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
            str(value).replace(
                "Z",
                "+00:00"
            )
        )

        return (
            dt
            if dt.tzinfo
            else dt.replace(
                tzinfo=timezone.utc
            )
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
        4
    )


# ============================================================
# 5. DNS LOOKUP
# ============================================================

@dataclass
class DNSResult:

    items: list
    status: str


def dns_lookup(
    host: str,
    kind: str
) -> DNSResult:

    resolver = dns.resolver.Resolver()

    resolver.timeout = DNS_TIMEOUT
    resolver.lifetime = DNS_TIMEOUT

    try:

        answers = list(
            resolver.resolve(
                host,
                kind
            )
        )

        return DNSResult(
            answers,
            "ok"
        )

    except dns.resolver.NoAnswer:

        return DNSResult(
            [],
            "noanswer"
        )

    except dns.resolver.NXDOMAIN:

        return DNSResult(
            [],
            "nxdomain"
        )

    except Exception as error:

        print(
            f"  DNS lookup failed for {kind} ({host}): "
            f"{type(error).__name__}: {error}"
        )

        return DNSResult(
            [],
            "error"
        )


def dns_count(result: DNSResult):

    if result.status in (
        "ok",
        "noanswer"
    ):
        return len(result.items)

    return np.nan


def dns_has(result: DNSResult):

    count = dns_count(result)

    if pd.notna(count):
        return int(count > 0)

    return np.nan


def resolved_related_ips(
    names: list[str]
):

    ips = set()
    attempted = False

    for name in names:

        for kind in (
            "A",
            "AAAA"
        ):

            result = dns_lookup(
                name,
                kind
            )

            if result.status == "error":
                continue

            attempted = True

            ips.update(
                str(item)
                for item in result.items
            )

    if attempted or not names:
        return len(ips)

    return np.nan


# ============================================================
# 6. COLLECT DNS FEATURES
# ============================================================

def collect_dns(
    host: str,
    features: dict
) -> dict:

    heading(
        "DNS INTELLIGENCE"
    )

    kinds = (
        "A",
        "AAAA",
        "MX",
        "NS",
        "TXT",
        "CNAME",
        "SOA"
    )

    records = {

        kind: dns_lookup(
            host,
            kind
        )

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

    # Historical DNS source/TTL metadata cannot
    # be reproduced from an ordinary live lookup.
    # Leave those feature values missing.

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
                "noanswer"
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
                "noanswer"
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
                "noanswer"
            )
            else np.nan
        ),

        dns_ipv6_count=(
            len(ipv6)
            if records["AAAA"].status in (
                "ok",
                "noanswer"
            )
            else np.nan
        ),

        dns_unique_ip_count=(
            len(ips)
            if (
                records["A"].status in (
                    "ok",
                    "noanswer"
                )
                and records["AAAA"].status in (
                    "ok",
                    "noanswer"
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
        "CNAME"
    ):

        result = records[kind]

        shown = (
            str(len(result.items))
            if result.status in (
                "ok",
                "noanswer"
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
# 7. RDAP HTTP SESSION
# ============================================================

def rdap_session() -> requests.Session:

    """
    Retry temporary HTTP/transport errors.

    404 is not retried.
    """

    session = requests.Session()

    retries = Retry(

        total=1,

        connect=1,

        read=1,

        status=1,

        backoff_factor=0.5,

        status_forcelist=(
            429,
            500,
            502,
            503,
            504
        ),

        allowed_methods=frozenset(
            ("GET",)
        ),

        respect_retry_after_header=True
    )

    session.mount(

        "https://",

        HTTPAdapter(
            max_retries=retries
        )
    )

    session.headers.update({

        "Accept":
            "application/rdap+json, "
            "application/json;q=0.9",

        "User-Agent":
            "AcademicDomainIntelligencePrototype/1.1",
    })

    return session


def request_json(
    session: requests.Session,
    url: str,
    *,
    timeout=RDAP_REQUEST_TIMEOUT,
    diagnostic: list[str] | None = None,
) -> dict | None:

    """
    Return JSON only when HTTP and parsing succeed.

    Otherwise store a helpful diagnostic message.
    """

    try:

        result = session.get(
            url,
            timeout=timeout
        )

        if result.status_code != 200:

            if diagnostic is not None:

                diagnostic.append(
                    f"HTTP {result.status_code}: {url}"
                )

            return None

        payload = result.json()

        if not isinstance(
            payload,
            dict
        ):

            if diagnostic is not None:

                diagnostic.append(
                    f"Non-object JSON: {url}"
                )

            return None

        return payload

    except (
        requests.RequestException,
        ValueError
    ) as error:

        if diagnostic is not None:

            diagnostic.append(
                f"{type(error).__name__}: "
                f"{url}: {str(error)[:130]}"
            )

        return None


def fetch_rdap(url: str):

    """
    Simple RDAP lookup used for IP enrichment.
    """

    with rdap_session() as session:

        return request_json(

            session,

            url,

            timeout=(4, 6)
        )


# ============================================================
# 8. IANA RDAP BOOTSTRAP
# ============================================================

def read_iana_bootstrap(
    session: requests.Session,
    messages: list[str]
) -> dict | None:

    """
    Download or reuse IANA domain RDAP
    bootstrap information.

    Cache duration: 24 hours.
    """

    cache = None

    if IANA_BOOTSTRAP_CACHE.is_file():

        try:

            cache = json.loads(

                IANA_BOOTSTRAP_CACHE.read_text(
                    encoding="utf-8"
                )
            )

            if (
                not isinstance(cache, dict)
                or not isinstance(
                    cache.get("services"),
                    list
                )
            ):

                cache = None

        except (
            OSError,
            ValueError
        ):

            cache = None

    if cache is not None:

        age = (
            time.time()
            - IANA_BOOTSTRAP_CACHE.stat().st_mtime
        )

        if age < BOOTSTRAP_CACHE_SECONDS:

            messages.append(
                "IANA bootstrap: using cached registry"
            )

            return cache

    fresh = request_json(

        session,

        IANA_BOOTSTRAP_URL,

        timeout=(5, 10),

        diagnostic=messages
    )

    if (
        fresh is not None
        and isinstance(
            fresh.get("services"),
            list
        )
    ):

        try:

            RESULTS.mkdir(
                parents=True,
                exist_ok=True
            )

            IANA_BOOTSTRAP_CACHE.write_text(

                json.dumps(fresh),

                encoding="utf-8"
            )

        except OSError:
            pass

        messages.append(
            "IANA bootstrap: retrieved registry"
        )

        return fresh

    if cache is not None:

        messages.append(
            "IANA bootstrap: network failed; "
            "using older cached registry"
        )

        return cache

    messages.append(
        "IANA bootstrap unavailable; "
        "will try rdap.org fallback"
    )

    return None


def bootstrap_base_urls(
    domain: str,
    bootstrap: dict | None
) -> list[str]:

    """
    Match the longest domain suffix.

    Example:
        example.com -> .com registry servers
    """

    if not bootstrap:
        return []

    labels = (
        domain.lower()
        .strip(".")
        .split(".")
    )

    best_len = -1

    matches = []

    for entry in bootstrap.get(
        "services",
        []
    ):

        if (
            not isinstance(entry, list)
            or len(entry) != 2
        ):
            continue

        suffixes, base_urls = entry

        if (
            not isinstance(suffixes, list)
            or not isinstance(base_urls, list)
        ):
            continue

        for suffix in suffixes:

            suffix_parts = (
                str(suffix)
                .lower()
                .strip(".")
                .split(".")
            )

            if (
                len(suffix_parts) <= len(labels)
                and labels[-len(suffix_parts):]
                == suffix_parts
            ):

                if len(suffix_parts) > best_len:

                    matches = []

                    best_len = len(
                        suffix_parts
                    )

                if len(suffix_parts) == best_len:

                    matches.extend(
                        base_urls
                    )

    # Only accept HTTPS registry endpoints.

    return list(
        dict.fromkeys(

            url

            for url in matches

            if (
                isinstance(url, str)
                and url.lower().startswith(
                    "https://"
                )
            )
        )
    )


# ============================================================
# 9. VALIDATE RDAP DOMAIN RESPONSE
# ============================================================

def rdap_valid_domain(
    data: dict,
    requested: str
) -> bool:

    """
    Avoid accepting an RDAP error response
    as a valid domain record.
    """

    if data.get(
        "objectClassName"
    ) == "domain":

        name = str(

            data.get("ldhName")
            or data.get("unicodeName")
            or ""

        ).lower().rstrip(".")

        return (
            not name
            or name == requested.lower()
        )

    name = str(

        data.get("ldhName")
        or ""

    ).lower().rstrip(".")

    return bool(

        name == requested.lower()

        and data.get("events")
    )


# ============================================================
# 10. FIND DOMAIN RDAP
# ============================================================

def follow_related_domain_rdap(
    session: requests.Session,
    registry_data: dict,
    domain: str,
    messages: list[str],
) -> dict:
    """Overlay one matching registrar response, as whoisit 2.7.7 does.

    Dataset source: nesfit/domainradar-dib collector/resolvers/rdap.py and
    collector/requirements.txt at d1f1a84cd388e55d64c9ca7431b146cb99935097.
    Failed referrals retain registry data with a diagnostic. WHOIS fallback
    is handled by collect_rdap when no parseable RDAP record is available.
    """
    links = registry_data.get("links", [])
    if not isinstance(links, list):
        return registry_data
    for link in links:
        if not isinstance(link, dict):
            continue
        if link.get("rel") not in ("related", "registration"):
            continue
        referral = link.get("href")
        if not isinstance(referral, str) or not referral:
            continue
        if urlparse(referral).scheme.lower() != "https":
            messages.append("Related RDAP not followed: HTTPS required; registry data only")
            return registry_data
        related = request_json(session, referral, diagnostic=messages)
        if related is None or not rdap_valid_domain(related, domain):
            messages.append(f"Related RDAP unavailable or mismatched: {referral}; registry data only")
            return registry_data
        merged = copy.deepcopy(registry_data)
        merge_collector_rdap(merged, related)
        messages.append(f"Successful related RDAP response: {referral}")
        return merged
    return registry_data


def normalize_domain_rdap(data: dict, domain: str = "") -> dict:
    """Parse offline with the version specified by the dataset collector.

    Domain registry/registrar endpoints have no regional IP registry name, so
    a fresh Bootstrap suffices for these domain features without network I/O.
    """
    return RDAPDomainParser(RDAPBootstrap(), data, domain).parse()


def find_domain_rdap(
    host: str,
    debug: bool = False
):

    """
    Try IANA-discovered registry servers first.

    Use rdap.org as the final fallback.

    Keep diagnostic messages for failures.
    """

    messages: list[str] = []

    with rdap_session() as session:

        bootstrap = read_iana_bootstrap(
            session,
            messages
        )

        labels = host.split(".")

        candidates = [

            ".".join(labels[i:])

            for i in range(
                len(labels) - 1
            )
        ]

        for name in candidates:

            bases = bootstrap_base_urls(
                name,
                bootstrap
            )

            if not bases:

                messages.append(

                    "No HTTPS IANA bootstrap "
                    f"match for {name}"
                )

            urls = [

                base.rstrip("/")
                + "/domain/"
                + quote(
                    name,
                    safe=""
                )

                for base in bases
            ]

            urls.append(

                "https://rdap.org/domain/"
                + quote(
                    name,
                    safe=""
                )
            )

            for url in dict.fromkeys(
                urls
            ):

                response = request_json(

                    session,

                    url,

                    diagnostic=messages
                )

                if (

                    response is not None

                    and rdap_valid_domain(
                        response,
                        name
                    )
                ):

                    messages.append(
                        f"Successful RDAP response: {url}"
                    )

                    response = follow_related_domain_rdap(
                        session, response, name, messages
                    )

                    if debug:

                        for message in messages:

                            print(
                                "  RDAP debug:",
                                message
                            )

                    return (

                        response,

                        name,

                        url,

                        messages
                    )

                if response is not None:

                    messages.append(

                        "Not a matching domain "
                        f"RDAP object: {url}"
                    )

    if debug:

        for message in messages:

            print(
                "  RDAP debug:",
                message
            )

    return (
        None,
        None,
        None,
        messages
    )


# ============================================================
# 11. NORMALIZE RDAP ENTITY ROLES
# ============================================================

def role_counts(data: dict) -> dict[str, int]:
    """Count normalized lists, including nested and empty-email contacts.

    The published parser deduplicates complete parsed contacts, not handles.
    Entities with no parseable fields are omitted by the parser itself.
    """
    normalized = data if isinstance(data.get("entities"), dict) else normalize_domain_rdap(data)
    entities = normalized["entities"]
    return {
        role: len(entities.get(role, []))
        for role in ("registrar", "registrant", "abuse", "admin", "administrative", "technical")
    }


WHOIS_TIMEOUT = 10.0
WHOIS_MAX_BYTES = 262144
# Authoritative referrals verified against IANA's .us/.de delegation records.
# DENIC also documents whois.denic.de at /en/services/whois-service/.
WHOIS_SERVER_CACHE = {"us": "whois.nic.us", "de": "whois.denic.de"}


def query_whois_server(server: str, query: str) -> str:
    """Read a bounded port-43 reply without requiring a platform WHOIS CLI."""
    if not re.fullmatch(r"[a-zA-Z0-9.-]+", server) or not re.fullmatch(r"[a-zA-Z0-9.-]+", query):
        raise ValueError("Invalid WHOIS server or query")
    deadline = time.monotonic() + WHOIS_TIMEOUT
    chunks = []
    size = 0
    # DENIC's bare-domain query returns availability only. Its documented
    # technical-record query includes nameservers and the last change date.
    wire_query = "-T dn " + query if server.lower().rstrip(".") == "whois.denic.de" else query
    with socket.create_connection((server, 43), timeout=WHOIS_TIMEOUT) as connection:
        connection.sendall((wire_query + "\r\n").encode("ascii"))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("WHOIS response timed out")
            connection.settimeout(remaining)
            part = connection.recv(min(16384, WHOIS_MAX_BYTES + 1 - size))
            if not part:
                break
            chunks.append(part)
            size += len(part)
            if size > WHOIS_MAX_BYTES:
                raise ValueError("WHOIS response exceeded size limit")
    return b"".join(chunks).decode("utf-8", errors="replace")


def discover_whois_server(tld: str) -> str | None:
    """Use IANA's referral for unsupported RDAP TLDs; cache only successes."""
    if tld in WHOIS_SERVER_CACHE:
        return WHOIS_SERVER_CACHE[tld]
    response = query_whois_server("whois.iana.org", tld)
    match = re.search(r"^(?:whois|refer):\s*([a-zA-Z0-9.-]+)\s*$", response, re.MULTILINE | re.IGNORECASE)
    if match:
        WHOIS_SERVER_CACHE[tld] = match.group(1).lower()
        return WHOIS_SERVER_CACHE[tld]
    return None


def parse_collector_whois(text: str, domain: str):
    """Validate a registry reply, then use the original collector's parser.

    Some TLD patterns match the queried name inside an error message and a
    constant registrar in the registry banner. Reject those replies before
    parsing. NIC Chile's current positive reply omits Domain Name; its exact
    registry lookup link supplies identity for the older parser instead.
    """
    if re.search(
        r"^\s*(?:%+\s*)?(?:invalid\s+domain(?:\s+name)?|no\s+(?:match|entries|data|such\s+domain)"
        r"|not\s+found|not\s+registered|quota\s+exceeded|query\s+rate\s+limit"
        r"|too\s+many\s+(?:requests|queries))\b",
        text, re.MULTILINE | re.IGNORECASE,
    ):
        return None
    if domain.lower().endswith(".cl") and not re.search(
        r"^\s*Domain\s+Name\s*:", text, re.MULTILINE | re.IGNORECASE
    ):
        identity = re.search(
            r"^\s*%%\s*See\s+https://www\.nic\.cl/registry/Whois\.do\?d=([a-z0-9.-]+)\s*$",
            text, re.MULTILINE | re.IGNORECASE,
        )
        if identity is None or identity.group(1).lower().rstrip(".") != domain.lower().rstrip("."):
            return None
        # This name is backed by the registry response, not the input alone.
        text = f"Domain Name: {identity.group(1)}\n" + text
    labels = domain.split(".")
    pattern = filterTldToSupportedPattern(domain, labels)
    if pattern is None:
        return None
    parameters = WhoisParameters()
    context = WhoisDataContext(domain)
    context.dList = labels
    context.tldString = pattern
    context.whoisStr = text
    parser = WhoisParser(parameters, context)
    parser.getThisTld(pattern)
    parser.init()
    parsed, _ = parser.parse(WhoisDomain(parameters, context))
    if parsed is None or str(parsed.name).lower().rstrip(".") != domain.lower().rstrip("."):
        return None
    if not any((parsed.creation_date, parsed.expiration_date, parsed.registrar, parsed.name_servers)):
        return None
    return parsed


def whois_to_collector_record(domain) -> dict:
    """Reproduce domainradar-dib's whois_to_rdap_domain conversion.

    Its four entity lists each contain one object even when the corresponding
    contact is empty. Its handle, WHOIS server and description are empty,
    and its normalized WHOIS result does not include a DNSSEC field.
    These definitions preserve training semantics rather than inventing
    equivalence with raw RDAP fields.
    """
    def field(name):
        value = getattr(domain, name, None)
        return "" if value is None else str(value)
    statuses = [re.sub(r"([A-Z])", r" \1", status.split()[0]).lower()
                for status in domain.statuses if status.split()]
    return {
        "handle": "", "parent_handle": "", "name": domain.name,
        "whois_server": "", "type": "domain", "terms_of_service_url": "",
        "copyright_notice": "", "description": [],
        "last_changed_date": domain.last_updated,
        "registration_date": domain.creation_date,
        "expiration_date": domain.expiration_date, "rir": "", "url": "",
        "entities": {
            "registrant": [{"name": field("registrant")}],
            "abuse": [{"email": field("abuse_contact")}],
            "admin": [{"name": field("admin")}],
            "registrar": [{"name": field("registrar")}],
        },
        "nameservers": [name.upper() for name in domain.name_servers],
        "status": list(dict.fromkeys(statuses)),
    }


def find_domain_whois(host: str, diagnostics: list[str]):
    """Try authoritative WHOIS only after RDAP is unavailable/unparseable."""
    labels = host.split(".")
    try:
        server = discover_whois_server(labels[-1])
    except (OSError, ValueError) as error:
        diagnostics.append(f"WHOIS server discovery failed: {type(error).__name__}")
        return None, None, None
    if server is None:
        diagnostics.append(f"No IANA WHOIS referral for .{labels[-1]}")
        return None, None, None
    for index in range(len(labels) - 1):
        candidate = ".".join(labels[index:])
        try:
            text = query_whois_server(server, candidate)
            parsed = parse_collector_whois(text, candidate)
            if parsed is not None:
                source = f"whois://{server}/{candidate}"
                diagnostics.append(f"Successful WHOIS fallback: {source}")
                return whois_to_collector_record(parsed), candidate, source
            diagnostics.append(f"No matching usable WHOIS record for {candidate} from {server}")
        except (OSError, ValueError, TypeError, KeyError, AttributeError,
                whois_errors.WhoisException) as error:
            diagnostics.append(f"WHOIS query/parse failed for {candidate}: {type(error).__name__}")
            break
    return None, None, None


def collect_rdap(host: str, features: dict, debug: bool = False) -> dict:
    heading("RDAP / WHOIS INTELLIGENCE")
    data, queried_domain, url, diagnostics = find_domain_rdap(host, debug)
    normalized = None
    method = "RDAP"
    if data is not None:
        try:
            normalized = normalize_domain_rdap(data, queried_domain)
        except (WhoisItError, ValueError, TypeError, KeyError, AttributeError) as error:
            diagnostics.append(f"RDAP normalization failed: {type(error).__name__}; trying WHOIS")
    if normalized is None:
        print("  RDAP unavailable or unparseable; trying WHOIS fallback.")
        normalized, queried_domain, url = find_domain_whois(host, diagnostics)
        method = "WHOIS"
    if normalized is None:
        print("  Registration data unavailable; features remain missing.")
        for message in diagnostics[-4:]:
            print("   -", message)
        return {"available": False, "diagnostics": diagnostics}

    today = datetime.now(timezone.utc)
    created = parsed_utc_date(normalized.get("registration_date"))
    expires = parsed_utc_date(normalized.get("expiration_date"))
    updated = parsed_utc_date(normalized.get("last_changed_date"))
    roles = role_counts(normalized)
    put(
        features,
        rdap_available=1,
        domain_age_days=days_between(created, today),
        registration_period_days=days_between(created, expires),
        days_until_expiration=days_between(today, expires),
        days_since_last_change=days_between(updated, today),
        rdap_has_registration_date=int(created is not None),
        rdap_has_expiration_date=int(expires is not None),
        rdap_has_last_changed_date=int(updated is not None),
        rdap_has_whois_server=int(bool(normalized.get("whois_server"))),
        rdap_has_name=int(bool(normalized.get("name"))),
        rdap_has_handle=int(bool(normalized.get("handle"))),
        rdap_registrar_count=roles["registrar"],
        rdap_registrant_count=roles["registrant"],
        rdap_abuse_count=roles["abuse"],
        rdap_admin_count=roles["admin"] + roles["administrative"],
        rdap_technical_count=roles["technical"],
        rdap_noc_count=len(normalized["entities"].get("noc", [])),
        rdap_has_registrar=int(roles["registrar"] > 0),
        rdap_has_registrant=int(roles["registrant"] > 0),
        rdap_has_abuse_contact=int(roles["abuse"] > 0),
        rdap_nameserver_count=len(normalized.get("nameservers", [])),
        rdap_status_count=len(normalized.get("status", [])),
        # False DNSSEC is present information. Training tests presence,
        # rather than whether the domain actually enables DNSSEC.
        rdap_dnssec_information_present=int(normalized.get("dnssec") is not None),
        rdap_description_count=len(normalized.get("description", [])),
    )
    print("  Registration method:", method)
    print("  Registration domain:", queried_domain)
    print("  Registration source:", url)
    print("  Domain age:", days_between(created, today), "days")
    print("  RDAP statuses:", len(normalized.get("status", [])))
    print("  Entity roles:", ", ".join(f"{key}={value}" for key, value in roles.items()))
    normalization = "whoisit-2.7.7" if method == "RDAP" else "whoisdomain-1.20240129.1 / collector WHOIS mapping"
    print("  Registration normalization:", normalization)
    partial = method == "RDAP" and any("registry data only" in message for message in diagnostics)
    if partial:
        print("  Related registrar data unavailable; registry-only features may be incomplete.")
    return {
        "available": True, "source": url, "diagnostics": diagnostics,
        "normalization": normalization, "method": method, "registry_only_fallback": partial,
    }


# ============================================================
# TLS HELPER
# ============================================================


def tls_connect(
    host: str,
    verify: bool
):

    context = (

        ssl.create_default_context()

        if verify

        else ssl._create_unverified_context()
    )

    with socket.create_connection(

        (host, 443),

        timeout=TLS_TIMEOUT

    ) as sock:

        sock.settimeout(
            TLS_TIMEOUT
        )

        with context.wrap_socket(

            sock,

            server_hostname=host

        ) as secure:

            return (

                secure.getpeercert(
                    binary_form=True
                ),

                secure.version(),

                secure.cipher()
            )


def cert_has_name(
    cert: x509.Certificate,
    oid
) -> int:

    return int(

        bool(

            cert.subject.get_attributes_for_oid(
                oid
            )
        )
    )


# ============================================================
# 14. COLLECT TLS FEATURES
# ============================================================

def collect_tls(
    host: str,
    features: dict
) -> dict:

    heading(
        "TLS CERTIFICATE INTELLIGENCE"
    )

    validation_error = None

    try:

        der, protocol, cipher = tls_connect(
            host,
            verify=True
        )

    except ssl.SSLCertVerificationError as error:

        validation_error = str(
            error
        )

        print(

            "  Certificate validation FAILED. "
            "Retrying only to inspect metadata."
        )

        try:

            der, protocol, cipher = tls_connect(
                host,
                verify=False
            )

        except (
            OSError,
            ssl.SSLError
        ) as second:

            print(
                "  TLS unavailable:",
                second
            )

            return {
                "available": False
            }

    except (
        OSError,
        ssl.SSLError
    ) as error:

        print(
            "  TLS unavailable:",
            error
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
        None
    )

    end = getattr(
        cert,
        "not_valid_after_utc",
        None
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

        # Full certificate chain counts cannot
        # be reliably collected here.

        tls_has_leaf_certificate=1,

        tls_has_common_name=cert_has_name(
            cert,
            NameOID.COMMON_NAME
        ),

        tls_has_organization=cert_has_name(
            cert,
            NameOID.ORGANIZATION_NAME
        ),

        tls_has_country=cert_has_name(
            cert,
            NameOID.COUNTRY_NAME
        ),

        tls_certificate_age_days=days_between(
            start,
            now
        ),

        tls_certificate_validity_days=days_between(
            start,
            end
        ),

        tls_certificate_days_remaining=days_between(
            now,
            end
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
        f"validity: "
        f"{days_between(start, end)} days"
    )

    print(
        "  Certificate verified:",
        validation_error is None
    )

    if validation_error:

        print(
            "  Verification warning:",
            validation_error[:180]
        )

    return {

        "available": True,

        "certificate_verified": (
            validation_error is None
        )
    }


# ============================================================
# 15. COLLECT IP / HOSTING FEATURES
# ============================================================

def collect_ip(
    dns_data: dict,
    features: dict,
    do_rdap: bool = True
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
        set(
            a_ips + aaaa_ips
        )
    )

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

            ip_rdap_available_count=found
        )

        print(

            f"  IP RDAP lookups: "
            f"{found}/{len(ips)} successful"
        )

    elif len(ips) > MAX_IP_RDAP_LOOKUPS:

        print(

            "  IP RDAP not collected: "
            f"more than {MAX_IP_RDAP_LOOKUPS} IPs"
        )

    return {

        "available": bool(ips)
    }


# ============================================================
# 16. FEATURE QUALITY
# ============================================================

def nonmissing(value) -> bool:

    try:

        return bool(
            pd.notna(value)
        )

    except (
        TypeError,
        ValueError
    ):

        return False


def feature_quality(
    features: dict,
    group_status: dict,
    details: bool
):

    heading(
        "FEATURE QUALITY / AVAILABILITY"
    )

    for name, keys in GROUPS.items():

        selected = [

            feature

            for feature in keys

            if feature in features
        ]

        observed = [

            feature

            for feature in selected

            if nonmissing(
                features[feature]
            )
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

                key

                for key in selected

                if key not in observed
            ]

            print(

                "         Missing:",

                ", ".join(missing)

                if missing

                else "none"
            )

    count = sum(

        nonmissing(value)

        for value in features.values()
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
            "(no live features collected)"
        )

        return "INSUFFICIENT DATA"

    if (
        groups < 3
        or not group_status["RDAP"]["available"]
    ):

        reason = (

            "RDAP missing"

            if not group_status["RDAP"]["available"]

            else "too few groups"
        )

        print(

            f"  DATA QUALITY: LIMITED "
            f"({reason})"
        )

        return "LIMITED"

    print(

        "  DATA QUALITY: PARTIAL/AVAILABLE "
        "(still subject to training-live mismatch)"
    )

    return "PARTIAL"


# ============================================================
# 17. LOAD MODEL AND PREPROCESSOR
# ============================================================

def load_pair(
    model_path: Path,
    preprocess_path: Path,
    label: str
):

    missing = [

        str(path)

        for path in (
            model_path,
            preprocess_path
        )

        if not path.exists()
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
        "feature_names_in_"
    ):

        raise RuntimeError(

            f"{label} preprocessor has "
            "no feature_names_in_"
        )

    return (
        model,
        prep
    )


def predict(
    model,
    preprocess,
    features: dict
):

    names = list(
        preprocess.feature_names_in_
    )

    data = pd.DataFrame(

        [

            {
                key: features[key]

                for key in names
            }
        ],

        columns=names
    )

    for name in (
        "tls_protocol",
        "tls_cipher"
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

    return (
        probability,
        transformed
    )


# ============================================================
# BINARY OUTPUT AND DATA AVAILABILITY
# ============================================================

def binary_prediction(score: float) -> str:
    """Keep the existing threshold and expose only the two trained classes."""
    if not np.isfinite(score):
        raise ValueError("Model returned a non-finite phishing score")
    return "PHISHING" if score >= THRESHOLD else "LEGITIMATE"


def canonical_reputation_url(raw: str) -> str | None:
    """Normalize authority/default port; preserve path/query distinctions."""
    try:
        parsed = urlsplit(raw.strip())
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None:
            return None
        host = normalize_host(raw)
        port = parsed.port
        authority = host
        if port is not None and port != (443 if scheme == "https" else 80):
            authority += f":{port}"
        return urlunsplit((scheme, authority, parsed.path or "/", parsed.query, ""))
    except (ValueError, UnicodeError):
        return None


def validate_reputation_snapshot(payload: dict, *, cache_used: bool = False) -> dict:
    """Reject stale, future-dated or corrupt cache data before using verdicts."""
    if payload.get("source") != REPUTATION_SOURCE:
        raise ValueError("Unexpected reputation source")
    retrieved = parsed_utc_date(payload.get("retrieved_at_utc"))
    if retrieved is None:
        raise ValueError("Reputation snapshot has no valid retrieval time")
    age = (datetime.now(timezone.utc) - retrieved).total_seconds()
    if age < -60 or age > REPUTATION_MAX_AGE_SECONDS:
        raise ValueError("Reputation snapshot is stale or future-dated")
    raw_urls = payload.get("urls")
    if not isinstance(raw_urls, list) or not raw_urls or len(raw_urls) > 100000:
        raise ValueError("Reputation snapshot has no usable URLs")
    normalized = {canonical_reputation_url(value) for value in raw_urls if isinstance(value, str)}
    if len([value for value in raw_urls if isinstance(value, str)]) != len(raw_urls) or None in normalized:
        raise ValueError("Invalid reputation URL")
    urls = sorted(normalized)
    if payload.get("url_count") != len(urls) or not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("feed_sha256", ""))):
        raise ValueError("Invalid reputation snapshot metadata")
    digest = hashlib.sha256("\n".join(urls).encode("utf-8")).hexdigest()
    if digest != payload.get("normalized_urls_sha256"):
        raise ValueError("Reputation snapshot checksum mismatch")
    return {**payload, "available": True, "urls": frozenset(urls),
            "hosts": frozenset(urlsplit(value).hostname for value in urls),
            "cache_used": cache_used, "snapshot_age_seconds": max(0, age)}


def load_reputation_snapshot(cache_path: Path = REPUTATION_CACHE, *, force_refresh: bool = False) -> dict:
    """Fetch only the official feed; never navigate to a listed phishing URL."""
    cached = None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        cached = validate_reputation_snapshot(payload, cache_used=True)
        if not force_refresh:
            return cached
    except (OSError, ValueError, TypeError, KeyError):
        pass
    try:
        with requests.get(REPUTATION_SOURCE, timeout=(5, 12), stream=True, allow_redirects=False,
                          headers={"User-Agent": "DomainIntelligenceResearch/1.0"}) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("Unexpected reputation feed status")
            if urlsplit(response.url).scheme != "https" or urlsplit(response.url).hostname != "raw.githubusercontent.com":
                raise ValueError("Unexpected reputation feed redirect")
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise ValueError("Reputation feed returned HTML")
            data = bytearray()
            deadline = time.monotonic() + 30
            for chunk in response.iter_content(chunk_size=65536):
                if time.monotonic() > deadline:
                    raise ValueError("Reputation feed exceeded collection deadline")
                data.extend(chunk)
                if len(data) > REPUTATION_MAX_BYTES:
                    raise ValueError("Reputation feed exceeded size limit")
            lines = [line.strip() for line in data.decode("utf-8-sig").splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
            parsed_urls = [canonical_reputation_url(line) for line in lines]
            valid_urls = sorted({value for value in parsed_urls if value is not None})
            if not valid_urls or sum(value is not None for value in parsed_urls) < 0.9 * len(lines):
                raise ValueError("Reputation feed has too few valid URLs")
            payload = {"source": REPUTATION_SOURCE,
                       "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                       "feed_sha256": hashlib.sha256(data).hexdigest(),
                       "normalized_urls_sha256": hashlib.sha256("\n".join(valid_urls).encode("utf-8")).hexdigest(),
                       "url_count": len(valid_urls), "urls": valid_urls}
        snapshot = validate_reputation_snapshot(payload)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as error:
            snapshot["cache_write_error"] = type(error).__name__
        return snapshot
    except (requests.RequestException, OSError, ValueError, TypeError, KeyError) as error:
        if cached is not None:
            return {**cached, "refresh_error": type(error).__name__}
        return {"available": False, "source": REPUTATION_SOURCE, "error": type(error).__name__}


def check_reputation(raw_input: str, snapshot: dict | None) -> dict:
    """Match an exact host for domain inputs, or an exact URL for URL inputs."""
    if snapshot is None:
        return {"available": False, "matched": False, "status": "DISABLED",
                "reason": "Reputation check disabled; final decision uses Tuned 80."}
    if not snapshot.get("available"):
        return {"available": False, "matched": False, "status": "UNAVAILABLE",
                "source": snapshot.get("source"), "error": snapshot.get("error"),
                "reason": "No fresh reputation snapshot available; final decision uses Tuned 80."}
    try:
        fresh = validate_reputation_snapshot({**snapshot, "urls": list(snapshot["urls"])},
                                            cache_used=snapshot.get("cache_used", False))
    except (ValueError, TypeError, KeyError):
        return {"available": False, "matched": False, "status": "UNAVAILABLE",
                "reason": "Reputation snapshot is invalid or expired; final decision uses Tuned 80."}
    raw = raw_input.strip()
    parsed = urlsplit(raw if "://" in raw else "//" + raw)
    host = normalize_host(raw)
    is_url = "://" in raw or parsed.path not in {"", "/"} or bool(parsed.query)
    if is_url:
        candidates = [raw] if "://" in raw else ["https://" + raw, "http://" + raw]
        matched_urls = [canonical_reputation_url(value) for value in candidates]
        matched = any(value in fresh["urls"] for value in matched_urls if value is not None)
        scope = "EXACT_URL"
        reason = "A reported phishing URL exactly matches this URL."
    else:
        matched = host in fresh["hosts"]
        scope = "EXACT_HOST"
        reason = "A reported phishing URL uses this exact host."
    no_match_reason = "No exact feed match; final decision uses Tuned 80."
    if is_url and host in fresh["hosts"] and not matched:
        no_match_reason = "This URL has no exact feed match; other URLs on this host are listed. Final decision uses Tuned 80."
    return {"available": True, "matched": matched, "status": "MATCH" if matched else "NO_MATCH",
            "match_scope": scope, "host_has_reported_urls": host in fresh["hosts"],
            "source": fresh["source"], "retrieved_at_utc": fresh["retrieved_at_utc"],
            "feed_sha256": fresh["feed_sha256"], "snapshot_age_seconds": fresh["snapshot_age_seconds"],
            "cache_used": fresh["cache_used"], "url_count": fresh["url_count"],
            "reason": reason if matched else no_match_reason}


def resolve_prediction(raw_input: str, score: float, snapshot: dict | None = None) -> dict:
    """Retain the ML verdict and expose the separate evidence-based decision."""
    ml_prediction = binary_prediction(score)
    reputation = check_reputation(raw_input, snapshot)
    return {"model_prediction": ml_prediction,
            "prediction": "PHISHING" if reputation["matched"] else ml_prediction,
            "decision_source": "OPENPHISH_" + reputation["match_scope"] if reputation["matched"] else "TUNED_80",
            "reputation": reputation}


def prediction_data_note(quality: str, registry_only: bool = False) -> str | None:
    """Describe missing evidence separately from the classification label."""
    if quality == "INSUFFICIENT DATA":
        return "No live features were collected; the result uses saved preprocessing defaults."
    if quality == "LIMITED":
        return "Limited live information; missing inputs use saved preprocessing defaults."
    if registry_only:
        return "Related registrar data were unavailable; registry information was used."
    return None


# ============================================================
# 18. EXPLAIN MODEL PREDICTION
# ============================================================

def explain_prediction(
    model,
    preprocessor,
    matrix,
    label: str
):

    try:

        contrib = model.booster_.predict(

            matrix,

            pred_contrib=True
        )

        if hasattr(
            contrib,
            "toarray"
        ):

            contrib = contrib.toarray()

        values = np.asarray(
            contrib
        ).ravel()[:-1]

        names = (
            preprocessor.get_feature_names_out()
        )

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
# 19. LOAD SELECTED MODEL
# ============================================================

def load_selected_model():
    """Load Tuned 80 independently of the two archived comparison models."""
    model, prep = load_pair(TUNED_MODEL, TUNED_PREPROCESSOR, ACTIVE_MODEL_LABEL)
    names = list(prep.feature_names_in_)
    if len(names) != 80 or len(set(names)) != 80:
        raise RuntimeError("TUNED 80 preprocessor must have 80 unique raw features")
    known_features = {feature for keys in GROUPS.values() for feature in keys}
    unrecognized = set(names) - known_features
    if unrecognized:
        raise RuntimeError("Unrecognized selected model features: " + ", ".join(sorted(unrecognized)))
    processed_count = len(prep.get_feature_names_out())
    trained_count = getattr(model, "n_features_in_", None)
    if trained_count is not None and int(trained_count) != processed_count:
        raise RuntimeError(
            f"TUNED 80 expects {trained_count} processed features; preprocessor outputs {processed_count}"
        )
    return ACTIVE_MODEL_LABEL, model, prep, names


# Research compatibility only. The live commands use load_selected_model.
def load_three_models():

    configurations = (

        (
            "ORIGINAL 80",
            ORIGINAL_MODEL,
            ORIGINAL_PREPROCESSOR,
            80
        ),

        (
            "TUNED 80",
            TUNED_MODEL,
            TUNED_PREPROCESSOR,
            80
        ),

        (
            "TUNED 40",
            FEATURE40_MODEL,
            FEATURE40_PREPROCESSOR,
            40
        ),
    )

    bundles = []

    for (
        label,
        model_path,
        prep_path,
        expected_count
    ) in configurations:

        model, prep = load_pair(

            model_path,
            prep_path,
            label
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

            None
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
                names
            )
        )

    original_names = bundles[0][3]

    if set(
        bundles[1][3]
    ) != set(original_names):

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

    if FEATURE40_LIST.exists():

        try:

            with FEATURE40_LIST.open(

                "r",
                encoding="utf-8"

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
            TypeError
        ) as error:

            raise RuntimeError(

                "Cannot read 40-feature "
                f"selection JSON: {error}"
            ) from error

    return bundles


# ============================================================
# 20. MODEL-SPECIFIC FEATURE COVERAGE
# ============================================================

def model_coverage(
    features: dict,
    bundles,
    details: bool
) -> None:

    heading(
        "MODEL-SPECIFIC FEATURE AVAILABILITY"
    )

    for (
        label,
        _model,
        _prep,
        names
    ) in bundles:

        observed = [

            name

            for name in names

            if nonmissing(
                features[name]
            )
        ]

        missing = [

            name

            for name in names

            if not nonmissing(
                features[name]
            )
        ]

        print(

            f"  {label:12s}: "
            f"observed "
            f"{len(observed):2}/{len(names):2} "
            "raw features"
        )

        if details:

            print(

                "      Missing:",

                ", ".join(missing)

                if missing

                else "none"
            )


# ============================================================
# 21. RUN LIVE TEST
# ============================================================

def run(
    domain: str,
    details: bool,
    explain: bool,
    save_features: bool,
    skip_ip_rdap: bool,
    rdap_debug: bool,
    rdap_ablation: bool,
    model_only: bool = False
):

    host = normalize_host(
        domain
    )

    heading(

        "08 - LIVE DOMAIN INTELLIGENCE: "
        "TUNED 80"
    )

    print(
        "  Domain:",
        host
    )

    bundles = [load_selected_model()]
    reputation_snapshot = None if model_only else load_reputation_snapshot()

    print(

        "  Loaded: TUNED 80 and its matching preprocessor"
    )

    selected_cols = bundles[0][3]

    features = {

        name: np.nan

        for name in selected_cols
    }

    group_status = {}

    # Collect features once.
    # Keep all four intelligence groups for the selected model.

    group_status["DNS"] = collect_dns(
        host,
        features
    )

    group_status["RDAP"] = collect_rdap(

        host,
        features,

        debug=rdap_debug
    )

    group_status["TLS"] = collect_tls(
        host,
        features
    )

    group_status["IP"] = collect_ip(

        group_status["DNS"],

        features,

        do_rdap=not skip_ip_rdap
    )

    quality = feature_quality(

        features,
        group_status,
        details
    )

    model_coverage(

        features,
        bundles,
        details
    )

    # Save raw features if requested.

    if save_features:

        RESULTS.mkdir(

            exist_ok=True,

            parents=True
        )

        safe_name = re.sub(

            r"[^A-Za-z0-9_.-]+",

            "_",

            host
        )

        dest = (

            RESULTS

            / f"live_features_{safe_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.csv"
        )

        pd.DataFrame(

            [features],

            columns=selected_cols

        ).to_csv(

            dest,

            index=False,
            mode="x"
        )

        print(
            "  Features saved:",
            dest
        )

    # Binary output also runs when collection provides no features.
    # Fitted preprocessors supply their existing missing-value defaults.

    predictions = []

    for (
        label,
        model,
        prep,
        _names
    ) in bundles:

        probability, transformed = predict(

            model,
            prep,
            features
        )

        predictions.append(

            (
                label,
                probability,
                model,
                prep,
                transformed
            )
        )

    # --------------------------------------------------------
    # SELECTED MODEL RESULT
    # --------------------------------------------------------

    heading(
        "MODEL RESULT"
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
        _transformed
    ) in predictions:

        prediction = binary_prediction(probability)

        print(

            f"  {label:13s}     "
            f"{probability:8.4f}             "
            f"{prediction}"
        )

    # --------------------------------------------------------
    # SELECTED MODEL SCORE
    # --------------------------------------------------------
    print(

        "  Tuned 80 score is RAW; it has no fitted probability calibrator."
    )

    print(

        "  Raw scores use the same 0.50 "
        "decision threshold; do not treat them as"
    )

    print(

        "  automatically calibrated or "
        "as verified real-world risk percentages."
    )

    print(
        "  Data quality:",
        quality
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    note = prediction_data_note(
        quality, group_status["RDAP"].get("registry_only_fallback", False)
    )
    if note:
        print("\n  Data availability note:", note)

    decision = resolve_prediction(domain, predictions[0][1], reputation_snapshot)
    print("\n  Reputation check:", decision["reputation"]["status"])
    print("  Decision evidence:", decision["reputation"]["reason"])
    print("  Decision source:", decision["decision_source"])
    print("  FINAL PREDICTION:", decision["prediction"])

    # --------------------------------------------------------
    # RDAP ABLATION EXPERIMENT
    # --------------------------------------------------------

    if (

        rdap_ablation

        and group_status["RDAP"]["available"]
    ):

        heading(
            "RDAP ABLATION (DIAGNOSTIC ONLY)"
        )

        print(

            "  Comparing model scores with "
            "collected RDAP features versus "
            "all RDAP fields missing."
        )

        print(

            "  This does NOT prove why "
            "a historical prediction was wrong."
        )

        without_rdap = dict(
            features
        )

        for name in GROUPS["RDAP"]:

            if name in without_rdap:

                without_rdap[name] = np.nan

        for (
            label,
            probability,
            model,
            prep,
            _transformed
        ) in predictions:

            reduced_score, _ = predict(

                model,
                prep,
                without_rdap
            )

            print(

                f"  {label:13s} "

                f"with RDAP={probability:.4f}  "

                f"without RDAP={reduced_score:.4f}"
            )

    # --------------------------------------------------------
    # LIGHTGBM EXPLANATIONS
    # --------------------------------------------------------

    if explain:

        for (
            label,
            _probability,
            model,
            prep,
            transformed
        ) in predictions:

            explain_prediction(

                model,
                prep,
                transformed,
                label
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
# 22. COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(

            "Classify a live domain with the selected Tuned 80 LightGBM model"
        )
    )

    parser.add_argument(

        "domain",

        nargs="?",

        help="Domain name or URL"
    )

    parser.add_argument(

        "--details",

        action="store_true",

        help="Show missing raw feature names"
    )

    parser.add_argument(

        "--explain",

        action="store_true",

        help="Show LightGBM feature contributions"
    )

    parser.add_argument(

        "--save-features",

        action="store_true",

        help="Save collected live features to CSV"
    )

    parser.add_argument(

        "--skip-ip-rdap",

        action="store_true",

        help="Skip optional IP RDAP lookups"
    )

    parser.add_argument(

        "--rdap-debug",

        action="store_true",

        help="Show all RDAP attempts and errors"
    )

    parser.add_argument(

        "--rdap-ablation",

        action="store_true",

        help="Compare predictions with and without RDAP"
    )

    parser.add_argument("--model-only", action="store_true",
                        help="Use only the ML model, disabling the phishing reputation check")

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

        args.rdap_debug,

        args.rdap_ablation,
        args.model_only
    )


# ============================================================
# 23. MAIN
# ============================================================

if __name__ == "__main__":
    main()
