from __future__ import annotations

import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

import ijson


# ============================================================
# 01 - FULL DOMAIN INTELLIGENCE DATASET PREPROCESSING
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\SLIIT\final reserch chamath"
)

RAW_DIR = PROJECT_ROOT / "raw"
PROCESSED_DIR = PROJECT_ROOT / "processed"

BENIGN_FILE = RAW_DIR / "benign_cesnet.json"
PHISHING_FILE = RAW_DIR / "phishing.json"

OUTPUT_FILE = (
    PROCESSED_DIR
    / "domain_intelligence_full.csv"
)


# ============================================================
# SETTINGS
# ============================================================

PROGRESS_INTERVAL = 10_000


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def to_binary(value: Any) -> int:
    return 1 if bool(value) else 0


def safe_number(
    value: Any
) -> float | None:

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(
        value,
        (
            int,
            float,
            Decimal
        )
    ):
        try:
            return float(value)
        except Exception:
            return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError
    ):
        return None


def safe_mean(
    values: list[float]
) -> float | None:

    cleaned = []

    for value in values:

        number = safe_number(value)

        if number is None:
            continue

        if math.isnan(number):
            continue

        cleaned.append(number)

    if not cleaned:
        return None

    return (
        sum(cleaned)
        / len(cleaned)
    )


def count_structure(
    value: Any
) -> int:

    if value is None:
        return 0

    if isinstance(
        value,
        (
            list,
            dict
        )
    ):
        return len(value)

    return 1


# ============================================================
# DATE HANDLING
# ============================================================

def extract_date_value(
    value: Any
) -> str | None:

    if value is None:
        return None

    if isinstance(value, str):
        return value

    if not isinstance(
        value,
        dict
    ):
        return None

    date_value = value.get(
        "$date"
    )

    if isinstance(
        date_value,
        str
    ):
        return date_value

    if isinstance(
        date_value,
        dict
    ):

        milliseconds = (
            date_value.get(
                "$numberLong"
            )
        )

        if milliseconds is not None:

            try:

                dt = datetime.fromtimestamp(
                    int(milliseconds) / 1000,
                    tz=timezone.utc
                )

                return dt.isoformat()

            except (
                ValueError,
                TypeError,
                OverflowError
            ):
                return None

    return None


def parse_datetime(
    value: Any
) -> datetime | None:

    raw_date = extract_date_value(
        value
    )

    if not raw_date:
        return None

    try:

        raw_date = (
            raw_date.replace(
                "Z",
                "+00:00"
            )
        )

        dt = datetime.fromisoformat(
            raw_date
        )

        if dt.tzinfo is None:

            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt

    except (
        ValueError,
        TypeError
    ):
        return None


def days_between(
    start: datetime | None,
    end: datetime | None
) -> float | None:

    if (
        start is None
        or end is None
    ):
        return None

    try:

        return round(
            (
                end - start
            ).total_seconds()
            / 86400,
            4
        )

    except Exception:
        return None


# ============================================================
# JSON STREAMING
# ============================================================

def first_non_whitespace_byte(
    path: Path
) -> bytes:

    with path.open(
        "rb"
    ) as file:

        while True:

            char = file.read(1)

            if not char:

                raise ValueError(
                    f"Empty dataset: {path}"
                )

            if not char.isspace():
                return char


def stream_json_array(
    path: Path
) -> Iterator[dict]:

    with path.open(
        "rb"
    ) as file:

        for record in ijson.items(
            file,
            "item"
        ):

            if isinstance(
                record,
                dict
            ):
                yield record


def stream_json_lines(
    path: Path
) -> Iterator[dict]:

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

                if isinstance(
                    record,
                    dict
                ):
                    yield record

            except json.JSONDecodeError:

                if line_number <= 5:

                    print(
                        f"\nWarning: invalid JSON "
                        f"at line {line_number}"
                    )


def stream_records(
    path: Path
) -> Iterator[dict]:

    first_char = (
        first_non_whitespace_byte(
            path
        )
    )

    if first_char == b"[":

        yield from stream_json_array(
            path
        )

    elif first_char == b"{":

        yield from stream_json_lines(
            path
        )

    else:

        raise ValueError(
            f"Unsupported JSON structure: "
            f"{path}"
        )


# ============================================================
# DNS FEATURE EXTRACTION
# ============================================================

def extract_dns_features(
    record: dict
) -> dict:

    dns = safe_dict(
        record.get("dns")
    )

    # --------------------------------------------------------
    # A
    # --------------------------------------------------------

    a_raw = dns.get("A")

    if isinstance(
        a_raw,
        list
    ):
        a_records = a_raw
    else:
        a_records = []

    # --------------------------------------------------------
    # AAAA
    # --------------------------------------------------------

    aaaa_raw = dns.get(
        "AAAA"
    )

    if isinstance(
        aaaa_raw,
        list
    ):
        aaaa_records = aaaa_raw
    else:
        aaaa_records = []

    # --------------------------------------------------------
    # MX
    # --------------------------------------------------------

    mx_raw = dns.get(
        "MX"
    )

    if isinstance(
        mx_raw,
        dict
    ):
        mx_records = mx_raw
    else:
        mx_records = {}

    # --------------------------------------------------------
    # NS
    # --------------------------------------------------------

    ns_raw = dns.get(
        "NS"
    )

    if isinstance(
        ns_raw,
        dict
    ):
        ns_records = ns_raw
    else:
        ns_records = {}

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    txt_raw = dns.get(
        "TXT"
    )

    txt_count = (
        count_structure(
            txt_raw
        )
    )

    # --------------------------------------------------------
    # CNAME
    # --------------------------------------------------------

    cname = dns.get(
        "CNAME"
    )

    # --------------------------------------------------------
    # A / AAAA UNIQUE IPS
    # --------------------------------------------------------

    ipv4 = set()

    for value in a_records:

        if isinstance(
            value,
            dict
        ):
            ip = value.get(
                "value"
            )

        else:
            ip = value

        if ip:

            ipv4.add(
                str(ip)
            )

    ipv6 = set()

    for value in aaaa_records:

        if isinstance(
            value,
            dict
        ):
            ip = value.get(
                "value"
            )

        else:
            ip = value

        if ip:

            ipv6.add(
                str(ip)
            )

    all_dns_ips = (
        ipv4 | ipv6
    )

    # --------------------------------------------------------
    # MX DETAILS
    # --------------------------------------------------------

    mx_related_ips = set()
    mx_priorities = []

    for _, mx_info in (
        mx_records.items()
    ):

        if not isinstance(
            mx_info,
            dict
        ):
            continue

        priority = safe_number(
            mx_info.get(
                "priority"
            )
        )

        if priority is not None:

            mx_priorities.append(
                priority
            )

        related_ips = safe_list(
            mx_info.get(
                "related_ips"
            )
        )

        for item in related_ips:

            if isinstance(
                item,
                dict
            ):

                ip = item.get(
                    "value"
                )

            else:
                ip = item

            if ip:

                mx_related_ips.add(
                    str(ip)
                )

    mx_min_priority = None
    mx_max_priority = None
    mx_mean_priority = None

    if mx_priorities:

        mx_min_priority = min(
            mx_priorities
        )

        mx_max_priority = max(
            mx_priorities
        )

        mx_mean_priority = (
            sum(mx_priorities)
            / len(mx_priorities)
        )

    # --------------------------------------------------------
    # NS DETAILS
    # --------------------------------------------------------

    ns_related_ips = set()

    for _, ns_info in (
        ns_records.items()
    ):

        if not isinstance(
            ns_info,
            dict
        ):
            continue

        related_ips = safe_list(
            ns_info.get(
                "related_ips"
            )
        )

        for item in related_ips:

            if isinstance(
                item,
                dict
            ):

                ip = item.get(
                    "value"
                )

            else:
                ip = item

            if ip:

                ns_related_ips.add(
                    str(ip)
                )

    # --------------------------------------------------------
    # EXTRA DNS INFORMATION
    # --------------------------------------------------------

    dnssec = dns.get(
        "dnssec"
    )

    soa = dns.get(
        "SOA"
    )

    zone_soa = dns.get(
        "zone_SOA"
    )

    dns_sources = dns.get(
        "sources"
    )

    dns_ttls = dns.get(
        "ttls"
    )

    return {

        "dns_available":
            to_binary(bool(dns)),

        "dns_a_count":
            len(a_records),

        "dns_aaaa_count":
            len(aaaa_records),

        "dns_has_a":
            to_binary(
                bool(a_records)
            ),

        "dns_has_aaaa":
            to_binary(
                bool(aaaa_records)
            ),

        "dns_mx_count":
            len(mx_records),

        "dns_has_mx":
            to_binary(
                bool(mx_records)
            ),

        "dns_mx_related_ip_count":
            len(mx_related_ips),

        "dns_mx_min_priority":
            mx_min_priority,

        "dns_mx_max_priority":
            mx_max_priority,

        "dns_mx_mean_priority":
            mx_mean_priority,

        "dns_ns_count":
            len(ns_records),

        "dns_has_ns":
            to_binary(
                bool(ns_records)
            ),

        "dns_ns_related_ip_count":
            len(ns_related_ips),

        "dns_txt_count":
            txt_count,

        "dns_has_txt":
            to_binary(
                txt_count > 0
            ),

        "dns_has_cname":
            to_binary(
                bool(cname)
            ),

        "dns_ipv4_count":
            len(ipv4),

        "dns_ipv6_count":
            len(ipv6),

        "dns_unique_ip_count":
            len(all_dns_ips),

        "dnssec_information_present":
            to_binary(
                dnssec is not None
            ),

        "dns_has_soa":
            to_binary(
                bool(soa)
            ),

        "dns_has_zone_soa":
            to_binary(
                bool(zone_soa)
            ),

        "dns_source_count":
            count_structure(
                dns_sources
            ),

        "dns_ttl_information_count":
            count_structure(
                dns_ttls
            )
    }


# ============================================================
# RDAP / WHOIS FEATURE EXTRACTION
# ============================================================

def extract_rdap_features(
    record: dict
) -> dict:

    rdap = safe_dict(
        record.get("rdap")
    )

    evaluated_on = (
        parse_datetime(
            record.get(
                "evaluated_on"
            )
        )
    )

    registration_date = (
        parse_datetime(
            rdap.get(
                "registration_date"
            )
        )
    )

    expiration_date = (
        parse_datetime(
            rdap.get(
                "expiration_date"
            )
        )
    )

    last_changed_date = (
        parse_datetime(
            rdap.get(
                "last_changed_date"
            )
        )
    )

    domain_age_days = (
        days_between(
            registration_date,
            evaluated_on
        )
    )

    registration_period_days = (
        days_between(
            registration_date,
            expiration_date
        )
    )

    days_until_expiration = (
        days_between(
            evaluated_on,
            expiration_date
        )
    )

    days_since_last_change = (
        days_between(
            last_changed_date,
            evaluated_on
        )
    )

    entities = safe_dict(
        rdap.get(
            "entities"
        )
    )

    registrar = safe_list(
        entities.get(
            "registrar"
        )
    )

    registrant = safe_list(
        entities.get(
            "registrant"
        )
    )

    abuse = safe_list(
        entities.get(
            "abuse"
        )
    )

    admin = safe_list(
        entities.get(
            "admin"
        )
    )

    administrative = safe_list(
        entities.get(
            "administrative"
        )
    )

    technical = safe_list(
        entities.get(
            "technical"
        )
    )

    noc = safe_list(
        entities.get(
            "noc"
        )
    )

    nameservers = (
        rdap.get(
            "nameservers"
        )
    )

    status = rdap.get(
        "status"
    )

    rdap_dnssec = (
        rdap.get(
            "dnssec"
        )
    )

    description = (
        rdap.get(
            "description"
        )
    )

    return {

        "rdap_available":
            to_binary(
                bool(rdap)
            ),

        "domain_age_days":
            domain_age_days,

        "registration_period_days":
            registration_period_days,

        "days_until_expiration":
            days_until_expiration,

        "days_since_last_change":
            days_since_last_change,

        "rdap_has_registration_date":
            to_binary(
                registration_date
                is not None
            ),

        "rdap_has_expiration_date":
            to_binary(
                expiration_date
                is not None
            ),

        "rdap_has_last_changed_date":
            to_binary(
                last_changed_date
                is not None
            ),

        "rdap_has_whois_server":
            to_binary(
                bool(
                    rdap.get(
                        "whois_server"
                    )
                )
            ),

        "rdap_has_name":
            to_binary(
                bool(
                    rdap.get(
                        "name"
                    )
                )
            ),

        "rdap_has_handle":
            to_binary(
                bool(
                    rdap.get(
                        "handle"
                    )
                )
            ),

        "rdap_registrar_count":
            len(registrar),

        "rdap_registrant_count":
            len(registrant),

        "rdap_abuse_count":
            len(abuse),

        "rdap_admin_count":
            (
                len(admin)
                + len(administrative)
            ),

        "rdap_technical_count":
            len(technical),

        "rdap_noc_count":
            len(noc),

        "rdap_has_registrar":
            to_binary(
                bool(registrar)
            ),

        "rdap_has_registrant":
            to_binary(
                bool(registrant)
            ),

        "rdap_has_abuse_contact":
            to_binary(
                bool(abuse)
            ),

        "rdap_nameserver_count":
            count_structure(
                nameservers
            ),

        "rdap_status_count":
            count_structure(
                status
            ),

        "rdap_dnssec_information_present":
            to_binary(
                rdap_dnssec
                is not None
            ),

        "rdap_description_count":
            count_structure(
                description
            )
    }


# ============================================================
# TLS FEATURE EXTRACTION
# ============================================================

def extract_tls_features(
    record: dict
) -> dict:

    tls = safe_dict(
        record.get("tls")
    )

    certificates = safe_list(
        tls.get(
            "certificates"
        )
    )

    root_certificates = []
    non_root_certificates = []

    for cert in certificates:

        if not isinstance(
            cert,
            dict
        ):
            continue

        if cert.get(
            "is_root"
        ) is True:

            root_certificates.append(
                cert
            )

        else:

            non_root_certificates.append(
                cert
            )

    leaf = (
        non_root_certificates[0]
        if non_root_certificates
        else {}
    )

    evaluated_on = (
        parse_datetime(
            record.get(
                "evaluated_on"
            )
        )
    )

    validity_start = (
        parse_datetime(
            leaf.get(
                "validity_start"
            )
        )
    )

    validity_end = (
        parse_datetime(
            leaf.get(
                "validity_end"
            )
        )
    )

    certificate_age_days = (
        days_between(
            validity_start,
            evaluated_on
        )
    )

    certificate_days_remaining = (
        days_between(
            evaluated_on,
            validity_end
        )
    )

    # --------------------------------------------------------
    # IMPORTANT FIX:
    # valid_len is measured in seconds.
    # Convert it into DAYS.
    # --------------------------------------------------------

    valid_len = safe_number(
        leaf.get(
            "valid_len"
        )
    )

    if valid_len is not None:

        certificate_validity_days = (
            valid_len / 86400
        )

    else:

        certificate_validity_days = (
            days_between(
                validity_start,
                validity_end
            )
        )

    expired = None

    if (
        evaluated_on is not None
        and validity_end is not None
    ):

        expired = int(
            validity_end
            < evaluated_on
        )

    extension_count = safe_number(
        leaf.get(
            "extension_count"
        )
    )

    extensions = safe_list(
        leaf.get(
            "extensions"
        )
    )

    critical_extension_count = 0

    for extension in extensions:

        if not isinstance(
            extension,
            dict
        ):
            continue

        critical = extension.get(
            "critical"
        )

        if critical in (
            1,
            True
        ):
            critical_extension_count += 1

    return {

        "tls_available":
            to_binary(
                bool(tls)
            ),

        "tls_certificate_count":
            len(certificates),

        "tls_non_root_certificate_count":
            len(
                non_root_certificates
            ),

        "tls_root_certificate_count":
            len(
                root_certificates
            ),

        "tls_has_leaf_certificate":
            to_binary(
                bool(leaf)
            ),

        "tls_has_common_name":
            to_binary(
                bool(
                    leaf.get(
                        "common_name"
                    )
                )
            ),

        "tls_has_organization":
            to_binary(
                bool(
                    leaf.get(
                        "organization"
                    )
                )
            ),

        "tls_has_country":
            to_binary(
                bool(
                    leaf.get(
                        "country"
                    )
                )
            ),

        "tls_certificate_age_days":
            certificate_age_days,

        "tls_certificate_validity_days":
            certificate_validity_days,

        "tls_certificate_days_remaining":
            certificate_days_remaining,

        "tls_certificate_expired":
            expired,

        "tls_extension_count":
            extension_count,

        "tls_critical_extension_count":
            critical_extension_count,

        "tls_protocol":
            tls.get(
                "protocol"
            ),

        "tls_cipher":
            tls.get(
                "cipher"
            )
    }


# ============================================================
# IP / HOSTING FEATURE EXTRACTION
# ============================================================

def extract_ip_features(
    record: dict
) -> dict:

    ip_data = safe_list(
        record.get(
            "ip_data"
        )
    )

    unique_ips = set()

    source_counts = {
        "A": 0,
        "AAAA": 0,
        "CNAME": 0,
        "MX": 0,
        "NS": 0
    }

    alive_count = 0

    rtt_values = []

    asn_numbers = set()
    asn_organizations = set()

    countries = set()
    country_codes = set()

    rir_values = set()

    ip_rdap_count = 0

    ipv4_count = 0
    ipv6_count = 0

    for item in ip_data:

        if not isinstance(
            item,
            dict
        ):
            continue

        ip = item.get(
            "ip"
        )

        if ip:

            ip_string = str(ip)

            unique_ips.add(
                ip_string
            )

            if ":" in ip_string:
                ipv6_count += 1
            else:
                ipv4_count += 1

        source = item.get(
            "from_record"
        )

        if source in source_counts:

            source_counts[
                source
            ] += 1

        remarks = safe_dict(
            item.get(
                "remarks"
            )
        )

        if remarks.get(
            "is_alive"
        ) is True:

            alive_count += 1

        rtt = safe_number(
            remarks.get(
                "average_rtt"
            )
        )

        if rtt is not None:

            rtt_values.append(
                rtt
            )

        asn = safe_dict(
            item.get(
                "asn"
            )
        )

        asn_number = (
            asn.get(
                "asn"
            )
        )

        if asn_number is not None:

            asn_numbers.add(
                str(asn_number)
            )

        as_org = asn.get(
            "as_org"
        )

        if as_org:

            asn_organizations.add(
                str(as_org)
            )

        geo = safe_dict(
            item.get(
                "geo"
            )
        )

        geo_country = (
            geo.get(
                "country"
            )
        )

        if geo_country:

            countries.add(
                str(
                    geo_country
                )
            )

        geo_country_code = (
            geo.get(
                "country_code"
            )
        )

        if geo_country_code:

            country_codes.add(
                str(
                    geo_country_code
                )
            )

        ip_rdap = safe_dict(
            item.get(
                "rdap"
            )
        )

        if ip_rdap:

            ip_rdap_count += 1

        rir = ip_rdap.get(
            "rir"
        )

        if rir:

            rir_values.add(
                str(rir)
            )

        rdap_country = (
            ip_rdap.get(
                "country"
            )
        )

        if rdap_country:

            countries.add(
                str(
                    rdap_country
                )
            )

    total_ips = len(
        ip_data
    )

    alive_ratio = None

    if total_ips > 0:

        alive_ratio = (
            alive_count
            / total_ips
        )

    return {

        "ip_data_available":
            to_binary(
                bool(ip_data)
            ),

        "ip_record_count":
            len(ip_data),

        "unique_ip_count":
            len(
                unique_ips
            ),

        "ip_ipv4_count":
            ipv4_count,

        "ip_ipv6_count":
            ipv6_count,

        "ip_from_a_count":
            source_counts["A"],

        "ip_from_aaaa_count":
            source_counts["AAAA"],

        "ip_from_cname_count":
            source_counts["CNAME"],

        "ip_from_mx_count":
            source_counts["MX"],

        "ip_from_ns_count":
            source_counts["NS"],

        "ip_alive_count":
            alive_count,

        "ip_alive_ratio":
            alive_ratio,

        "ip_average_rtt":
            safe_mean(
                rtt_values
            ),

        "ip_rdap_available_count":
            ip_rdap_count,

        "asn_count":
            len(
                asn_numbers
            ),

        "asn_organization_count":
            len(
                asn_organizations
            ),

        "country_count":
            len(
                countries
            ),

        "country_code_count":
            len(
                country_codes
            ),

        "rir_count":
            len(
                rir_values
            )
    }


# ============================================================
# COMBINE ALL FEATURES
# ============================================================

def extract_features(
    record: dict,
    label: int,
    class_name: str
) -> dict:

    row = {

        "domain_name":
            record.get(
                "domain_name"
            ),

        "source":
            record.get(
                "source"
            ),

        "label":
            label,

        "class_name":
            class_name
    }

    row.update(
        extract_dns_features(
            record
        )
    )

    row.update(
        extract_rdap_features(
            record
        )
    )

    row.update(
        extract_tls_features(
            record
        )
    )

    row.update(
        extract_ip_features(
            record
        )
    )

    return row


# ============================================================
# OUTPUT COLUMN ORDER
# ============================================================

FIELDNAMES = [

    "domain_name",
    "source",

    # DNS
    "dns_available",
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

    # RDAP / WHOIS
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
    "rdap_noc_count",

    "rdap_has_registrar",
    "rdap_has_registrant",
    "rdap_has_abuse_contact",

    "rdap_nameserver_count",
    "rdap_status_count",
    "rdap_dnssec_information_present",
    "rdap_description_count",

    # TLS
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

    # IP / HOSTING
    "ip_data_available",
    "ip_record_count",
    "unique_ip_count",

    "ip_ipv4_count",
    "ip_ipv6_count",

    "ip_from_a_count",
    "ip_from_aaaa_count",
    "ip_from_cname_count",
    "ip_from_mx_count",
    "ip_from_ns_count",

    "ip_alive_count",
    "ip_alive_ratio",
    "ip_average_rtt",

    "ip_rdap_available_count",

    "asn_count",
    "asn_organization_count",

    "country_count",
    "country_code_count",

    "rir_count",

    # TARGET
    "label",
    "class_name"
]


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    dataset_path: Path,
    writer: csv.DictWriter,
    label: int,
    class_name: str
) -> tuple[int, int]:

    print()
    print("=" * 78)

    print(
        f"Processing: "
        f"{dataset_path.name}"
    )

    print(
        f"Class:      "
        f"{class_name}"
    )

    print(
        f"Label:      "
        f"{label}"
    )

    print("=" * 78)

    processed = 0
    skipped = 0

    start_time = time.time()

    for record_number, record in enumerate(
        stream_records(
            dataset_path
        ),
        start=1
    ):

        try:

            domain_name = (
                record.get(
                    "domain_name"
                )
            )

            if not domain_name:

                skipped += 1
                continue

            row = extract_features(
                record=record,
                label=label,
                class_name=class_name
            )

            writer.writerow(
                row
            )

            processed += 1

        except Exception as error:

            skipped += 1

            if skipped <= 10:

                print()

                print(
                    f"Warning at record "
                    f"{record_number:,}: "
                    f"{error}"
                )

        if (
            processed > 0
            and processed
            % PROGRESS_INTERVAL
            == 0
        ):

            elapsed = (
                time.time()
                - start_time
            )

            speed = (
                processed / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"Processed "
                f"{processed:,} "
                f"{class_name} records "
                f"| skipped "
                f"{skipped:,} "
                f"| {speed:,.1f} "
                f"records/sec"
            )

    elapsed = (
        time.time()
        - start_time
    )

    print()

    print(
        f"Finished "
        f"{dataset_path.name}"
    )

    print(
        f"Successfully processed: "
        f"{processed:,}"
    )

    print(
        f"Skipped:                "
        f"{skipped:,}"
    )

    print(
        f"Time:                   "
        f"{elapsed / 60:.2f} minutes"
    )

    return (
        processed,
        skipped
    )


# ============================================================
# ENVIRONMENT VALIDATION
# ============================================================

def validate_environment() -> None:

    print()
    print("=" * 78)

    print(
        "01 - FULL DOMAIN INTELLIGENCE "
        "DATASET PREPROCESSING"
    )

    print("=" * 78)

    print(
        f"Python version: "
        f"{sys.version.split()[0]}"
    )

    print(
        f"Project root:   "
        f"{PROJECT_ROOT}"
    )

    print()

    for dataset in [
        BENIGN_FILE,
        PHISHING_FILE
    ]:

        if not dataset.exists():

            raise FileNotFoundError(
                "\nDataset not found:\n"
                f"{dataset}"
            )

        size_gb = (
            dataset.stat().st_size
            / (1024 ** 3)
        )

        print(
            f"{dataset.name:<25}"
            f"{size_gb:.2f} GB"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    validate_environment()

    print()

    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "Processing ALL available "
        "benign and phishing records."
    )

    print(
        "Existing output CSV will be "
        "overwritten."
    )

    start_all = time.time()

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as output:

        writer = csv.DictWriter(
            output,
            fieldnames=FIELDNAMES,
            extrasaction="ignore"
        )

        writer.writeheader()

        # ----------------------------------------------------
        # BENIGN
        # ----------------------------------------------------

        benign_processed, benign_skipped = (
            process_dataset(
                dataset_path=BENIGN_FILE,
                writer=writer,
                label=0,
                class_name="benign"
            )
        )

        output.flush()

        # ----------------------------------------------------
        # PHISHING
        # ----------------------------------------------------

        phishing_processed, phishing_skipped = (
            process_dataset(
                dataset_path=PHISHING_FILE,
                writer=writer,
                label=1,
                class_name="phishing"
            )
        )

        output.flush()

    total_processed = (
        benign_processed
        + phishing_processed
    )

    total_skipped = (
        benign_skipped
        + phishing_skipped
    )

    total_time = (
        time.time()
        - start_all
    )

    print()
    print("=" * 78)

    print(
        "FULL PREPROCESSING COMPLETED"
    )

    print("=" * 78)

    print(
        f"Benign records:   "
        f"{benign_processed:,}"
    )

    print(
        f"Phishing records: "
        f"{phishing_processed:,}"
    )

    print(
        f"Total records:    "
        f"{total_processed:,}"
    )

    print(
        f"Skipped records:  "
        f"{total_skipped:,}"
    )

    print(
        f"Total runtime:    "
        f"{total_time / 60:.2f} minutes"
    )

    print()

    print(
        "Processed dataset saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        f"Total extracted columns: "
        f"{len(FIELDNAMES)}"
    )


if __name__ == "__main__":
    main()