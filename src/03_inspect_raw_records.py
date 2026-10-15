from __future__ import annotations

import json
from pathlib import Path
from pprint import pprint

import ijson


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\SLIIT\final reserch chamath")

RAW_DIR = PROJECT_ROOT / "raw"

BENIGN_FILE = RAW_DIR / "benign_cesnet.json"
PHISHING_FILE = RAW_DIR / "phishing.json"


# Number of records to inspect from each dataset
MAX_RECORDS = 10


# ============================================================
# STREAM JSON
# ============================================================

def first_non_whitespace_byte(path: Path) -> bytes:

    with path.open("rb") as file:

        while True:

            char = file.read(1)

            if not char:
                raise ValueError(
                    f"Empty file: {path}"
                )

            if not char.isspace():
                return char


def stream_records(path: Path):

    first_char = first_non_whitespace_byte(path)

    # Standard JSON array
    if first_char == b"[":

        with path.open("rb") as file:

            for record in ijson.items(
                file,
                "item"
            ):

                if isinstance(record, dict):
                    yield record

    # NDJSON / JSON Lines
    elif first_char == b"{":

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                    if isinstance(record, dict):
                        yield record

                except json.JSONDecodeError:
                    continue

    else:

        raise ValueError(
            f"Unsupported JSON format: {path}"
        )


# ============================================================
# INSPECTION
# ============================================================

def inspect_file(path: Path, class_name: str):

    print()
    print("=" * 80)
    print(f"INSPECTING: {class_name.upper()}")
    print(f"FILE: {path.name}")
    print("=" * 80)

    dns_keys_found = set()
    ip_keys_found = set()
    rdap_keys_found = set()
    tls_keys_found = set()

    mx_examples = []
    ns_examples = []
    ip_examples = []

    count = 0

    for record in stream_records(path):

        count += 1

        dns = record.get("dns")

        if isinstance(dns, dict):

            dns_keys_found.update(
                dns.keys()
            )

            # Capture possible MX values
            for key, value in dns.items():

                if "mx" in key.lower():

                    mx_examples.append(
                        {
                            "key": key,
                            "value": value
                        }
                    )

                if (
                    key.lower() == "ns"
                    or "nameserver" in key.lower()
                    or "name_server" in key.lower()
                ):

                    ns_examples.append(
                        {
                            "key": key,
                            "value": value
                        }
                    )

        rdap = record.get("rdap")

        if isinstance(rdap, dict):

            rdap_keys_found.update(
                rdap.keys()
            )

        tls = record.get("tls")

        if isinstance(tls, dict):

            tls_keys_found.update(
                tls.keys()
            )

        ip_data = record.get("ip_data")

        if isinstance(ip_data, list):

            for item in ip_data:

                if isinstance(item, dict):

                    ip_keys_found.update(
                        item.keys()
                    )

                    if len(ip_examples) < 3:
                        ip_examples.append(item)

        if count >= MAX_RECORDS:
            break

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("DNS KEYS")
    print("-" * 80)

    pprint(
        sorted(dns_keys_found)
    )

    print()
    print("RDAP KEYS")
    print("-" * 80)

    pprint(
        sorted(rdap_keys_found)
    )

    print()
    print("TLS KEYS")
    print("-" * 80)

    pprint(
        sorted(tls_keys_found)
    )

    print()
    print("IP DATA KEYS")
    print("-" * 80)

    pprint(
        sorted(ip_keys_found)
    )

    print()
    print("MX EXAMPLES")
    print("-" * 80)

    if mx_examples:

        for example in mx_examples[:5]:
            pprint(example)

    else:

        print(
            "No MX-related fields found "
            "in inspected records."
        )

    print()
    print("NS / NAMESERVER EXAMPLES")
    print("-" * 80)

    if ns_examples:

        for example in ns_examples[:5]:
            pprint(example)

    else:

        print(
            "No NS-related fields found "
            "in inspected records."
        )

    print()
    print("IP DATA EXAMPLES")
    print("-" * 80)

    if ip_examples:

        for example in ip_examples:
            pprint(example)
            print()

    else:

        print(
            "No IP data examples found."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("RAW DOMAIN INTELLIGENCE STRUCTURE INSPECTION")
    print("=" * 80)

    inspect_file(
        BENIGN_FILE,
        "benign"
    )

    inspect_file(
        PHISHING_FILE,
        "phishing"
    )

    print()
    print("=" * 80)
    print("INSPECTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()