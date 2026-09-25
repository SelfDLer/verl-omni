# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export a byte-bounded schema brief from an existing local summary.json."""

import argparse
import json
from collections import Counter
from pathlib import Path


def line(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def render(report, max_bytes=2048):
    """Deduplicate schema records and fit complete JSON lines inside a byte cap."""
    if max_bytes < 512:
        raise ValueError("--max-bytes must be at least 512")
    if report.get("format_version") != 1 or not isinstance(report.get("inventory"), list):
        raise ValueError("expected format_version=1 profile summary.json with inventory")
    inventory = report["inventory"]
    schemas = {}
    formats = Counter()
    errors = 0
    for entry in inventory:
        fmt = entry.get("format", "unknown")
        # Fixed categories: arbitrary input strings never become file paths in output.
        fmt = fmt if fmt in ("sqlite", "csv", "json", "raw_or_unsupported", "unreadable") else "unknown"
        formats[fmt] += 1
        errors += int(bool(entry.get("capability_gap")))
        candidates = []
        for table in entry.get("tables", []):
            candidates.append(
                {
                    "format": fmt,
                    "table": table["name"],
                    "columns": [[c["name"], c.get("type", "")] for c in table.get("columns", [])],
                    "upstream_columns_omitted": table.get("columns_truncated", 0),
                }
            )
        for field in ("columns", "first_event_fields", "first_args_fields", "json_key_sample"):
            if field in entry:
                candidates.append({"format": fmt, "field": field, "columns": entry[field]})
        for schema in candidates:
            key = line(schema)
            if key not in schemas:
                schemas[key] = {**schema, "occurrences": 0}
            schemas[key]["occurrences"] += 1

    def priority(schema):
        name = str(schema.get("table", schema.get("field", ""))).lower()
        relevant = any(word in name for word in ("memory", "alloc", "hccl", "comm", "task", "op", "event"))
        return (not relevant, name, line(schema))

    counts = report.get("counts", {})
    header = line(
        {
            "brief": 1,
            "scope": "schema sample only; not performance or root-cause evidence",
            "discovered": counts.get("discovered_files"),
            "inventory_entries": len(inventory),
            "upstream_inventory_omitted": counts.get("inventory_files_omitted"),
            "formats": dict(formats),
            "entries_with_gap": errors,
            "upstream_gap_kinds": len(report.get("capability_gaps", {})),
            "unique_schemas": len(schemas),
        }
    )
    records = []
    for index, schema in enumerate(sorted(schemas.values(), key=priority)):
        columns = schema["columns"]
        base = {k: v for k, v in schema.items() if k != "columns"}
        # Repeated identity makes every retained chunk interpretable independently.
        for offset in range(0, max(1, len(columns)), 4):
            records.append(
                line(
                    {
                        "schema": index,
                        **base,
                        "column_total": len(columns),
                        "offset": offset,
                        "columns": columns[offset : offset + 4],
                    }
                )
            )
    # Reserve enough room for the largest possible omission footer before fitting.
    reserve = len(line({"schema_records_total": len(records), "schema_records_omitted": len(records)}))
    if len(header) + reserve > max_bytes:
        raise ValueError("metadata exceeds budget; increase --max-bytes")
    output = bytearray(header)
    omitted = 0
    for record in records:
        if len(output) + len(record) + reserve <= max_bytes:
            output.extend(record)
        else:
            omitted += 1
    output.extend(line({"schema_records_total": len(records), "schema_records_omitted": omitted}))
    return bytes(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=2048)
    args = parser.parse_args()
    if args.summary.resolve() == args.output.resolve():
        parser.error("output must differ from input")
    try:
        report = json.loads(args.summary.read_text(encoding="utf-8-sig"))
        content = render(report, args.max_bytes)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    # Exclusive creation also protects existing reports and symlinks from overwrite.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as dst:
        dst.write(content)
    print(f"{args.output}: {len(content)} bytes (limit {args.max_bytes})")


if __name__ == "__main__":
    main()
