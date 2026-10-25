"""
ingest_data.py - secure bulk ingestion of the dataset into MongoDB `raw_posts`.

* pre-validates every record (required fields, types, ISO timestamps)
* casts timestamps to datetime objects (naive UTC) and counts to int
* enforces a unique index on `post_id`; duplicates are skipped, never fatal
* prints a data-integrity report
Without a reachable MongoDB the script still validates the file (dry-run mode).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import config

REQUIRED_FIELDS = ("post_id", "user_id", "platform", "text", "hashtags", "timestamp")


def parse_iso(value) -> datetime:
    """ISO-8601 string -> naive UTC datetime (MongoDB's native representation)."""
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def validate_record(raw: dict) -> tuple[dict | None, str | None]:
    """Return (typed_record, None) when valid, else (None, reason)."""
    if not isinstance(raw, dict):
        return None, "not_an_object"
    for field in REQUIRED_FIELDS:
        if field not in raw or raw[field] in (None, ""):
            if field == "hashtags" and isinstance(raw.get(field), list):
                continue
            return None, f"missing_{field}"
    if not isinstance(raw["hashtags"], list) or not all(isinstance(t, str) for t in raw["hashtags"]):
        return None, "bad_hashtags"
    try:
        ts = parse_iso(raw["timestamp"])
    except (ValueError, TypeError):
        return None, "bad_timestamp"
    try:
        likes, shares = int(raw.get("likes", 0)), int(raw.get("shares", 0))
    except (ValueError, TypeError):
        return None, "bad_counts"
    return {
        "post_id": str(raw["post_id"]),
        "user_id": str(raw["user_id"]),
        "platform": str(raw["platform"]),
        "text": str(raw["text"]),
        "hashtags": list(raw["hashtags"]),
        "timestamp": ts,
        "likes": likes,
        "shares": shares,
    }, None


def read_records(path: Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"__broken_line__": line_no}


def ingest(path: Path | str = config.RAW_DATASET_PATH, batch_size: int = 5000,
           reset: bool = False, dry_run: bool = False, verbose: bool = True) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python dataset_generator.py` first")
    db = None if dry_run else config.get_db()
    collection = None
    if db is not None:
        collection = db[config.COLLECTIONS["raw_posts"]]
        if reset:
            collection.drop()
        collection.create_index("post_id", unique=True, name="uq_post_id")
        collection.create_index("timestamp", name="idx_timestamp")

    report = {"mode": "mongodb" if collection is not None else "validation-only", "read": 0, "valid": 0,
              "invalid": Counter(), "duplicates_in_file": 0, "duplicates_in_db": 0, "inserted": 0,
              "first_timestamp": None, "last_timestamp": None}
    seen: set[str] = set()
    batch: list[dict] = []
    started = time.time()

    def flush() -> None:
        if not batch or collection is None:
            batch.clear()
            return
        from pymongo.errors import BulkWriteError

        try:
            res = collection.insert_many(batch, ordered=False)
            report["inserted"] += len(res.inserted_ids)
        except BulkWriteError as exc:
            details = exc.details
            dup = sum(1 for e in details.get("writeErrors", []) if e.get("code") == 11000)
            report["duplicates_in_db"] += dup
            report["inserted"] += details.get("nInserted", 0)
            other = len(details.get("writeErrors", [])) - dup
            if other:
                report["invalid"]["db_write_error"] += other
        batch.clear()

    for raw in read_records(path):
        report["read"] += 1
        if "__broken_line__" in raw:
            report["invalid"]["broken_json"] += 1
            continue
        record, reason = validate_record(raw)
        if record is None:
            report["invalid"][reason] += 1
            continue
        if record["post_id"] in seen:
            report["duplicates_in_file"] += 1
            continue
        seen.add(record["post_id"])
        report["valid"] += 1
        ts = record["timestamp"]
        if report["first_timestamp"] is None or ts < report["first_timestamp"]:
            report["first_timestamp"] = ts
        if report["last_timestamp"] is None or ts > report["last_timestamp"]:
            report["last_timestamp"] = ts
        batch.append(record)
        if len(batch) >= batch_size:
            flush()
    flush()

    report["invalid"] = dict(report["invalid"])
    report["elapsed_sec"] = round(time.time() - started, 2)
    if verbose:
        print_report(report)
    return report


def print_report(r: dict) -> None:
    bad = sum(r["invalid"].values())
    print("=" * 60)
    print(" DATA INTEGRITY REPORT  (raw_posts)")
    print("=" * 60)
    print(f" mode                 : {r['mode']}")
    print(f" records read         : {r['read']:,}")
    print(f" valid                : {r['valid']:,}")
    print(f" rejected             : {bad:,}  {r['invalid'] or ''}")
    print(f" duplicates (file/db) : {r['duplicates_in_file']:,} / {r['duplicates_in_db']:,}")
    print(f" inserted             : {r['inserted']:,}")
    print(f" time span            : {r['first_timestamp']} -> {r['last_timestamp']}")
    print(f" elapsed              : {r['elapsed_sec']}s")
    if r["mode"] == "validation-only":
        print(" NOTE: MongoDB not reachable - file validated only (nothing inserted).")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the dataset into MongoDB raw_posts")
    parser.add_argument("--file", default=str(config.RAW_DATASET_PATH))
    parser.add_argument("--reset", action="store_true", help="drop raw_posts first")
    parser.add_argument("--dry-run", action="store_true", help="validate only")
    args = parser.parse_args()
    ingest(args.file, reset=args.reset, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
