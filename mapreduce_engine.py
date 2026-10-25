"""
mapreduce_engine.py - MapReduce over `topic_mentions`.

1. MongoDB engine   : genuine server-side mapReduce command (JavaScript map/reduce
                       executed by the server through db.command == db.runCommand)
                       writing natively into `topic_period_counts`.
2. Python fallback  : functional Map -> Shuffle -> Reduce that mirrors the JS logic
                       exactly, so the pipeline runs without a MongoDB server.

Both produce {(topic, 'YYYY-MM'): mentions}.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import config

# --------------------------------------------------------------------------- #
# JavaScript executed inside mongod
# --------------------------------------------------------------------------- #
MAP_JS = """
function () {
    var d = this.timestamp;
    var month = ("0" + (d.getUTCMonth() + 1)).slice(-2);
    var period = d.getUTCFullYear() + "-" + month;
    emit({ topic: this.topic, period: period }, 1);
}
"""

REDUCE_JS = """
function (key, values) {
    var total = 0;
    for (var i = 0; i < values.length; i++) {
        total += values[i];
    }
    return total;
}
"""


def run_mapreduce_mongo(db=None) -> dict:
    """Execute the server-side mapReduce command and replace `topic_period_counts`."""
    from bson.code import Code
    from bson.son import SON

    db = db if db is not None else config.get_db()
    if db is None:
        raise RuntimeError("MongoDB is not reachable")
    command = SON([
        ("mapReduce", config.COLLECTIONS["topic_mentions"]),
        ("map", Code(MAP_JS)),
        ("reduce", Code(REDUCE_JS)),
        ("out", {"replace": config.COLLECTIONS["topic_period_counts"]}),
        ("query", {}),
    ])
    started = time.time()
    result = db.command(command)              # == db.runCommand({mapReduce: ...})
    return {"engine": "mongodb", "counts": result.get("counts", {}), "seconds": round(time.time() - started, 2)}


# --------------------------------------------------------------------------- #
# Pure-Python Map -> Shuffle -> Reduce (mirror of the JS above)
# --------------------------------------------------------------------------- #
def py_map(mention: dict):
    ts = mention["timestamp"]
    period = f"{ts.year}-{ts.month:02d}"
    yield (mention["topic"], period), 1


def py_shuffle(pairs) -> dict:
    groups: dict = defaultdict(list)
    for key, value in pairs:
        groups[key].append(value)
    return groups


def py_reduce(key, values) -> int:
    total = 0
    for v in values:
        total += v
    return total


def run_mapreduce_python(mentions) -> dict:
    """mentions: iterable of {'topic', 'timestamp'(datetime)} -> {(topic, period): count}."""
    def mapped():
        for m in mentions:
            yield from py_map(m)

    grouped = py_shuffle(mapped())
    return {key: py_reduce(key, values) for key, values in grouped.items()}


# --------------------------------------------------------------------------- #
# Persistence / loading
# --------------------------------------------------------------------------- #
def save_counts_local(counts: dict, path: Path | str = config.PERIOD_COUNTS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"topic": t, "period": p, "value": v} for (t, p), v in sorted(counts.items())]
    path.write_text(json.dumps(rows), encoding="utf-8")


def rows_to_series(rows, months=config.MONTHS) -> dict[str, list[int]]:
    """Iterable of (topic, period, value) -> {topic: [count per month]} (zero filled)."""
    index = {m: i for i, m in enumerate(months)}
    series: dict[str, list[int]] = {}
    for topic, period, value in rows:
        if period not in index:
            continue
        series.setdefault(topic, [0] * len(months))[index[period]] += int(round(value))
    return dict(sorted(series.items()))


def _mongo_period_counts_nonempty() -> bool:
    db = config.get_db()
    return db is not None and db[config.COLLECTIONS["topic_period_counts"]].estimated_document_count() > 0


def period_counts_available() -> bool:
    return _mongo_period_counts_nonempty() or config.PERIOD_COUNTS_PATH.exists()


def data_source() -> str:
    return "MongoDB MapReduce" if _mongo_period_counts_nonempty() else "Python MapReduce fallback"


def load_period_counts() -> dict[str, list[int]]:
    """Return {topic: [36 monthly counts]} from MongoDB output or the local cache."""
    if _mongo_period_counts_nonempty():
        db = config.get_db()
        cursor = db[config.COLLECTIONS["topic_period_counts"]].find({})
        return rows_to_series((d["_id"]["topic"], d["_id"]["period"], d["value"]) for d in cursor)
    if config.PERIOD_COUNTS_PATH.exists():
        rows = json.loads(config.PERIOD_COUNTS_PATH.read_text(encoding="utf-8"))
        return rows_to_series((r["topic"], r["period"], r["value"]) for r in rows)
    raise FileNotFoundError("topic_period_counts not found - run mapreduce_engine.py first")


def run(force_python: bool = False, verbose: bool = True) -> dict:
    """Run on MongoDB when possible, otherwise transparently use the Python engine."""
    db = None if force_python else config.get_db()
    if db is not None and db[config.COLLECTIONS["topic_mentions"]].estimated_document_count() > 0:
        try:
            info = run_mapreduce_mongo(db)
            if verbose:
                print(f"[mapreduce] MongoDB server-side job done: {info['counts']} in {info['seconds']}s")
            return info
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[mapreduce] MongoDB job failed ({exc}); falling back to Python engine")
    if not config.TOPIC_MENTIONS_PATH.exists():
        raise FileNotFoundError("topic mentions missing - run preprocess.py first")
    import preprocess

    started = time.time()
    counts = run_mapreduce_python(preprocess.iter_local_mentions())
    save_counts_local(counts)
    info = {"engine": "python", "counts": {"output": len(counts), "emit": sum(counts.values())},
            "seconds": round(time.time() - started, 2)}
    if verbose:
        print(f"[mapreduce] Python Map->Shuffle->Reduce: {len(counts):,} keys, {info['counts']['emit']:,} emits")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MapReduce job")
    parser.add_argument("--force-python", action="store_true", help="skip MongoDB, use the Python engine")
    args = parser.parse_args()
    run(force_python=args.force_python)


if __name__ == "__main__":
    main()
