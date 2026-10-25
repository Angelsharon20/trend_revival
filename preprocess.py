"""
preprocess.py - multi-core cleanup pipeline.

raw_posts  --clean / tokenise / extract-->  clean_posts
           and  -->  topic_mentions {post_id, topic, timestamp}

Two execution modes share the same pure function `process_record`:
  * run_mongo : raw_posts -> clean_posts / topic_mentions inside MongoDB
  * run_local : JSONL dataset -> data/processed/*.jsonl (no database needed)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import config

STOPWORDS = frozenset(
    "a an the and or but if of to in on at for with by from as is are was were be been it its this that these "
    "those i me my we our you your they them their he she his her not no so do does did have has had just really "
    "very than then too can will would should about into over out up down all any some more most also here there "
    "what when where who how why".split()
)
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_PUNCT = re.compile(r"[^\w\s]|_")
_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Pure helpers (picklable, side-effect free)
# --------------------------------------------------------------------------- #
def parse_timestamp(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def clean_text(text: str) -> str:
    """Lowercase, drop URLs / @mentions, strip special punctuation, collapse spaces."""
    t = (text or "").lower()
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def tokenize(clean: str) -> list[str]:
    return [w for w in clean.split() if len(w) > 1 and w not in STOPWORDS and not w.isdigit()]


def extract_hashtags(text: str, tags: list[str] | None) -> list[str]:
    """Unique, lowercase, sorted hashtags from the free text and the hashtag field."""
    found = {"#" + t.lower() for t in _HASHTAG.findall(text or "")}
    for tag in tags or []:
        tag = str(tag).strip().lower()
        if tag:
            found.add(tag if tag.startswith("#") else "#" + tag)
    return sorted(found)


def process_record(rec: dict) -> tuple[dict, list[dict]]:
    """raw record -> (clean_post_doc, [topic_mention_docs]); never mutates `rec`."""
    ts = parse_timestamp(rec["timestamp"])
    cleaned = clean_text(rec.get("text", ""))
    tokens = tokenize(cleaned)
    hashtags = extract_hashtags(rec.get("text", ""), rec.get("hashtags"))
    clean_doc = {
        "post_id": rec["post_id"],
        "user_id": rec.get("user_id"),
        "platform": rec.get("platform"),
        "clean_text": cleaned,
        "tokens": tokens,
        "hashtags": hashtags,
        "n_tokens": len(tokens),
        "timestamp": ts,
        "likes": int(rec.get("likes", 0)),
        "shares": int(rec.get("shares", 0)),
    }
    mentions = [{"post_id": rec["post_id"], "topic": tag, "timestamp": ts} for tag in hashtags]
    return clean_doc, mentions


def _iso(doc: dict) -> dict:
    return {k: (v.strftime("%Y-%m-%dT%H:%M:%SZ") if isinstance(v, datetime) else v) for k, v in doc.items()}


def _process_lines(lines: list[str]):
    clean_out, mention_out, topics = [], [], set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        clean, mentions = process_record(json.loads(line))
        clean_out.append(json.dumps(_iso(clean), ensure_ascii=False))
        for m in mentions:
            mention_out.append(json.dumps(_iso(m)))
            topics.add(m["topic"])
    return clean_out, mention_out, topics


def _process_docs(docs: list[dict]):
    clean_docs, mention_docs = [], []
    for d in docs:
        clean, mentions = process_record(d)
        clean_docs.append(clean)
        mention_docs.extend(mentions)
    return clean_docs, mention_docs


def _default_workers(workers: int | None) -> int:
    return workers if workers else max(1, (os.cpu_count() or 2) - 1)


def _chunked(iterable, size: int):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _map_chunks(func, chunks, workers: int):
    if workers > 1:
        with mp.Pool(workers) as pool:
            yield from pool.imap(func, chunks)
    else:
        for chunk in chunks:
            yield func(chunk)


# --------------------------------------------------------------------------- #
# Local (no database) pipeline
# --------------------------------------------------------------------------- #
def run_local(dataset_path: Path | str = config.RAW_DATASET_PATH, out_dir: Path | str = config.PROCESSED_DIR,
              workers: int | None = None, chunk_lines: int = 5000, verbose: bool = True) -> dict:
    dataset_path, out_dir = Path(dataset_path), Path(out_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found - run dataset_generator.py first")
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = _default_workers(workers)
    started = time.time()
    n_posts = n_mentions = 0
    topics: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as src, \
            (out_dir / "clean_posts.jsonl").open("w", encoding="utf-8") as clean_f, \
            (out_dir / "topic_mentions.jsonl").open("w", encoding="utf-8") as ment_f:
        for clean_out, mention_out, chunk_topics in _map_chunks(_process_lines, _chunked(src, chunk_lines), workers):
            clean_f.write("\n".join(clean_out) + "\n" if clean_out else "")
            ment_f.write("\n".join(mention_out) + "\n" if mention_out else "")
            n_posts += len(clean_out)
            n_mentions += len(mention_out)
            topics |= chunk_topics
    stats = _write_stats(out_dir, n_posts, len(topics), n_mentions, "local", round(time.time() - started, 2))
    if verbose:
        print(f"[preprocess/local] {n_posts:,} posts -> {n_mentions:,} mentions, {len(topics):,} topics ({workers} workers)")
    return stats


def iter_local_mentions(path: Path | str = config.TOPIC_MENTIONS_PATH):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                doc = json.loads(line)
                doc["timestamp"] = parse_timestamp(doc["timestamp"])
                yield doc


def iter_local_clean_posts(path: Path | str = config.CLEAN_POSTS_PATH):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                doc = json.loads(line)
                doc["timestamp"] = parse_timestamp(doc["timestamp"])
                yield doc


# --------------------------------------------------------------------------- #
# MongoDB pipeline
# --------------------------------------------------------------------------- #
def run_mongo(db=None, workers: int | None = None, batch: int = 20000, verbose: bool = True) -> dict:
    db = db if db is not None else config.get_db()
    if db is None:
        raise RuntimeError("MongoDB is not reachable")
    raw = db[config.COLLECTIONS["raw_posts"]]
    clean = db[config.COLLECTIONS["clean_posts"]]
    mentions = db[config.COLLECTIONS["topic_mentions"]]
    if raw.estimated_document_count() == 0:
        raise RuntimeError("raw_posts is empty - run ingest_data.py first")
    clean.drop()
    mentions.drop()
    workers = _default_workers(workers)
    started = time.time()
    n_posts = n_mentions = 0
    topics: set[str] = set()

    def sub_chunks(docs):
        return list(_chunked(docs, 2000))

    cursor = raw.find({}, {"_id": 0}, batch_size=batch)
    for docs in _chunked(cursor, batch):
        for clean_docs, mention_docs in _map_chunks(_process_docs, sub_chunks(docs), workers):
            if clean_docs:
                clean.insert_many(clean_docs, ordered=False)
            if mention_docs:
                mentions.insert_many(mention_docs, ordered=False)
            n_posts += len(clean_docs)
            n_mentions += len(mention_docs)
            topics.update(m["topic"] for m in mention_docs)
    clean.create_index("post_id", unique=True, name="uq_post_id")
    clean.create_index("timestamp", name="idx_timestamp")
    mentions.create_index([("topic", 1), ("timestamp", 1)], name="idx_topic_timestamp")
    mentions.create_index("post_id", name="idx_post_id")
    stats = _write_stats(config.PROCESSED_DIR, n_posts, len(topics), n_mentions, "mongodb", round(time.time() - started, 2))
    if verbose:
        print(f"[preprocess/mongo] {n_posts:,} posts -> {n_mentions:,} mentions, {len(topics):,} topics; indexes created")
    return stats


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def _write_stats(out_dir: Path, posts: int, topics: int, mentions: int, engine: str, seconds: float) -> dict:
    stats = {"total_posts": posts, "unique_topics": topics, "total_mentions": mentions, "months": config.N_MONTHS,
             "period_start": config.MONTHS[0], "period_end": config.MONTHS[-1], "engine": engine, "seconds": seconds}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_stats() -> dict:
    if config.STATS_PATH.exists():
        return json.loads(config.STATS_PATH.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw posts and build topic_mentions")
    parser.add_argument("--mode", choices=["auto", "mongo", "local"], default="auto")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    db = config.get_db() if args.mode in ("auto", "mongo") else None
    use_mongo = db is not None and db[config.COLLECTIONS["raw_posts"]].estimated_document_count() > 0
    if args.mode == "mongo" and not use_mongo:
        raise SystemExit("MongoDB unavailable or raw_posts empty (run ingest_data.py)")
    if use_mongo and args.mode != "local":
        run_mongo(db, workers=args.workers)
    else:
        print("[preprocess] using local mode (no MongoDB data found)")
        run_local(workers=args.workers)


if __name__ == "__main__":
    main()
