"""
config.py - Trend Revival: central configuration.

Holds MongoDB connection parameters, collection schemas, dataset / detector
constants and the global UI theme used by the Streamlit dashboard and charts.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DATASET_PATH = DATA_DIR / "social_posts.jsonl"
CLEAN_POSTS_PATH = PROCESSED_DIR / "clean_posts.jsonl"
TOPIC_MENTIONS_PATH = PROCESSED_DIR / "topic_mentions.jsonl"
PERIOD_COUNTS_PATH = PROCESSED_DIR / "topic_period_counts.json"
COMEBACK_RESULTS_PATH = PROCESSED_DIR / "comeback_results.json"
STATS_PATH = PROCESSED_DIR / "stats.json"

# --------------------------------------------------------------------------- #
# MongoDB
# --------------------------------------------------------------------------- #
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "trend_revival_db"
MONGO_TIMEOUT_MS = 1500
FORCE_LOCAL = os.getenv("TREND_REVIVAL_FORCE_LOCAL", "0") == "1"

COLLECTIONS = {
    "raw_posts": "raw_posts",
    "clean_posts": "clean_posts",
    "topic_mentions": "topic_mentions",
    "topic_period_counts": "topic_period_counts",
    "comeback_results": "comeback_results",
}

COLLECTION_SCHEMAS = {
    "raw_posts": {
        "description": "Validated social posts exactly as ingested (typed).",
        "fields": {
            "post_id": "string (unique)",
            "user_id": "string",
            "platform": "string",
            "text": "string",
            "hashtags": "array<string>",
            "timestamp": "ISODate",
            "likes": "int",
            "shares": "int",
        },
        "indexes": ["post_id (unique)", "timestamp"],
    },
    "clean_posts": {
        "description": "Normalised posts: lowercase text, tokens, unique hashtags.",
        "fields": {
            "post_id": "string (unique)",
            "user_id": "string",
            "platform": "string",
            "clean_text": "string",
            "tokens": "array<string>",
            "hashtags": "array<string> (unique, lowercase)",
            "n_tokens": "int",
            "timestamp": "ISODate",
            "likes": "int",
            "shares": "int",
        },
        "indexes": ["post_id (unique)", "timestamp"],
    },
    "topic_mentions": {
        "description": "One document per (post, hashtag) - exploded mentions.",
        "fields": {"post_id": "string", "topic": "string (#tag)", "timestamp": "ISODate"},
        "indexes": ["{topic: 1, timestamp: 1}", "post_id"],
    },
    "topic_period_counts": {
        "description": "MapReduce output: mentions per topic per YYYY-MM.",
        "fields": {"_id": "{topic: string, period: 'YYYY-MM'}", "value": "number"},
        "indexes": ["_id (composite key)"],
    },
    "comeback_results": {
        "description": "Per-topic three-stage comeback verdicts with priority tier.",
        "fields": {
            "topic": "string",
            "historical_peak": "int",
            "peak_period": "YYYY-MM",
            "quiet_months": "int",
            "dormancy_coeff": "float",
            "recent_count": "float",
            "spike_velocity": "float",
            "is_comeback": "bool",
            "priority": "High | Medium | Low | None",
            "score": "float 0-100",
        },
        "indexes": ["topic (unique)", "is_comeback", "priority"],
    },
}

_DB_CACHE: dict = {"checked": False, "db": None}


def get_db(refresh: bool = False):
    """Return a connected pymongo Database, or None when MongoDB is unreachable."""
    if FORCE_LOCAL:
        return None
    if _DB_CACHE["checked"] and not refresh:
        return _DB_CACHE["db"]
    _DB_CACHE.update(checked=True, db=None)
    try:
        from pymongo import MongoClient

        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=MONGO_TIMEOUT_MS)
        client.admin.command("ping")
        _DB_CACHE["db"] = client[DB_NAME]
    except Exception:  # noqa: BLE001 - any failure means "no MongoDB"
        _DB_CACHE["db"] = None
    return _DB_CACHE["db"]


# --------------------------------------------------------------------------- #
# Dataset constants
# --------------------------------------------------------------------------- #
SEED = 42
TARGET_RECORDS = 124_862
START_YEAR = 2022
N_MONTHS = 36
MONTHS = [f"{START_YEAR + i // 12}-{i % 12 + 1:02d}" for i in range(N_MONTHS)]

# --------------------------------------------------------------------------- #
# Comeback detector defaults (three historical windows)
# --------------------------------------------------------------------------- #
DETECTOR = {
    "early_months": 12,          # Stage 1: Early Peak window  (2022)
    "resurgence_months": 6,      # Stage 3: Resurgence window  (last 6 months)
    "recent_avg_months": 3,      # recovery measured on the final 3 months
    "dormancy_ratio": 0.25,      # valley must stay below 25% of the peak
    "recovery_ratio": 0.50,      # final period must recover >= 50% of the peak
    "min_quiet_months": 6,       # minimum continuous valley length
    "min_peak": 20,              # ignore noise-level topics
    "high_ratio": 1.00,
    "high_quiet": 9,
    "medium_ratio": 0.75,
}

# --------------------------------------------------------------------------- #
# UI theme
# --------------------------------------------------------------------------- #
COLORS = {
    "bg": "#0D1117",
    "card": "#161B22",
    "card_alt": "#10151C",
    "line": "#263041",
    "grid": "#222B38",
    "text": "#E6EDF3",
    "muted": "#8B98A9",
    "cyan": "#00E5FF",
    "purple": "#B388FF",
    "green": "#39FF88",
    "pink": "#FF4DA6",
    "amber": "#FFC24D",
    "red": "#FF5C7A",
}
CHART_COLORS = [COLORS["cyan"], COLORS["purple"], COLORS["green"], COLORS["pink"], COLORS["amber"], "#4D9DFF", "#FF8A4D", "#7CFFEA"]
PRIORITY_COLORS = {"High": COLORS["pink"], "Medium": COLORS["amber"], "Low": COLORS["cyan"], "None": COLORS["muted"]}

# Pretty names for the hard-coded benchmark trends (topics are stored lowercase).
TOPIC_DISPLAY = {
    "#retrogaming": "#RetroGaming",
    "#booktok": "#BookTok",
    "#y2k": "#Y2K",
    "#cottagecore": "#Cottagecore",
    "#ai": "#AI",
    "#sustainableliving": "#SustainableLiving",
    "#mentalhealth": "#MentalHealth",
}


def display_name(topic: str) -> str:
    return TOPIC_DISPLAY.get(topic, topic)
