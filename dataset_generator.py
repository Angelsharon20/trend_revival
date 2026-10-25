"""
dataset_generator.py - synthetic social-media dataset with real time-series signatures.

Generates EXACTLY 124,862 posts between Jan 2022 and Dec 2024. Every topic owns a
36-month intensity curve; post counts per month are drawn with a multinomial so the
per-topic totals (and therefore the global total) are exact.

Signatures
  comeback : Initial Peak -> Decay -> Dormancy -> Resurgence
  modest   : same shape, weaker resurgence
  growth   : continuous exponential growth          (not a comeback)
  decline  : deep ongoing decline                   (not a comeback)
  flat     : steady baseline                        (not a comeback)

Output: JSON Lines (one post per line) at data/social_posts.jsonl
"""
from __future__ import annotations

import argparse
import calendar
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import config

N = config.N_MONTHS
GENERIC_TAGS = ["#viral", "#fyp", "#trending", "#explore"]
PLATFORMS = ["TikTok", "Instagram", "X", "Reddit", "YouTube"]
EPOCH = datetime(1970, 1, 1)

# Hard-coded benchmark trends used to test detection precision.
NAMED_PROFILES = {
    "#RetroGaming": dict(shape="comeback", total=9800, peak_month=3, width=2.2, base=0.05, resurgence=1.45, ramp=31.0, expected="High"),
    "#BookTok": dict(shape="comeback", total=8600, peak_month=4, width=2.4, base=0.06, resurgence=1.35, ramp=31.5, expected="High"),
    "#Y2K": dict(shape="modest", total=6200, peak_month=5, width=2.3, base=0.07, resurgence=0.72, ramp=31.0, expected="Low"),
    "#Cottagecore": dict(shape="modest", total=5600, peak_month=5, width=2.3, base=0.07, resurgence=0.62, ramp=31.5, expected="Low"),
    "#AI": dict(shape="growth", total=11000, rate=0.13, expected=None),
    "#SustainableLiving": dict(shape="decline", total=7400, rate=0.13, expected=None),
    "#MentalHealth": dict(shape="flat", total=9000, phase=0, expected=None),
}

ADJECTIVES = ["neon", "vintage", "urban", "cosmic", "mellow", "digital", "wild", "golden", "silent", "electric",
              "rustic", "lunar", "velvet", "pixel", "crystal", "midnight", "solar", "analog", "cyber", "floral",
              "glitch", "cozy", "minimal", "bold", "retro", "hyper", "indie", "coastal", "alpine", "nordic",
              "tropical", "stellar", "ember", "arctic", "mystic", "lofi", "punk", "atomic", "dreamy", "vivid"]
NOUNS = ["vinyl", "skate", "garden", "synth", "brunch", "denim", "sneaker", "podcast", "yoga", "bakery",
         "comic", "drone", "thrift", "fitness", "camera", "anime", "festival", "hiking", "cafe", "poetry",
         "tarot", "crochet", "gadget", "soundtrack", "cinema", "kitchen", "travel", "fashion", "fandom", "startup",
         "puzzle", "origami", "aquarium", "mixtape", "wellness", "woodwork", "pottery", "sketch", "remix", "rooftop"]

# synthetic population (1,237 topics + 7 named + 4 generic tags = 1,248 unique topics)
N_SYNTHETIC = 1237
SYN_COMEBACK = {"High": 8, "Medium": 7, "Low": 8}      # 23 synthetic + 4 named = 27 candidates
SYN_GROWTH, SYN_DECLINE = 40, 90
SYN_TOTALS = {"comeback": 900, "growth": 350, "decline": 180}

TEMPLATES = [
    "just discovered {tag} and honestly cannot stop thinking about it",
    "{tag} is everywhere on my feed lately",
    "hot take: {tag} was better back in the day",
    "who else is deep into {tag} this week?",
    "spent my whole weekend on {tag} no regrets",
    "the {tag} community never disappoints",
    "okay but {tag} deserves way more credit",
    "{tag} content hits different at 2am",
    "my friends keep sending me {tag} posts",
    "thread: why {tag} keeps coming back around",
    "tried {tag} for the first time and wow",
    "unpopular opinion but {tag} is overrated",
    "{tag} vibes only today",
    "anyone else remember when {tag} first blew up?",
]
FILLERS = ["lol", "omg", "wow", "so good", "can't even", "no cap", "100%", "must watch", "link in bio", "save this",
           "thoughts?", "big mood", "love this", "so true", "wait for it"]
EMOJIS = ["", "", "", " :)", " <3", " !!!", " ...", " ??", " lol"]


# --------------------------------------------------------------------------- #
# Intensity curves
# --------------------------------------------------------------------------- #
def build_curve(profile: dict) -> np.ndarray:
    m = np.arange(N, dtype=float)
    shape = profile["shape"]
    if shape in ("comeback", "modest"):
        early = np.exp(-((m - profile["peak_month"]) ** 2) / (2 * profile["width"] ** 2))
        late = profile["resurgence"] / (1 + np.exp(-(m - profile["ramp"]) / 1.1))
        return early + profile["base"] + late
    if shape == "growth":
        return 0.03 + np.exp(profile["rate"] * (m - (N - 1)))
    if shape == "decline":
        return np.exp(-profile["rate"] * m) + 0.02
    if shape == "flat":
        return 1.0 + 0.08 * np.sin(2 * np.pi * (m + profile.get("phase", 0)) / 12)
    raise ValueError(f"unknown shape {shape}")


def _synthetic_names(rng: random.Random) -> list[str]:
    combos = [f"#{a}{n}" for a in ADJECTIVES for n in NOUNS]
    rng.shuffle(combos)
    return combos[:N_SYNTHETIC]


def build_profiles(seed: int = config.SEED) -> dict[str, dict]:
    """Return {topic_display_tag: profile} for all 1,244 non-generic topics."""
    rng = random.Random(seed)
    profiles: dict[str, dict] = {k: dict(v) for k, v in NAMED_PROFILES.items()}
    names = _synthetic_names(rng)
    r_ranges = {"High": (1.30, 1.80), "Medium": (0.92, 1.05), "Low": (0.62, 0.80)}
    idx = 0
    for tier, count in SYN_COMEBACK.items():
        for _ in range(count):
            lo, hi = r_ranges[tier]
            profiles[names[idx]] = dict(
                shape="comeback", total=SYN_TOTALS["comeback"], peak_month=rng.randint(2, 7),
                width=rng.uniform(1.8, 2.8), base=rng.uniform(0.04, 0.08),
                resurgence=rng.uniform(lo, hi), ramp=rng.uniform(30.5, 32.0), expected=tier)
            idx += 1
    for _ in range(SYN_GROWTH):
        profiles[names[idx]] = dict(shape="growth", total=SYN_TOTALS["growth"], rate=rng.uniform(0.11, 0.15), expected=None)
        idx += 1
    for _ in range(SYN_DECLINE):
        profiles[names[idx]] = dict(shape="decline", total=SYN_TOTALS["decline"], rate=rng.uniform(0.11, 0.15), expected=None)
        idx += 1
    flat_names = names[idx:]
    used = sum(p["total"] for p in profiles.values())
    flat_total = config.TARGET_RECORDS - used
    base, extra = divmod(flat_total, len(flat_names))
    for i, name in enumerate(flat_names):
        profiles[name] = dict(shape="flat", total=base + (1 if i < extra else 0), phase=rng.randint(0, 11), expected=None)
    assert sum(p["total"] for p in profiles.values()) == config.TARGET_RECORDS
    return profiles


# --------------------------------------------------------------------------- #
# Text / record synthesis
# --------------------------------------------------------------------------- #
def _noisy_text(rng: random.Random, tags: list[str]) -> str:
    text = rng.choice(TEMPLATES).format(tag=tags[0])
    for extra in tags[1:]:
        text += f" {extra}"
    text += " " + rng.choice(FILLERS) + rng.choice(EMOJIS)
    roll = rng.random()
    if roll < 0.12:
        text += f" https://t.co/{rng.randrange(16**6):06x}"
    elif roll < 0.22:
        text = f"@user{rng.randint(1, 9999)} " + text
    if rng.random() < 0.06:
        text = text.upper()
    if rng.random() < 0.05:
        text = "  " + text.replace(" ", "   ", 1)
    return text


def _month_start(i: int) -> datetime:
    return datetime(config.START_YEAR + i // 12, i % 12 + 1, 1)


def generate_dataset(path: Path | str = config.RAW_DATASET_PATH, seed: int = config.SEED) -> dict:
    """Write the dataset and return a summary dictionary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profiles = build_profiles(seed)
    np_rng = np.random.default_rng(seed)
    rng = random.Random(seed)

    topics = list(profiles)
    ts_chunks, topic_chunks = [], []
    month_epochs = [calendar.timegm(_month_start(i).timetuple()) for i in range(N)]
    month_secs = [
        int((_month_start(i + 1) - _month_start(i)).total_seconds()) if i < N - 1
        else 31 * 86400 for i in range(N)
    ]
    for t_idx, topic in enumerate(topics):
        curve = build_curve(profiles[topic])
        counts = np_rng.multinomial(profiles[topic]["total"], curve / curve.sum())
        for m, n in enumerate(counts):
            if n:
                ts_chunks.append(month_epochs[m] + np_rng.integers(0, month_secs[m], int(n)))
                topic_chunks.append(np.full(int(n), t_idx, dtype=np.int32))
    all_ts = np.concatenate(ts_chunks)
    all_topic = np.concatenate(topic_chunks)
    order = np.argsort(all_ts, kind="stable")
    all_ts, all_topic = all_ts[order], all_topic[order]
    assert len(all_ts) == config.TARGET_RECORDS

    with path.open("w", encoding="utf-8") as fh:
        for i, (ts, t_idx) in enumerate(zip(all_ts.tolist(), all_topic.tolist()), start=1):
            primary = topics[t_idx]
            tags = [primary] + [g for g in GENERIC_TAGS if rng.random() < 0.12]
            text_tags = [t.upper() if (j == 0 and rng.random() < 0.05) else t for j, t in enumerate(tags)]
            record = {
                "post_id": f"POST-{i:07d}",
                "user_id": f"U{rng.randint(1, 48000):05d}",
                "platform": rng.choice(PLATFORMS),
                "text": _noisy_text(rng, text_tags),
                "hashtags": tags,
                "timestamp": (EPOCH + timedelta(seconds=ts)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "likes": int(rng.lognormvariate(3.2, 1.3)),
                "shares": int(rng.lognormvariate(1.8, 1.2)),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "path": str(path),
        "records": len(all_ts),
        "unique_topics": len(topics) + len(GENERIC_TAGS),
        "start": "2022-01",
        "end": "2024-12",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Trend Revival dataset")
    parser.add_argument("--out", default=str(config.RAW_DATASET_PATH))
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    summary = generate_dataset(args.out, args.seed)
    print(f"[dataset] wrote {summary['records']:,} posts / {summary['unique_topics']:,} topics -> {summary['path']}")


if __name__ == "__main__":
    main()
