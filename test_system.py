"""
test_system.py - pytest unit / integration tests for Trend Revival.

Run:  pytest -v
Tests are fully local: they never touch your MongoDB data (everything is written to
pytest temp folders and the pure-Python MapReduce engine is used).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime

import pytest

os.environ.setdefault("TREND_REVIVAL_FORCE_LOCAL", "1")

import comeback_detector as cd  # noqa: E402
import config  # noqa: E402
import dataset_generator  # noqa: E402
import ingest_data  # noqa: E402
import mapreduce_engine as mre  # noqa: E402
import preprocess  # noqa: E402


# --------------------------------------------------------------------------- #
# Shared fixtures (built once per test session)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw") / "posts.jsonl"
    summary = dataset_generator.generate_dataset(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return path, summary, records


@pytest.fixture(scope="session")
def processed(dataset, tmp_path_factory):
    path, _, _ = dataset
    out = tmp_path_factory.mktemp("processed")
    stats = preprocess.run_local(path, out, workers=2, verbose=False)
    return out, stats


@pytest.fixture(scope="session")
def series(processed):
    out, _ = processed
    counts = mre.run_mapreduce_python(preprocess.iter_local_mentions(out / "topic_mentions.jsonl"))
    rows = ((t, p, v) for (t, p), v in counts.items())
    return mre.rows_to_series(rows), counts


# --------------------------------------------------------------------------- #
# 1. Dataset
# --------------------------------------------------------------------------- #
def test_dataset_row_count(dataset):
    _, summary, records = dataset
    assert len(records) > 50_000
    assert len(records) == config.TARGET_RECORDS == 124_862 == summary["records"]


def test_dataset_schema_and_integrity(dataset):
    _, summary, records = dataset
    assert len({r["post_id"] for r in records}) == len(records)
    for r in records[:2000]:
        assert set(r) >= {"post_id", "user_id", "platform", "text", "hashtags", "timestamp", "likes", "shares"}
        assert isinstance(r["hashtags"], list) and r["hashtags"]
    stamps = [r["timestamp"] for r in records]
    assert stamps == sorted(stamps)
    assert stamps[0].startswith("2022-01") and stamps[-1].startswith("2024-12")
    topics = {t.lower() for r in records for t in r["hashtags"]}
    assert len(topics) == summary["unique_topics"] == 1248
    for tag in config.TOPIC_DISPLAY:
        assert tag in topics


def test_dataset_signatures(series):
    s, _ = series
    retro, ai, health, decline = s["#retrogaming"], s["#ai"], s["#mentalhealth"], s["#sustainableliving"]
    peak = max(retro[:12])
    assert sum(retro[14:29]) / 15 < 0.25 * peak           # deep valley
    assert sum(retro[-3:]) / 3 >= 0.5 * peak               # resurgence
    assert ai[-1] > 5 * max(ai[:12])                       # exponential growth
    assert all(b >= 0.5 * a for a, b in zip(ai[:-3], ai[3:]))   # no collapse anywhere
    mean = sum(health) / len(health)
    assert max(health) < 1.4 * mean and min(health) > 0.6 * mean   # flat baseline
    assert decline[-1] < 0.1 * decline[0]


# --------------------------------------------------------------------------- #
# 2. Ingestion validation
# --------------------------------------------------------------------------- #
def test_validate_record_type_casting():
    rec, why = ingest_data.validate_record({"post_id": "1", "user_id": "u", "platform": "X", "text": "hi", "hashtags": ["#a"],
                                            "timestamp": "2023-05-06T07:08:09Z", "likes": "12", "shares": 3.0})
    assert why is None and rec["timestamp"] == datetime(2023, 5, 6, 7, 8, 9) and rec["likes"] == 12 and rec["shares"] == 3


@pytest.mark.parametrize("bad,reason", [
    ({"post_id": "1", "user_id": "u", "platform": "X", "text": "t", "hashtags": ["#a"], "timestamp": "yesterday"}, "bad_timestamp"),
    ({"post_id": "1", "user_id": "u", "platform": "X", "text": "t", "hashtags": "#a", "timestamp": "2023-01-01T00:00:00Z"}, "bad_hashtags"),
    ({"user_id": "u", "platform": "X", "text": "t", "hashtags": [], "timestamp": "2023-01-01T00:00:00Z"}, "missing_post_id"),
])
def test_validate_record_rejects(bad, reason):
    assert ingest_data.validate_record(bad) == (None, reason)


def test_ingest_dry_run_report(dataset):
    path, _, _ = dataset
    report = ingest_data.ingest(path, dry_run=True, verbose=False)
    assert report["valid"] == 124_862 and report["invalid"] == {} and report["duplicates_in_file"] == 0


# --------------------------------------------------------------------------- #
# 3. Preprocessing / pipeline isolation
# --------------------------------------------------------------------------- #
def test_process_record_is_pure_and_clean():
    raw = {"post_id": "P1", "user_id": "u", "platform": "X", "timestamp": "2023-02-03T04:05:06Z", "likes": 1, "shares": 0,
           "text": "OMG!!! @bob Check https://t.co/abc123 #RetroGaming is BACK #retrogaming #Y2K :)", "hashtags": ["#RetroGaming", "#fyp"]}
    snapshot = json.dumps(raw, sort_keys=True)
    clean, mentions = preprocess.process_record(raw)
    assert json.dumps(raw, sort_keys=True) == snapshot            # input untouched
    assert clean["clean_text"] == clean["clean_text"].lower()
    assert not any(ch in clean["clean_text"] for ch in "!@#:)/")
    assert "http" not in clean["clean_text"] and "bob" not in clean["clean_text"]
    assert clean["hashtags"] == ["#fyp", "#retrogaming", "#y2k"]      # unique, lowercase, sorted
    assert [m["topic"] for m in mentions] == clean["hashtags"]
    assert all(set(m) == {"post_id", "topic", "timestamp"} for m in mentions)


def test_preprocess_pipeline_consistency(processed):
    out, stats = processed
    clean = list(preprocess.iter_local_clean_posts(out / "clean_posts.jsonl"))
    mentions = list(preprocess.iter_local_mentions(out / "topic_mentions.jsonl"))
    assert len(clean) == stats["total_posts"] == 124_862
    assert len(mentions) == sum(len(c["hashtags"]) for c in clean) == stats["total_mentions"]
    assert len({(m["post_id"], m["topic"]) for m in mentions}) == len(mentions)       # exploded uniquely
    assert stats["unique_topics"] == 1248


# --------------------------------------------------------------------------- #
# 4. MapReduce correctness
# --------------------------------------------------------------------------- #
def test_mapreduce_hand_computed():
    mentions = [{"topic": "#a", "timestamp": datetime(2022, 1, 5)}, {"topic": "#a", "timestamp": datetime(2022, 1, 28)},
                {"topic": "#a", "timestamp": datetime(2022, 2, 1)}, {"topic": "#b", "timestamp": datetime(2022, 1, 9)}]
    assert mre.run_mapreduce_python(mentions) == {("#a", "2022-01"): 2, ("#a", "2022-02"): 1, ("#b", "2022-01"): 1}


def test_mapreduce_matches_counter_and_conserves_total(processed, series):
    out, stats = processed
    _, counts = series
    expected = Counter((m["topic"], f"{m['timestamp'].year}-{m['timestamp'].month:02d}")
                       for m in preprocess.iter_local_mentions(out / "topic_mentions.jsonl"))
    assert counts == dict(expected)
    assert sum(counts.values()) == stats["total_mentions"]


def test_shuffle_reduce_units():
    groups = mre.py_shuffle([(("t", "2022-01"), 1), (("t", "2022-01"), 1), (("u", "2022-02"), 1)])
    assert groups[("t", "2022-01")] == [1, 1]
    assert mre.py_reduce(("t", "2022-01"), [1, 1, 5]) == 7


def test_js_blocks_mirror_python_logic():
    assert "emit(" in mre.MAP_JS and "this.topic" in mre.MAP_JS and "getUTCFullYear" in mre.MAP_JS
    assert "for (" in mre.REDUCE_JS and "total += values[i]" in mre.REDUCE_JS


def test_series_zero_filled(series):
    s, _ = series
    assert all(len(v) == 36 for v in s.values())
    assert len(s) == 1248


# --------------------------------------------------------------------------- #
# 5. Comeback mathematics
# --------------------------------------------------------------------------- #
def test_classic_comeback_is_detected_with_metrics():
    counts = [100, 90, 80, 50, 30, 20, 10, 10, 10, 10, 10, 10] + [8] * 18 + [30, 60, 120, 125, 130, 135]
    assert len(counts) == 36
    r = cd.analyze_series("#x", counts)
    assert r["historical_peak"] == 100 and r["peak_period"] == "2022-01"
    assert r["quiet_months"] == 25
    assert r["spike_velocity"] == pytest.approx(130 / 100)
    assert r["is_comeback"] and r["priority"] == "High"
    assert r["dormancy_coeff"] < 0.25


@pytest.mark.parametrize("name,counts", [
    ("growth", [int(5 * 1.1 ** i) for i in range(36)]),
    ("decline", [int(200 * 0.88 ** i) for i in range(36)]),
    ("flat", [100] * 36),
])
def test_non_comebacks_rejected(name, counts):
    r = cd.analyze_series(f"#{name}", counts)
    assert not r["is_comeback"] and r["priority"] == "None"


def test_stage_thresholds_and_tiers():
    base = [100] + [10] * 29 + [0] * 6
    def with_recent(v):
        return base[:33] + [v] * 3
    assert cd.analyze_series("#t", with_recent(49))["is_comeback"] is False      # 49% < 50%
    assert cd.analyze_series("#t", with_recent(50))["priority"] == "Low"          # boundary inclusive
    assert cd.analyze_series("#t", with_recent(80))["priority"] == "Medium"
    assert cd.analyze_series("#t", with_recent(100))["priority"] == "High"
    short_valley = [100] + [10] * 3 + [60] * 26 + [0] * 6
    assert cd.analyze_series("#t", short_valley[:33] + [100] * 3)["is_comeback"] is False
    assert cd.analyze_series("#noise", [5] + [0] * 29 + [5] * 6)["stage_early_peak"] is False


def test_valley_must_stay_below_25_percent():
    counts = [100] + [24] * 29 + [100] * 6
    assert cd.analyze_series("#a", counts)["stage_dormancy"] is True
    counts = [100] + [25] * 29 + [100] * 6          # exactly 25% is NOT below 25%
    assert cd.analyze_series("#b", counts)["stage_dormancy"] is False


def test_longest_run_helper():
    assert cd.longest_run_below([5, 1, 1, 9, 1, 1, 1, 8], 3) == (3, 4, 6)
    assert cd.longest_run_below([9, 9], 3) == (0, None, None)


def test_benchmark_topics_classified_as_designed(series):
    s, _ = series
    res = {r["topic"]: r for r in cd.detect_all(s)}
    assert res["#retrogaming"]["priority"] == "High" and res["#booktok"]["priority"] == "High"
    assert res["#y2k"]["is_comeback"] and res["#y2k"]["priority"] in ("Low", "Medium")
    assert res["#cottagecore"]["is_comeback"] and res["#cottagecore"]["priority"] in ("Low", "Medium")
    for tag in ("#ai", "#sustainableliving", "#mentalhealth"):
        assert not res[tag]["is_comeback"], tag
    assert sum(r["is_comeback"] for r in res.values()) == 27
    assert res["#ai"]["spike_velocity"] > 1 and not res["#ai"]["stage_dormancy"]
    assert res["#sustainableliving"]["stage_dormancy"] and not res["#sustainableliving"]["stage_resurgence"]


def test_results_sorted_candidates_first(series):
    s, _ = series
    results = cd.detect_all(s)
    flags = [r["is_comeback"] for r in results]
    assert flags == sorted(flags, reverse=True)
    cand_scores = [r["score"] for r in results if r["is_comeback"]]
    assert cand_scores == sorted(cand_scores, reverse=True)
