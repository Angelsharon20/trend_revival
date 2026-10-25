# Trend Revival: Social Media Trend Comeback Detection Using MongoDB MapReduce

A modular, end-to-end Python and MongoDB project that finds topics which peaked, went quiet, and came back.
It builds a 124,862-post dataset, runs a real server-side MongoDB MapReduce job (with a pure-Python fallback),
applies a three-window comeback algorithm, and serves everything in a custom dark analytics dashboard.

```
trend_revival/
├── config.py             # DB params, collection schemas, detector constants, UI theme
├── dataset_generator.py  # 124,862 posts, Jan 2022 - Dec 2024, 5 trend signatures
├── ingest_data.py        # validated, typed bulk insert into raw_posts (+ integrity report)
├── preprocess.py         # multi-core clean / tokenise / hashtag explode / indexes
├── mapreduce_engine.py   # server-side MongoDB mapReduce + Python Map->Shuffle->Reduce fallback
├── comeback_detector.py  # 3-stage window comeback engine, High / Medium / Low tiers
├── app.py                # Streamlit + Plotly dashboard (5 views)
└── test_system.py        # 24 pytest tests
```

## 1. Setup and run

Requires Python 3.10+ and, optionally, a MongoDB server (4.4 to 8.x) on `localhost:27017`.

```bash
# 1. environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install streamlit pymongo plotly pandas numpy pytest

# 2. tests (no MongoDB needed)
pytest -v

# 3. launch the dashboard (builds the whole pipeline automatically on first run)
streamlit run app.py
```

Run the pipeline step by step instead:

```bash
python dataset_generator.py              # writes data/social_posts.jsonl (124,862 rows)
python ingest_data.py                    # MongoDB raw_posts (add --dry-run to only validate)
python preprocess.py                     # clean_posts + topic_mentions + indexes
python mapreduce_engine.py               # MongoDB mapReduce -> topic_period_counts
python comeback_detector.py              # comeback_results
```

* **No MongoDB?** Everything still works. `preprocess.py` and `mapreduce_engine.py` detect the missing server and
  use the local engine (add `--force-python` to choose it explicitly). Set `TREND_REVIVAL_FORCE_LOCAL=1` to skip the connection check.
* **Custom server:** `export MONGO_URI="mongodb://user:pass@host:27017"`.
* MongoDB deprecated the `mapReduce` command in 5.0, but it is still available and is what the assignment requires.

## 2. How it works

| Stage | What happens |
|---|---|
| Generate | Each topic owns a 36-month intensity curve; monthly counts are multinomial draws, so totals are exact (124,862). |
| Ingest | Strict validation, ISO string to datetime casting, unique index on `post_id`, duplicates skipped. |
| Preprocess | Lowercase, strip URLs/mentions/punctuation, tokenise, unique hashtags, explode to `{post_id, topic, timestamp}`, index `{topic: 1, timestamp: 1}`. |
| MapReduce | Map emits `{topic, period: "YYYY-MM"} -> 1`; Reduce sums with a loop; output replaces `topic_period_counts`. |
| Detect | Stage 1 early peak (2022), stage 2 dormancy valley, stage 3 resurgence (last 3 months vs peak). |

Detector rules: the valley must stay **below 25% of the peak** for at least 6 consecutive months, and the final
3-month average must be **at least 50% of the peak**. Tiers: High (recovery 100% or more and 9+ quiet months),
Medium (75% or more), Low (50% or more).

## 3. Summary metrics for the submission report

Measured on the delivered seed (`SEED = 42`):

| Metric | Value |
|---|---|
| Posts | 124,862 (Jan 2022 - Dec 2024, 36 months) |
| Unique topics | 1,248 (1,237 synthetic + 7 benchmark + 4 generic tags) |
| Topic mentions (exploded) | 185,127 |
| MapReduce output keys (topic x month) | 18,733 |
| Comeback candidates | 27 (High 11, Medium 7, Low 9) |
| Benchmark `#RetroGaming` | High, recovery 134% of peak after 22 quiet months |
| Benchmark `#BookTok` | High, recovery 115% after 21 quiet months |
| Benchmark `#Y2K` / `#Cottagecore` | Low (modest resurgence, 69% / 55%) |
| Benchmark `#AI` / `#SustainableLiving` / `#MentalHealth` | Correctly rejected (growth / decline / flat) |
| Automated tests | 24 passing |

Detection precision on the 7 benchmark trends is 7/7. The 23 synthetic comeback topics were all recovered
(no false positives among the other 1,221 topics).

## 4. Presentation outline (10 slides)

1. **Title and motivation.** "Trend Revival": why brands and analysts care about trends that come back; one-sentence goal.
2. **Problem statement.** Existing dashboards show what is trending now, not what is returning; define "comeback" (peak, dormancy, resurgence).
3. **Dataset.** 124,862 posts, 36 months, 1,248 topics, five engineered trend signatures; show the lifecycle chart.
4. **System architecture.** Pipeline diagram: generate, ingest, preprocess, MapReduce, detect, visualise; MongoDB collections and indexes.
5. **MapReduce design.** Show the map and reduce JavaScript, the composite key `{topic, period}`, and why this scales (shuffle by key).
6. **Comeback algorithm.** Three windows, 25% dormancy line, 50% recovery line, the priority tiers and the composite score.
7. **Dashboard demo.** KPI cards, lifecycle chart with stage overlays, insights feed, ranked comeback table (screenshots).
8. **Results and validation.** 27 candidates, benchmark outcomes, confusion on growth / decline / flat controls, pytest evidence.
9. **Engineering quality.** Modular code, Mongo and Python engines producing identical counts, fault-tolerant ingest, 24 automated tests.
10. **Limitations and future work.** Synthetic data, fixed windows, deprecated `mapReduce` (migrate to aggregation `$merge`), real API ingestion, seasonality-aware detection, live streaming.

## 5. Notes

* The dashboard KPIs are computed from the data, not hard-coded, and match the targets above for the delivered seed.
* Generated files live in `data/` and can be deleted at any time; the app rebuilds them.
* Suggested `.gitignore`: `data/`, `.venv/`, `__pycache__/`, `.pytest_cache/`.
