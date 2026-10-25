"""
comeback_detector.py - three-stage time-series comeback recognition.

Stage 1  Early Peak window   : first `early_months` months -> Historical Peak
Stage 2  Dormancy window     : after the peak until the resurgence window starts;
                               needs a continuous valley below 25% of the peak
                               lasting >= min_quiet_months
Stage 3  Resurgence window   : final months; the recent average (last 3 months)
                               must recover >= 50% of the peak  (Spike Velocity)

Metrics
  historical_peak : max monthly count in stage 1
  dormancy_coeff  : mean valley level / peak (lower = deeper dormancy)
  spike_velocity  : recent average / peak
Priority tiers
  High   : spike_velocity >= 1.00 and quiet >= 9 months
  Medium : spike_velocity >= 0.75
  Low    : spike_velocity >= 0.50
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import config


@dataclass(frozen=True)
class Params:
    early_months: int = config.DETECTOR["early_months"]
    resurgence_months: int = config.DETECTOR["resurgence_months"]
    recent_avg_months: int = config.DETECTOR["recent_avg_months"]
    dormancy_ratio: float = config.DETECTOR["dormancy_ratio"]
    recovery_ratio: float = config.DETECTOR["recovery_ratio"]
    min_quiet_months: int = config.DETECTOR["min_quiet_months"]
    min_peak: int = config.DETECTOR["min_peak"]
    high_ratio: float = config.DETECTOR["high_ratio"]
    high_quiet: int = config.DETECTOR["high_quiet"]
    medium_ratio: float = config.DETECTOR["medium_ratio"]


DEFAULT_PARAMS = Params()


def longest_run_below(values, threshold: float, offset: int = 0) -> tuple[int, int | None, int | None]:
    """Longest continuous run of values strictly below threshold -> (length, start_idx, end_idx)."""
    best = (0, None, None)
    start = None
    for i, v in enumerate(list(values) + [float("inf")]):
        if v < threshold:
            if start is None:
                start = i
        elif start is not None:
            length = i - start
            if length > best[0]:
                best = (length, start + offset, i - 1 + offset)
            start = None
    return best


def priority_for(spike: float, quiet: int, params: Params) -> str:
    if spike >= params.high_ratio and quiet >= params.high_quiet:
        return "High"
    if spike >= params.medium_ratio:
        return "Medium"
    return "Low"


def analyze_series(topic: str, counts, params: Params = DEFAULT_PARAMS, months=config.MONTHS) -> dict:
    counts = [int(c) for c in counts]
    n = len(counts)
    early = counts[: min(params.early_months, n)]
    peak = max(early) if early else 0
    peak_idx = early.index(peak) if early else 0
    res_start = n - params.resurgence_months

    # Stage 2 - dormancy valley
    threshold = params.dormancy_ratio * peak
    quiet_len, q_start, q_end = longest_run_below(counts[peak_idx + 1: res_start], threshold, offset=peak_idx + 1)
    valley = counts[q_start: q_end + 1] if quiet_len else []
    dormancy_coeff = (sum(valley) / len(valley) / peak) if valley and peak else 1.0

    # Stage 3 - resurgence
    recent_vals = counts[-params.recent_avg_months:]
    recent = sum(recent_vals) / len(recent_vals) if recent_vals else 0.0
    spike = recent / peak if peak else 0.0

    stage1 = peak >= params.min_peak
    stage2 = quiet_len >= params.min_quiet_months and peak > 0
    stage3 = spike >= params.recovery_ratio
    is_comeback = bool(stage1 and stage2 and stage3)

    if not stage1:
        reason = f"Peak below noise floor ({peak} < {params.min_peak})"
    elif not stage2:
        reason = f"No continuous valley below {params.dormancy_ratio:.0%} of peak for {params.min_quiet_months}+ months"
    elif not stage3:
        reason = f"Recovery {spike:.0%} below required {params.recovery_ratio:.0%}"
    else:
        reason = f"Recovered to {spike:.0%} of peak after {quiet_len} quiet months"

    priority = priority_for(spike, quiet_len, params) if is_comeback else "None"
    score = 0.0
    if is_comeback:
        depth = 1 - min(dormancy_coeff / params.dormancy_ratio, 1.0)
        score = 100 * (0.5 * min(spike / 1.5, 1.0) + 0.3 * min(quiet_len / 18, 1.0) + 0.2 * depth)

    return {
        "topic": topic,
        "historical_peak": int(peak),
        "peak_period": months[peak_idx] if early else None,
        "quiet_months": int(quiet_len),
        "quiet_start": months[q_start] if quiet_len else None,
        "quiet_end": months[q_end] if quiet_len else None,
        "dormancy_coeff": round(dormancy_coeff, 4),
        "recent_count": round(recent, 1),
        "spike_velocity": round(spike, 4),
        "stage_early_peak": bool(stage1),
        "stage_dormancy": bool(stage2),
        "stage_resurgence": bool(stage3),
        "is_comeback": is_comeback,
        "priority": priority,
        "score": round(score, 1),
        "reason": reason,
    }


def detect_all(series: dict[str, list[int]], params: Params = DEFAULT_PARAMS) -> list[dict]:
    results = [analyze_series(t, c, params) for t, c in series.items()]
    results.sort(key=lambda r: (not r["is_comeback"], -r["score"], -r["spike_velocity"]))
    return results


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_results(results: list[dict]) -> None:
    config.COMEBACK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.COMEBACK_RESULTS_PATH.write_text(json.dumps(results), encoding="utf-8")
    db = config.get_db()
    if db is not None:
        col = db[config.COLLECTIONS["comeback_results"]]
        col.drop()
        col.insert_many([dict(r) for r in results], ordered=False)
        col.create_index("topic", unique=True)
        col.create_index("is_comeback")
        col.create_index("priority")


def load_results() -> list[dict]:
    db = config.get_db()
    if db is not None:
        col = db[config.COLLECTIONS["comeback_results"]]
        if col.estimated_document_count() > 0:
            docs = list(col.find({}, {"_id": 0}))
            docs.sort(key=lambda r: (not r["is_comeback"], -r["score"], -r["spike_velocity"]))
            return docs
    if config.COMEBACK_RESULTS_PATH.exists():
        return json.loads(config.COMEBACK_RESULTS_PATH.read_text(encoding="utf-8"))
    return []


def results_available() -> bool:
    return bool(load_results())


def run(verbose: bool = True) -> list[dict]:
    import mapreduce_engine

    series = mapreduce_engine.load_period_counts()
    results = detect_all(series)
    save_results(results)
    if verbose:
        cand = [r for r in results if r["is_comeback"]]
        tiers = {t: sum(1 for r in cand if r["priority"] == t) for t in ("High", "Medium", "Low")}
        print(f"[detector] {len(results):,} topics evaluated -> {len(cand)} comeback candidates {tiers}")
        for r in cand[:5]:
            print(f"   {config.display_name(r['topic']):<22} {r['priority']:<6} spike={r['spike_velocity']:.2f} quiet={r['quiet_months']}m")
    return results


def main() -> None:
    argparse.ArgumentParser(description="Detect trend comebacks").parse_args()
    run()


if __name__ == "__main__":
    main()
