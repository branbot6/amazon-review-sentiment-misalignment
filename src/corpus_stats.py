#!/usr/bin/env python3
"""Corpus-wide misalignment statistics.

Streams every review in all three categories and reports how far apart star
ratings and text sentiment run. Standard library only -- no pandas -- so the
headline figures reproduce on a bare Python install and 1.5M rows never land
in memory at once.

Writes ``results/corpus_stats.json``.

Usage:
    python src/corpus_stats.py
    REVIEW_DATA_DIR=/path/to/csvs python src/corpus_stats.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CATEGORIES, MISALIGNMENT_THRESHOLD, RESULTS_DIR, category_path  # noqa: E402


def _as_int(value: str) -> int:
    """Ratings are stored as '5.0'; int() rejects that directly."""
    return int(float(value))


def analyse() -> dict:
    gap_hist: Counter[int] = Counter()
    rating_hist: Counter[int] = Counter()
    predicted_hist: Counter[int] = Counter()
    groups: Counter[str] = Counter()
    verified = {0: [0, 0], 1: [0, 0]}  # flag -> [misaligned, total]
    per_category: dict[str, dict] = {}
    total = 0
    misaligned = 0

    for category in CATEGORIES:
        path = category_path(category)
        if not path.exists():
            raise SystemExit(
                f"Missing {path}.\nSee data/README.md for how to obtain and score the corpus."
            )

        rows = 0
        cat_misaligned = 0
        cat_gap: Counter[int] = Counter()

        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            for row in csv.DictReader(handle):
                try:
                    rating = _as_int(row["rating"])
                    predicted = _as_int(row["predict_stars"])
                except (KeyError, TypeError, ValueError):
                    continue

                gap = rating - predicted
                rows += 1
                total += 1
                cat_gap[gap] += 1
                gap_hist[gap] += 1
                rating_hist[rating] += 1
                predicted_hist[predicted] += 1

                severe = abs(gap) >= MISALIGNMENT_THRESHOLD
                if severe:
                    cat_misaligned += 1
                    misaligned += 1

                groups["A_rating_above_text" if gap > 0
                       else "B_rating_below_text" if gap < 0
                       else "C_aligned"] += 1

                try:
                    flag = _as_int(row["verified_purchase"])
                except (KeyError, TypeError, ValueError):
                    continue
                if flag in verified:
                    verified[flag][1] += 1
                    if severe:
                        verified[flag][0] += 1

        per_category[category] = {
            "reviews": rows,
            "misaligned": cat_misaligned,
            "misaligned_pct": round(100 * cat_misaligned / rows, 4),
            "mean_gap": round(sum(k * v for k, v in cat_gap.items()) / rows, 4),
        }

    return {
        "reviews": total,
        "misaligned": misaligned,
        "misaligned_pct": round(100 * misaligned / total, 4),
        "mean_gap": round(sum(k * v for k, v in gap_hist.items()) / total, 4),
        "threshold": MISALIGNMENT_THRESHOLD,
        "gap_histogram": dict(sorted(gap_hist.items())),
        "rating_histogram": dict(sorted(rating_hist.items())),
        "predicted_histogram": dict(sorted(predicted_hist.items())),
        "groups": dict(groups),
        "verified": {
            "verified": {"reviews": verified[1][1], "misaligned": verified[1][0],
                         "misaligned_pct": round(100 * verified[1][0] / verified[1][1], 4)},
            "unverified": {"reviews": verified[0][1], "misaligned": verified[0][0],
                           "misaligned_pct": round(100 * verified[0][0] / verified[0][1], 4)},
        },
        "per_category": per_category,
    }


def main() -> None:
    stats = analyse()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "corpus_stats.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    total = stats["reviews"]
    print(f"reviews              {total:>12,}")
    print(f"severely misaligned  {stats['misaligned']:>12,}  ({stats['misaligned_pct']:.2f}%)")
    print(f"mean rating_gap      {stats['mean_gap']:>+12.4f}")
    print()
    print("rating_gap = rating - predict_stars")
    for gap, count in stats["gap_histogram"].items():
        print(f"  {gap:+d}  {count:>10,}  {100 * count / total:>6.2f}%")
    print()
    v = stats["verified"]
    print(f"verified    {v['verified']['misaligned_pct']:.2f}%  (n={v['verified']['reviews']:,})")
    print(f"unverified  {v['unverified']['misaligned_pct']:.2f}%  (n={v['unverified']['reviews']:,})")
    print(f"difference  {abs(v['unverified']['misaligned_pct'] - v['verified']['misaligned_pct']):.2f}pp")
    print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
