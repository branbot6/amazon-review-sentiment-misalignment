#!/usr/bin/env python3
"""RQ2 -- which review characteristics predict sentiment-rating misalignment.

TF-IDF vectorises the review bodies, K-Means groups them into semantic
clusters, and each cluster is profiled for misalignment rate, mean rating gap,
review length and verification share. The widest-gap cluster is then broken
down by review length crossed with verification status.

Cluster ids are not stable across refits, so the cluster of interest is
selected by its properties rather than a hard-coded index, and the breakdown
is printed sorted by rate with the leading group named explicitly.

Usage:
    python src/rq2_topics.py --category Clothing_Shoes_and_Jewelry
    python src/rq2_topics.py --category Electronics --sample 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    LONG_REVIEW_WORDS,
    MISALIGNMENT_THRESHOLD,
    RANDOM_SEED,
    RESULTS_DIR,
    category_path,
)
from lib.text import for_bagofwords, word_count  # noqa: E402

N_CLUSTERS = 5
MAX_FEATURES = 1000


def load(category: str, sample: int | None) -> pd.DataFrame:
    path = category_path(category, with_text=True)
    if not path.exists():
        raise SystemExit(f"Missing {path}. See data/README.md.")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["review_text"] = (
        frame["title"].fillna("") + " " + frame["text"].fillna("")
    ).map(for_bagofwords)
    frame = frame[frame["review_text"].str.len() > 20].copy()

    if sample and len(frame) > sample:
        frame = frame.sample(n=sample, random_state=RANDOM_SEED).copy()

    frame["word_count"] = frame["review_text"].map(word_count)
    frame["rating_gap"] = frame["rating"] - frame["predict_stars"]
    frame["misaligned"] = frame["rating_gap"].abs() >= MISALIGNMENT_THRESHOLD
    frame["length"] = frame["word_count"].ge(LONG_REVIEW_WORDS).map(
        {True: "Long", False: "Short"}
    )
    frame["verification"] = frame["verified_purchase"].astype(int).map(
        {1: "Verified", 0: "Unverified"}
    )
    return frame


def cluster(frame: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer, KMeans]:
    vectoriser = TfidfVectorizer(
        stop_words="english", max_df=0.95, min_df=10, max_features=MAX_FEATURES
    )
    matrix = vectoriser.fit_transform(frame["review_text"])
    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    frame = frame.assign(topic=model.fit_predict(matrix))
    return frame, vectoriser, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Clothing_Shoes_and_Jewelry")
    parser.add_argument("--sample", type=int, default=None,
                        help="Row cap. Omit to use the full category.")
    args = parser.parse_args()

    frame = load(args.category, args.sample)
    frame, vectoriser, model = cluster(frame)

    print(f"{args.category}: {len(frame):,} reviews, k={N_CLUSTERS}\n")

    terms = vectoriser.get_feature_names_out()
    centroids = model.cluster_centers_.argsort()[:, ::-1]

    per_topic = (
        frame.groupby("topic")
        .agg(reviews=("misaligned", "size"),
             misaligned_pct=("misaligned", lambda s: 100 * s.mean()),
             mean_gap=("rating_gap", "mean"),
             mean_words=("word_count", "mean"),
             verified_pct=("verified_purchase", lambda s: 100 * s.mean()))
        .sort_values("mean_gap", key=abs, ascending=False)
    )

    print("Clusters, ordered by |mean rating gap|:")
    for topic, row in per_topic.iterrows():
        words = ", ".join(terms[i] for i in centroids[topic, :8])
        print(f"  topic {topic}  n={row.reviews:>7,.0f}  "
              f"misaligned={row.misaligned_pct:5.2f}%  gap={row.mean_gap:+.3f}  "
              f"words={row.mean_words:5.1f}  verified={row.verified_pct:5.1f}%")
        print(f"           {words}")

    # Selected by widest mean gap rather than by index: K-Means numbers its
    # clusters arbitrarily, so the same semantic group takes a different id
    # whenever the model is refit or the input changes.
    focus = int(per_topic.index[0])
    subset = frame[frame["topic"] == focus]
    print(f"\nWidest-gap cluster: topic {focus} ({len(subset):,} reviews, "
          f"{100 * subset['verified_purchase'].mean():.2f}% verified)\n")

    breakdown = (
        subset.groupby(["length", "verification"])
        .agg(reviews=("misaligned", "size"),
             misaligned_pct=("misaligned", lambda s: 100 * s.mean()))
        .reset_index()
        .sort_values("misaligned_pct", ascending=False)
    )

    print("Misalignment rate by length x verification, highest first:")
    for _, row in breakdown.iterrows():
        print(f"  {row.length:<5} + {row.verification:<10} "
              f"{row.misaligned_pct:5.2f}%   n={row.reviews:>6,}")

    top = breakdown.iloc[0]
    print(f"\n  -> highest: {top.length} + {top.verification} at {top.misaligned_pct:.2f}%")
    print("     Read this table at the review level. A cluster's overall")
    print("     verification share says nothing about which reviews inside it")
    print("     are misaligned; run src/corpus_stats.py for the corpus-wide split.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"rq2_{args.category}.json"
    out.write_text(json.dumps({
        "category": args.category,
        "reviews": len(frame),
        "k": N_CLUSTERS,
        "focus_topic": focus,
        "per_topic": per_topic.reset_index().to_dict("records"),
        "breakdown": breakdown.to_dict("records"),
        "highest_group": f"{top.length} + {top.verification}",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
