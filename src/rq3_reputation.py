#!/usr/bin/env python3
"""RQ3 -- do misaligned reviews move product reputation and sales?

Scores every review with VADER, aggregates to product level, computes a
reputation risk index per product, and fits random forests for that index and
for a sales proxy.

Products need a minimum number of reviews to be scored, and the correlation
between the two targets is reported alongside the models, since both derive
in part from review volume.

Writes ``results/rq3_reputation.json``, the product-level table to
``results/rq3_products.csv``, and the two scatter figures.

Usage:
    python src/rq3_reputation.py
    python src/rq3_reputation.py --from-cache   # replot without rescoring
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CATEGORIES,
    FIGURES_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    category_path,
)
from lib.metrics import reputation_risk, sales_proxy  # noqa: E402
from lib.text import for_vader  # noqa: E402

MIN_REVIEWS_PER_PRODUCT = 5
FEATURES = [
    "rating",
    "sentiment",
    "text_word_length",
    "helpful_vote",
    "verified_purchase",
    "time_weight",
]


def load() -> pd.DataFrame:
    frames = []
    for category in CATEGORIES:
        path = category_path(category, with_text=True)
        if not path.exists():
            raise SystemExit(f"Missing {path}. See data/README.md.")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["category"] = category
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def score_sentiment(frame: pd.DataFrame) -> pd.DataFrame:
    analyser = SentimentIntensityAnalyzer()
    # for_vader keeps punctuation, casing and emoji, which carry the intensity.
    text = (frame["title"].fillna("") + ". " + frame["text"].fillna("")).map(for_vader)
    frame["sentiment"] = [analyser.polarity_scores(t)["compound"] for t in text]
    return frame


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame["timestamp"], unit="ms", errors="coerce")
    reference = timestamps.max()
    frame["time_weight"] = np.exp(-(reference - timestamps).dt.days / 365).fillna(0)
    frame["verified_purchase"] = frame["verified_purchase"].astype(int)

    grouped = frame.groupby("asin")
    products = grouped.agg({f: "mean" for f in FEATURES}).reset_index()
    products["review_count"] = grouped.size().reindex(products["asin"]).to_numpy()

    products = products[products["review_count"] >= MIN_REVIEWS_PER_PRODUCT].copy()

    max_reviews = int(products["review_count"].max())
    sentiment_by_asin = grouped["sentiment"].apply(list)
    products["reputation_risk"] = [
        reputation_risk(sentiment_by_asin[a], max_reviews=max_reviews,
                        min_reviews=MIN_REVIEWS_PER_PRODUCT)
        for a in products["asin"]
    ]
    products["sales_proxy"] = products["review_count"].map(sales_proxy)
    return products.dropna(subset=["reputation_risk"])


def fit(products: pd.DataFrame, target: str) -> dict:
    X = products[FEATURES]
    y = products[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    return {
        "target": target,
        "r2": round(float(r2_score(y_test, predicted)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, predicted))), 4),
        "importance": {
            f: round(float(i), 4)
            for f, i in sorted(zip(FEATURES, model.feature_importances_),
                               key=lambda kv: -kv[1])
        },
    }


def plot(products: pd.DataFrame) -> None:
    """Scatter the two product-level targets against mean rating."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panels = [
        ("reputation_risk", "Reputation risk", "rq3_rating_vs_reputation_risk.png",
         "Product rating vs. reputation risk"),
        ("sales_proxy", "Sales proxy (log review count)", "rq3_rating_vs_sales_proxy.png",
         "Product rating vs. sales proxy"),
    ]
    for column, ylabel, filename, title in panels:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
        points = ax.scatter(
            products["rating"], products[column],
            c=products["sentiment"], cmap="coolwarm_r",
            s=7, alpha=0.45, linewidths=0, vmin=-1, vmax=1,
        )
        ax.set_xlabel("Mean star rating")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.15, linewidth=0.6)
        ax.set_axisbelow(True)
        fig.colorbar(points, ax=ax, label="Mean compound sentiment")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / filename)
        plt.close(fig)
        print(f"wrote {FIGURES_DIR / filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-cache", action="store_true",
                        help="Reuse results/rq3_products.csv instead of rescoring.")
    args = parser.parse_args()

    cache = RESULTS_DIR / "rq3_products.csv"
    if args.from_cache:
        if not cache.exists():
            raise SystemExit(f"No cache at {cache}. Run without --from-cache first.")
        products = pd.read_csv(cache)
        review_total = int(products["review_count"].sum())
        print(f"cached: {len(products):,} products\n")
    else:
        frame = score_sentiment(load())
        products = aggregate(frame)
        review_total = len(frame)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        products.to_csv(cache, index=False)
        print(f"{review_total:,} reviews -> {len(products):,} products "
              f"(>= {MIN_REVIEWS_PER_PRODUCT} reviews each)\n")

    results = [fit(products, "sales_proxy"), fit(products, "reputation_risk")]
    for result in results:
        print(f"{result['target']}:  R2={result['r2']:.3f}  RMSE={result['rmse']:.3f}")
        for feature, importance in result["importance"].items():
            print(f"    {feature:<20} {importance:.4f}")
        print()

    # Both targets draw on review volume -- sales_proxy is log review count and
    # the risk index carries a volume-scaled exposure term -- so report how
    # coupled they actually are rather than assuming independence.
    coupling = products["reputation_risk"].corr(products["sales_proxy"])
    print(f"corr(reputation_risk, sales_proxy) = {coupling:+.3f}")

    # Mean sentiment is an input to the risk index, so a high R2 on that target
    # partly reflects the model recovering its own definition. Read the feature
    # ranking, not the R2.
    print("  Note: sentiment feeds the risk index; read its R2 as consistency,")
    print("  not predictive skill.")

    worst = products.nlargest(10, "reputation_risk")[
        ["asin", "review_count", "rating", "sentiment", "reputation_risk"]
    ]
    print("\nHighest reputation risk:")
    print(worst.to_string(index=False))

    plot(products)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "rq3_reputation.json"
    out.write_text(json.dumps({
        "reviews": review_total,
        "products": int(len(products)),
        "min_reviews_per_product": MIN_REVIEWS_PER_PRODUCT,
        "models": results,
        "target_correlation": round(float(coupling), 4),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
