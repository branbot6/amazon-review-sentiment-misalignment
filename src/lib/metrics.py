"""Product-level reputation metrics."""

from __future__ import annotations

import math
from typing import Sequence

__all__ = ["reputation_risk", "sales_proxy"]


def reputation_risk(
    sentiments: Sequence[float],
    *,
    max_reviews: int,
    min_reviews: int = 5,
    volume_weight: float = 0.5,
) -> float:
    """Reputation risk for one product, on a 0-1 scale. Higher is worse.

    The score combines two severity signals with an exposure term:

    * **Negative share** -- the fraction of reviews carrying negative
      sentiment. Complaint volume relative to total feedback.
    * **Sentiment penalty** -- mean compound sentiment mapped from [-1, 1] to
      [1, 0], so the *direction* of sentiment drives the score, not just its
      magnitude. Two products with equal complaint counts but opposite overall
      tone score differently.
    * **Exposure** -- log-scaled review volume. Five hundred angry reviews
      represent more reputational damage than five, but log scaling keeps
      popularity from swamping severity. ``volume_weight=0`` removes the term.

    Products below ``min_reviews`` return ``nan`` rather than being scored on
    thin evidence; a product with two reviews carries no reliable signal and
    should not enter a regression next to one with four hundred.

    Args:
        sentiments: Per-review compound sentiment scores in [-1, 1].
        max_reviews: Review count of the busiest product in the corpus, used to
            normalise exposure. Pass the corpus-wide maximum so that scores are
            comparable across products.
        min_reviews: Evidence floor. Products below it return ``nan``.
        volume_weight: How strongly exposure modulates severity, in [0, 1].

    Returns:
        Risk in [0, 1], or ``nan`` when there is too little evidence.
    """
    total = len(sentiments)
    if total < min_reviews:
        return float("nan")

    negative_share = sum(1 for s in sentiments if s < 0) / total
    mean_sentiment = sum(sentiments) / total
    sentiment_penalty = (1.0 - mean_sentiment) / 2.0

    severity = 0.5 * negative_share + 0.5 * sentiment_penalty

    if max_reviews <= 1 or volume_weight <= 0:
        return severity

    exposure = math.log1p(total) / math.log1p(max_reviews)
    return severity * ((1.0 - volume_weight) + volume_weight * exposure)


def sales_proxy(review_count: int) -> float:
    """Log review count, used as a stand-in for unit sales.

    Sales figures are not published in the Amazon Reviews dataset, so review
    volume serves as a proxy. It is a popularity signal rather than a sales
    figure, and it also feeds the exposure term of :func:`reputation_risk`, so
    the two product-level targets are not independent. Models trained on both
    should report the coupling; :mod:`rq3_reputation` does.
    """
    return math.log1p(review_count)
