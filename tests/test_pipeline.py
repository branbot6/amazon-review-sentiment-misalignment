"""Unit tests for the metrics and text-cleaning helpers.

Neither the dataset nor any third-party package is required:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lib.metrics import reputation_risk, sales_proxy  # noqa: E402
from lib.text import for_bagofwords, for_vader, word_count  # noqa: E402

# Four synthetic products spanning the reputation range.
FLAWLESS = [0.9] * 10                    # no negative reviews
HEALTHY = [0.8] * 9 + [-0.5]             # one negative
TROUBLED = [-0.8] * 9 + [0.5]            # nine negative
DISASTER = [-0.9] * 10                   # all negative

MAX_REVIEWS = 10


def risk(sentiments, **kwargs):
    kwargs.setdefault("max_reviews", MAX_REVIEWS)
    kwargs.setdefault("min_reviews", 5)
    return reputation_risk(sentiments, **kwargs)


class ReputationRisk(unittest.TestCase):
    def test_rises_as_reputation_falls(self):
        self.assertLess(risk(FLAWLESS), risk(HEALTHY))
        self.assertLess(risk(HEALTHY), risk(TROUBLED))
        self.assertLess(risk(TROUBLED), risk(DISASTER))

    def test_bounded_between_zero_and_one(self):
        for product in (FLAWLESS, HEALTHY, TROUBLED, DISASTER):
            self.assertGreaterEqual(risk(product), 0.0)
            self.assertLessEqual(risk(product), 1.0)

    def test_sentiment_direction_separates_equal_complaint_counts(self):
        """Both products have 10 reviews and 5 negative ones. Only the
        direction of overall sentiment distinguishes them."""
        mildly_positive = [0.9] * 5 + [-0.1] * 5   # mean +0.4
        mildly_negative = [-0.9] * 5 + [0.1] * 5   # mean -0.4
        self.assertLess(risk(mildly_positive), risk(mildly_negative))

    def test_volume_breaks_ties(self):
        self.assertLess(
            reputation_risk([-0.8] * 5, max_reviews=500),
            reputation_risk([-0.8] * 500, max_reviews=500),
        )

    def test_volume_does_not_outweigh_severity(self):
        """A busy but well-liked product still ranks below a quiet angry one."""
        self.assertLess(
            reputation_risk([0.9] * 500, max_reviews=500),
            reputation_risk([-0.8] * 5, max_reviews=500),
        )

    def test_volume_weight_zero_disables_exposure(self):
        few = reputation_risk([-0.8] * 5, max_reviews=500, volume_weight=0)
        many = reputation_risk([-0.8] * 500, max_reviews=500, volume_weight=0)
        self.assertAlmostEqual(few, many, places=9)

    def test_thin_evidence_returns_nan(self):
        self.assertTrue(math.isnan(reputation_risk([-0.9] * 3, max_reviews=500)))

    def test_sales_proxy_is_monotone(self):
        self.assertLess(sales_proxy(1), sales_proxy(10))
        self.assertLess(sales_proxy(10), sales_proxy(1000))
        self.assertEqual(sales_proxy(0), 0.0)


class AbbreviationExpansion(unittest.TestCase):
    def test_does_not_match_inside_words(self):
        self.assertEqual(for_bagofwords("I bought a lollipop"), "i bought a lollipop")
        self.assertEqual(for_bagofwords("Lolita dress"), "lolita dress")
        self.assertEqual(for_bagofwords("my brbq grill"), "my brbq grill")

    def test_expands_standalone_tokens(self):
        self.assertEqual(
            for_bagofwords("lol what a product"),
            "laughing out loud what a product",
        )

    def test_is_case_insensitive(self):
        self.assertEqual(for_vader("LOL"), "laughing out loud")


class TextCleaning(unittest.TestCase):
    def test_vader_variant_preserves_intensity_markers(self):
        self.assertEqual(for_vader("This is AMAZING!!!"), "This is AMAZING!!!")

    def test_bagofwords_variant_strips_them(self):
        self.assertEqual(for_bagofwords("This is AMAZING!!!"), "this is amazing")

    def test_markup_and_entities_are_resolved(self):
        self.assertEqual(for_vader("<p>Great &amp; cheap</p>"), "Great & cheap")
        self.assertEqual(for_bagofwords("<p>Great &amp; cheap</p>"), "great cheap")

    def test_whitespace_is_normalised(self):
        self.assertEqual(for_vader("  too   many\n\nspaces "), "too many spaces")

    def test_non_string_input_is_tolerated(self):
        for value in (None, float("nan"), 42):
            self.assertEqual(for_vader(value), "")
            self.assertEqual(for_bagofwords(value), "")
            self.assertEqual(word_count(value), 0)

    def test_word_count(self):
        self.assertEqual(word_count("fits true to size"), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
