"""Paths and shared constants.

All data locations resolve from ``DATA_DIR``, overridable with the
``REVIEW_DATA_DIR`` environment variable so the pipeline runs unchanged on a
laptop, a Colab mount or a GPU box.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("REVIEW_DATA_DIR", REPO_ROOT / "data"))
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"

# All three research questions run over the same three categories, so their
# results are directly comparable.
CATEGORIES = {
    "Electronics": "Electronics.csv",
    "Clothing_Shoes_and_Jewelry": "Clothing_Shoes_and_Jewelry.csv",
    "Health_and_Personal_Care": "Health_and_Personal_Care.csv",
}

# `predict_stars` is the model's 1-5 reading of the review text; `rating` is
# what the reviewer actually clicked.
REQUIRED_COLUMNS = [
    "rating",
    "text_word_length",
    "helpful_vote",
    "verified_purchase",
    "predict_stars",
    "asin",
    "parent_asin",
    "user_id",
    "timestamp",
]
TEXT_COLUMNS = ["title", "text"]

SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

# A gap of this many stars or more between the reviewer and the classifier is
# treated as severe misalignment. This is a screening threshold on a model
# output, not a judgement about the reviewer: a general-purpose sentiment
# classifier is routinely a star out, so a two-star gap flags a case worth
# examining rather than proving intent. See `data/README.md`.
MISALIGNMENT_THRESHOLD = 2

LONG_REVIEW_WORDS = 50  # long/short split used throughout RQ2
RANDOM_SEED = 42


def category_path(category: str, *, with_text: bool = False) -> Path:
    """Resolve a category CSV.

    ``with_text=True`` selects the variant carrying the raw ``title`` and
    ``text`` columns, needed wherever review bodies are re-vectorised.
    """
    name = CATEGORIES[category]
    if with_text:
        name = name.replace(".csv", "_with_text.csv")
    return DATA_DIR / name
