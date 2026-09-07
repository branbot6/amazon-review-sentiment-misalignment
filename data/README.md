# Data

The corpus is **not committed**. The three scored CSVs total roughly 600 MB and
the largest single file is 196 MB, past GitHub's 100 MB per-file limit.

## Source

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab,
UCSD). Three category files:

| Category | Reviews used |
|---|---|
| Electronics | 500,000 |
| Clothing, Shoes and Jewelry | 500,000 |
| Health and Personal Care | 494,121 |

Reviews are the first *n* records of each category's `review` JSONL.

## Expected files

Place these in `data/`, or point `REVIEW_DATA_DIR` elsewhere.

| File | Columns |
|---|---|
| `Electronics.csv` | the nine base columns below |
| `Clothing_Shoes_and_Jewelry.csv` | " |
| `Health_and_Personal_Care.csv` | " |
| `*_with_text.csv` | base columns **plus `title`, `text`** |

Base columns:

```
rating              1-5, as clicked by the reviewer (stored as "5.0")
text_word_length    whitespace token count of the review body
helpful_vote        helpful votes received
verified_purchase   0 / 1
predict_stars       1-5 model reading of the review text  <- generated, see below
asin                product id
parent_asin         product-variant group id
user_id             reviewer id
timestamp           epoch milliseconds
```

`corpus_stats.py` needs only the base files. `rq2_topics.py` and
`rq3_reputation.py` need the `_with_text` variants, since they re-vectorise the
review bodies.

## Generating `predict_stars`

`predict_stars` is not part of the source dataset. Each review body is scored
with

```
nlptown/bert-base-multilingual-uncased-sentiment
```

which emits a 1-5 star label. Reviews are split into sentences with NLTK,
scored in batches, and averaged to one value per review. On a CUDA GPU this
takes a few hours for 1.5M reviews; it is the one expensive step in the
project, and everything downstream runs on a laptop.

> `predict_stars` is a model output, not a label. Its accuracy on this corpus
> is unmeasured, and a general-purpose sentiment classifier is routinely a star
> out. Treat `|rating - predict_stars| >= 2` as a screening threshold for cases
> worth examining, not as evidence a review is dishonest.
