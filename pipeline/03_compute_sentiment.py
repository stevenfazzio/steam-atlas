"""Add a Steam-style review sentiment label to games.parquet.

Mirrors the labels Steam shows on store pages (Overwhelmingly Positive / Very Positive /
Mostly Positive / Mixed / Mostly Negative / Very Negative / Overwhelmingly Negative).
Bucketing approximates Valve's documented thresholds on positive ratio + sample size.

Saves us a separate appreviews crawl: the published review_score_desc is just a function
of (positive, negative), and FronkonGames already has both.
"""

import pandas as pd
from config import GAMES_PARQUET


def sentiment_label(positive: int, negative: int) -> str:
    total = positive + negative
    if total == 0:
        return "No User Reviews"
    if total < 10:
        return "Too Few Reviews"
    ratio = positive / total
    if ratio >= 0.95 and total >= 500:
        return "Overwhelmingly Positive"
    if ratio >= 0.80:
        return "Very Positive" if total >= 50 else "Positive"
    if ratio >= 0.70:
        return "Mostly Positive"
    if ratio >= 0.40:
        return "Mixed"
    if ratio >= 0.20:
        return "Mostly Negative" if total >= 50 else "Negative"
    if total >= 500:
        return "Overwhelmingly Negative"
    if total >= 50:
        return "Very Negative"
    return "Negative"


def main():
    df = pd.read_parquet(GAMES_PARQUET)
    print(f"Loaded {len(df):,} games")

    df["sentiment_label"] = df.apply(
        lambda r: sentiment_label(int(r["positive"] or 0), int(r["negative"] or 0)),
        axis=1,
    )
    df.to_parquet(GAMES_PARQUET, index=False)

    print("Sentiment distribution:")
    for label, count in df["sentiment_label"].value_counts().items():
        print(f"  {label:>30s}  {count:>5,}")


if __name__ == "__main__":
    main()
