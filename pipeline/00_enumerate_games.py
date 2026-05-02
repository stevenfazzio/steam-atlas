"""Download FronkonGames steam-games-dataset and rank by total review count.

Source: https://huggingface.co/datasets/FronkonGames/steam-games-dataset
A periodic snapshot of the Steam catalog (last upload Feb 2026, ~124K games).
We use it as a hot-start enumeration: trim to the top N games by review count
and hand off to stage 01.
"""

import pandas as pd
from config import (
    CANDIDATE_OVERSHOOT_COUNT,
    CANDIDATES_PARQUET,
    FRONKONGAMES_FILENAME,
    FRONKONGAMES_REPO_ID,
)
from huggingface_hub import hf_hub_download

# FronkonGames uses "AppID" with capital letters; normalize to "appid" so
# downstream stages can assume a consistent column name.
COLUMN_RENAMES = {
    "AppID": "appid",
    "appID": "appid",
    "AppId": "appid",
    "appId": "appid",
}


def main():
    print(f"Downloading {FRONKONGAMES_REPO_ID} ({FRONKONGAMES_FILENAME})...")
    path = hf_hub_download(
        repo_id=FRONKONGAMES_REPO_ID,
        filename=FRONKONGAMES_FILENAME,
        repo_type="dataset",
    )
    print(f"  Cached at: {path}")

    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"Columns: {sorted(df.columns)}")

    # Normalize the appid column name
    renames = {k: v for k, v in COLUMN_RENAMES.items() if k in df.columns}
    if renames:
        df = df.rename(columns=renames)
    if "appid" not in df.columns:
        raise RuntimeError(f"Expected an appid-style column in FronkonGames; got: {sorted(df.columns)}")

    if "positive" not in df.columns or "negative" not in df.columns:
        raise RuntimeError(f"Expected 'positive' and 'negative' columns; got: {sorted(df.columns)}")
    df["total_reviews"] = df["positive"].fillna(0).astype(int) + df["negative"].fillna(0).astype(int)

    df = df.sort_values("total_reviews", ascending=False).head(CANDIDATE_OVERSHOOT_COUNT).reset_index(drop=True)

    print(
        f"\nTrimmed to top {len(df):,} by review count:"
        f"\n  median={df['total_reviews'].median():,.0f}"
        f"\n  min={df['total_reviews'].min():,.0f}"
        f"\n  max={df['total_reviews'].max():,.0f}"
    )

    CANDIDATES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CANDIDATES_PARQUET, index=False)
    print(f"\nSaved {len(df):,} candidates to {CANDIDATES_PARQUET}")


if __name__ == "__main__":
    main()
