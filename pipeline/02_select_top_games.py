"""Trim games.parquet to top TARGET_GAME_COUNT by total review count.

Mirrors github-map's pipeline/02_select_top_repos.py. Backs up the pre-trim file so
the larger fetched pool stays available if we ever want to relax the cut.
"""

import shutil

import pandas as pd
from config import GAMES_PARQUET, GAMES_PRETRIM_PARQUET, TARGET_GAME_COUNT


def main():
    df = pd.read_parquet(GAMES_PARQUET)
    print(f"Loaded {len(df):,} games")

    if len(df) <= TARGET_GAME_COUNT:
        print(f"Already at or below target ({TARGET_GAME_COUNT:,}); nothing to trim.")
        return

    df = df.sort_values("total_reviews", ascending=False).head(TARGET_GAME_COUNT).reset_index(drop=True)

    shutil.copy2(GAMES_PARQUET, GAMES_PRETRIM_PARQUET)
    print(f"Backed up original to {GAMES_PRETRIM_PARQUET}")

    df.to_parquet(GAMES_PARQUET, index=False)
    print(
        f"Trimmed to top {TARGET_GAME_COUNT:,} by review count"
        f" (min: {df['total_reviews'].min():,}, max: {df['total_reviews'].max():,})"
    )


if __name__ == "__main__":
    main()
