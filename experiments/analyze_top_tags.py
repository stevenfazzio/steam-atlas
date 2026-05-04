"""Frequency of each SteamSpy tag appearing in the top-3 of our 10k games.

Used to decide which tags (if any) are too generic to surface as hovercard chips.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from config import GAMES_PARQUET  # noqa: E402

TOP_N = 3
N_GAMES = None  # use full corpus


def top_n_tags(tags_dict, n: int = TOP_N) -> list[str]:
    if not isinstance(tags_dict, dict):
        return []
    counted = [(k, v) for k, v in tags_dict.items() if isinstance(v, (int, float)) and v is not None]
    counted.sort(key=lambda x: -x[1])
    return [k for k, _ in counted[:n]]


def main() -> None:
    df = pd.read_parquet(GAMES_PARQUET)
    if N_GAMES:
        df = df.head(N_GAMES)
    n = len(df)
    print(f"Corpus: {n:,} games\n")

    rank_counters: list[Counter] = [Counter() for _ in range(TOP_N)]
    any_top3 = Counter()

    games_with_tags = 0
    for tags in df["tags"]:
        top = top_n_tags(tags, TOP_N)
        if not top:
            continue
        games_with_tags += 1
        for i, t in enumerate(top):
            rank_counters[i][t] += 1
            any_top3[t] += 1

    print(f"Games with at least one tag: {games_with_tags:,} ({games_with_tags / n:.1%})\n")

    print(f"=== Top 30 tags by 'appears in top-{TOP_N}' frequency ===")
    print(f"{'tag':<28} {'top3 %':>8} {'#1 %':>8} {'#2 %':>8} {'#3 %':>8}")
    for tag, count in any_top3.most_common(30):
        r1 = rank_counters[0].get(tag, 0) / games_with_tags
        r2 = rank_counters[1].get(tag, 0) / games_with_tags
        r3 = rank_counters[2].get(tag, 0) / games_with_tags
        pct = count / games_with_tags
        print(f"{tag:<28} {pct:>7.1%} {r1:>7.1%} {r2:>7.1%} {r3:>7.1%}")

    print()
    print(f"=== Tags that appear as #1 in at least 1% of games ===")
    print(f"{'tag':<28} {'#1 %':>8}")
    for tag, count in rank_counters[0].most_common():
        pct = count / games_with_tags
        if pct < 0.01:
            break
        print(f"{tag:<28} {pct:>7.1%}")

    print()
    print(f"=== Distinct tags ever in top-{TOP_N}: {len(any_top3):,} ===")

    # How redundant is top-3? Average size of unique-tag set across the slot positions
    # tells us if the same handful of tags are filling all 3 slots.
    print()
    print("=== Most common (tag1, tag2, tag3) triples ===")
    triples = Counter()
    for tags in df["tags"]:
        top = top_n_tags(tags, TOP_N)
        if len(top) == TOP_N:
            triples[tuple(top)] += 1
    for triple, count in triples.most_common(20):
        print(f"{count:>4}  {' / '.join(triple)}")


if __name__ == "__main__":
    main()
