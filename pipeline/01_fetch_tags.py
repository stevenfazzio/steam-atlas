"""Fetch user-applied tags from SteamSpy and merge into games.parquet.

Reads candidates.parquet, takes top FETCH_OVERSHOOT_COUNT by review count, and hits
https://steamspy.com/api.php?request=appdetails&appid=<id> per game. Tags come back
as {tag_name: vote_count}; we attach them to the FronkonGames row data.

SteamSpy rate limit: 1 request/second. For 12K games ~ 3.3 hours wall clock.
Resumable: rows already present in games.parquet are skipped.

For a quick smoke test, temporarily lower FETCH_OVERSHOOT_COUNT in config.py.
"""

import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests
from config import (
    CANDIDATES_PARQUET,
    FETCH_OVERSHOOT_COUNT,
    GAMES_PARQUET,
    STEAM_USER_AGENT,
)
from tqdm import tqdm

STEAMSPY_API = "https://steamspy.com/api.php"
STEAMSPY_DELAY_SEC = 1.05  # 1 req/sec limit; small safety margin
STEAMSPY_MAX_RETRIES = 3
STEAMSPY_BACKOFF_SEC = 30
CHECKPOINT_EVERY = 100

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": STEAM_USER_AGENT})


def _fetch_tags(appid: int) -> dict | None:
    """Fetch SteamSpy tags. Returns {tag_name: vote_count}, {} if no tags, None on failure."""
    for attempt in range(STEAMSPY_MAX_RETRIES):
        try:
            resp = SESSION.get(
                STEAMSPY_API,
                params={"request": "appdetails", "appid": appid},
                timeout=15,
            )
        except requests.exceptions.RequestException as e:
            wait = STEAMSPY_BACKOFF_SEC * (attempt + 1)
            print(f"\n  appid={appid}: {type(e).__name__}, retry in {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            wait = STEAMSPY_BACKOFF_SEC * (attempt + 1)
            print(f"\n  appid={appid}: HTTP {resp.status_code}, retry in {wait}s")
            time.sleep(wait)
            continue

        try:
            body = resp.json()
        except ValueError:
            return None

        tags = body.get("tags")
        # SteamSpy returns [] when there are no tags, dict when there are
        if isinstance(tags, dict):
            return tags
        return {}

    print(f"\n  appid={appid}: gave up after {STEAMSPY_MAX_RETRIES} retries")
    return None


def _safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomic parquet write: tmp file in same dir, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _checkpoint(candidates: pd.DataFrame, tags_by_appid: dict) -> None:
    """Build games.parquet from candidates + freshly-fetched SteamSpy tags."""
    df = candidates[candidates["appid"].astype(int).isin(tags_by_appid.keys())].copy()
    df["tags"] = df["appid"].astype(int).map(tags_by_appid)
    _safe_write_parquet(df, GAMES_PARQUET)


def main():
    candidates = pd.read_parquet(CANDIDATES_PARQUET)
    if "appid" not in candidates.columns:
        raise RuntimeError(f"candidates.parquet missing 'appid' column. Got: {sorted(candidates.columns)}")

    candidates = (
        candidates.sort_values("total_reviews", ascending=False).head(FETCH_OVERSHOOT_COUNT).reset_index(drop=True)
    )
    print(f"Fetching tags for top {len(candidates):,} candidates")

    # Resume: load any existing games.parquet
    tags_by_appid: dict[int, dict] = {}
    if GAMES_PARQUET.exists():
        existing_df = pd.read_parquet(GAMES_PARQUET)
        if "tags" in existing_df.columns:
            for _, row in existing_df.iterrows():
                tags_by_appid[int(row["appid"])] = row["tags"]
            print(f"Resuming: {len(tags_by_appid):,} games already have tag data")

    todo = [a for a in candidates["appid"].astype(int) if a not in tags_by_appid]
    print(f"To fetch: {len(todo):,} games (estimated {len(todo) * STEAMSPY_DELAY_SEC / 60:.0f} minutes)")

    if not todo:
        print("Nothing to do. (delete data/games.parquet to refetch)")
        return

    fetched = 0
    failed = 0
    with tqdm(total=len(todo), desc="Fetching SteamSpy tags") as pbar:
        for i, appid in enumerate(todo):
            tags = _fetch_tags(appid)
            if tags is not None:
                tags_by_appid[appid] = tags
                fetched += 1
            else:
                failed += 1

            time.sleep(STEAMSPY_DELAY_SEC)
            pbar.update(1)
            pbar.set_postfix({"ok": fetched, "drop": failed})

            if (i + 1) % CHECKPOINT_EVERY == 0:
                _checkpoint(candidates, tags_by_appid)

    _checkpoint(candidates, tags_by_appid)

    n_with_tags = sum(1 for v in tags_by_appid.values() if v)
    print(f"\nDone. {fetched:,} fetched, {failed:,} failed.")
    print(f"  {n_with_tags:,} games have at least one tag")
    print(f"Saved {len(tags_by_appid):,} games to {GAMES_PARQUET}")


if __name__ == "__main__":
    main()
