"""Fetch fresh appdetails from Steam storefront. Not in the v1 pipeline.

Reserved for future use if FronkonGames staleness becomes a problem (currently 3 months
behind on prices and missing games released since the snapshot). Fetches one app per
request at the storefront's ~200 req / 5 min IP throttle, so a 12K crawl is ~5 hours.

The v1 pipeline relies on FronkonGames data plus a SteamSpy tag fetch instead. To wire
this back in, copy back to pipeline/ as the appropriate stage and join on appid.
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
    STEAM_API_BASE,
    STEAM_LANG,
    STEAM_MAX_RETRIES,
    STEAM_REGION_CC,
    STEAM_REQUEST_DELAY_SEC,
    STEAM_RETRY_BACKOFF_SEC,
    STEAM_USER_AGENT,
)
from tqdm import tqdm

CHECKPOINT_EVERY = 100

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": STEAM_USER_AGENT})


def _fetch_appdetails(appid: int) -> dict | None:
    """Fetch one game's appdetails. Returns inner data dict on success, None otherwise."""
    url = f"{STEAM_API_BASE}/appdetails"
    params = {"appids": appid, "cc": STEAM_REGION_CC, "l": STEAM_LANG}

    for attempt in range(STEAM_MAX_RETRIES):
        try:
            resp = SESSION.get(url, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            wait = min(STEAM_RETRY_BACKOFF_SEC * (attempt + 1), 300)
            print(f"\n  appid={appid}: {type(e).__name__}, retry in {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            wait = STEAM_RETRY_BACKOFF_SEC * (attempt + 1)
            print(f"\n  appid={appid}: 429 rate-limited, sleeping {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            wait = min(STEAM_RETRY_BACKOFF_SEC * (attempt + 1), 300)
            print(f"\n  appid={appid}: HTTP {resp.status_code}, retry in {wait}s")
            time.sleep(wait)
            continue

        try:
            body = resp.json()
        except ValueError:
            return None

        entry = body.get(str(appid)) if isinstance(body, dict) else None
        if not entry or not entry.get("success"):
            return None
        return entry.get("data")

    print(f"\n  appid={appid}: gave up after {STEAM_MAX_RETRIES} retries")
    return None


def _flatten(appid: int, data: dict) -> dict:
    """Flatten an appdetails 'data' object into a parquet-friendly row dict."""
    price = data.get("price_overview") or {}
    platforms = data.get("platforms") or {}
    release = data.get("release_date") or {}
    metacritic = data.get("metacritic") or {}
    recs = data.get("recommendations") or {}
    achievements = data.get("achievements") or {}
    descriptors = data.get("content_descriptors") or {}

    return {
        "appid": appid,
        "name": data.get("name", ""),
        "type": data.get("type", ""),
        "required_age": int(data.get("required_age") or 0),
        "is_free": bool(data.get("is_free", False)),
        "controller_support": data.get("controller_support") or "",
        "short_description": data.get("short_description") or "",
        "detailed_description": data.get("detailed_description") or "",
        "about_the_game": data.get("about_the_game") or "",
        "supported_languages": data.get("supported_languages") or "",
        "header_image": data.get("header_image") or "",
        "developers": list(data.get("developers") or []),
        "publishers": list(data.get("publishers") or []),
        "categories": [c.get("description", "") for c in (data.get("categories") or [])],
        "genres": [g.get("description", "") for g in (data.get("genres") or [])],
        "price_currency": price.get("currency", ""),
        "price_initial_cents": int(price.get("initial") or 0),
        "price_final_cents": int(price.get("final") or 0),
        "discount_percent": int(price.get("discount_percent") or 0),
        "windows": bool(platforms.get("windows", False)),
        "mac": bool(platforms.get("mac", False)),
        "linux": bool(platforms.get("linux", False)),
        "release_date": release.get("date", ""),
        "release_coming_soon": bool(release.get("coming_soon", False)),
        "metacritic_score": int(metacritic["score"]) if metacritic.get("score") is not None else None,
        "recommendations_total": int(recs["total"]) if recs.get("total") is not None else None,
        "achievements_total": int(achievements["total"]) if achievements.get("total") is not None else None,
        "content_descriptor_ids": list(descriptors.get("ids") or []),
        "website": data.get("website") or "",
    }


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


def main():
    candidates = pd.read_parquet(CANDIDATES_PARQUET)
    if "appid" not in candidates.columns:
        raise RuntimeError(f"candidates.parquet missing 'appid' column. Got: {sorted(candidates.columns)}")

    candidates = candidates.sort_values("total_reviews", ascending=False).head(FETCH_OVERSHOOT_COUNT)
    appids_wanted = candidates["appid"].astype(int).tolist()

    # Resumability: load existing rows
    existing: dict[int, dict] = {}
    if GAMES_PARQUET.exists():
        df_existing = pd.read_parquet(GAMES_PARQUET)
        for row in df_existing.to_dict("records"):
            existing[int(row["appid"])] = row
        print(f"Resuming: {len(existing):,} games already fetched")

    todo = [a for a in appids_wanted if a not in existing]
    print(f"To fetch: {len(todo):,} games (estimated {len(todo) * STEAM_REQUEST_DELAY_SEC / 60:.0f} minutes)")

    if not todo:
        print("Nothing to do.")
        return

    fetched = 0
    failed = 0
    with tqdm(total=len(todo), desc="Fetching appdetails") as pbar:
        for i, appid in enumerate(todo):
            data = _fetch_appdetails(appid)
            if data is not None:
                existing[appid] = _flatten(appid, data)
                fetched += 1
            else:
                failed += 1

            time.sleep(STEAM_REQUEST_DELAY_SEC)
            pbar.update(1)
            pbar.set_postfix({"ok": fetched, "drop": failed})

            if (i + 1) % CHECKPOINT_EVERY == 0:
                df = pd.DataFrame(list(existing.values()))
                _safe_write_parquet(df, GAMES_PARQUET)

    df = pd.DataFrame(list(existing.values()))
    _safe_write_parquet(df, GAMES_PARQUET)

    print(f"\nDone. {fetched:,} fetched, {failed:,} dropped (delisted/region-locked/non-game).")
    print(f"Saved {len(df):,} total rows to {GAMES_PARQUET}")
    if "type" in df.columns:
        print(f"Types: {dict(df['type'].value_counts())}")


if __name__ == "__main__":
    main()
