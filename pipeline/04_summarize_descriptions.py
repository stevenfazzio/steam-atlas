"""Generate gamer-flavored taglines and neutral summaries via Claude Haiku.

Strips HTML from FronkonGames detailed_description, sends to Haiku, extracts a tagline
(short noun phrase) and a 2-3 sentence summary. These are used downstream for hover
tooltips, as input documents to Toponymy (stages 08 and design_facets), and as the text
the per-game facet labeler reads in stage 07.
"""

import asyncio
import html
import json
import os
import re
import shutil
import tempfile

import anthropic
import pandas as pd
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_CONCURRENCY,
    ANTHROPIC_MODEL_SUMMARIZE,
    GAMES_PARQUET,
)
from tqdm import tqdm

MAX_DESC_CHARS = 4_000
CHECKPOINT_EVERY = 200
MAX_RETRIES = 5

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

SYSTEM_PROMPT = (
    "You are given the description of a Steam game. Return a JSON object with two fields:\n"
    '- "tagline": A short noun phrase (3-7 words) identifying what the game IS. '
    "Write for a knowledgeable gamer audience. A category or identity, not a feature list.\n"
    "  Bad: 'Cozy farming game with crafting, romance, mining, fishing, and cooking'\n"
    "  Good: 'Cozy farm-life sim'\n"
    '- "summary": 2-3 sentences in a neutral voice (not marketing copy). Focus on the '
    "gameplay loop and what makes the game distinct. Skip press quotes, calls to action, "
    "franchise marketing, and story spoilers.\n\n"
    "Respond with only the JSON object, no markdown fencing."
)


def strip_html(s: str) -> str:
    """Strip HTML tags and unescape entities. Returns whitespace-collapsed plain text."""
    if not s:
        return ""
    text = HTML_TAG_RE.sub(" ", s)
    text = html.unescape(text)
    return WHITESPACE_RE.sub(" ", text).strip()


def safe_write_parquet(df: pd.DataFrame, path) -> None:
    """Atomically write a parquet file via tmp + verify + rename."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(df)
        os.replace(tmp_path, str(path))
    except Exception:
        os.unlink(tmp_path)
        raise


async def _summarize_one(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    text: str,
    pbar: tqdm,
) -> tuple[str, str]:
    """Return (tagline, summary) for one game's description."""
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.messages.create(
                    model=ANTHROPIC_MODEL_SUMMARIZE,
                    max_tokens=400,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": text[:MAX_DESC_CHARS]}],
                )
                break
            except anthropic.RateLimitError:
                wait = min(2**attempt * 5, 60)
                await asyncio.sleep(wait)
            except (anthropic.APIStatusError, anthropic.APIConnectionError):
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = min(2**attempt * 5, 60)
                await asyncio.sleep(wait)

    pbar.update(1)
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        tagline = (obj.get("tagline") or "").strip().strip("'\"")
        summary = (obj.get("summary") or "").strip()
    except (json.JSONDecodeError, AttributeError):
        tagline = ""
        summary = raw
    return tagline, summary


async def main():
    df = pd.read_parquet(GAMES_PARQUET)
    if "tagline" not in df.columns:
        df["tagline"] = ""
    if "summary" not in df.columns:
        df["summary"] = ""

    needs = df["tagline"].fillna("").eq("") | df["summary"].fillna("").eq("")
    todo_indices = df.index[needs].tolist()
    print(f"Loaded {len(df):,} games. To summarize: {len(todo_indices):,}")
    if not todo_indices:
        print("All games already summarized.")
        return

    backup = str(GAMES_PARQUET) + ".bak"
    shutil.copy2(GAMES_PARQUET, backup)
    print(f"Backed up {GAMES_PARQUET} → {backup}")

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(ANTHROPIC_CONCURRENCY)

    pbar = tqdm(total=len(todo_indices), desc="Summarizing")
    count = 0

    for chunk_start in range(0, len(todo_indices), CHECKPOINT_EVERY):
        chunk = todo_indices[chunk_start : chunk_start + CHECKPOINT_EVERY]
        tasks = []
        for idx in chunk:
            desc_raw = df.at[idx, "detailed_description"]
            if not isinstance(desc_raw, str):
                desc_raw = ""
            text = strip_html(desc_raw)
            if not text:
                df.at[idx, "tagline"] = ""
                df.at[idx, "summary"] = ""
                pbar.update(1)
                continue
            tasks.append((idx, text))

        async def _process(idx, text):
            try:
                return idx, await _summarize_one(client, sem, text, pbar)
            except Exception as e:
                print(f"\n  Error on idx {idx}: {e}")
                pbar.update(1)
                return idx, None

        results = await asyncio.gather(*[_process(i, t) for i, t in tasks])
        for idx, res in results:
            if res is not None:
                tagline, summary = res
                df.at[idx, "tagline"] = tagline
                df.at[idx, "summary"] = summary

        count += len(chunk)
        safe_write_parquet(df, GAMES_PARQUET)
        print(f"  Checkpoint: {count}/{len(todo_indices)}")

    pbar.close()
    print(f"Done. Summarized {len(todo_indices)} games. Saved to {GAMES_PARQUET}")


if __name__ == "__main__":
    asyncio.run(main())
