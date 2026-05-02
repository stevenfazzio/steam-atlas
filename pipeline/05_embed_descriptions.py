"""Embed game descriptions + top user tags via Cohere embed-v4.0.

The text per game is the HTML-stripped detailed_description plus the top 20 SteamSpy
tags by vote count. We embed full descriptions (not LLM summaries) to mirror the
github-map pipeline's choice and let detailed-description texture inform clustering;
tags add crowd vocabulary so the embedding is grounded in what players actually call
the game.
"""

import html
import re

import cohere
import numpy as np
import pandas as pd
from config import (
    CO_API_KEY,
    COHERE_BATCH_SIZE,
    COHERE_EMBED_DIMENSION,
    COHERE_EMBED_MODEL,
    EMBEDDINGS_NPZ,
    GAMES_PARQUET,
)
from tqdm import tqdm

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
MAX_DESC_CHARS = 8_000
TOP_N_TAGS = 20


def strip_html(s: str) -> str:
    if not s:
        return ""
    text = HTML_TAG_RE.sub(" ", s)
    text = html.unescape(text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _build_text(row) -> str:
    """HTML-stripped description + top-N tags joined by commas."""
    desc = strip_html(row.get("detailed_description") or "")
    if not desc:
        desc = strip_html(row.get("short_description") or "")
    desc = desc[:MAX_DESC_CHARS]

    tags = row.get("tags")
    tag_str = ""
    if isinstance(tags, dict) and tags:
        # SteamSpy returns top-20 tags with vote counts plus a long tail of
        # ever-applied tags with None counts; only the counted ones matter.
        counted = [(k, v) for k, v in tags.items() if isinstance(v, (int, float))]
        top = sorted(counted, key=lambda x: -x[1])[:TOP_N_TAGS]
        tag_str = ", ".join(t for t, _ in top)

    if desc and tag_str:
        return f"{desc}\n\nTags: {tag_str}"
    return desc or tag_str or (row.get("name") or "")


def main():
    df = pd.read_parquet(GAMES_PARQUET)
    print(f"Loaded {len(df):,} games")

    texts = [_build_text(row) for _, row in df.iterrows()]
    empty_count = sum(1 for t in texts if not t)
    if empty_count:
        print(f"Warning: {empty_count} games have empty embedding text")

    co = cohere.ClientV2(api_key=CO_API_KEY)

    all_embeddings = []
    for i in tqdm(range(0, len(texts), COHERE_BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + COHERE_BATCH_SIZE]
        resp = co.embed(
            texts=batch,
            model=COHERE_EMBED_MODEL,
            input_type="clustering",
            embedding_types=["float"],
            output_dimension=COHERE_EMBED_DIMENSION,
        )
        all_embeddings.extend(resp.embeddings.float_)

    embeddings = np.array(all_embeddings)
    np.savez(EMBEDDINGS_NPZ, embeddings=embeddings)
    print(f"Saved embeddings {embeddings.shape} to {EMBEDDINGS_NPZ}")


if __name__ == "__main__":
    main()
