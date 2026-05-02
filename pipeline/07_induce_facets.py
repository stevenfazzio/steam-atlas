"""Discover orthogonal facets via Typologist with metadata erasure.

Inputs: games.parquet (descriptions, tags, genres, sentiment, etc.) + Cohere embeddings.
Pre-erases existing taxonomic structure (primary_genre, top_tag, free_or_paid, era,
sentiment) so the discovered facets are orthogonal to what Steam already labels for us.

Outputs: a JSON schema describing the discovered facets, and a parquet of per-game
facet labels keyed by appid. Each becomes a colormap dropdown in the final viz.
"""

import json
import re

import numpy as np
import pandas as pd
from config import (
    EMBEDDINGS_NPZ,
    FACETS_PARQUET,
    GAMES_PARQUET,
    SCHEMA_JSON,
    TYPOLOGIST_LABELING_MODEL,
    TYPOLOGIST_N_FACETS,
    TYPOLOGIST_NAMING_MODEL,
    TYPOLOGIST_SCHEMA_MODEL,
)
from sentence_transformers import SentenceTransformer
from typologist import Typologist

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _build_documents(df: pd.DataFrame) -> list[str]:
    """Name + tagline + summary per game. Short, neutral text for the labeling LLM."""
    docs = []
    for _, row in df.iterrows():
        name = (row.get("name") or "").strip()
        tagline = (row.get("tagline") or "").strip()
        summary = (row.get("summary") or "").strip()
        if tagline and summary:
            text = f"{name} - {tagline}\n{summary}"
        elif summary:
            text = f"{name}\n{summary}"
        else:
            text = name
        docs.append(text)
    return docs


def _first_genre(g) -> str:
    if isinstance(g, (list, np.ndarray)) and len(g):
        return str(g[0])
    return "Unknown"


def _top_tag(t) -> str:
    if isinstance(t, dict) and t:
        counted = [(k, v) for k, v in t.items() if isinstance(v, (int, float))]
        if counted:
            return max(counted, key=lambda x: x[1])[0]
    return "Unknown"


def _era(date_str) -> str:
    if not isinstance(date_str, str):
        return "Unknown"
    m = YEAR_RE.search(date_str)
    if not m:
        return "Unknown"
    year = int(m.group())
    if year < 2010:
        return "Pre-2010"
    if year < 2015:
        return "2010-2014"
    if year < 2020:
        return "2015-2019"
    if year < 2025:
        return "2020-2024"
    return "2025+"


def _build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Existing taxonomic axes Typologist will erase before discovering new facets."""
    return pd.DataFrame(
        {
            "primary_genre": df["genres"].apply(_first_genre),
            "top_tag": df["tags"].apply(_top_tag),
            "free_or_paid": np.where(df["price"].fillna(0) == 0, "Free", "Paid"),
            "era": df["release_date"].apply(_era),
            "sentiment": df["sentiment_label"].fillna("Unknown"),
        }
    )


def main():
    df = pd.read_parquet(GAMES_PARQUET)
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]
    print(f"Loaded {len(df):,} games and embeddings {embeddings.shape}")

    documents = _build_documents(df)
    metadata = _build_metadata(df)
    print(f"Erasing {len(metadata.columns)} metadata axes: {list(metadata.columns)}")
    for col in metadata.columns:
        n_unique = metadata[col].nunique()
        print(f"  {col}: {n_unique} unique values")
    print(f"Discovering {TYPOLOGIST_N_FACETS} orthogonal facets")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    t = Typologist(
        n_facets=TYPOLOGIST_N_FACETS,
        topic_embedder=embedder,
        naming_llm=TYPOLOGIST_NAMING_MODEL,
        schema_llm=TYPOLOGIST_SCHEMA_MODEL,
        labeling_llm=TYPOLOGIST_LABELING_MODEL,
        object_description="Steam games",
        corpus_description=f"top {len(df):,} most-reviewed Steam games",
        random_state=42,
        verbose=True,
    ).fit(documents, embeddings, metadata=metadata)

    SCHEMA_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEMA_JSON, "w") as f:
        json.dump(t.schema_, f, indent=2)
    print(f"\nSaved schema ({len(t.schema_)} facets) to {SCHEMA_JSON}")

    labels = t.labels_df_.copy()
    labels.insert(0, "appid", df["appid"].astype(int).reset_index(drop=True))
    labels.to_parquet(FACETS_PARQUET, index=False)
    print(f"Saved per-game facet labels to {FACETS_PARQUET}")

    print("\nDiscovered facets:")
    for facet in t.schema_:
        print(f"  {facet['name']}: {facet.get('values', [])}")


if __name__ == "__main__":
    main()
