"""Generate hierarchical region labels via Toponymy with Claude Sonnet.

Mirrors github-map's 06_label_topics.py. Toponymy partitions the embedding+UMAP space
into a hierarchy of clusters and asks an LLM to name each one. The output is a per-game
label at multiple zoom levels, fed to DataMapPlot's `label_layers=` for the floating
region names on the map.

Documents are the name+tagline+summary text (cleaner than raw descriptions for the
LLM to reason about cluster identity).
"""

import joblib

# Workaround: nested asyncio.run() calls fail with "Event loop is closed".
# nest_asyncio patches the loop to allow re-entrant calls.
import nest_asyncio
import numpy as np
import pandas as pd
from toponymy import Toponymy, ToponymyClusterer
from toponymy.embedding_wrappers import CohereEmbedder
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()

from config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL_NAMING,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    EMBEDDINGS_NPZ,
    GAMES_PARQUET,
    LABELS_PARQUET,
    TARGET_GAME_COUNT,
    TOPONYMY_MODEL_JOBLIB,
    UMAP_COORDS_NPZ,
)


def _build_documents(df: pd.DataFrame) -> list[str]:
    """Name - tagline + summary per game. Falls back to raw short_description if missing."""
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
            short = (row.get("short_description") or "").strip()
            text = f"{name}\n{short}" if short else name
        docs.append(text)
    return docs


def main():
    df = pd.read_parquet(GAMES_PARQUET)
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]
    coords = np.load(UMAP_COORDS_NPZ)["coords"]
    print(f"Loaded {len(df):,} games, embeddings {embeddings.shape}, coords {coords.shape}")

    documents = _build_documents(df)

    llm = AsyncAnthropicNamer(api_key=ANTHROPIC_API_KEY, model=ANTHROPIC_MODEL_NAMING)
    embedder = CohereEmbedder(api_key=CO_API_KEY, model=COHERE_EMBED_MODEL)
    clusterer = ToponymyClusterer(min_clusters=4)

    np.random.seed(42)
    clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)

    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="Steam game descriptions",
        corpus_description=f"collection of the top {TARGET_GAME_COUNT:,} most-reviewed games on Steam",
        exemplar_delimiters=['    * """', '"""\n'],
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    topic_model.fit(
        objects=documents,
        embedding_vectors=embeddings,
        clusterable_vectors=coords,
    )

    n_layers = len(topic_model.cluster_layers_)
    if n_layers == 0:
        raise ValueError("Toponymy produced 0 cluster layers")
    print(f"Toponymy produced {n_layers} cluster layer(s)")

    # DataMapPlot expects coarsest first
    labels_dict = {"appid": df["appid"].astype(int).reset_index(drop=True)}
    for i, layer in enumerate(reversed(topic_model.cluster_layers_)):
        labels_dict[f"label_layer_{i}"] = layer.topic_name_vector

    labels_df = pd.DataFrame(labels_dict)
    labels_df.to_parquet(LABELS_PARQUET, index=False)
    print(f"Saved labels to {LABELS_PARQUET}")

    try:
        joblib.dump(topic_model, TOPONYMY_MODEL_JOBLIB)
        print(f"Saved model to {TOPONYMY_MODEL_JOBLIB}")
    except TypeError:
        print("Skipped saving model (async client is not picklable)")


if __name__ == "__main__":
    main()
