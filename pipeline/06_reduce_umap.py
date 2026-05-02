"""Reduce embeddings to 2D with UMAP. Same params as github-map for shape consistency."""

import numpy as np
import umap
from config import EMBEDDINGS_NPZ, UMAP_COORDS_NPZ


def main():
    data = np.load(EMBEDDINGS_NPZ)
    embeddings = data["embeddings"]
    print(f"Loaded embeddings: {embeddings.shape}")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    np.savez(UMAP_COORDS_NPZ, coords=coords)
    print(f"Saved 2D coords {coords.shape} to {UMAP_COORDS_NPZ}")


if __name__ == "__main__":
    main()
