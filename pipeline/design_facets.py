"""One-time discovery of orthogonal categorical facets via Toponymy + EVoC.

Clusters the full-dim Cohere embeddings (not the 2D UMAP projection) so the cluster
structure reflects actual semantic axes, including ones UMAP collapses for layout.
Toponymy labels each cluster hierarchically; we hand the hierarchy plus sample
in-cluster taglines to Claude Opus and ask for 3-5 categorical facets (4-8 mutually
exclusive values each), suitable as colormap dropdowns on the final viz.

Output: pipeline/facets_schema.json (committed). Stage 07 reads this to label
individual games. Re-run only when you want a new schema; the committed file is
the source of truth between runs.
"""

import json
import os
import re
import tempfile

import nest_asyncio
import numpy as np
import pandas as pd
from toponymy import Toponymy
from toponymy.clustering import EVoCClusterer
from toponymy.embedding_wrappers import CohereEmbedder
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()

import anthropic  # noqa: E402
from config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL_NAMING,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    EMBEDDINGS_NPZ,
    FACET_DESIGN_MODEL,
    FACETS_SCHEMA_JSON,
    GAMES_PARQUET,
)

SAMPLES_PER_CLUSTER = 8
MAX_LAYERS_TO_SHOW = 2

OTHER_VALUE = {
    "name": "Other",
    "description": (
        "None of the above values clearly fit. When picking Other, also provide a "
        "2-5 word phrase in the corresponding _other field describing what would "
        "actually fit."
    ),
}

SYSTEM_PROMPT = (
    "You design categorical facet schemas for interactive data visualizations. "
    "Given a hierarchical cluster summary of a corpus of objects, propose 3-5 "
    "categorical facets that capture orthogonal axes of variation. Each facet "
    "has 4-8 mutually exclusive values.\n\n"
    "The facets are colormap dropdowns on an interactive 2D map; each one should "
    "surface a meaningfully different way to slice the corpus. Two facets that "
    "encode the same dimension are wasted slots. Within a facet, values must be "
    "mutually exclusive (each object gets exactly one) and collectively cover "
    "the corpus (every object has at least a plausible fit).\n\n"
    "Do NOT include an 'Other', 'Unknown', or 'Miscellaneous' value; the system "
    "appends an Other escape valve automatically after your proposal.\n\n"
    "Each value description must give a labeler enough information to confidently "
    "pick that value when reading the object's name and short summary.\n\n"
    "Return only a JSON array, no markdown fencing. Each element:\n"
    '  {"name": "<title-cased noun phrase, 1-3 words>",\n'
    '   "description": "<one-sentence explanation of what this facet captures>",\n'
    '   "values": [\n'
    '     {"name": "<short label>", "description": "<labeler-facing explanation>"},\n'
    "     ... (4-8 values total)\n"
    "   ]}"
)


def _build_documents(df: pd.DataFrame) -> list[str]:
    """Mirror stage 08's document construction so cluster identity is consistent."""
    docs = []
    for _, row in df.iterrows():
        name = (row.get("name") or "").strip()
        tagline = (row.get("tagline") or "").strip()
        summary = (row.get("summary") or "").strip()
        if tagline and summary:
            docs.append(f"{name} - {tagline}\n{summary}")
        elif summary:
            docs.append(f"{name}\n{summary}")
        else:
            short = (row.get("short_description") or "").strip()
            docs.append(f"{name}\n{short}" if short else name)
    return docs


def _format_hierarchy(topic_model: Toponymy, df: pd.DataFrame) -> str:
    """Render the coarsest layers as a bullet hierarchy with sample game taglines."""
    layers = list(reversed(topic_model.cluster_layers_))[:MAX_LAYERS_TO_SHOW]
    names = df["name"].fillna("").tolist()
    taglines = df.get("tagline", pd.Series([""] * len(df))).fillna("").tolist()

    lines = []
    for level_idx, layer in enumerate(layers):
        level_name = "Top-level" if level_idx == 0 else f"Level {level_idx + 1}"
        n_clusters = len(layer.topic_names)
        lines.append(f"\n## {level_name} regions ({n_clusters} clusters)\n")
        for cluster_id, topic_name in enumerate(layer.topic_names):
            members = np.where(layer.cluster_labels == cluster_id)[0]
            if len(members) == 0:
                continue
            sample_indices = members[: min(SAMPLES_PER_CLUSTER, len(members))]
            samples = []
            for j in sample_indices:
                tag = taglines[j].strip() if isinstance(taglines[j], str) else ""
                game_name = names[j].strip() if isinstance(names[j], str) else ""
                if tag and game_name:
                    samples.append(f"{game_name} ({tag})")
                elif game_name:
                    samples.append(game_name)
            sample_str = "; ".join(samples)
            lines.append(f"- **{topic_name}** ({len(members)} games): {sample_str}")
    return "\n".join(lines)


def _validate_schema(schema) -> None:
    if not isinstance(schema, list):
        raise ValueError(f"Schema must be a list, got {type(schema).__name__}")
    if not (3 <= len(schema) <= 5):
        raise ValueError(f"Schema must have 3-5 facets, got {len(schema)}")
    for i, facet in enumerate(schema):
        if not isinstance(facet, dict):
            raise ValueError(f"Facet {i} is not an object: {facet!r}")
        for key in ("name", "description", "values"):
            if key not in facet:
                raise ValueError(f"Facet {i} missing key '{key}'")
        values = facet["values"]
        if not isinstance(values, list) or not (4 <= len(values) <= 8):
            raise ValueError(f"Facet '{facet['name']}' must have 4-8 values, got {len(values)}")
        for j, v in enumerate(values):
            if not isinstance(v, dict) or "name" not in v or "description" not in v:
                raise ValueError(f"Facet '{facet['name']}' value {j} malformed: {v!r}")


def _atomic_write_json(obj, path) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".json.tmp")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp_path, str(path))
    except Exception:
        os.unlink(tmp_path)
        raise


def main():
    df = pd.read_parquet(GAMES_PARQUET)
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]
    print(f"Loaded {len(df):,} games, embeddings {embeddings.shape}")

    documents = _build_documents(df)

    namer = AsyncAnthropicNamer(api_key=ANTHROPIC_API_KEY, model=ANTHROPIC_MODEL_NAMING)
    embedder = CohereEmbedder(api_key=CO_API_KEY, model=COHERE_EMBED_MODEL)
    clusterer = EVoCClusterer(min_clusters=4, base_min_cluster_size=20)

    np.random.seed(42)
    # EVoC ignores `clusterable_vectors` and clusters directly on `embedding_vectors`.
    # Pass full-dim embeddings on purpose: facet axes that survive UMAP-to-2D are
    # exactly the axes we DO NOT need to discover.
    clusterer.fit(clusterable_vectors=embeddings, embedding_vectors=embeddings)

    topic_model = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="Steam game descriptions",
        corpus_description=f"top {len(df):,} most-reviewed Steam games",
        exemplar_delimiters=['    * """', '"""\n'],
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    topic_model.fit(
        objects=documents,
        embedding_vectors=embeddings,
        clusterable_vectors=embeddings,
    )

    n_layers = len(topic_model.cluster_layers_)
    if n_layers == 0:
        raise ValueError("EVoC produced 0 cluster layers")
    print(f"EVoC produced {n_layers} cluster layer(s)")

    hierarchy = _format_hierarchy(topic_model, df)
    print(f"Hierarchy summary: {len(hierarchy):,} chars")

    user_message = (
        f"Corpus: top {len(df):,} most-reviewed Steam games, embedded by description "
        f"and clustered hierarchically by EVoC on the full-dim embeddings.\n\n"
        f"Cluster hierarchy:\n{hierarchy}\n\n"
        f"Design 3-5 orthogonal categorical facets that a curious gamer would find "
        f"useful as colormap dropdowns when exploring an interactive map of these games."
    )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    print(f"Asking {FACET_DESIGN_MODEL} for facet schema...")
    resp = client.messages.create(
        model=FACET_DESIGN_MODEL,
        max_tokens=4_000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        schema = json.loads(raw)
    except json.JSONDecodeError:
        print(f"\nLLM returned invalid JSON:\n{raw}\n")
        raise

    _validate_schema(schema)

    for facet in schema:
        existing_names = {v["name"].lower() for v in facet["values"]}
        if OTHER_VALUE["name"].lower() not in existing_names:
            facet["values"].append(dict(OTHER_VALUE))

    FACETS_SCHEMA_JSON.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(schema, FACETS_SCHEMA_JSON)

    print(f"\nSaved {len(schema)} facets to {FACETS_SCHEMA_JSON}\n")
    for facet in schema:
        values = ", ".join(v["name"] for v in facet["values"])
        print(f"  {facet['name']}: {values}")


if __name__ == "__main__":
    main()
