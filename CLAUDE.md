# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python data pipeline that produces an interactive 2D semantic map of the top ~10,000 most-reviewed games on Steam, paralleling `../semantic-github-map/`. The final artifact is `docs/index.html` (deployed to GitHub Pages); a local copy lands at `data/steam_map.html`.

The intended audience is data-curious gamers who would wander a map of their hobby. That shapes choices: rich hover (capsule images, sentiment, summary), gamer-flavored taglines, multiple colormap axes, click-to-Steam-store.

## Pipeline architecture

Ten sequential stages, each a standalone script. Run via `make pipeline` or individually:

```
00 enumerate_games.py         FronkonGames HF dataset      -> data/candidates.parquet
01 fetch_tags.py              SteamSpy /api.php (~3.3 hr crawl) -> data/games.parquet
02 select_top_games.py        Trim to TARGET_GAME_COUNT (10K)
03 compute_sentiment.py       Steam-style review label from positive/negative
04 summarize_descriptions.py  Claude Haiku tagline + 2-3 sent summary  (~$5-10)
05 embed_descriptions.py      Cohere embed-v4.0, description + top tags (512-dim)
06 reduce_umap.py             UMAP cosine, n_neighbors=15, min_dist=0.05
07 induce_facets.py           Typologist with metadata erasure  (~$30-60)
08 label_topics.py            Toponymy hierarchical region labels via Claude Sonnet
09 visualize.py               DataMapPlot interactive HTML, capsule images on hover
```

Three logical layers: **data** (00-03), **semantics** (04-06), **labels + render** (07-09).

`pipeline/config.py` is the central hub. Every stage imports paths and constants from it. Edit `TARGET_GAME_COUNT` / `FETCH_OVERSHOOT_COUNT` there for smoke tests; don't add CLI args.

### Data flow

All outputs land in `data/` (gitignored):

```
candidates.parquet  ->  games.parquet  -+->  embeddings.npz  ->  umap_coords.npz  ->  labels.parquet
                                        |                                                  |
                                        +-> facets.parquet (stage 07)                      |
                                        |                                                  |
                                        +--------------------------------------------------+--> steam_map.html
```

`games.parquet` is enriched in-place by stages 03 and 04 (they add columns); other stages produce separate files.

## Why no fresh appdetails refetch

The original plan included a 5-hour `appdetails` refetch (stage 01) for fresh prices, capsule URLs, descriptions. Dropped after discovering FronkonGames already has all of that. The only thing it doesn't have is the live `review_score_desc` ("Very Positive", "Mixed", etc.), which we recompute locally in stage 03 from `positive` and `negative` counts using Valve's documented bucketing thresholds. Net: ~5 hours saved with no material data loss.

The appdetails fetcher is preserved at `experiments/fetch_appdetails.py` for future use if FronkonGames staleness becomes a problem (currently 3 months behind).

## Why stage 07 is currently skipped at runtime

Stage 07 uses Typologist (the user's own library, `../typologist/`) to discover orthogonal categorical facets via concept erasure. The script and Makefile entry exist, but the user has paused on it pending a facet-design decision (Typologist-discovered vs hand-authored from existing FronkonGames metadata).

Stage 09 is already patched to tolerate missing `data/facets.parquet`, so removing or commenting out the `07_induce_facets.py` line in the Makefile lets the rest of the pipeline run cleanly. To re-enable, just uncomment.

## Gotchas (learned the hard way)

### FronkonGames data quirks

Source: `https://huggingface.co/datasets/FronkonGames/steam-games-dataset` (last upload 2026-02-02).

- Column is `appID` (capital ID). Stage 00 normalizes to lowercase `appid`.
- The `tags` column exists in the schema but is **empty for every row**. Don't trust it. Use SteamSpy (stage 01).
- There is no `is_free` field. Free-to-play check: `price == 0` (where `price` is a float, USD).

### SteamSpy tag dict structure

`https://steamspy.com/api.php?request=appdetails&appid=<id>` returns roughly 20 tags with integer vote counts **plus a long tail of 400+ "ever-applied" tags with `None` vote counts**. Filter to non-None before sorting or taking max:

```python
counted = [(k, v) for k, v in tags.items() if isinstance(v, (int, float))]
top = sorted(counted, key=lambda x: -x[1])[:TOP_N_TAGS]
```

### appid dtype mismatch across parquets

FronkonGames stores `appid` as **string**. Stages 08 and 07 cast to **int** when they write derived parquets (labels, facets). Stage 09 must cast both sides to int before merging or pandas raises `ValueError: trying to merge on object and int64 columns`. Don't `merge(on="appid")` without normalizing dtypes first.

### Dependency pins that matter

- **`datasets>=3.0`** in `pyproject.toml`. Without this, uv's resolver picks `datasets==1.1.1` (Sept 2020), which crashes on modern pyarrow's removed `PyExtensionType`. Pulled in transitively via `sentence_transformers`.
- **`fast-hdbscan==0.2.2`**. `toponymy==0.5.0`'s clustering calls `parallel_boruvka(tree, ...)` without an `n_threads` argument. `fast-hdbscan>=0.3.0` made that argument required, so 0.5.0 + 0.3.x is broken at runtime. We need toponymy 0.5.x for typologist compatibility (typologist requires `toponymy>=0.5.0,<0.6.0`), so pin fast-hdbscan back to 0.2.2 instead.

### Pandas API drift

`Series.clip(min=N)` was renamed in newer pandas. Use `Series.clip(lower=N)`. (Numpy ndarrays' `.clip(min=...)` still works, so check whether you're holding a Series or an ndarray.)

### Typologist 0.0.1 API differs from its README

- The README shows `from typologist import AnthropicLLM`. The published 0.0.1 wheel has **no public `AnthropicLLM` class**.
- Actual API: `Typologist(naming_llm="claude-haiku-4-5", schema_llm="claude-opus-4-7", labeling_llm="claude-haiku-4-5", ...)`. LLM args are bare model-name strings; the library resolves them via internal `_resolve_llm`.
- `metadata=` accepts a `pd.DataFrame` of categorical columns; the library one-hot encodes them automatically before LEACE.
- Read the actual source under `.venv/lib/python3.11/site-packages/typologist/` if behavior surprises you, since the README documents an aspirational API.

## Common commands

```bash
make install           # uv sync --extra dev
make lint              # ruff check + ruff format --check
make format            # ruff format
make test              # pytest (no tests authored yet)
make pipeline          # run all stages in sequence

# Run a single stage (config.py paths cascade naturally):
uv run python pipeline/04_summarize_descriptions.py

# Long-running stages can be detached so they survive the shell:
nohup uv run python pipeline/01_fetch_tags.py > /tmp/steam-map-fetch-tags.log 2>&1 &

# Smoke-test a stage with reduced N: temporarily lower TARGET_GAME_COUNT or
# FETCH_OVERSHOOT_COUNT in pipeline/config.py, run the stage, restore.
```

## Resumability and atomic writes

Stages 01 (SteamSpy) and 04 (Haiku) are resumable: rerunning skips rows already present in `games.parquet`. Both checkpoint every N rows (100 for 01, 200 for 04) via atomic tmp+rename, so a kill mid-run leaves a consistent partial file.

Stage 02 backs up `games.parquet` to `games_pretrim.parquet` before trimming. Stage 04 writes a `.bak` copy of `games.parquet` before its first batch.

Treat `data/*.parquet` as expensive to regenerate. Never overwrite without a tmp+rename or a backup; the SteamSpy crawl alone is 3+ hours.

## PROJECT_NAME

`PROJECT_NAME` and `PROJECT_TAGLINE` in `pipeline/config.py` are the single source of truth for the user-facing name. The repo directory is `steam-map` as a working name; renaming later should be a one-line change in `config.py`, not a sweep across files. Don't bake the project name into multiple identifiers.

## Required env vars

In `.env` (loaded by `python-dotenv` in `config.py`):

- `ANTHROPIC_API_KEY`: stages 04, 07, 08
- `CO_API_KEY`: stages 05, 08

Stages 00, 01, 02, 03, 06, 09 need no external auth (FronkonGames is a public HF parquet, SteamSpy is unauthenticated).

## Stage 09 (visualize) is currently minimum-viable

Stage 09 is roughly 250 lines vs github-map's 1400+. It has: capsule images on hover, Toponymy region labels at multiple zoom levels, click-to-Steam, search by name, and 5 colormaps (sentiment, genre, F2P, review count, plus one per Typologist facet when stage 07 has run).

Not yet ported from github-map: mobile-specific UI, custom filter panel beyond DataMapPlot's built-in colormap dropdown, edge-bundling background image, per-point text labels at zoom, hand-authored About page (with `<!-- DATA_AS_OF -->` placeholder pattern), Open Graph / social-preview tags, Plausible analytics. `../semantic-github-map/pipeline/07_visualize.py` is the reference when adding these.

## Sibling repos

- `../semantic-github-map/`: the parallel project for GitHub repos. Same skeleton; the atomic-write, resumability, two-phase fetch, Toponymy + DataMapPlot wiring patterns were lifted directly from there.
- `../typologist/`: the user's own library, used in stage 07. Alpha 0.0.1; expect rough edges and API churn between releases.
