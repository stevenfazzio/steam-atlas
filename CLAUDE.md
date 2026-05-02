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
07 label_facets.py            LLM-per-game label against committed schema (~$5-10)
08 label_topics.py            Toponymy hierarchical region labels via Claude Sonnet
09 visualize.py               DataMapPlot interactive HTML, capsule images on hover
```

Three logical layers: **data** (00-03), **semantics** (04-06), **labels + render** (07-09).

**Out-of-band: facet schema design.** `pipeline/design_facets.py` is a one-shot script (run via `make design-facets`, NOT in `make pipeline`). It clusters the full-dim embeddings with EVoC, hands the hierarchical Toponymy labels + sample taglines to Opus, and writes `pipeline/facets_schema.json` (committed). Stage 07 reads that schema. Re-run the design step only when you intentionally want a new schema.

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

## Why facet design is out-of-band

Schema discovery (what facets exist, what their values are) is conceptually one-time work. Per-game labeling against an existing schema is per-run work. We split them so the schema becomes a committed artifact and per-run cost is bounded to the labeling pass.

The design step (`pipeline/design_facets.py`) runs Toponymy on the full-dim Cohere embeddings using `EVoCClusterer` (not the 2D UMAP coords used by stage 08). UMAP-to-2D collapses orthogonal axes for layout coherence; facet discovery wants exactly the axes it collapses, so we cluster in the original embedding space. Output is `pipeline/facets_schema.json`, committed.

Stage 07 (`07_label_facets.py`) reads the committed schema and labels each game with one Haiku call returning all facet values as JSON. Resumable, atomic-write checkpointed every 200 rows, mirrors stage 04's pattern.

If `pipeline/facets_schema.json` is absent, stage 07 fails fast with a clear error. Stage 09 already tolerates a missing `data/facets.parquet`, so removing/commenting `07_label_facets.py` from `make pipeline` is the way to skip facets entirely (e.g. on a fresh repo before the design step has been run).

## Why no fresh appdetails refetch

The original plan included a 5-hour `appdetails` refetch (stage 01) for fresh prices, capsule URLs, descriptions. Dropped after discovering FronkonGames already has all of that. The only thing it doesn't have is the live `review_score_desc` ("Very Positive", "Mixed", etc.), which we recompute locally in stage 03 from `positive` and `negative` counts using Valve's documented bucketing thresholds. Net: ~5 hours saved with no material data loss.

The appdetails fetcher is preserved at `experiments/fetch_appdetails.py` for future use if FronkonGames staleness becomes a problem (currently 3 months behind).

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
- **`fast-hdbscan==0.2.2`**. `toponymy==0.5.0`'s clustering calls `parallel_boruvka(tree, ...)` without an `n_threads` argument. `fast-hdbscan>=0.3.0` made that argument required, so 0.5.0 + 0.3.x is broken at runtime. Pin fast-hdbscan back to 0.2.2 until toponymy 0.6+ is out.

### Pandas API drift

`Series.clip(min=N)` was renamed in newer pandas. Use `Series.clip(lower=N)`. (Numpy ndarrays' `.clip(min=...)` still works, so check whether you're holding a Series or an ndarray.)

### Schema and parquet column lifecycle in stage 07

Facet column names in `data/facets.parquet` are derived from facet names in `pipeline/facets_schema.json` via `re.sub(r"[^a-zA-Z0-9]+", "_", name).lower()`. If the schema is re-designed (renamed or different facets), stage 07 detects the column-set mismatch and discards the prior parquet entirely on the next run. There is no migration path, just a clean rebuild. Re-running the design step is therefore a destructive op for any partial labeling progress.

### Facet field-name collisions in stage 09

DataMapPlot generates RGBA columns per colormap (`<field>_r`, `<field>_g`, etc.). Two colormaps using the same `field` name produce duplicate columns and pyarrow raises `ValueError: Duplicate column names found`. The facet schema can rediscover an existing built-in colormap (e.g. the LLM independently proposes "Primary Genre" alongside the existing genre dropdown). Stage 09 prefixes all facet colormap fields with `facet_` to side-step this; the existing built-in genre dropdown was relabeled "Primary Genre (Steam)" so the user can tell the two apart. Don't drop the prefix when refactoring or you'll rediscover this bug as soon as the schema overlaps with a built-in.

### DataMapPlot region-label rendering bugs (stage 09)

The bundled `datamap.js` ships with two latent bugs that bite our dark-theme render. Both have post-render workarounds in `pipeline/09_visualize.py`:

1. **`characterSet:"auto"` is not auto-discovered**. The TextLayer is created with `characterSet:"auto"` but deck.gl in this version consumes the literal string as a 4-character set `['a','u','t','o']` instead of triggering its auto-discovery path. We post-process the rendered HTML to replace it with an explicit array built from the actual region-label data.

2. **`waitForFont()` is called but not awaited** (`datamap.js:627`). The labelLayer is created and its first `updateState` runs before the WebFont finishes loading. The SDF font atlas mapping is correct, but the GPU texture upload silently fails — region labels are present in the data but render as blank boxes. We side-step the *cause* by injecting `<link rel="stylesheet">` for the Google Font into `<head>` so the font finishes loading before deck.gl runs. We have NOT found a reliable in-page workaround for the *symptom*: cloning, addLabels, setProps, redraw, etc. from a `setTimeout` or event handler doesn't rebuild the texture, even though the same code from the JS console does. If labels are missing on initial load, paste this into devtools to fix:

   ```js
   const ll=datamap.labelLayer,d=ll.props.data,i=datamap.layers.indexOf(ll);
   const n=ll.clone({id:'lbl',data:[...d]});datamap.layers[i]=n;
   datamap.labelLayer=n;datamap.deckgl.setProps({layers:[...datamap.layers]});
   ```

   The proper fix is upstream: make DataMapPlot await `waitForFont` before constructing the TextLayer.

## Common commands

```bash
make install           # uv sync --extra dev
make lint              # ruff check + ruff format --check
make format            # ruff format
make test              # pytest (no tests authored yet)
make pipeline          # run all stages in sequence
make design-facets     # one-shot: rebuild pipeline/facets_schema.json (rare)

# Run a single stage (config.py paths cascade naturally):
uv run python pipeline/04_summarize_descriptions.py

# Long-running stages can be detached so they survive the shell:
nohup uv run python pipeline/01_fetch_tags.py > /tmp/steam-map-fetch-tags.log 2>&1 &

# Smoke-test a stage with reduced N: temporarily lower TARGET_GAME_COUNT or
# FETCH_OVERSHOOT_COUNT in pipeline/config.py, run the stage, restore.
```

## Resumability and atomic writes

Stages 01 (SteamSpy), 04 (Haiku summary), and 07 (Haiku facet labels) are resumable: rerunning skips rows already present in their output. All three checkpoint every N rows (100 for 01, 200 for 04 and 07) via atomic tmp+rename, so a kill mid-run leaves a consistent partial file.

Stage 02 backs up `games.parquet` to `games_pretrim.parquet` before trimming. Stage 04 writes a `.bak` copy of `games.parquet` before its first batch.

Treat `data/*.parquet` as expensive to regenerate. Never overwrite without a tmp+rename or a backup; the SteamSpy crawl alone is 3+ hours.

## PROJECT_NAME

`PROJECT_NAME` and `PROJECT_TAGLINE` in `pipeline/config.py` are the single source of truth for the user-facing name. The repo directory is `steam-map` as a working name; renaming later should be a one-line change in `config.py`, not a sweep across files. Don't bake the project name into multiple identifiers.

## Required env vars

In `.env` (loaded by `python-dotenv` in `config.py`):

- `ANTHROPIC_API_KEY`: stages 04, 07, 08, and `design_facets.py`
- `CO_API_KEY`: stages 05, 08, and `design_facets.py`

Stages 00, 01, 02, 03, 06, 09 need no external auth (FronkonGames is a public HF parquet, SteamSpy is unauthenticated).

## Stage 09 (visualize) is currently minimum-viable

Stage 09 is roughly 250 lines vs github-map's 1400+. It has: capsule images on hover, Toponymy region labels at multiple zoom levels, click-to-Steam, search by name, and 4+ colormaps (sentiment, genre, F2P, review count, plus one per facet when stage 07 has run).

Not yet ported from github-map: mobile-specific UI, custom filter panel beyond DataMapPlot's built-in colormap dropdown, edge-bundling background image, per-point text labels at zoom, hand-authored About page (with `<!-- DATA_AS_OF -->` placeholder pattern), Open Graph / social-preview tags, Plausible analytics. `../semantic-github-map/pipeline/07_visualize.py` is the reference when adding these.

## Sibling repos

- `../semantic-github-map/`: the parallel project for GitHub repos. Same skeleton; the atomic-write, resumability, two-phase fetch, Toponymy + DataMapPlot wiring patterns were lifted directly from there. Note that github-map hand-authors its facet schema (`PROJECT_TYPES`, `TARGET_AUDIENCES` constants in `03_summarize_readmes.py`) and folds per-game labeling into stage 03's existing summary call; we instead discover the schema via clustering (`design_facets.py`) and label in a dedicated stage 07.
