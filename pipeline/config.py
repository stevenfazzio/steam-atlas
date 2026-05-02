"""Shared paths, constants, and env var loading."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Project naming ───────────────────────────────────────────────────────────
# Single source of truth for the user-facing name. Tweak here when we rename.
PROJECT_NAME = "Steam Map"
PROJECT_TAGLINE = "A semantic map of the top games on Steam"

# ── Directories ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

# ── File paths ───────────────────────────────────────────────────────────────
CANDIDATES_PARQUET = DATA_DIR / "candidates.parquet"
GAMES_PARQUET = DATA_DIR / "games.parquet"
GAMES_PRETRIM_PARQUET = DATA_DIR / "games_pretrim.parquet"
EMBEDDINGS_NPZ = DATA_DIR / "embeddings.npz"
UMAP_COORDS_NPZ = DATA_DIR / "umap_coords.npz"
FACETS_PARQUET = DATA_DIR / "facets.parquet"
TOPONYMY_MODEL_JOBLIB = DATA_DIR / "toponymy_model.joblib"

# Facet schema lives next to the pipeline scripts, not in DATA_DIR. It is committed
# to the repo and serves as the contract between design_facets.py (writes) and
# stage 07 (reads). Editing this path means breaking that contract.
FACETS_SCHEMA_JSON = Path(__file__).parent / "facets_schema.json"
LABELS_PARQUET = DATA_DIR / "labels.parquet"
STEAM_MAP_HTML = DATA_DIR / "steam_map.html"
ABOUT_HTML = DATA_DIR / "about.html"

DOCS_INDEX_HTML = DOCS_DIR / "index.html"
DOCS_ABOUT_HTML = DOCS_DIR / "about.html"

# ── API keys ─────────────────────────────────────────────────────────────────
CO_API_KEY = os.environ.get("CO_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Pipeline constants ───────────────────────────────────────────────────────
TARGET_GAME_COUNT = 10_000
FETCH_OVERSHOOT_COUNT = 12_000  # buffer for delisted / non-game / failed fetches
CANDIDATE_OVERSHOOT_COUNT = 25_000  # how many top-by-reviews to keep from FronkonGames

# ── FronkonGames Hugging Face dataset ────────────────────────────────────────
FRONKONGAMES_REPO_ID = "FronkonGames/steam-games-dataset"
FRONKONGAMES_FILENAME = "data/train-00000-of-00001.parquet"

# ── Steam storefront API ─────────────────────────────────────────────────────
# Rate limit is undocumented but widely observed at ~200 req / 5 min per IP
# (≈ 40/min). We pace at ~37/min for safety.
STEAM_API_BASE = "https://store.steampowered.com/api"
STEAM_REQUEST_DELAY_SEC = 1.6
STEAM_RETRY_BACKOFF_SEC = 60
STEAM_MAX_RETRIES = 5
STEAM_USER_AGENT = "steam-map/0.1 (+https://github.com/stevenfazzio/steam-map)"
STEAM_REGION_CC = "us"
STEAM_LANG = "english"

# ── Cohere embeddings ────────────────────────────────────────────────────────
COHERE_BATCH_SIZE = 96
COHERE_EMBED_DIMENSION = 512
COHERE_EMBED_MODEL = "embed-v4.0"

# ── Anthropic models ─────────────────────────────────────────────────────────
ANTHROPIC_MODEL_SUMMARIZE = "claude-haiku-4-5"
ANTHROPIC_MODEL_NAMING = "claude-sonnet-4-6"  # Toponymy region naming
ANTHROPIC_CONCURRENCY = 30

# ── Facet design and labeling ────────────────────────────────────────────────
# Design is one-shot (Opus, expensive but worth it for schema quality).
# Labeling is per-game (Haiku, ~10K calls so cost matters).
FACET_DESIGN_MODEL = "claude-opus-4-7"
FACET_LABELING_MODEL = "claude-haiku-4-5"
