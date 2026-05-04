"""Render the interactive 2D Steam map via DataMapPlot.

Composes UMAP coords + Toponymy region labels + raw FronkonGames metadata + per-game
facet labels into an interactive HTML page with capsule images on hover, multiple
colormap dropdowns, and click-to-open-Steam-store.

This is a v1 minimum-viable version. Polish (edge bundling, mobile-specific UI, custom
filter panel, point-level text labels at zoom, hand-authored About page) comes in later
iterations once the basic map is verified.
"""

import json
import re
from html import escape
from pathlib import Path

import datamapplot
import glasbey
import numpy as np
import pandas as pd
from config import (
    DOCS_INDEX_HTML,
    FACETS_PARQUET,
    FACETS_SCHEMA_JSON,
    GAMES_PARQUET,
    LABELS_PARQUET,
    PROJECT_NAME,
    PROJECT_TAGLINE,
    STEAM_ATLAS_HTML,
    UMAP_COORDS_NPZ,
)


def _facet_slug(facet_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", facet_name).strip("_").lower()


# Tiled SVG turbulence used as a film-grain overlay on body::before. Rendered
# inline as a data URI so there's no external asset; opacity and blend mode are
# tuned in the body::before rule.
_GRAIN_SVG_DATA_URI = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 240 240'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 0.06 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>"  # noqa: E501


# ColorBrewer 9-class RdYlGn (diverging). Aligns with Steam's green-for-positive
# convention; pale yellow midpoint at Mixed ramps out to saturated red at
# Overwhelmingly Negative and saturated dark green at Overwhelmingly Positive.
# Adjacent tiers stay adjacent in color space (no jumps between Mostly Positive
# and Mixed).
#
# Dict order is the legend order (most positive at top); preserve it.
SENTIMENT_COLORS = {
    "Overwhelmingly Positive": "#1a9850",  # dark green
    "Very Positive": "#66bd63",  # mid green
    "Positive": "#a6d96a",  # light green
    "Mostly Positive": "#d9ef8b",  # pale yellow-green
    "Mixed": "#ffffbf",  # pale yellow (midpoint)
    "Mostly Negative": "#fee08b",  # pale yellow-orange
    "Negative": "#fdae61",  # orange
    "Very Negative": "#f46d43",  # red-orange
    "Overwhelmingly Negative": "#d73027",  # red
    "Too Few Reviews": "#7a8190",
    "No User Reviews": "#5a6172",
}


def _format_price(row) -> str:
    price = row.get("price")
    if price is None or pd.isna(price) or price == 0:
        return "Free"
    return f"${price:.2f}"


def _format_reviews(n) -> str:
    n = int(n or 0)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _store_url(appid) -> str:
    return f"https://store.steampowered.com/app/{int(appid)}/"


# SteamSpy "Early Access" votes accumulate during a game's EA period and don't
# decay after release, so a game shipped years ago can still surface it as a
# top tag. FronkonGames is also ~3 months stale. Strip it from chip selection.
TAG_CHIP_STOPLIST = {"Early Access"}
N_TAG_CHIPS = 3


def _top_tag_chips(tags_dict) -> str:
    """Render the top-N SteamSpy tags as chip <span>s, with stoplist applied."""
    if not isinstance(tags_dict, dict):
        return ""
    counted = [(k, v) for k, v in tags_dict.items() if isinstance(v, (int, float)) and v is not None]
    counted.sort(key=lambda x: -x[1])
    chosen: list[str] = []
    for tag, _ in counted:
        if tag in TAG_CHIP_STOPLIST:
            continue
        chosen.append(tag)
        if len(chosen) >= N_TAG_CHIPS:
            break
    return "".join(f'<span class="hc-tag">{escape(t)}</span>' for t in chosen)


def main():
    df = pd.read_parquet(GAMES_PARQUET)
    coords = np.load(UMAP_COORDS_NPZ)["coords"]
    labels_df = pd.read_parquet(LABELS_PARQUET)
    print(f"Loaded {len(df):,} games, coords {coords.shape}")

    # FronkonGames stores appid as string; downstream stages cast to int.
    # Normalize on the merge key so joins don't trip on dtype mismatch.
    df["appid"] = df["appid"].astype(int)
    labels_df["appid"] = labels_df["appid"].astype(int)

    df = df.merge(labels_df, on="appid", how="left")

    if FACETS_PARQUET.exists():
        facets_df = pd.read_parquet(FACETS_PARQUET)
        facets_df["appid"] = facets_df["appid"].astype(int)
        df = df.merge(facets_df, on="appid", how="left")
        print(f"  + Facet colormaps: {[c for c in facets_df.columns if c != 'appid']}")
    else:
        facets_df = None
        print("  (no facets.parquet; skipping facet colormaps)")

    label_columns = sorted(c for c in df.columns if c.startswith("label_layer_"))
    topic_name_vectors = [df[c].fillna("").values for c in label_columns]

    hover_text = df["name"].tolist()

    name_col = df["name"].fillna("").apply(escape).values
    tagline_col = df.get("tagline", pd.Series([""] * len(df))).fillna("").apply(escape).values
    summary_col = df.get("summary", pd.Series([""] * len(df))).fillna("").apply(escape).values
    sentiment_col = df.get("sentiment_label", pd.Series(["Unknown"] * len(df))).fillna("Unknown").values
    sentiment_color_col = np.array([SENTIMENT_COLORS.get(s, "#999999") for s in sentiment_col])

    primary_genre_col = (
        df["genres"]
        .apply(lambda g: str(g[0]) if isinstance(g, (list, np.ndarray)) and len(g) else "Unknown")
        .apply(escape)
        .values
    )

    review_str_col = np.array([_format_reviews(n) for n in df["total_reviews"]])
    price_str_col = np.array([_format_price(row) for _, row in df.iterrows()])
    header_image_col = df["header_image"].fillna("").values
    tag_chips_col = df["tags"].apply(_top_tag_chips).values

    appids = df["appid"].astype(int).values
    store_urls = np.array([_store_url(a) for a in appids])

    # Marker sizes: log10(reviews), normalized. Log puts equal visual weight on
    # each order of magnitude, which matches Steam's log-normal review distribution
    # better than sqrt (which collapsed everything but CS2 to the floor).
    raw = np.log10(df["total_reviews"].values.astype(float).clip(min=1))
    marker_sizes = 3 + 12 * (raw - raw.min()) / max(raw.max() - raw.min(), 1)

    # Sentiment chip uses the per-row sentiment_color twice: as a glowing tinted
    # background (~20% opacity) and as the foreground text/border. Gives each chip
    # a luminous "this is the sentiment" feel against the dark card.
    hover_template = (
        '<div class="hc">'
        '  <div class="hc-img-wrap">'
        '    <img src="{header_image}" class="hc-img" alt="" />'
        "  </div>"
        '  <div class="hc-body">'
        '    <div class="hc-title">{name}</div>'
        '    <div class="hc-tagline">{tagline}</div>'
        '    <div class="hc-classify">'
        '      <span class="hc-sentiment" '
        'style="background:{sentiment_color}33; color:{sentiment_color}; '
        'border-color:{sentiment_color}66">{sentiment}</span>'
        '      <span class="hc-chip">{primary_genre}</span>'
        '      <span class="hc-chip hc-chip-num">{review_str}</span>'
        '      <span class="hc-chip hc-chip-num">{price_str}</span>'
        "    </div>"
        '    <div class="hc-tags">{tag_chips}</div>'
        '    <div class="hc-summary">{summary}</div>'
        "  </div>"
        "</div>"
    )

    extra_data = pd.DataFrame(
        {
            "name": name_col,
            "tagline": tagline_col,
            "summary": summary_col,
            "sentiment": sentiment_col,
            "sentiment_color": sentiment_color_col,
            "primary_genre": primary_genre_col,
            "review_str": review_str_col,
            "price_str": price_str_col,
            "header_image": header_image_col,
            "store_url": store_urls,
            "tag_chips": tag_chips_col,
        }
    )

    all_rawdata = []
    all_metadata = []

    # Sentiment legend order follows SENTIMENT_COLORS (most positive -> most negative).
    # Any unexpected values land at the end alphabetically.
    sentiments_in_data = set(sentiment_col)
    unique_sentiments = [s for s in SENTIMENT_COLORS if s in sentiments_in_data]
    unique_sentiments += sorted(sentiments_in_data - set(SENTIMENT_COLORS))
    sentiment_color_map = {s: SENTIMENT_COLORS.get(s, "#999999") for s in unique_sentiments}
    all_rawdata.append(sentiment_col)
    all_metadata.append(
        {
            "field": "sentiment",
            "description": "Review Sentiment",
            "kind": "categorical",
            "color_mapping": sentiment_color_map,
        }
    )

    # Primary genre (top 10 + Other)
    top_genres = pd.Series(primary_genre_col).value_counts().head(10).index.tolist()
    genres_capped = np.where(np.isin(primary_genre_col, top_genres), primary_genre_col, "Other")
    unique_genres = sorted(set(genres_capped))
    genre_palette = glasbey.create_palette(palette_size=len(unique_genres))
    genre_map = dict(zip(unique_genres, genre_palette))
    all_rawdata.append(genres_capped)
    all_metadata.append(
        {
            "field": "primary_genre",
            "description": "Primary Genre (Steam)",
            "kind": "categorical",
            "color_mapping": genre_map,
        }
    )

    # Free vs Paid (FronkonGames stores price as float; 0 means F2P)
    is_free_str = np.where(df["price"].fillna(0) == 0, "Free", "Paid")
    all_rawdata.append(is_free_str)
    all_metadata.append(
        {
            "field": "free_or_paid",
            "description": "Price Tier",
            "kind": "categorical",
            "color_mapping": {"Free": "#4caf50", "Paid": "#5c6bc0"},
        }
    )

    # Review count (continuous, log10)
    review_log = np.log10(df["total_reviews"].astype(float).clip(lower=1))
    all_rawdata.append(review_log)
    all_metadata.append(
        {
            "field": "reviews",
            "description": "Review Count (log10)",
            "kind": "continuous",
            "cmap": "viridis",
        }
    )

    # Per-game facets, one colormap each (skipped when facets.parquet absent).
    # The _other suffixed columns hold freeform labels for games in the Other
    # bucket; they are debug/iteration data, not categorical colormaps.
    # Field names are prefixed with `facet_` to avoid colliding with the
    # built-in colormaps above (e.g. the facet schema may rediscover Primary Genre).
    # Descriptions come from the schema's facet name when available so the dropdown
    # shows "Primary Genre" rather than the slug-derived "Primary Genre".
    facet_cols = (
        [c for c in facets_df.columns if c != "appid" and not c.endswith("_other")] if facets_df is not None else []
    )
    if facet_cols and FACETS_SCHEMA_JSON.exists():
        with open(FACETS_SCHEMA_JSON) as f:
            schema = json.load(f)
        slug_to_name = {_facet_slug(facet["name"]): facet["name"] for facet in schema}
    else:
        slug_to_name = {}
    for col in facet_cols:
        values = df[col].fillna("Other").astype(str).values
        unique_vals = sorted(set(values))
        palette = glasbey.create_palette(palette_size=len(unique_vals))
        cmap = dict(zip(unique_vals, palette))
        all_rawdata.append(values)
        all_metadata.append(
            {
                "field": f"facet_{col}",
                "description": slug_to_name.get(col, col.replace("_", " ").title()),
                "kind": "categorical",
                "color_mapping": cmap,
            }
        )

    # The .deck-tooltip rule body comes from this string; the rest of the chrome
    # is styled in custom_css below (kept together so the design tokens stay
    # in one place). We zero out the wrapper here so our .deck-tooltip rule in
    # custom_css fully owns the tooltip's container styling.
    tooltip_css = """
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    """

    # ── Design tokens (tactical-atlas) ──────────────────────────────────────
    # Page is a deep ink-navy with subtle radial light + film grain. UI panels
    # are translucent dark with hairline borders. Brass is the primary accent
    # (cartouche, rules, taglines); cyan is the cool interactive accent (focus,
    # store CTA glow). Sharper corners (4px) and tighter typographic rhythm
    # than the default DataMapPlot chrome to read more "instrument panel" and
    # less "dashboard widget".
    custom_css = r"""
    @import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;700;800;900&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;600&family=Fraunces:ital,opsz,wght@1,12..72,400;1,12..72,500&display=swap');

    :root {
        --ink: #0a0e15;
        --ink-2: #0f141d;
        --ink-3: #161c27;
        --ink-4: #1f2735;
        --rule: rgba(255, 255, 255, 0.08);
        --rule-strong: rgba(255, 255, 255, 0.16);
        --text: #e3e8f0;
        --text-dim: #97a0b3;
        --text-faint: #5d6678;
        --brass: #d8a657;
        --brass-soft: #b8893f;
        --brass-glow: rgba(216, 166, 87, 0.22);
        --cyan: #5fb3a1;
        --cyan-glow: rgba(95, 179, 161, 0.30);
    }

    /* ── Page atmosphere ────────────────────────────────────────────────── */
    body {
        background:
          radial-gradient(ellipse 90% 60% at 50% 30%, rgba(78, 102, 138, 0.18), transparent 70%),
          radial-gradient(ellipse 60% 60% at 85% 85%, rgba(95, 179, 161, 0.05), transparent 60%),
          var(--ink) !important;
        color: var(--text) !important;
        font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
    }
    /* Film grain overlay — a tiled SVG turbulence at low opacity. Sits behind
       the deck.gl canvas (which has z-index: 1) so it doesn't muddy the points,
       but adds tooth to the otherwise-flat dark fields. */
    body::before {
        content: "";
        position: fixed; inset: 0;
        pointer-events: none;
        z-index: 0;
        background-image: url("__GRAIN_SVG_DATA_URI__");
        opacity: 0.7;
        mix-blend-mode: screen;
    }

    /* ── Container chrome (panels, dropdowns, search) ──────────────────── */
    .container-box {
        background: rgba(15, 20, 29, 0.82) !important;
        border: 1px solid var(--rule) !important;
        backdrop-filter: blur(12px) saturate(140%);
        -webkit-backdrop-filter: blur(12px) saturate(140%);
        box-shadow:
            0 10px 28px rgba(0, 0, 0, 0.55),
            0 1px 0 rgba(255, 255, 255, 0.04) inset !important;
        border-radius: 4px !important;
    }
    .more-opaque {
        background-color: rgba(15, 20, 29, 0.94) !important;
    }

    /* ── Title cartouche ───────────────────────────────────────────────── */
    #title-container {
        margin: 18px !important;
        padding: 20px 24px 18px !important;
        max-width: 460px;
        line-height: 1.2 !important;
    }
    #main-title {
        display: block !important;
        font-family: 'Big Shoulders Display', sans-serif !important;
        font-size: 46pt !important;
        font-weight: 800 !important;
        line-height: 0.92 !important;
        letter-spacing: 0.025em !important;
        color: var(--text) !important;
        text-transform: uppercase;
    }
    /* Hairline rule under the title, in brass — the cartouche divider. */
    #main-title::after {
        content: "";
        display: block;
        width: 72px;
        height: 1px;
        background: var(--brass);
        margin: 14px 0 12px;
    }
    /* Sub-title is the second span inside #title-container. We hide the
       intervening <br> since the ::after rule handles the break. */
    #title-container > br { display: none !important; }
    #title-container > span:last-of-type {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 12.5pt !important;
        font-weight: 400 !important;
        color: var(--text-dim) !important;
        letter-spacing: 0.005em !important;
        line-height: 1.4 !important;
        display: block;
    }

    /* ── Search ─────────────────────────────────────────────────────────── */
    #search-container {
        padding: 8px 10px !important;
    }
    #text-search {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        color: var(--text) !important;
        background: var(--ink-3) !important;
        border: 1px solid var(--rule-strong) !important;
        border-radius: 3px !important;
        padding: 8px 12px !important;
        width: 240px !important;
        outline: none !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    #text-search::placeholder {
        color: var(--text-faint) !important;
        font-weight: 400;
    }
    #text-search:focus {
        border-color: var(--cyan) !important;
        box-shadow: 0 0 0 3px var(--cyan-glow) !important;
    }
    /* The Chrome/Safari search-cancel "x" — invert so it shows on dark. */
    #text-search::-webkit-search-cancel-button { filter: invert(0.85); }

    /* ── Colormap selector ─────────────────────────────────────────────── */
    #colormap-selector-container {
        padding: 10px 12px !important;
    }
    #selectedColorMapText {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 12.5px !important;
        font-weight: 500 !important;
        color: var(--text) !important;
        letter-spacing: 0.01em;
    }
    #colorMapOptions {
        background: var(--ink-2) !important;
        border: 1px solid var(--rule) !important;
        border-radius: 4px !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6) !important;
        padding: 4px !important;
    }
    .color-map-text {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 12px !important;
        color: var(--text-dim) !important;
        padding: 6px 10px !important;
        transition: color 0.1s;
    }
    /* DataMapPlot's row-level hover already paints a subtle background on
       the whole option (palette + label). We only nudge the label color so
       it brightens slightly — no second background, which read as a confusing
       "smaller redundant box" inside the row hover. */
    .color-map-text:hover {
        color: var(--text) !important;
    }

    /* ── Legend ─────────────────────────────────────────────────────────── */
    #legend-container {
        padding: 12px 14px !important;
        max-height: 60vh;
        overflow-y: auto;
    }
    #legend-container::-webkit-scrollbar { width: 6px; }
    #legend-container::-webkit-scrollbar-thumb {
        background: var(--rule-strong);
        border-radius: 3px;
    }
    /* No gap between swatch elements — adjacent squares should TOUCH so the
       palette reads as a single rectangular bar rather than five disconnected
       dots. The label gets explicit spacing via .color-map-text below. */
    .color-swatch {
        color: var(--text-dim) !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 11.5px !important;
        font-weight: 400 !important;
        padding: 3px 0 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0 !important;
    }
    .color-swatch:hover { color: var(--text) !important; }
    /* Sharp-cornered squares match DataMapPlot's default. In the colormap-
       trigger row and legend list, adjacent squares (gap: 0 above) read as
       a continuous palette bar — a single visual unit you can scan at a
       glance. Continuous colormaps like Review Count render as a smooth
       gradient. Circles or any spacing weakens the "this is a palette"
       affordance. Sharp corners also fit the cartographic instrument-panel
       feel of the surrounding chrome. */
    .color-swatch-box {
        border-radius: 0 !important;
        width: 10px !important;
        height: 10px !important;
        flex-shrink: 0;
    }

    /* ── Hover card ────────────────────────────────────────────────────── */
    /* The card sits inside .deck-tooltip (we zeroed its wrapper styles via
       tooltip_css above). The card itself paints the dark bg, hairline border,
       and shadow — this lets the rounded-corner image clip cleanly. */
    .deck-tooltip {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        max-width: 360px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    .hc {
        background: var(--ink-2);
        border: 1px solid var(--rule-strong);
        border-radius: 4px;
        overflow: hidden;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.7);
    }
    .hc-img-wrap {
        position: relative;
        background: var(--ink);
    }
    .hc-img {
        width: 100%;
        height: auto;
        aspect-ratio: 460 / 215;
        object-fit: cover;
        display: block;
    }
    /* Subtle vignette at bottom of capsule so title sits cleanly when capsule
       art is bright at its bottom edge. */
    .hc-img-wrap::after {
        content: "";
        position: absolute;
        left: 0; right: 0; bottom: 0;
        height: 40%;
        background: linear-gradient(to bottom, transparent, rgba(15, 20, 29, 0.55));
        pointer-events: none;
    }
    .hc-body { padding: 14px 16px 16px; }
    .hc-title {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        font-size: 15.5px;
        line-height: 1.25;
        letter-spacing: -0.005em;
        color: var(--text);
    }
    .hc-tagline {
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-weight: 400;
        font-size: 13.5px;
        line-height: 1.4;
        color: var(--brass);
        margin-top: 6px;
    }
    .hc-tagline:empty { display: none; }
    .hc-classify {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin-top: 12px;
    }
    .hc-sentiment, .hc-chip {
        display: inline-flex;
        align-items: center;
        padding: 3px 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border-radius: 2px;
        border: 1px solid transparent;
        white-space: nowrap;
    }
    .hc-sentiment {
        /* bg/color/border-color come inline from the hover_template; we just
           shape the chip here. The 1px transparent border above gets recolored
           to a translucent variant of the sentiment color. */
    }
    .hc-chip {
        background: var(--ink-4);
        color: var(--text-dim);
        border-color: var(--rule);
    }
    .hc-chip-num {
        font-variant-numeric: tabular-nums;
        color: var(--text);
    }
    /* SteamSpy top-tag chips. Visually quieter than the metadata chips above
       (no fill, dimmer text) so they read as user-applied descriptors rather
       than facts about the game. */
    .hc-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin-top: 8px;
    }
    .hc-tags:empty { display: none; }
    .hc-tag {
        display: inline-flex;
        align-items: center;
        padding: 3px 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border-radius: 2px;
        background: transparent;
        color: var(--text-faint);
        border: 1px solid var(--rule);
        white-space: nowrap;
    }
    .hc-summary {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 12.5px;
        font-weight: 400;
        line-height: 1.5;
        color: var(--text-dim);
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid var(--rule);
        display: -webkit-box;
        -webkit-line-clamp: 7;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .hc-summary:empty { display: none; }

    /* ── Loading ────────────────────────────────────────────────────────── */
    #loading {
        background: var(--ink) !important;
        color: var(--text) !important;
    }
    .datamapplot-progress-bar {
        background: var(--ink-3) !important;
    }
    .datamapplot-progress-bar-fill {
        background: var(--cyan) !important;
    }
    .datamapplot-progress-bar-text {
        color: var(--text-dim) !important;
    }
    """.replace("__GRAIN_SVG_DATA_URI__", _GRAIN_SVG_DATA_URI)

    # Speed up scroll-zoom — deck.gl's default (0.01) is sluggish for an
    # exploratory atlas where users want to dive in and out fast.
    #
    # KNOWN ISSUE — region labels invisible on initial page load:
    # DataMapPlot's bundled datamap.js calls waitForFont() but does NOT await
    # its Promise (line 627), so the labelLayer's SDF GPU texture is built
    # before the WebFont finishes loading, leaving an empty atlas. The atlas
    # mapping is correct but the GPU texture upload never happens, and no
    # amount of script-triggered cloning, addLabels-recreation, setProps, or
    # explicit redraw rebuilds it. The same code DOES rebuild the texture
    # when run from the JS devtools console after the page is fully idle
    # (probably a Chrome user-gesture or rendering-loop interaction we don't
    # understand). The source-level patches below (font preload + explicit
    # characterSet) cleanly fix two related bugs in the same area, but the
    # GPU texture race needs an upstream fix in DataMapPlot/deck.gl.
    # Workaround for users: paste this into the console after page load:
    #   const ll=datamap.labelLayer,d=ll.props.data,i=datamap.layers.indexOf(ll);
    #   const n=ll.clone({id:'lbl',data:[...d]});datamap.layers[i]=n;
    #   datamap.labelLayer=n;datamap.deckgl.setProps({layers:[...datamap.layers]});
    custom_js = r"""
    if (typeof datamap !== 'undefined' && datamap.deckgl) {
        datamap.deckgl.setProps({controller: {scrollZoom: {speed: 0.05, smooth: true}}});
    }
    """

    fig = datamapplot.create_interactive_plot(
        coords,
        *topic_name_vectors,
        hover_text=hover_text,
        hover_text_html_template=hover_template,
        marker_size_array=marker_sizes,
        extra_point_data=extra_data,
        on_click="window.open(`{store_url}`, '_blank')",
        colormap_rawdata=all_rawdata,
        colormap_metadata=all_metadata,
        title=PROJECT_NAME,
        # The "top 10,000" framing already lives in PROJECT_TAGLINE; the
        # About page (TODO) will explain what "top 10,000" means precisely
        # (most-reviewed via FronkonGames). Keeping the cartouche subtitle
        # short keeps it to one line at the cartouche width.
        sub_title=PROJECT_TAGLINE,
        enable_search=True,
        custom_css=custom_css,
        custom_js=custom_js,
        tooltip_css=tooltip_css,
        # Body / UI font. Title font is overridden in custom_css to Big Shoulders.
        font_family="IBM Plex Sans",
        tooltip_font_family="IBM Plex Sans",
        title_font_size=46,
        sub_title_font_size=13,
        font_weight=800,
        # Dark theme + ink-navy (not pure black) for atmosphere.
        darkmode=True,
        background_color="#0a0e15",
    )
    fig.save(str(STEAM_ATLAS_HTML))
    print(f"Saved interactive map to {STEAM_ATLAS_HTML}")

    # Two post-render HTML patches; both work around bugs in the bundled
    # DataMapPlot/deck.gl:
    #
    # 1. characterSet:"auto" is treated as the literal 4-char set
    #    ['a','u','t','o']. Replace with the explicit set computed from the
    #    label data so every glyph in the region labels can render.
    # 2. The bundled JS calls waitForFont() but does NOT await it, so the
    #    SDF font atlas can build before the Google Font finishes loading
    #    and ends up empty. Move the @import block to a <link> in <head>
    #    so the font starts loading at parse time, BEFORE deck.gl runs.
    chars = sorted({c for v in topic_name_vectors for s in v for c in str(s)})
    explicit_charset = "characterSet:" + json.dumps(chars, ensure_ascii=False)
    html = Path(STEAM_ATLAS_HTML).read_text()
    if 'characterSet:"auto"' not in html:
        raise RuntimeError(
            "characterSet:'auto' marker not found in rendered HTML. The "
            "DataMapPlot template may have changed; update the patch."
        )
    html = html.replace('characterSet:"auto"', explicit_charset, 1)

    # Inject font preload + stylesheet links into <head> so the WebFont
    # finishes loading before the deck.gl bundle starts building its SDF
    # font atlas. Without this, the atlas is built with a fallback font
    # (or no font at all) and region labels never appear.
    font_url = (
        "https://fonts.googleapis.com/css2?"
        "family=Big+Shoulders+Display:wght@600;700;800;900&"
        "family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&"
        "family=JetBrains+Mono:wght@400;500;600&"
        "family=Fraunces:ital,opsz,wght@1,12..72,400;1,12..72,500&"
        "display=block"  # block: wait up to 3s for font, no fallback
    )
    head_inject = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        f'<link rel="stylesheet" href="{font_url}">\n'
    )
    html = html.replace("</head>", head_inject + "</head>", 1)

    Path(STEAM_ATLAS_HTML).write_text(html)
    print(f"Patched labelLayer characterSet ({len(chars)} chars) + font preload")

    DOCS_INDEX_HTML.write_text(html)
    print(f"Copied to {DOCS_INDEX_HTML}")


if __name__ == "__main__":
    main()
