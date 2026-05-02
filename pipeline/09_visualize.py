"""Render the interactive 2D Steam map via DataMapPlot.

Composes UMAP coords + Toponymy region labels + raw FronkonGames metadata + Typologist
facets into an interactive HTML page with capsule images on hover, multiple colormap
dropdowns, and click-to-open-Steam-store.

This is a v1 minimum-viable version. Polish (edge bundling, mobile-specific UI, custom
filter panel, point-level text labels at zoom, hand-authored About page) comes in later
iterations once the basic map is verified.
"""

from html import escape
from pathlib import Path

import datamapplot
import glasbey
import numpy as np
import pandas as pd
from config import (
    DOCS_INDEX_HTML,
    FACETS_PARQUET,
    GAMES_PARQUET,
    LABELS_PARQUET,
    PROJECT_NAME,
    PROJECT_TAGLINE,
    STEAM_MAP_HTML,
    TARGET_GAME_COUNT,
    UMAP_COORDS_NPZ,
)

SENTIMENT_COLORS = {
    "Overwhelmingly Positive": "#1f8b4c",
    "Very Positive": "#3aa15b",
    "Mostly Positive": "#7bc370",
    "Positive": "#a5d99a",
    "Mixed": "#dabd57",
    "Mostly Negative": "#d99272",
    "Negative": "#c25656",
    "Very Negative": "#a83a3a",
    "Overwhelmingly Negative": "#7c1f1f",
    "Too Few Reviews": "#999999",
    "No User Reviews": "#bbbbbb",
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
        print(f"  + Typologist facets: {[c for c in facets_df.columns if c != 'appid']}")
    else:
        facets_df = None
        print("  (no facets.parquet; skipping Typologist colormaps)")

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

    appids = df["appid"].astype(int).values
    store_urls = np.array([_store_url(a) for a in appids])

    # Marker sizes: sqrt(reviews), normalized
    raw = np.sqrt(df["total_reviews"].values.astype(float).clip(min=1))
    marker_sizes = 3 + 12 * (raw - raw.min()) / max(raw.max() - raw.min(), 1)

    hover_template = (
        '<div class="hc">'
        '  <img src="{header_image}" class="hc-img" alt="" />'
        '  <div class="hc-body">'
        '    <div class="hc-title">{name}</div>'
        '    <div class="hc-tagline">{tagline}</div>'
        '    <div class="hc-classify">'
        '      <span class="hc-sentiment" '
        'style="background:{sentiment_color}1f; color:{sentiment_color}">{sentiment}</span>'
        '      <span class="hc-chip">{primary_genre}</span>'
        '      <span class="hc-chip">{review_str} reviews</span>'
        '      <span class="hc-chip">{price_str}</span>'
        "    </div>"
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
        }
    )

    all_rawdata = []
    all_metadata = []

    # Sentiment
    unique_sentiments = sorted(set(sentiment_col), key=lambda s: (s not in SENTIMENT_COLORS, s))
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
            "description": "Primary Genre",
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

    # Typologist facets, one colormap each (skipped when facets.parquet absent)
    facet_cols = [c for c in facets_df.columns if c != "appid"] if facets_df is not None else []
    for col in facet_cols:
        values = df[col].fillna("Other").astype(str).values
        unique_vals = sorted(set(values))
        palette = glasbey.create_palette(palette_size=len(unique_vals))
        cmap = dict(zip(unique_vals, palette))
        all_rawdata.append(values)
        all_metadata.append(
            {
                "field": col,
                "description": col.replace("_", " ").title(),
                "kind": "categorical",
                "color_mapping": cmap,
            }
        )

    tooltip_css = """
        font-family: 'IBM Plex Sans', system-ui, sans-serif;
        font-size: 13px;
        color: #1a1a2e !important;
        background: #ffffff !important;
        border: 1px solid rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        max-width: 360px;
        padding: 0 !important;
        overflow: hidden;
    """
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    .hc { display: flex; flex-direction: column; }
    .hc-img {
        width: 100%; height: auto; display: block;
        aspect-ratio: 460/215; object-fit: cover;
    }
    .hc-body { padding: 12px 14px; }
    .hc-title { font-weight: 600; font-size: 15px; color: #0d1117; line-height: 1.3; }
    .hc-tagline { font-size: 13px; font-style: italic; color: #656d76; margin-top: 4px; }
    .hc-tagline:empty { display: none; }
    .hc-classify { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .hc-sentiment, .hc-chip {
        display: inline-flex; align-items: center;
        padding: 2px 8px; border-radius: 6px;
        font-size: 11px; font-weight: 500;
    }
    .hc-chip { background: rgba(0, 0, 0, 0.05); color: #424a53; }
    .hc-summary {
        font-size: 12px; line-height: 1.5;
        color: #3d4752;
        margin-top: 10px;
        border-top: 1px solid rgba(0, 0, 0, 0.06);
        padding-top: 8px;
        display: -webkit-box; -webkit-line-clamp: 8; -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .hc-summary:empty { display: none; }
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
        sub_title=f"{PROJECT_TAGLINE} (top {TARGET_GAME_COUNT:,} most-reviewed)",
        enable_search=True,
        custom_css=custom_css,
        tooltip_css=tooltip_css,
        font_family="IBM Plex Sans",
        darkmode=False,
    )
    fig.save(str(STEAM_MAP_HTML))
    print(f"Saved interactive map to {STEAM_MAP_HTML}")

    docs_html = Path(STEAM_MAP_HTML).read_text()
    DOCS_INDEX_HTML.write_text(docs_html)
    print(f"Copied to {DOCS_INDEX_HTML}")


if __name__ == "__main__":
    main()
