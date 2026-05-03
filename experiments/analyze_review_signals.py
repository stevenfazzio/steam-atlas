"""Internal exploratory analysis: how do quality signals scale with review count?

Three questions this answers:

1. Where does the binomial noise on positive_rate make sentiment unreliable?
2. Which metadata fields (presence, cardinality) correlate with review count,
   and how strongly?
3. Is there a bimodal "shovelware vs. real games" split in a composite polish
   score, or is the catalog a smooth continuum?

Renders a single HTML to data/review_signals.html. Not part of the pipeline;
not shipped to docs/. Run on demand:

    uv run python experiments/analyze_review_signals.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from config import FRONKONGAMES_FILENAME, FRONKONGAMES_REPO_ID, TARGET_GAME_COUNT  # noqa: E402

OUTPUT_HTML = REPO_ROOT / "data" / "review_signals.html"

# Bucket edges shared across plots. Logarithmic, since the catalog spans 7 orders
# of magnitude in review count.
BUCKET_EDGES = [10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 1e9]
BUCKET_LABELS = ["10-50", "50-100", "100-500", "500-1k", "1k-5k", "5k-10k", "10k-50k", "50k+"]

# Six "polish" signals selected from the pre-check by Spearman rank correlation
# with review count (all rho >= 0.22). Mix of cardinality, length, and presence.
POLISH_SIGNALS = [
    ("n_languages", "Supported languages (count)"),
    ("n_categories", "Steam categories (count)"),
    ("n_screenshots", "Screenshots (count)"),
    ("detailed_desc_len", "Detailed description (chars)"),
    ("has_website", "Has marketing website"),
    ("has_metacritic", "Has Metacritic score"),
]


def alen(x):
    """Length of an iterable, treating None / scalars as 0."""
    if x is None:
        return 0
    try:
        return len(x)
    except TypeError:
        return 0


def parse_owners(s: str) -> float:
    """Parse FronkonGames' estimated_owners string ('0 - 20000') to bucket midpoint."""
    lo, hi = s.split(" - ")
    return (int(lo) + int(hi)) / 2


def load_and_enrich() -> pd.DataFrame:
    print(f"Downloading {FRONKONGAMES_REPO_ID}...")
    path = hf_hub_download(
        repo_id=FRONKONGAMES_REPO_ID,
        filename=FRONKONGAMES_FILENAME,
        repo_type="dataset",
    )
    df = pd.read_parquet(path)

    df["total_reviews"] = df["positive"].fillna(0).astype(int) + df["negative"].fillna(0).astype(int)
    df["pos_rate"] = df["positive"] / df["total_reviews"].replace(0, np.nan)

    # Restrict to games with enough reviews to compute a meaningful pos_rate.
    df = df[df["total_reviews"] >= 10].copy()
    print(f"Working set (>=10 reviews): {len(df):,} games")

    df["n_languages"] = df["supported_languages"].apply(alen)
    df["n_categories"] = df["categories"].apply(alen)
    df["n_screenshots"] = df["screenshots"].apply(alen)
    df["detailed_desc_len"] = df["detailed_description"].fillna("").str.len()
    df["has_website"] = (df["website"].fillna("").str.len() > 0).astype(int)
    df["has_metacritic"] = (df["metacritic_score"] > 0).astype(int)
    df["owners_mid"] = df["estimated_owners"].apply(parse_owners)

    df["rev_bucket"] = pd.cut(
        df["total_reviews"],
        bins=BUCKET_EDGES,
        right=False,
        labels=BUCKET_LABELS,
    )

    # Composite polish score: z-score each polish signal and sum. Used by both
    # the polish-composite histogram and the rank-disagreement scatter.
    z = pd.DataFrame(index=df.index)
    for col, _ in POLISH_SIGNALS:
        s = df[col].astype(float)
        z[col] = (s - s.mean()) / s.std()
    df["polish_score"] = z.sum(axis=1)

    return df


def stage_02_cutoff(df: pd.DataFrame) -> int:
    """Review count of the TARGET_GAME_COUNT-th game when ranked by reviews.

    Stage 02 selects the top N games by review count, so the equivalent
    review-count threshold is the value at rank N in the sorted catalog.
    """
    sorted_reviews = df["total_reviews"].sort_values(ascending=False).values
    return int(sorted_reviews[TARGET_GAME_COUNT - 1])


def cutoff_bucket_position(cutoff: int) -> float:
    """Map a review-count value to a continuous position on the categorical bucket axis.

    plotly maps category labels to integer x positions 0..N-1; we interpolate inside
    the matching bucket so the line appears at the right log-scaled location within it.
    """
    for i, (lo, hi) in enumerate(zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:])):
        if lo <= cutoff < hi:
            frac = (np.log(cutoff) - np.log(lo)) / (np.log(hi) - np.log(lo))
            # Subtract 0.5 so position 0 is the *center* of the first category.
            return i + frac - 0.5
    return len(BUCKET_LABELS) - 1


def fig_pos_rate_vs_reviews(df: pd.DataFrame, cutoff_reviews: int) -> go.Figure:
    """Density heatmap of pos_rate vs log10(reviews), with binomial CI envelope.

    The envelope shows the 95% range of observed pos_rate values you'd see if
    every game's true rate were the catalog median (~0.82). Inside the envelope
    = "you can't distinguish this game from average from review counts alone."
    """
    log_n = np.log10(df["total_reviews"])
    p_baseline = df["pos_rate"].median()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram2d(
            x=log_n,
            y=df["pos_rate"],
            xbins={"start": 1.0, "end": 7.0, "size": 0.05},
            ybins={"start": 0.0, "end": 1.0, "size": 0.01},
            colorscale="Viridis",
            colorbar={"title": "games"},
            zsmooth="best",
        )
    )

    n_grid = np.logspace(1, 7, 200)
    se = np.sqrt(p_baseline * (1 - p_baseline) / n_grid)
    upper = np.clip(p_baseline + 1.96 * se, 0, 1)
    lower = np.clip(p_baseline - 1.96 * se, 0, 1)
    log_n_grid = np.log10(n_grid)

    for y, name in [(upper, "95% upper"), (lower, "95% lower")]:
        fig.add_trace(
            go.Scatter(
                x=log_n_grid,
                y=y,
                mode="lines",
                line={"color": "white", "width": 2, "dash": "dash"},
                name=name,
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=log_n_grid,
            y=[p_baseline] * len(log_n_grid),
            mode="lines",
            line={"color": "white", "width": 1},
            name=f"baseline p={p_baseline:.2f}",
            showlegend=True,
        )
    )

    fig.add_vline(
        x=np.log10(cutoff_reviews),
        line={"color": "#ff6b6b", "width": 2, "dash": "solid"},
        annotation_text=f"Stage 02 cutoff (top {TARGET_GAME_COUNT:,} = {cutoff_reviews:,}+ reviews)",
        annotation_position="top right",
        annotation_font_color="#ff6b6b",
    )

    fig.update_layout(
        title=(
            "Positive review rate vs total reviews"
            f"<br><sup>White dashes: 95% binomial CI around catalog median p={p_baseline:.2f}. "
            "Inside the envelope = indistinguishable from average given sample size.</sup>"
        ),
        xaxis_title="log10(total reviews)",
        yaxis_title="positive rate",
        height=520,
        template="plotly_dark",
    )
    return fig


def fig_polish_signals(df: pd.DataFrame, cutoff_reviews: int) -> go.Figure:
    """6-panel small multiples: median + IQR of each polish signal vs review bucket."""
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[label for _, label in POLISH_SIGNALS],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    grouped = df.groupby("rev_bucket", observed=True)
    bucket_x = list(BUCKET_LABELS)
    cutoff_x = cutoff_bucket_position(cutoff_reviews)

    for i, (col, _label) in enumerate(POLISH_SIGNALS):
        row = i // 3 + 1
        column = i % 3 + 1

        q25 = grouped[col].quantile(0.25).reindex(bucket_x).values
        q50 = grouped[col].quantile(0.50).reindex(bucket_x).values
        q75 = grouped[col].quantile(0.75).reindex(bucket_x).values

        fig.add_trace(
            go.Scatter(
                x=bucket_x,
                y=q75,
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=column,
        )
        fig.add_trace(
            go.Scatter(
                x=bucket_x,
                y=q25,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(100, 180, 255, 0.25)",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=column,
        )
        fig.add_trace(
            go.Scatter(
                x=bucket_x,
                y=q50,
                mode="lines+markers",
                line={"color": "rgb(100, 180, 255)", "width": 2},
                marker={"size": 6},
                name=col,
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        fig.update_xaxes(tickangle=-45, row=row, col=column)
        # Annotate the cutoff line only on the top-left panel; redraw the line
        # itself on every panel for visual reference.
        fig.add_vline(
            x=cutoff_x,
            line={"color": "#ff6b6b", "width": 2},
            row=row,
            col=column,
            annotation_text=(f"top {TARGET_GAME_COUNT // 1000}k cutoff (~{cutoff_reviews:,})" if i == 0 else None),
            annotation_position="top left" if i == 0 else None,
            annotation_font_color="#ff6b6b",
        )

    fig.update_layout(
        title=(
            "Polish signals vs review-count bucket"
            "<br><sup>Median (line) and IQR (band). Red line: stage 02 keeps games to its right. "
            "Every signal climbs smoothly across the cutoff: no knee, no natural break.</sup>"
        ),
        height=620,
        template="plotly_dark",
    )
    return fig


def fig_polish_composite(df: pd.DataFrame, cutoff_reviews: int) -> go.Figure:
    """Composite polish score: z-score the 6 signals, sum. Histogram colored by bucket."""
    kept = df[df["total_reviews"] >= cutoff_reviews]
    dropped = df[df["total_reviews"] < cutoff_reviews]
    kept_median = kept["polish_score"].median()
    dropped_median = dropped["polish_score"].median()

    fig = go.Figure()
    palette = [
        "#2c3e50",
        "#34495e",
        "#3498db",
        "#1abc9c",
        "#f39c12",
        "#e67e22",
        "#e74c3c",
        "#c0392b",
    ]
    for label, color in zip(BUCKET_LABELS, palette):
        sub = df[df["rev_bucket"] == label]
        if len(sub) == 0:
            continue
        fig.add_trace(
            go.Histogram(
                x=sub["polish_score"],
                name=f"{label} (n={len(sub):,})",
                opacity=0.7,
                marker_color=color,
                xbins={"start": -8, "end": 20, "size": 0.5},
            )
        )

    fig.add_vline(
        x=kept_median,
        line={"color": "#ff6b6b", "width": 2},
        annotation_text=f"median of kept (top {TARGET_GAME_COUNT // 1000}k): {kept_median:+.2f}",
        annotation_position="top right",
        annotation_font_color="#ff6b6b",
    )
    fig.add_vline(
        x=dropped_median,
        line={"color": "#ffa94d", "width": 2, "dash": "dash"},
        annotation_text=f"median of dropped: {dropped_median:+.2f}",
        annotation_position="top left",
        annotation_font_color="#ffa94d",
    )

    fig.update_layout(
        title=(
            "Composite polish score, stacked by review-count bucket"
            "<br><sup>Z-summed across 6 signals. Red/orange lines: median polish score of "
            f"kept (n={len(kept):,}) vs dropped (n={len(dropped):,}) cohorts. Wide separation = "
            "review-count cutoff also strongly sorts on polish.</sup>"
        ),
        xaxis_title="polish score (sum of 6 z-scored signals)",
        yaxis_title="games",
        barmode="stack",
        height=520,
        template="plotly_dark",
        legend={"title": "review count"},
    )
    return fig


def fig_rank_disagreement(df: pd.DataFrame) -> go.Figure:
    """Per-game scatter of (review_rank, polish_rank), colored by which top-N each is in.

    Each axis reversed so rank 1 (the best) sits at top/right. Boundary lines at
    rank=TARGET_GAME_COUNT split the plot into four quadrants:
      - upper-right: kept by both metrics (the agreed cohort)
      - upper-left:  kept by polish, dropped by reviews
      - lower-right: kept by reviews, dropped by polish
      - lower-left:  in neither top-N
    """
    df = df.copy()
    df["review_rank"] = df["total_reviews"].rank(ascending=False, method="min")
    df["polish_rank"] = df["polish_score"].rank(ascending=False, method="min")

    in_review = df["review_rank"] <= TARGET_GAME_COUNT
    in_polish = df["polish_rank"] <= TARGET_GAME_COUNT

    cohorts = [
        ("both", in_review & in_polish, "rgba(46, 204, 113, 0.55)", "Kept by both"),
        ("review_only", in_review & ~in_polish, "rgba(52, 152, 219, 0.5)", "Review-only (popular but unpolished)"),
        ("polish_only", ~in_review & in_polish, "rgba(241, 196, 15, 0.5)", "Polish-only (polished but unpopular)"),
        ("neither", ~in_review & ~in_polish, "rgba(149, 165, 166, 0.18)", "In neither"),
    ]

    fig = go.Figure()
    for _, mask, color, label in cohorts:
        sub = df[mask]
        fig.add_trace(
            go.Scattergl(
                x=sub["review_rank"],
                y=sub["polish_rank"],
                mode="markers",
                marker={"size": 3, "color": color},
                name=f"{label} (n={len(sub):,})",
                hovertext=sub["name"],
                hoverinfo="text+x+y",
            )
        )

    fig.add_vline(
        x=TARGET_GAME_COUNT,
        line={"color": "white", "width": 1, "dash": "dash"},
        annotation_text=f"top {TARGET_GAME_COUNT:,} by reviews",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=TARGET_GAME_COUNT,
        line={"color": "white", "width": 1, "dash": "dash"},
        annotation_text=f"top {TARGET_GAME_COUNT:,} by polish",
        annotation_position="top left",
    )

    fig.update_layout(
        title=(
            "Rank disagreement: review count vs composite polish score"
            "<br><sup>Each point is one game. Axes reversed so rank 1 is at upper-right "
            "(better is up and to the right). Quadrants split at rank 10k.</sup>"
        ),
        xaxis_title="review rank (1 = most reviewed)",
        yaxis_title="polish rank (1 = most polished)",
        xaxis={"type": "log", "autorange": "reversed"},
        yaxis={"type": "log", "autorange": "reversed"},
        height=680,
        template="plotly_dark",
    )
    return fig


def fig_cohort_jaccard(df: pd.DataFrame) -> go.Figure:
    """Heatmap of pairwise Jaccard between top-N cohorts under different selection metrics.

    Two natural clusters typically emerge: {reviews, owners} (audience size) and
    {avg_playtime, med_playtime} (engagement depth). Cross-cluster agreement is much
    weaker, which is the substantive finding: these signals measure different things.
    """
    metrics = {
        "reviews": ("total_reviews", None),
        "owners": ("owners_mid", "total_reviews"),
        "peak_ccu": ("peak_ccu", "total_reviews"),
        "avg_playtime": ("average_playtime_forever", "total_reviews"),
        "med_playtime": ("median_playtime_forever", "total_reviews"),
    }

    cohorts: dict[str, set] = {}
    for label, (col, tiebreak) in metrics.items():
        sort_cols = [col, tiebreak] if tiebreak else [col]
        ranked = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        cohorts[label] = set(ranked.head(TARGET_GAME_COUNT)["appID"])

    labels = list(cohorts.keys())
    matrix = []
    text = []
    for a in labels:
        row_vals = []
        row_text = []
        for b in labels:
            i = len(cohorts[a] & cohorts[b])
            u = len(cohorts[a] | cohorts[b])
            j = i / u if u else 0.0
            row_vals.append(j)
            row_text.append(f"{j:.2f}<br><span style='font-size:10px'>n={i:,}</span>")
        matrix.append(row_vals)
        text.append(row_text)

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "white"},
            colorbar={"title": "Jaccard"},
        )
    )
    fig.update_layout(
        title=(
            f"Top-{TARGET_GAME_COUNT:,} cohort agreement across selection metrics"
            "<br><sup>Pairwise Jaccard between cohorts produced by ranking on each signal. "
            "Two visible clusters: audience-size (reviews + owners) and engagement-depth "
            "(avg + med playtime). Cross-cluster agreement is weak because these signals "
            "capture different things.</sup>"
        ),
        height=560,
        template="plotly_dark",
        xaxis={"side": "bottom"},
        yaxis={"autorange": "reversed"},
    )
    return fig


def render_html(figs: list[go.Figure], out_path: Path) -> None:
    parts = [
        "<html><head><title>Steam Atlas: review-count signals</title>",
        '<meta charset="utf-8"></head>',
        '<body style="background:#111;color:#eee;font-family:system-ui,sans-serif;'
        'max-width:1100px;margin:2em auto;padding:0 1em;">',
        "<h1>Quality signals as a function of review count</h1>",
        "<p>Internal analysis. See <code>experiments/analyze_review_signals.py</code>.</p>",
    ]
    for i, fig in enumerate(figs):
        include = "cdn" if i == 0 else False
        parts.append(fig.to_html(full_html=False, include_plotlyjs=include))
    parts.append("</body></html>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))
    print(f"Wrote {out_path}")


def main() -> None:
    df = load_and_enrich()
    cutoff_reviews = stage_02_cutoff(df)
    print(f"Stage 02 cutoff: top {TARGET_GAME_COUNT:,} games = {cutoff_reviews:,}+ reviews")
    figs = [
        fig_pos_rate_vs_reviews(df, cutoff_reviews),
        fig_polish_signals(df, cutoff_reviews),
        fig_polish_composite(df, cutoff_reviews),
        fig_rank_disagreement(df),
        fig_cohort_jaccard(df),
    ]
    render_html(figs, OUTPUT_HTML)


if __name__ == "__main__":
    main()
