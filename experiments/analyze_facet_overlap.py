"""How much do the 5 LLM facets overlap with Steam-native metadata?

Three questions:

1. For each LLM facet (stage 07), how is each value distributed over Steam tags,
   categories, and genres? Where is the LLM rediscovering Steam structure vs
   adding orthogonal axes?
2. Does positional order matter in `categories` / `genres`? Compare any-position
   to position-0 views. If position-0 is markedly sharper, order carries signal.
3. How well could a simple Steam-native lookup heuristic predict the LLM's labels?
   High predictability = facet is largely redundant; low = facet is doing real work.

Renders data/facet_overlap.html. Run on demand:

    uv run python experiments/analyze_facet_overlap.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from config import FACETS_PARQUET, GAMES_PARQUET  # noqa: E402

OUTPUT_HTML = REPO_ROOT / "data" / "facet_overlap.html"

# Top-K Steam values to display per contingency heatmap. Beyond ~15 the cells
# get too small to read; below 10 we truncate signal in the longer-tail views.
TOP_K_PER_VIEW = 15

FACETS = [
    ("primary_genre", "Primary Genre"),
    ("setting_aesthetic", "Setting & Aesthetic"),
    ("player_mode", "Player Mode"),
    ("pacing_intensity", "Pacing & Intensity"),
    ("session_structure", "Session Structure"),
]


def _extract_tags_top3(tags_dict):
    """Top 3 SteamSpy tags by vote count."""
    if not isinstance(tags_dict, dict):
        return []
    counted = [(k, v) for k, v in tags_dict.items() if isinstance(v, (int, float)) and v is not None]
    counted.sort(key=lambda x: -x[1])
    return [k for k, _ in counted[:3]]


def _extract_tags_any(tags_dict):
    """Every vote-counted SteamSpy tag (typically the top 20 SteamSpy returns)."""
    if not isinstance(tags_dict, dict):
        return []
    return [k for k, v in tags_dict.items() if isinstance(v, (int, float)) and v is not None]


def _extract_list_all(lst):
    if isinstance(lst, (list, np.ndarray)) and len(lst) > 0:
        return [str(x) for x in lst]
    return []


def _extract_list_pos0(lst):
    if isinstance(lst, (list, np.ndarray)) and len(lst) > 0:
        return [str(lst[0])]
    return []


# Order matters: pairs are placed side-by-side in the heatmap grid so each row
# of the figure compares two views of the same source (pos0 vs all-of-set, or
# top-3 vs any-vote-counted). Eyeballing left-to-right answers the order/depth
# question for that source.
VIEWS = [
    ("tags_top3", "Tags (top 3 by votes)", "tags", _extract_tags_top3),
    ("tags_any", "Tags (any vote-counted)", "tags", _extract_tags_any),
    ("categories_pos0", "Categories (position 0)", "categories", _extract_list_pos0),
    ("categories_all", "Categories (any position)", "categories", _extract_list_all),
    ("genres_pos0", "Genres (position 0)", "genres", _extract_list_pos0),
    ("genres_all", "Genres (any position)", "genres", _extract_list_all),
]


def load_data() -> pd.DataFrame:
    games = pd.read_parquet(GAMES_PARQUET)
    games["appid"] = games["appid"].astype(int)
    facets = pd.read_parquet(FACETS_PARQUET)
    facets["appid"] = facets["appid"].astype(int)
    df = games.merge(facets, on="appid", how="inner")
    print(f"Loaded {len(df):,} games with facet labels")
    return df


def compute_contingency(
    df: pd.DataFrame,
    facet_col: str,
    source_col: str,
    extract_fn,
    top_k: int = TOP_K_PER_VIEW,
):
    """Cell[i,j] = P(Steam value j present in game | facet value = i).

    Rows = LLM facet values (sorted by frequency, "Other" pinned last).
    Cols = top-K most-common Steam values across the working subset.
    """
    sub = df[df[facet_col].notna()].copy()
    if len(sub) == 0:
        return None, None, None
    sub["_steam_set"] = sub[source_col].apply(lambda x: set(extract_fn(x)))

    overall = Counter()
    for s in sub["_steam_set"]:
        overall.update(s)
    top_steam = [v for v, _ in overall.most_common(top_k)]

    counts = sub[facet_col].value_counts()
    facet_values = [v for v in counts.index if v != "Other"] + (["Other"] if "Other" in counts.index else [])

    matrix = np.zeros((len(facet_values), len(top_steam)))
    for i, f in enumerate(facet_values):
        f_games = sub[sub[facet_col] == f]
        n_f = len(f_games)
        if n_f == 0:
            continue
        for j, v in enumerate(top_steam):
            matrix[i, j] = sum(1 for s in f_games["_steam_set"] if v in s) / n_f
    return matrix, facet_values, top_steam


def compute_predictability(df: pd.DataFrame, facet_col: str, source_col: str, extract_fn) -> dict | None:
    """In-sample accuracy of a majority-vote-by-Steam-value rule.

    For each Steam value v, find the most common facet among games containing v.
    Then for each game, predict the facet by majority vote across its Steam values
    (ties broken by global majority). Report top-1 accuracy.

    In-sample, so this is an upper bound. Used only as a relative ranking across
    views: if tags_any scores 0.85 and genres_pos0 scores 0.45, tags_any is
    capturing more of the LLM's signal whether or not the absolute number is
    realistic out-of-sample.
    """
    sub = df[df[facet_col].notna()].copy()
    if len(sub) == 0:
        return None
    sub["_steam_list"] = sub[source_col].apply(extract_fn)

    pair_counts: dict = {}
    for steam_list, facet in zip(sub["_steam_list"], sub[facet_col]):
        for v in steam_list:
            if v not in pair_counts:
                pair_counts[v] = Counter()
            pair_counts[v][facet] += 1
    dominant = {v: c.most_common(1)[0][0] for v, c in pair_counts.items()}

    overall_majority = sub[facet_col].value_counts().index[0]
    correct = 0
    total = 0
    for steam_list, actual in zip(sub["_steam_list"], sub[facet_col]):
        votes: Counter = Counter()
        for v in steam_list:
            if v in dominant:
                votes[dominant[v]] += 1
        pred = votes.most_common(1)[0][0] if votes else overall_majority
        total += 1
        if pred == actual:
            correct += 1

    baseline = sub[facet_col].value_counts(normalize=True).iloc[0]
    return {"acc": correct / total, "baseline": baseline, "n": total}


def fig_predictability_summary(df: pd.DataFrame) -> go.Figure:
    """Heatmap: rows = facets, cols = views, cells = in-sample accuracy."""
    matrix = []
    baselines = []
    for facet_col, _ in FACETS:
        row = []
        for _, _, source, extract in VIEWS:
            res = compute_predictability(df, facet_col, source, extract)
            row.append(res["acc"] if res else 0.0)
        matrix.append(row)

        sub = df[df[facet_col].notna()]
        baselines.append(sub[facet_col].value_counts(normalize=True).iloc[0] if len(sub) else 0.0)

    matrix = np.array(matrix)
    text = [
        [
            f"{matrix[i, j]:.2f}<br><span style='font-size:9px'>(+{matrix[i, j] - baselines[i]:+.2f})</span>"
            for j in range(matrix.shape[1])
        ]
        for i in range(matrix.shape[0])
    ]
    facet_labels = [
        f"{label}<br><span style='font-size:10px;color:#aaa'>baseline {baselines[i]:.2f}</span>"
        for i, (_, label) in enumerate(FACETS)
    ]
    view_labels = [v[1] for v in VIEWS]

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=view_labels,
            y=facet_labels,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "white"},
            colorbar={"title": "in-sample acc"},
        )
    )
    fig.update_layout(
        title=(
            "<b>Predictability of LLM facets from Steam-native sources</b>"
            "<br><sup>In-sample accuracy of a majority-vote-by-Steam-value rule. "
            "Higher = facet is more recoverable from Steam metadata. "
            "(+x.xx) is gain over majority-class baseline (left margin).</sup>"
        ),
        height=460,
        template="plotly_dark",
        xaxis={"side": "bottom"},
        yaxis={"autorange": "reversed"},
        margin={"l": 220, "b": 120},
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def fig_position_distribution(df: pd.DataFrame, source_col: str, label: str, top_k: int = 30) -> go.Figure:
    """For each value of `source_col` (genres or categories), what fraction of
    its occurrences land at each position?

    Concentration at pos 0 = developers lead with this value. A flat distribution
    across positions = no positional convention. The redundancy check in the
    contingency tables only tells us whether pos-0 *predicts* facet labels; this
    figure tells us whether there's even a convention worth predicting from.
    """
    counter: dict[str, Counter] = {}
    totals: Counter = Counter()
    max_pos = 0
    for lst in df[source_col]:
        if not isinstance(lst, (list, np.ndarray)) or len(lst) == 0:
            continue
        for pos, v in enumerate(lst):
            v = str(v)
            counter.setdefault(v, Counter())[pos] += 1
            totals[v] += 1
            max_pos = max(max_pos, pos)

    values = [v for v, _ in totals.most_common(top_k)]
    matrix = np.zeros((len(values), max_pos + 1))
    for i, v in enumerate(values):
        for p in range(max_pos + 1):
            matrix[i, p] = counter[v].get(p, 0) / totals[v]

    text = [
        [f"{matrix[i, j]:.0%}" if matrix[i, j] >= 0.05 else "" for j in range(matrix.shape[1])]
        for i in range(matrix.shape[0])
    ]
    y_labels = [f"{v} (n={totals[v]:,})" for v in values]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[f"pos {i}" for i in range(max_pos + 1)],
            y=y_labels,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10, "color": "white"},
            hovertemplate="value=%{y}<br>position=%{x}<br>P(position|value)=%{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            f"<b>{label} position distribution</b>"
            f"<br><sup>For each {label.lower()} value (top {top_k} by total occurrences), "
            "fraction of its occurrences at each position. Bright pos-0 cell = "
            "developers consistently lead with this value. Spread across columns = "
            "no positional convention. Read as P(position | value).</sup>"
        ),
        height=max(360, 22 * len(values) + 200),
        template="plotly_dark",
        yaxis={"autorange": "reversed"},
        margin={"l": 240},
    )
    return fig


def fig_coverage_histograms(df: pd.DataFrame) -> go.Figure:
    sources = [
        ("tags (vote-counted)", "tags", _extract_tags_any),
        ("categories", "categories", _extract_list_all),
        ("genres", "genres", _extract_list_all),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[s[0] for s in sources])
    for i, (_, col, extract) in enumerate(sources):
        counts = df[col].apply(lambda x: len(extract(x)))
        fig.add_trace(
            go.Histogram(x=counts, nbinsx=30, marker_color="#5fb3a1", showlegend=False),
            row=1,
            col=i + 1,
        )
        median = counts.median()
        mean = counts.mean()
        fig.add_vline(
            x=median,
            line={"color": "#d8a657", "width": 2, "dash": "dash"},
            annotation_text=f"med={median:.0f}<br>mean={mean:.1f}",
            annotation_position="top right",
            row=1,
            col=i + 1,
        )
    fig.update_layout(
        title="Values per game across the three Steam-native sources",
        height=380,
        template="plotly_dark",
    )
    return fig


def fig_facet_contingencies(df: pd.DataFrame, facet_col: str, facet_label: str) -> go.Figure:
    """3x2 grid: each row is a (pos0, all-of-set) or (top-3, any-vote-counted) pair."""
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[v[1] for v in VIEWS],
        vertical_spacing=0.13,
        horizontal_spacing=0.22,
    )

    for i, (_, _, source, extract) in enumerate(VIEWS):
        matrix, facet_values, top_steam = compute_contingency(df, facet_col, source, extract)
        row = i // 2 + 1
        col = i % 2 + 1
        if matrix is None:
            continue
        text = [[f"{v:.0%}" if v >= 0.05 else "" for v in r] for r in matrix]
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=top_steam,
                y=facet_values,
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 9, "color": "white"},
                showscale=False,
                hovertemplate=("facet=%{y}<br>steam=%{x}<br>P(steam|facet)=%{z:.1%}<extra></extra>"),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(tickangle=-45, row=row, col=col)
        fig.update_yaxes(autorange="reversed", row=row, col=col)

    fig.update_layout(
        title=(
            f"<b>{facet_label}</b>: P(Steam value present | LLM facet value)"
            "<br><sup>Each cell: fraction of games with that facet value that "
            "contain the Steam value. Rows ordered by facet-value frequency "
            "(top = most common, 'Other' pinned last). Compare left vs right columns "
            "for the order-matters check.</sup>"
        ),
        height=1200,
        template="plotly_dark",
    )
    return fig


def render_html(figs_with_titles, out_path: Path) -> None:
    parts = [
        "<html><head><title>Steam Atlas: facet vs Steam-native overlap</title>",
        '<meta charset="utf-8"></head>',
        '<body style="background:#111;color:#eee;font-family:system-ui,sans-serif;'
        'max-width:1200px;margin:2em auto;padding:0 1em;">',
        "<h1>LLM facets vs Steam-native metadata: overlap analysis</h1>",
        "<p>Internal analysis. See <code>experiments/analyze_facet_overlap.py</code>.</p>",
    ]
    for i, (title, fig) in enumerate(figs_with_titles):
        if title:
            parts.append(f"<h2>{title}</h2>")
        include = "cdn" if i == 0 else False
        parts.append(fig.to_html(full_html=False, include_plotlyjs=include))
    parts.append("</body></html>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))
    print(f"Wrote {out_path}")


def main() -> None:
    df = load_data()
    figs: list[tuple[str, go.Figure]] = [
        ("TL;DR: predictability summary", fig_predictability_summary(df)),
        ("Coverage of Steam-native sources", fig_coverage_histograms(df)),
        ("Genre position distribution", fig_position_distribution(df, "genres", "Genres", top_k=28)),
        ("Category position distribution", fig_position_distribution(df, "categories", "Categories", top_k=30)),
    ]
    for facet_col, facet_label in FACETS:
        figs.append((f"{facet_label}: contingency tables", fig_facet_contingencies(df, facet_col, facet_label)))
    render_html(figs, OUTPUT_HTML)


if __name__ == "__main__":
    main()
