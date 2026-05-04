"""A/B ablation: does the LLM label better with the raw detailed_description vs
the stage-04 summary?

Tests the "display vs. semantic text split" hypothesis: stage 07 currently reads
a Haiku-distilled summary, but the embedding (stage 05) reads the full source.
That inconsistency may be costing us facet quality on games where the distillation
drops information that matters (utility software, hybrid genres, mood-vs-aesthetic
mismatches).

Same 100-game stratified sample as ablate_tags_in_prompt.py (RANDOM_SEED=42) so
results can be cross-referenced. Variants:

    summary:   Name + Tagline + Summary       (current stage-07 behavior)
    detailed:  Name + Tagline + detailed_description (HTML-stripped, truncated)

Trade-off being measured: information richness (detailed wins) vs marketing-copy
noise (summary wins, since stage 04 explicitly strips press quotes and franchise
marketing). The ablation tells us which dominates for facet classification.

Writes:
  data/ablation_detailed_vs_summary.parquet  per-game labels for both variants
  data/ablation_detailed_vs_summary.html     summary report

Run on demand:
    uv run python experiments/ablate_detailed_vs_summary.py
"""

import asyncio
import html
import json
import re
import sys
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    ANTHROPIC_CONCURRENCY,
    FACET_LABELING_MODEL,
    FACETS_PARQUET,
    FACETS_SCHEMA_JSON,
    GAMES_PARQUET,
)

OUTPUT_PARQUET = REPO_ROOT / "data" / "ablation_detailed_vs_summary.parquet"
OUTPUT_HTML = REPO_ROOT / "data" / "ablation_detailed_vs_summary.html"

SAMPLE_SIZE = 100
RANDOM_SEED = 42
MAX_RETRIES = 5
# 6000 chars is generous: mean detailed_description is 1820, p95 around 4500.
# Stage 04's distillation pipeline caps at 4000; we bump for the audit so we're
# not penalizing detailed by truncating away its tail.
MAX_DETAILED_CHARS = 6_000
MAX_SUMMARY_CHARS = 1_500
OTHER_VALUE = "Other"

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def strip_html(s: str) -> str:
    if not s:
        return ""
    text = HTML_TAG_RE.sub(" ", s)
    text = html.unescape(text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _field_name(facet_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", facet_name).strip("_").lower()


def _other_field_name(facet_name: str) -> str:
    return f"{_field_name(facet_name)}_other"


def _build_system_prompt(schema: list[dict], variant: str) -> str:
    """variant: 'summary' or 'detailed'. Only the input description differs."""
    facet_lines = []
    for facet in schema:
        facet_lines.append(f"\n**{facet['name']}**: {facet['description']}")
        for v in facet["values"]:
            facet_lines.append(f"  - {v['name']}: {v['description']}")
    facets_text = "\n".join(facet_lines)

    response_lines = []
    for facet in schema:
        col = _field_name(facet["name"])
        value_options = ", ".join(f'"{v["name"]}"' for v in facet["values"])
        response_lines.append(f'- "{col}": one of {value_options}')
        response_lines.append(f'- "{col}_other": 2-5 word phrase if {col} is "{OTHER_VALUE}"; omit otherwise.')
    response_schema = "\n".join(response_lines)

    if variant == "detailed":
        input_desc = (
            "You will receive a Steam game's name, tagline, and the full Steam store "
            "description text. The description is editorial copy and may contain marketing "
            "language, press quotes, or franchise pitches; focus on what the game itself "
            "actually does and is, not promotional framing. "
        )
    else:
        input_desc = "You will receive a Steam game's name, tagline, and short summary. "

    return (
        f"{input_desc}"
        "Classify it on each of the categorical facets below. For every facet, "
        "pick the single value that best fits. If a value clearly applies, pick it. "
        f'If no value clearly fits, pick "{OTHER_VALUE}" and provide a 2-5 word phrase '
        f"in the corresponding _other field describing what would actually fit.\n\n"
        f"## Facets\n{facets_text}\n\n"
        "Return only a JSON object with these keys, no markdown:\n"
        f"{response_schema}"
    )


def _build_text_summary(row) -> str:
    name = (row.get("name") or "").strip()
    tagline = (row.get("tagline") or "").strip()
    summary = (row.get("summary") or "").strip()
    if tagline and summary:
        text = f"Name: {name}\nTagline: {tagline}\nSummary: {summary}"
    elif summary:
        text = f"Name: {name}\nSummary: {summary}"
    else:
        text = f"Name: {name}"
    return text[:MAX_SUMMARY_CHARS]


def _build_text_detailed(row) -> str:
    name = (row.get("name") or "").strip()
    tagline = (row.get("tagline") or "").strip()
    detailed = strip_html(row.get("detailed_description") or "")
    if tagline and detailed:
        text = f"Name: {name}\nTagline: {tagline}\nDescription: {detailed}"
    elif detailed:
        text = f"Name: {name}\nDescription: {detailed}"
    elif tagline:
        text = f"Name: {name}\nTagline: {tagline}"
    else:
        text = f"Name: {name}"
    return text[:MAX_DETAILED_CHARS]


async def _label_one(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    text: str,
    system_prompt: str,
    pbar: tqdm,
) -> dict | None:
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.messages.create(
                    model=FACET_LABELING_MODEL,
                    max_tokens=400,
                    system=system_prompt,
                    messages=[{"role": "user", "content": text}],
                )
                break
            except anthropic.RateLimitError:
                wait = min(2**attempt * 5, 60)
                await asyncio.sleep(wait)
            except (anthropic.APIStatusError, anthropic.APIConnectionError):
                if attempt == MAX_RETRIES - 1:
                    pbar.update(1)
                    return None
                wait = min(2**attempt * 5, 60)
                await asyncio.sleep(wait)
        else:
            pbar.update(1)
            return None
    pbar.update(1)
    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _coerce(appid: int, llm: dict | None, schema, valid_per_field, prefix: str) -> dict:
    out: dict = {"appid": appid}
    for facet in schema:
        col = _field_name(facet["name"])
        ocol = _other_field_name(facet["name"])
        raw = (llm or {}).get(col)
        if raw in valid_per_field[col]:
            out[f"{prefix}_{col}"] = raw
            if raw == OTHER_VALUE:
                ff = (llm or {}).get(ocol)
                out[f"{prefix}_{ocol}"] = ff.strip() if isinstance(ff, str) and ff.strip() else None
            else:
                out[f"{prefix}_{ocol}"] = None
        else:
            out[f"{prefix}_{col}"] = None
            out[f"{prefix}_{ocol}"] = None
    return out


def _stratified_sample(games: pd.DataFrame, facets: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratify on existing primary_genre labels so we cover all values evenly.

    Identical implementation to ablate_tags_in_prompt.py with the same seed, so
    the sampled games are the same set across both ablations.
    """
    facets = facets.copy()
    facets["appid"] = facets["appid"].astype(int)
    games = games.copy()
    games["appid"] = games["appid"].astype(int)
    pool = games.merge(facets[["appid", "primary_genre"]], on="appid", how="inner")
    pool = pool[pool["primary_genre"].notna()]

    values = pool["primary_genre"].value_counts().index.tolist()
    per_value = max(1, n // len(values))
    rng = np.random.default_rng(seed)
    chunks = []
    for v in values:
        sub = pool[pool["primary_genre"] == v]
        take = min(per_value, len(sub))
        idx = rng.choice(sub.index, size=take, replace=False)
        chunks.append(sub.loc[idx])
    out = pd.concat(chunks, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    elif len(out) < n:
        remainder = pool[~pool["appid"].isin(out["appid"])]
        extra = remainder.sample(n=min(n - len(out), len(remainder)), random_state=seed)
        out = pd.concat([out, extra], ignore_index=True)
    return out


async def run_labeling(sample: pd.DataFrame, schema: list[dict], valid_per_field: dict) -> pd.DataFrame:
    sys_summary = _build_system_prompt(schema, "summary")
    sys_detailed = _build_system_prompt(schema, "detailed")

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(ANTHROPIC_CONCURRENCY)
    pbar = tqdm(total=len(sample) * 2, desc="Labeling")

    async def _one(appid: int, text: str, system: str, prefix: str):
        res = await _label_one(client, sem, text, system, pbar)
        return prefix, appid, res

    tasks = []
    for _, row in sample.iterrows():
        appid = int(row["appid"])
        text_s = _build_text_summary(row)
        text_d = _build_text_detailed(row)
        tasks.append(_one(appid, text_s, sys_summary, "summary"))
        tasks.append(_one(appid, text_d, sys_detailed, "detailed"))

    raw_results = await asyncio.gather(*tasks)
    pbar.close()

    by_appid: dict[int, dict] = {}
    for prefix, appid, res in raw_results:
        merged = _coerce(appid, res, schema, valid_per_field, prefix)
        if appid not in by_appid:
            by_appid[appid] = {"appid": appid}
        by_appid[appid].update({k: v for k, v in merged.items() if k != "appid"})

    rows = []
    for _, row in sample.iterrows():
        appid = int(row["appid"])
        out = by_appid.get(appid, {"appid": appid})
        out["name"] = row["name"]
        out["summary_chars"] = len((row.get("summary") or "").strip())
        out["detailed_chars"] = len(strip_html(row.get("detailed_description") or ""))
        rows.append(out)
    return pd.DataFrame(rows)


def fig_movement_summary(df: pd.DataFrame, schema: list[dict]) -> go.Figure:
    facet_names = [f["name"] for f in schema]
    facet_cols = [_field_name(n) for n in facet_names]
    move_rates = []
    summary_other = []
    detailed_other = []
    for col in facet_cols:
        s = df[f"summary_{col}"]
        d = df[f"detailed_{col}"]
        both = s.notna() & d.notna()
        move_rates.append(((s != d) & both).sum() / max(1, both.sum()))
        summary_other.append((s == OTHER_VALUE).sum() / max(1, s.notna().sum()))
        detailed_other.append((d == OTHER_VALUE).sum() / max(1, d.notna().sum()))

    fig = go.Figure()
    fig.add_bar(name="Movement rate (label changed)", x=facet_names, y=move_rates, marker_color="#d8a657")
    fig.add_bar(name='"Other" rate, summary', x=facet_names, y=summary_other, marker_color="#7daea3")
    fig.add_bar(name='"Other" rate, detailed', x=facet_names, y=detailed_other, marker_color="#a9b665")
    fig.update_layout(
        title=(
            "<b>Per-facet ablation summary</b>"
            "<br><sup>Movement rate = fraction of games where the label changed when "
            "switching from summary to detailed_description as facet input. "
            'Watch "Other" rate especially: if it goes UP under detailed (as we expect '
            "for utility/hybrid games where the source is more honest), the source-text "
            "variant is preserving signal that distillation rounded off.</sup>"
        ),
        barmode="group",
        height=440,
        template="plotly_dark",
        yaxis={"title": "rate", "tickformat": ".0%"},
    )
    return fig


def fig_confusion_per_facet(df: pd.DataFrame, schema: list[dict]) -> go.Figure:
    facets = [(f["name"], _field_name(f["name"]), [v["name"] for v in f["values"]]) for f in schema]
    fig = make_subplots(
        rows=len(facets),
        cols=1,
        subplot_titles=[f[0] for f in facets],
        vertical_spacing=0.06,
    )
    for i, (label, col, values) in enumerate(facets, start=1):
        s = df[f"summary_{col}"]
        d = df[f"detailed_{col}"]
        mask = s.notna() & d.notna()
        s = s[mask]
        d = d[mask]
        matrix = np.zeros((len(values), len(values)))
        for sv, dv in zip(s, d):
            if sv in values and dv in values:
                matrix[values.index(sv), values.index(dv)] += 1
        text = [
            [str(int(matrix[r, c])) if matrix[r, c] > 0 else "" for c in range(len(values))] for r in range(len(values))
        ]
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=values,
                y=values,
                colorscale="Viridis",
                showscale=False,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 11, "color": "white"},
                hovertemplate="summary=%{y}<br>detailed=%{x}<br>n=%{z}<extra></extra>",
            ),
            row=i,
            col=1,
        )
        fig.update_xaxes(title_text="detailed label", tickangle=-30, row=i, col=1)
        fig.update_yaxes(title_text="summary label", autorange="reversed", row=i, col=1)
    fig.update_layout(
        title=(
            "<b>Per-facet confusion matrices</b>"
            "<br><sup>Rows = summary label, columns = detailed label. Diagonal = label "
            "agreed; off-diagonal = label moved when switching to source text.</sup>"
        ),
        height=400 * len(facets),
        template="plotly_dark",
        margin={"l": 200},
    )
    return fig


def fig_disagreements_table(df: pd.DataFrame, schema: list[dict], max_rows: int = 80) -> go.Figure:
    facet_cols = [_field_name(f["name"]) for f in schema]
    moved_mask = pd.Series(False, index=df.index)
    for col in facet_cols:
        moved_mask |= (
            (df[f"summary_{col}"].notna())
            & (df[f"detailed_{col}"].notna())
            & (df[f"summary_{col}"] != df[f"detailed_{col}"])
        )
    moved = df[moved_mask].copy().head(max_rows)

    rows = [moved["name"].tolist(), moved["detailed_chars"].astype(str).tolist()]
    headers = ["name", "detailed_chars"]
    for col, fname in zip(facet_cols, [f["name"] for f in schema]):
        diffs = []
        for _, r in moved.iterrows():
            s = r[f"summary_{col}"]
            d = r[f"detailed_{col}"]
            if pd.isna(s) or pd.isna(d) or s == d:
                diffs.append("")
            else:
                diffs.append(f"{s} → {d}")
        rows.append(diffs)
        headers.append(fname)

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": headers,
                    "fill_color": "#222",
                    "font": {"color": "#ddd", "size": 12},
                    "align": "left",
                },
                cells={
                    "values": rows,
                    "fill_color": "#111",
                    "font": {"color": "#eee", "size": 11},
                    "align": "left",
                    "height": 26,
                },
            )
        ]
    )
    fig.update_layout(
        title=f"<b>Disagreements (first {len(moved)} of {moved_mask.sum()} games where any facet moved)</b>",
        height=min(900, 60 + 28 * len(moved)),
        template="plotly_dark",
        margin={"l": 10, "r": 10},
    )
    return fig


def render_html(figs_with_titles, out_path: Path) -> None:
    parts = [
        "<html><head><title>Steam Atlas: detailed_description vs summary ablation</title>",
        '<meta charset="utf-8"></head>',
        '<body style="background:#111;color:#eee;font-family:system-ui,sans-serif;'
        'max-width:1200px;margin:2em auto;padding:0 1em;">',
        "<h1>detailed_description vs summary ablation: which makes a better facet-labeling input?</h1>",
        "<p>Audit. See <code>experiments/ablate_detailed_vs_summary.py</code>.</p>",
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


def print_summary(df: pd.DataFrame, schema: list[dict]) -> None:
    print()
    print(f"=== Per-facet summary (n={len(df)}) ===")
    print(f"{'facet':<25} {'moved':>9}  {'summary_Other':>17}  {'detailed_Other':>18}")
    for f in schema:
        col = _field_name(f["name"])
        s = df[f"summary_{col}"]
        d = df[f"detailed_{col}"]
        both = s.notna() & d.notna()
        moved = ((s != d) & both).sum()
        s_other = (s == OTHER_VALUE).sum()
        d_other = (d == OTHER_VALUE).sum()
        s_n = max(1, s.notna().sum())
        d_n = max(1, d.notna().sum())
        moved_str = f"{moved:>4}/{both.sum():<3}"
        s_other_str = f"{s_other:>4}/{s_n:<3} ({s_other / s_n:>5.1%})"
        d_other_str = f"{d_other:>4}/{d_n:<3} ({d_other / d_n:>5.1%})"
        print(f"{f['name']:<25} {moved_str:>9}  {s_other_str:>17}  {d_other_str:>18}")

    print()
    print(
        f"detailed_description char count: mean {df['detailed_chars'].mean():.0f}, "
        f"p50 {df['detailed_chars'].median():.0f}, "
        f"p95 {df['detailed_chars'].quantile(0.95):.0f}, "
        f"max {df['detailed_chars'].max():.0f}"
    )


async def main():
    with open(FACETS_SCHEMA_JSON) as f:
        schema = json.load(f)
    valid_per_field = {_field_name(f["name"]): {v["name"] for v in f["values"]} for f in schema}

    games = pd.read_parquet(GAMES_PARQUET)
    games["appid"] = games["appid"].astype(int)
    facets = pd.read_parquet(FACETS_PARQUET)
    sample = _stratified_sample(games, facets, SAMPLE_SIZE, RANDOM_SEED)
    n_strata = sample["primary_genre"].nunique()
    print(f"Sampled {len(sample)} games stratified across {n_strata} primary-genre values")

    if OUTPUT_PARQUET.exists():
        existing = pd.read_parquet(OUTPUT_PARQUET)
        if set(existing["appid"].astype(int)) == set(sample["appid"].astype(int)):
            print(f"Reusing {OUTPUT_PARQUET} ({len(existing)} rows)")
            results = existing
        else:
            print("Sample composition changed; rerunning")
            results = await run_labeling(sample, schema, valid_per_field)
            results.to_parquet(OUTPUT_PARQUET, index=False)
            print(f"Wrote {OUTPUT_PARQUET}")
    else:
        results = await run_labeling(sample, schema, valid_per_field)
        results.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"Wrote {OUTPUT_PARQUET}")

    print_summary(results, schema)

    figs = [
        ("Movement and Other-rate summary", fig_movement_summary(results, schema)),
        ("Confusion matrices", fig_confusion_per_facet(results, schema)),
        ("Sample disagreements", fig_disagreements_table(results, schema)),
    ]
    render_html(figs, OUTPUT_HTML)


if __name__ == "__main__":
    asyncio.run(main())
