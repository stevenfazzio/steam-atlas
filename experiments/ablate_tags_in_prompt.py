"""A/B ablation: does giving the labeling LLM the top-3 SteamSpy tags improve labels?

Mirrors `pipeline/07_label_facets.py` exactly except for the prompt input. Samples
N games stratified across Primary Genre (so we cover the distribution rather than
oversampling the majority class), runs two variants in parallel:

    baseline:  Name + Tagline + Summary
    +tags:     Name + Tagline + Summary + Top 3 SteamSpy tags by vote count

Writes:

  data/ablation_tags.parquet  per-game labels for both variants
  data/ablation_tags.html     summary report

The HTML quantifies, per facet:

  - movement rate: fraction of games where the label changed between variants
  - "Other" rate per variant: did exposing tags reduce hedging?
  - confusion matrix: which baseline values flow to which +tags values
  - parroting check: does predictability of +tags labels from tags jump (bad sign)
  - sample disagreements: ~30 games where labels moved, for human review

This is an audit, not a pipeline stage. Run on demand:

    uv run python experiments/ablate_tags_in_prompt.py
"""

import asyncio
import json
import re
import sys
from collections import Counter
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

OUTPUT_PARQUET = REPO_ROOT / "data" / "ablation_tags.parquet"
OUTPUT_HTML = REPO_ROOT / "data" / "ablation_tags.html"

SAMPLE_SIZE = 100
TOP_N_TAGS = 3
RANDOM_SEED = 42
MAX_RETRIES = 5
MAX_TEXT_CHARS = 1_500
OTHER_VALUE = "Other"


def _field_name(facet_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", facet_name).strip("_").lower()


def _other_field_name(facet_name: str) -> str:
    return f"{_field_name(facet_name)}_other"


def _build_system_prompt(schema: list[dict], variant: str) -> str:
    """variant: 'baseline' or 'plus_tags'. Only the input description differs."""
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

    if variant == "plus_tags":
        input_desc = (
            "You will receive a Steam game's name, tagline, short summary, and top "
            "player-applied tags by vote count. The tags are crowdsourced and noisy: "
            "use them as additional signal, not as ground truth. "
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


def _top_tags(tags_dict, n=TOP_N_TAGS) -> list[str]:
    if not isinstance(tags_dict, dict):
        return []
    counted = [(k, v) for k, v in tags_dict.items() if isinstance(v, (int, float)) and v is not None]
    counted.sort(key=lambda x: -x[1])
    return [k for k, _ in counted[:n]]


def _build_text_baseline(row) -> str:
    name = (row.get("name") or "").strip()
    tagline = (row.get("tagline") or "").strip()
    summary = (row.get("summary") or "").strip()
    if tagline and summary:
        text = f"Name: {name}\nTagline: {tagline}\nSummary: {summary}"
    elif summary:
        text = f"Name: {name}\nSummary: {summary}"
    else:
        text = f"Name: {name}"
    return text[:MAX_TEXT_CHARS]


def _build_text_plus_tags(row, top_tags: list[str]) -> str:
    base = _build_text_baseline(row)
    if top_tags:
        return f"{base}\nTop tags: {', '.join(top_tags)}"[:MAX_TEXT_CHARS]
    return base


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
    """Flatten a single LLM response into prefixed columns. Out-of-vocab -> None."""
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
    """Stratify on existing primary_genre labels so we cover all values evenly."""
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
    sys_baseline = _build_system_prompt(schema, "baseline")
    sys_plus_tags = _build_system_prompt(schema, "plus_tags")

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(ANTHROPIC_CONCURRENCY)
    pbar = tqdm(total=len(sample) * 2, desc="Labeling")

    async def _one(appid: int, text: str, system: str, prefix: str):
        res = await _label_one(client, sem, text, system, pbar)
        return prefix, appid, res

    tasks = []
    for _, row in sample.iterrows():
        appid = int(row["appid"])
        top_tags = _top_tags(row["tags"])
        text_b = _build_text_baseline(row)
        text_t = _build_text_plus_tags(row, top_tags)
        tasks.append(_one(appid, text_b, sys_baseline, "baseline"))
        tasks.append(_one(appid, text_t, sys_plus_tags, "plus_tags"))

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
        out["top_tags"] = ", ".join(_top_tags(row["tags"])) or ""
        rows.append(out)
    return pd.DataFrame(rows)


def fig_movement_summary(df: pd.DataFrame, schema: list[dict]) -> go.Figure:
    """Per-facet bar chart: movement rate, baseline-Other rate, plus_tags-Other rate."""
    facet_names = [f["name"] for f in schema]
    facet_cols = [_field_name(n) for n in facet_names]
    move_rates = []
    baseline_other = []
    plus_tags_other = []
    for col in facet_cols:
        b = df[f"baseline_{col}"]
        t = df[f"plus_tags_{col}"]
        both = b.notna() & t.notna()
        move_rates.append(((b != t) & both).sum() / max(1, both.sum()))
        baseline_other.append((b == OTHER_VALUE).sum() / max(1, b.notna().sum()))
        plus_tags_other.append((t == OTHER_VALUE).sum() / max(1, t.notna().sum()))

    fig = go.Figure()
    fig.add_bar(name="Movement rate (label changed)", x=facet_names, y=move_rates, marker_color="#d8a657")
    fig.add_bar(name='"Other" rate, baseline', x=facet_names, y=baseline_other, marker_color="#7daea3")
    fig.add_bar(name='"Other" rate, +tags', x=facet_names, y=plus_tags_other, marker_color="#a9b665")
    fig.update_layout(
        title=(
            "<b>Per-facet ablation summary</b>"
            "<br><sup>Movement rate = fraction of games where the label changed when "
            'tags were added. "Other" rate = fraction picking the fallback bucket. '
            "If +tags reduces Other rate without ballooning movement, tags are helping "
            "on edge cases. If movement is high but Other rate barely moves, tags are "
            "shifting confident labels (review disagreements to judge).</sup>"
        ),
        barmode="group",
        height=440,
        template="plotly_dark",
        yaxis={"title": "rate", "tickformat": ".0%"},
    )
    return fig


def fig_confusion_per_facet(df: pd.DataFrame, schema: list[dict]) -> go.Figure:
    """One confusion matrix per facet: rows=baseline, cols=+tags, cells=count."""
    facets = [(f["name"], _field_name(f["name"]), [v["name"] for v in f["values"]]) for f in schema]
    fig = make_subplots(
        rows=len(facets),
        cols=1,
        subplot_titles=[f[0] for f in facets],
        vertical_spacing=0.06,
    )
    for i, (label, col, values) in enumerate(facets, start=1):
        b = df[f"baseline_{col}"]
        t = df[f"plus_tags_{col}"]
        mask = b.notna() & t.notna()
        b = b[mask]
        t = t[mask]
        matrix = np.zeros((len(values), len(values)))
        for bv, tv in zip(b, t):
            if bv in values and tv in values:
                matrix[values.index(bv), values.index(tv)] += 1
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
                hovertemplate="baseline=%{y}<br>+tags=%{x}<br>n=%{z}<extra></extra>",
            ),
            row=i,
            col=1,
        )
        fig.update_xaxes(title_text="+tags label", tickangle=-30, row=i, col=1)
        fig.update_yaxes(title_text="baseline label", autorange="reversed", row=i, col=1)
    fig.update_layout(
        title=(
            "<b>Per-facet confusion matrices</b>"
            "<br><sup>Rows = baseline label, columns = +tags label. Diagonal = label "
            "agreed; off-diagonal = label moved. Read each row to see where +tags "
            "redistributes the baseline value.</sup>"
        ),
        height=400 * len(facets),
        template="plotly_dark",
        margin={"l": 200},
    )
    return fig


def fig_disagreements_table(df: pd.DataFrame, schema: list[dict], max_rows: int = 60) -> go.Figure:
    """Sample of games where any facet label moved. For human review."""
    facet_cols = [_field_name(f["name"]) for f in schema]
    moved_mask = pd.Series(False, index=df.index)
    for col in facet_cols:
        moved_mask |= (
            (df[f"baseline_{col}"].notna())
            & (df[f"plus_tags_{col}"].notna())
            & (df[f"baseline_{col}"] != df[f"plus_tags_{col}"])
        )
    moved = df[moved_mask].copy()
    moved = moved.head(max_rows)

    rows = [moved["name"].tolist(), moved["top_tags"].tolist()]
    headers = ["name", "top tags"]
    for col, fname in zip(facet_cols, [f["name"] for f in schema]):
        diffs = []
        for _, r in moved.iterrows():
            b = r[f"baseline_{col}"]
            t = r[f"plus_tags_{col}"]
            if pd.isna(b) or pd.isna(t) or b == t:
                diffs.append("")
            else:
                diffs.append(f"{b} → {t}")
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


def fig_parroting_check(df: pd.DataFrame, schema: list[dict]) -> go.Figure:
    """Predictability of each variant's labels from the top-3 tags.

    Same in-sample majority-vote-by-tag rule used in analyze_facet_overlap.py, but
    applied to BOTH variants. If +tags predictability jumps far above baseline,
    the LLM is parroting tags. If they're similar, +tags is using tags as one
    input among several.
    """

    def predict(label_col: str, df: pd.DataFrame) -> float:
        sub = df[df[label_col].notna() & (df["top_tags"] != "")].copy()
        if len(sub) == 0:
            return 0.0
        sub["_tags"] = sub["top_tags"].str.split(", ")
        pair_counts: dict[str, Counter] = {}
        for tags, lab in zip(sub["_tags"], sub[label_col]):
            for t in tags:
                pair_counts.setdefault(t, Counter())[lab] += 1
        dominant = {t: c.most_common(1)[0][0] for t, c in pair_counts.items()}
        majority = sub[label_col].value_counts().index[0]
        correct = 0
        for tags, actual in zip(sub["_tags"], sub[label_col]):
            votes: Counter = Counter()
            for t in tags:
                if t in dominant:
                    votes[dominant[t]] += 1
            pred = votes.most_common(1)[0][0] if votes else majority
            if pred == actual:
                correct += 1
        return correct / len(sub)

    facet_names = [f["name"] for f in schema]
    facet_cols = [_field_name(f["name"]) for f in schema]
    baseline_pred = [predict(f"baseline_{c}", df) for c in facet_cols]
    plus_tags_pred = [predict(f"plus_tags_{c}", df) for c in facet_cols]

    fig = go.Figure()
    fig.add_bar(name="baseline labels", x=facet_names, y=baseline_pred, marker_color="#7daea3")
    fig.add_bar(name="+tags labels", x=facet_names, y=plus_tags_pred, marker_color="#a9b665")
    fig.update_layout(
        title=(
            "<b>Parroting check: predictability of labels from top-3 tags</b>"
            "<br><sup>In-sample accuracy of a majority-vote-by-tag rule, applied to "
            "each variant's labels. If +tags is much higher than baseline, the LLM is "
            "regurgitating tags. If similar, the LLM is using tags as one signal "
            "among several.</sup>"
        ),
        barmode="group",
        height=440,
        template="plotly_dark",
        yaxis={"title": "in-sample accuracy", "tickformat": ".0%", "range": [0, 1]},
    )
    return fig


def render_html(figs_with_titles, out_path: Path) -> None:
    parts = [
        "<html><head><title>Steam Atlas: tags-in-prompt ablation</title>",
        '<meta charset="utf-8"></head>',
        '<body style="background:#111;color:#eee;font-family:system-ui,sans-serif;'
        'max-width:1200px;margin:2em auto;padding:0 1em;">',
        "<h1>Tags-in-prompt ablation: does adding top-3 tags improve facet labels?</h1>",
        "<p>Audit. See <code>experiments/ablate_tags_in_prompt.py</code>.</p>",
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
        ("Parroting check", fig_parroting_check(results, schema)),
        ("Confusion matrices", fig_confusion_per_facet(results, schema)),
        ("Sample disagreements", fig_disagreements_table(results, schema)),
    ]
    render_html(figs, OUTPUT_HTML)


def print_summary(df: pd.DataFrame, schema: list[dict]) -> None:
    print()
    print(f"=== Per-facet summary (n={len(df)}) ===")
    print(f"{'facet':<25} {'moved':>9}  {'baseline_Other':>17}  {'+tags_Other':>17}")
    for f in schema:
        col = _field_name(f["name"])
        b = df[f"baseline_{col}"]
        t = df[f"plus_tags_{col}"]
        both = b.notna() & t.notna()
        moved = ((b != t) & both).sum()
        b_other = (b == OTHER_VALUE).sum()
        t_other = (t == OTHER_VALUE).sum()
        b_n = max(1, b.notna().sum())
        t_n = max(1, t.notna().sum())
        moved_str = f"{moved:>4}/{both.sum():<3}"
        b_other_str = f"{b_other:>4}/{b_n:<3} ({b_other / b_n:>5.1%})"
        t_other_str = f"{t_other:>4}/{t_n:<3} ({t_other / t_n:>5.1%})"
        print(f"{f['name']:<25} {moved_str:>9}  {b_other_str:>17}  {t_other_str:>17}")


if __name__ == "__main__":
    asyncio.run(main())
