"""Per-game facet labeling against the committed schema.

Reads pipeline/facets_schema.json and games.parquet, asks Claude Haiku (one call
per game) to assign each game to a value of each facet, writes data/facets.parquet
keyed by appid with one column per facet plus one <facet>_other column for the
freeform label when a game lands in the schema's Other bucket.

Input text is name + tagline + detailed_description (HTML-stripped). The summary
column is intentionally NOT used here: it's a Haiku-distilled rewrite for hovercard
display, and ablation showed using the source text instead lifts label quality on
~50% of disagreements. See experiments/ablate_detailed_vs_summary.py.

Resumable: rerunning skips appids already labeled. If the schema has changed since
the prior run, the existing parquet is discarded and labeling restarts from scratch.
Checkpoints atomically every CHECKPOINT_EVERY rows so a kill mid-run leaves a
consistent partial file.
"""

import asyncio
import html
import json
import os
import re
import tempfile

import anthropic
import pandas as pd
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_CONCURRENCY,
    FACET_LABELING_MODEL,
    FACETS_PARQUET,
    FACETS_SCHEMA_JSON,
    GAMES_PARQUET,
)
from tqdm import tqdm

CHECKPOINT_EVERY = 200
MAX_RETRIES = 5
MAX_TEXT_CHARS = 6_000
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
    """'Combat Pacing' -> 'combat_pacing'. Used as the parquet column name."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", facet_name).strip("_").lower()


def _other_field_name(facet_name: str) -> str:
    return f"{_field_name(facet_name)}_other"


def _build_system_prompt(schema: list[dict]) -> str:
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

    return (
        "You will receive a Steam game's name, tagline, and the full Steam store "
        "description text. The description is editorial copy and may contain marketing "
        "language, press quotes, or franchise pitches; focus on what the game itself "
        "actually does and is, not promotional framing. "
        "Classify it on each of the categorical facets below. For every facet, "
        "pick the single value that best fits. If a value clearly applies, pick it. "
        f'If no value clearly fits, pick "{OTHER_VALUE}" and provide a 2-5 word phrase '
        f"in the corresponding _other field describing what would actually fit.\n\n"
        f"## Facets\n{facets_text}\n\n"
        "Return only a JSON object with these keys, no markdown:\n"
        f"{response_schema}"
    )


def _build_text(row) -> str:
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
    return text[:MAX_TEXT_CHARS]


def _expected_columns(schema: list[dict]) -> list[str]:
    cols = []
    for facet in schema:
        cols.append(_field_name(facet["name"]))
        cols.append(_other_field_name(facet["name"]))
    return cols


def _empty_facets_df(expected_cols: list[str]) -> pd.DataFrame:
    columns = {"appid": pd.Series(dtype=int)}
    for col in expected_cols:
        columns[col] = pd.Series(dtype=object)
    return pd.DataFrame(columns)


def safe_write_parquet(df: pd.DataFrame, path) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(df)
        os.replace(tmp_path, str(path))
    except Exception:
        os.unlink(tmp_path)
        raise


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


def _coerce_row(appid: int, llm_response: dict | None, schema: list[dict], valid_per_field: dict) -> dict:
    """Convert raw LLM output into a clean parquet row, enforcing the schema."""
    row: dict = {"appid": appid}
    for facet in schema:
        col = _field_name(facet["name"])
        other_col = _other_field_name(facet["name"])
        raw_value = (llm_response or {}).get(col)
        if raw_value in valid_per_field[col]:
            row[col] = raw_value
            if raw_value == OTHER_VALUE:
                freeform = (llm_response or {}).get(other_col)
                row[other_col] = freeform.strip() if isinstance(freeform, str) and freeform.strip() else None
            else:
                row[other_col] = None
        else:
            row[col] = None
            row[other_col] = None
    return row


async def main():
    with open(FACETS_SCHEMA_JSON) as f:
        schema = json.load(f)
    print(f"Loaded {len(schema)} facets from {FACETS_SCHEMA_JSON}")
    for facet in schema:
        values = ", ".join(v["name"] for v in facet["values"])
        print(f"  {facet['name']}: {values}")

    expected_cols = _expected_columns(schema)
    valid_per_field = {_field_name(f["name"]): {v["name"] for v in f["values"]} for f in schema}

    games = pd.read_parquet(GAMES_PARQUET)
    games["appid"] = games["appid"].astype(int)

    if FACETS_PARQUET.exists():
        existing = pd.read_parquet(FACETS_PARQUET)
        existing["appid"] = existing["appid"].astype(int)
        existing_cols = [c for c in existing.columns if c != "appid"]
        if set(existing_cols) != set(expected_cols):
            print(f"Schema changed; discarding {len(existing):,} rows from prior run")
            existing = _empty_facets_df(expected_cols)
        else:
            existing = existing[["appid"] + expected_cols]
    else:
        existing = _empty_facets_df(expected_cols)

    done_ids = set(existing["appid"].tolist())
    todo_indices = [i for i, aid in enumerate(games["appid"]) if int(aid) not in done_ids]
    print(f"Loaded {len(games):,} games. Already labeled: {len(done_ids):,}. To label: {len(todo_indices):,}")
    if not todo_indices:
        print("All games already labeled.")
        return

    system_prompt = _build_system_prompt(schema)
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(ANTHROPIC_CONCURRENCY)
    pbar = tqdm(total=len(todo_indices), desc="Labeling")

    new_rows = []
    for chunk_start in range(0, len(todo_indices), CHECKPOINT_EVERY):
        chunk = todo_indices[chunk_start : chunk_start + CHECKPOINT_EVERY]
        tasks = []
        for idx in chunk:
            row = games.iloc[idx]
            tasks.append((int(row["appid"]), _build_text(row)))

        async def _process(appid: int, text: str):
            result = await _label_one(client, sem, text, system_prompt, pbar)
            return appid, result

        results = await asyncio.gather(*[_process(a, t) for a, t in tasks])
        for appid, res in results:
            new_rows.append(_coerce_row(appid, res, schema, valid_per_field))

        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        safe_write_parquet(combined, FACETS_PARQUET)
        print(f"  Checkpoint: {len(combined):,} rows written to {FACETS_PARQUET}")

    pbar.close()
    print(f"Done. Labeled {len(new_rows):,} games. Saved to {FACETS_PARQUET}")


if __name__ == "__main__":
    asyncio.run(main())
