.PHONY: install lint format test pipeline design-facets clean

install:
	uv sync --extra dev

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest

pipeline:
	uv run python pipeline/00_enumerate_games.py
	uv run python pipeline/01_fetch_tags.py
	uv run python pipeline/02_select_top_games.py
	uv run python pipeline/03_compute_sentiment.py
	uv run python pipeline/04_summarize_descriptions.py
	uv run python pipeline/05_embed_descriptions.py
	uv run python pipeline/06_reduce_umap.py
	uv run python pipeline/07_label_facets.py
	uv run python pipeline/08_label_topics.py
	uv run python pipeline/09_visualize.py

# One-shot facet schema discovery. Not part of `make pipeline` because the schema
# is committed; rerun only when you intentionally want a new schema.
design-facets:
	uv run python pipeline/design_facets.py

clean:
	@echo "This will remove all files in data/. Press Ctrl+C to cancel."
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf data/*
