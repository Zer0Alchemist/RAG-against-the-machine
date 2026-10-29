install:
	@uv sync

run:
	@uv run python3 -m src index --max_chunk_size 2000
debug:
	@python3 -m pdb -m src
clean:
	@rm -rf .mypy_cache __pycache__ src/__pycache__ src/augmenter/__pycache__ \
	src/evaluator/__pycache__ src/generator/__pycache__ src/indexer/__pycache__ \
	src/models/__pycache__ src/retriever/__pycache__ src/utils/__pycache__ \
	*/__pycache__
lint:
	@uv run python3 -m flake8 src/
	@uv run python3 -m mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@uv run python3 -m flake8 src/
	@uv run python3 -m mypy src/ --strict

create:
	@mkdir data
	@mkdir data/raw
	@mkdir data/datasets
