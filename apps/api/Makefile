.PHONY: test lint format proxy_redis

######################
# TESTING AND COVERAGE
######################

test:
	uv run pytest tests/

######################
# LINTING AND FORMATTING
######################

lint:
	uv run ruff check .
	uv run ruff format . --diff
	uv run mypy .

format:
	uv run ruff format .
	uv run ruff check --select I --fix .
