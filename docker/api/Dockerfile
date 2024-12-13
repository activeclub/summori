FROM python:3.10-slim-buster AS python-base
COPY --from=ghcr.io/astral-sh/uv:0.5.9 /uv /uvx /bin/
ENV \
  # python
  PYTHONUNBUFFERED=1 \
  # prevents python creating .pyc files
  PYTHONDONTWRITEBYTECODE=1 \
  \
  # pip
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

FROM python-base
RUN apt-get update \
  && apt-get install --no-install-recommends -y \
  # deps for building python deps
  build-essential

COPY . ./app
WORKDIR /app

RUN uv sync --no-dev
