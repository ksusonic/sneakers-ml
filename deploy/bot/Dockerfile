FROM python:3.9-slim as builder

LABEL authors="Slava and Daniil, miem-refugees"

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR  \
    poetry install --only bot --no-interaction --no-ansi --no-root

FROM python:3.9-slim-buster as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /app
COPY sneakers_ml /app/sneakers_ml

ENTRYPOINT [ "python", "sneakers_ml/bot/main.py" ]
