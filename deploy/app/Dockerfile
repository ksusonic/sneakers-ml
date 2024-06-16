FROM python:3.9-bullseye AS base

LABEL authors="Slava and Daniil, miem-refugees"

RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    make ffmpeg libsm6 libxext6

FROM python:3.9-slim AS builder

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR  \
    poetry install --with api

FROM base as runtime

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY config /app/config
COPY sneakers_ml /app/sneakers_ml

# Models binaries
#ADD https://storage.yandexcloud.net/sneaker-ml-models/brand-cls-models.tar.gz ./
# or use local add
ADD ./build/brand-cls-models.tar.gz ./

EXPOSE 8000
ENTRYPOINT [ "uvicorn", "sneakers_ml.app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
