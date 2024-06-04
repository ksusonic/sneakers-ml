FROM debian:bullseye as base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    # OpenCV dependencies
    make ca-certificates wget ffmpeg libsm6 libxext6


FROM python:3.9-buster as builder

LABEL authors="Slava and Daniil, miem-refugees"

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

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY config ./config
COPY sneakers_ml ./sneakers_ml
COPY Makefile ./Makefile

EXPOSE 8000
CMD [ "make", "load-models" ]
ENTRYPOINT [ "uvicorn", "sneakers_ml.app.main:app", "--proxy-headers", "--host 0.0.0.0", "--port", "8000"
