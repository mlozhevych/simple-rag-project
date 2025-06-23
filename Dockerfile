# --- Етап 1: Збірка (Builder) ---
FROM python:3.12-slim as builder

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# --- Оптимізація кешування ---
# 1. Копіюємо тільки файли залежностей
COPY pyproject.toml poetry.lock* ./
# 2. Встановлюємо залежності
RUN poetry install --no-ansi -vvv --no-root
# 3. Тепер копіюємо решту коду
COPY . .

# --- Етап 2: Фінальний образ ---
FROM python:3.12-slim as final

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/src /app/src

ENV PYTHONPATH=/app/src

# Ця тека буде використовуватись для тимчасових файлів метрик від воркерів
RUN mkdir -p /app/prometheus_multiproc_dir

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "3", "simple_rag_project.app:app"]