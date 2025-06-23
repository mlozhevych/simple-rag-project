# Simple RAG Project

Цей проєкт — це повноцінна реалізація системи Retrieval-Augmented Generation (RAG), розроблена для відповідей на
запитання на основі наданого PDF-документа. Він побудований з використанням сучасного стека технологій, включаючи Flask,
LangChain та OpenAI, і повністю контейнеризований за допомогою Docker.

Особлива увага приділяється моніторингу та спостереженню (observability). Проєкт інтегровано з Prometheus, OpenTelemetry
та Grafana, що забезпечує глибоке розуміння продуктивності та поведінки системи.

## 🌟 Ключові особливості

- **🧠 Основна функціональність RAG**: Відповідає на запитання, використовуючи як базу знань PDF-документ "Zalando REST
  API Guidelines".
- **🐍 Сучасний стек Python**: Використовує Flask для API, LangChain для побудови RAG-ланцюжка, Qdrant як векторну базу
  даних, HuggingFace Transformers для створення ембедингів та OpenAI `gpt-3.5-turbo` для генерації відповідей.
- **🐳 Повна контейнеризація**: Усі сервіси (API, база даних, моніторинг) налаштовані в `docker-compose.yml` для легкого
  запуску локально.
- **📊 Комплексний моніторинг**:
    - **Метрики**: Prometheus збирає системні та кастомні метрики додатку (кількість запитів, затримка).
    - **Трейсинг**: OpenTelemetry інструментує додаток для розподіленого трейсингу, експортуючи дані до ClickHouse.
    - **Візуалізація**: Попередньо налаштована панель у Grafana для візуалізації метрик з Prometheus та аналітики
      запитів з ClickHouse.
- **⚙️ CI/CD**: Готовий до використання пайплайн GitHub Actions для автоматичного тестування та збірки Docker-образу.
- **🧪 Тестування**: Включає інтеграційні тести для API та пошуку у векторній базі даних, які виконуються у власному
  CI-середовищі.

## 🚀 Початок роботи

### Передумови

- **Docker** та **Docker Compose**
- **Git** для клонування репозиторію
- **OpenAI API Key**

### Встановлення та запуск

1. **Клонуйте репозиторій:**
   ```bash
   git clone https://github.com/mlozhevych/simple-rag-project
   cd simple-rag-project
   ```

2. **Створіть `.env` файл:**
   Створіть файл `.env` у кореневій директорії проєкту та додайте свій OpenAI API ключ та інші змінні.

   ```dotenv
   # .env
    OPENAI_API_KEY=...
    QDRANT_HOST=localhost
    QDRANT_PORT=6333
    COLLECTION_NAME=documents
    DATA_PATH=data/
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    CHUNK_SIZE=600
    CHUNK_OVERLAP=150
    OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318/v1/traces
   ```
   Цей файл використовується сервісом `simple-rag-project` у `docker-compose.yml`.

3. **Запустіть повний стек додатку:**
   Ця команда запустить RAG API та всі сервіси моніторингу.
   ```bash
   docker-compose up -d
   ```
4. **Створення ембедингів (індексація даних):**
   Цей крок завантажує PDF-документ, розбиває його на частини та зберігає їхні векторні представлення (ембединги) у
   Qdrant.
   a. Запустіть скрипт індексації. Він виконається всередині тимчасового контейнера, використовуючи конфігурацію з
   `pyproject.toml`:
   ```bash
   poetry run create_embeddings
   ```
   Ви повинні побачити лог процесу, який завершиться повідомленням про успішне завантаження чанків до Qdrant.

Тепер застосунок повністю готов до використання.

### Як користуватися

Після успішного запуску, наступні сервіси будуть доступні:

- **RAG API (Swagger UI)**: `http://localhost:8080/apidocs/`
    - Інтерактивна документація API, де можна тестувати ендпоінти.

- **Grafana**: `http://localhost:3000`
    - **Логін**: `admin`
    - **Пароль**: `admin`
    - Перейдіть до **Dashboards** -> **RAG Service Monitoring**, щоб побачити попередньо налаштовану панель.

- **Prometheus**: `http://localhost:9090`
    - Переглядайте зібрані метрики.

- **Qdrant Web UI**: `http://localhost:6334/dashboard`
    - Досліджуйте стан колекції векторів.

#### Приклад запиту до API

Ви можете надіслати запит до ендпоінту `/ask` за допомогою `curl` або будь-якого іншого HTTP-клієнта:

```bash
curl -X POST "http://localhost:8080/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the API First principle?"}'
```

Очікувана відповідь:

```json
{
  "answer": {
    "query": "What is the API First principle?",
    "result": "The API First principle requires two aspects: first, to define APIs before coding the implementation using a standard specification language, and second, to get early review feedback from peers and client developers. This approach facilitates early review feedback and a development discipline that focuses service interface design on a profound understanding of the domain, generalized business entities/resources, and a clear separation of WHAT vs. HOW concerns."
  }
}
```

## 📊 Моніторинг та спостереження

Стек моніторингу є ключовою частиною цього проєкту. Панель **RAG Service Monitoring** у Grafana надає повне уявлення про
стан системи:

- **System & Service Metrics (з Prometheus)**:
    - **CPU Usage**: Використання процесора.
    - **Memory Usage**: Використання пам'яті.
    - **Requests Per Second (RPS)**: Кількість запитів в секунду до ендпоінту `/ask`.
    - **Average RAG Chain Latency (p95)**: 95-й перцентиль затримки обробки запиту RAG-ланцюжком.

- **RAG Application Analytics (з трейсів у ClickHouse)**:
    - **Queries per Minute**: Кількість запитів до `/ask` за хвилину.
    - **Average RAG Chain Processing Time (ms)**: Середній час виконання логіки RAG-ланцюжка.
    - **Total Queries**: Загальна кількість оброблених запитів.
    - **Recent Questions**: Таблиця з останніми запитаннями, фрагментами відповідей та часом обробки.

## 🧪 Тестування

Проєкт містить набір інтеграційних тестів. Для їх запуску використовується окремий файл `docker-compose.ci.yml`, який
створює ізольоване середовище.

Для запуску тестів локально:

1. **Переконайтесь, що у вас є OpenAI API ключ у середовищі або в `.env` файлі.**

2. **Запустіть сервіси для тестування:**
   ```bash
   docker-compose -f docker-compose.ci.yml up -d --build
   ```

3. **Запустіть тести за допомогою Poetry:**
   Ця команда виконає тести з `tests/` директорії.
   ```bash
   poetry run pytest
   ```

4. **Зупиніть тестові сервіси:**
   ```bash
   docker-compose -f docker-compose.ci.yml down
   ```

Ці кроки автоматизовані в пайплайні CI/CD (`.github/workflows/ci-cd.yml`).

## 📁 Структура проєкту

```
.
├── .github/workflows/          # Конфігурація CI/CD для GitHub Actions
│   └── ci-cd.yml
├── infrastructure/qdrant/      # Docker Compose для запуску Qdrant окремо
├── monitoring/                 # Конфігурації для стека моніторингу
│   ├── grafana/                # Дашборди та джерела даних для Grafana
│   ├── otel-collector-config.yml # Конфігурація OpenTelemetry Collector
│   └── prometheus.yml          # Конфігурація Prometheus
├── src/simple_rag_project/     # Основний код додатку
│   ├── data/                   # Директорія для PDF-документів
│   │   └── zalando_rest_api_guidelines.pdf
│   ├── app.py                  # Основний файл Flask API
│   └── create_embeddings.py    # Скрипт для індексації даних
├── tests/                      # Автоматизовані тести
│   ├── conftest.py             # Фікстури для Pytest
│   ├── test_api.py             # Тести для Flask API
│   └── test_qdrant_search.py   # Тести для пошуку в Qdrant
├── .gitignore                  # Файли та папки, що ігноруються Git
├── docker-compose.ci.yml       # Docker Compose для середовища CI
├── docker-compose.yml          # Основний Docker Compose для локальної розробки
├── Dockerfile                  # Dockerfile для збірки образу додатку
└── pyproject.toml              # Файл конфігурації проєкту та залежностей (Poetry)
```