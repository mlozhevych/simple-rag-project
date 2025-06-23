import logging
import os
import time

from dotenv import load_dotenv
from flasgger import Swagger, swag_from
from flask import Flask, request, jsonify
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram
from prometheus_flask_exporter import PrometheusMetrics
from qdrant_client import QdrantClient

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY не знайдено! Перевірте ваш .env файл.")


# --- Tracing (OpenTelemetry) ---
def setup_telemetry():
    """Налаштовує OpenTelemetry для трейсингу."""
    service_name = os.getenv("OTEL_SERVICE_NAME", "simple-rag-project")
    exporter_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    resource = Resource(attributes={"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Використовуємо OTLP HTTP експортер
    exporter = OTLPSpanExporter(endpoint=exporter_endpoint)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    logger.info(f"Телеметрію налаштовано для сервісу '{service_name}' з експортером '{exporter_endpoint}'")


# Ініціалізуємо телеметрію
setup_telemetry()
tracer = trace.get_tracer(__name__)

# --- Flask ДОДАТОК ---
app = Flask(__name__)

# --- Prometheus ---
metrics = PrometheusMetrics(app, group_by='path')

# Кастомні метрики
rag_queries_total = Counter('rag_queries_total', 'Total number of RAG queries')
rag_query_latency = Histogram('rag_query_latency_seconds', 'Latency of the RAG chain execution')

# Автоматична інструментація Flask для трейсингу
FlaskInstrumentor().instrument_app(app)

# Додаємо конфігурацію Swagger
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"  # URL для доступу до Swagger UI
}
swagger = Swagger(app, config=swagger_config)

# ----- кастомний промпт -----
DEFAULT_TEMPLATE = """
Відповідай українською. Використовуй наведені контексти, щоб дати точну та стислу відповідь.
Якщо відповіді немає в контекстах — скажи "На жаль, не знайшов відповіді".
----------------
{context}
----------------
Питання: {question}
Відповідь:"""

prompt = PromptTemplate(template=DEFAULT_TEMPLATE, input_variables=["context", "question"])

# Змінні для lazy init
_embedding_model = None
_qa_chain = None


def get_qa_chain():
    global _embedding_model, _qa_chain
    if _qa_chain is None:
        with tracer.start_as_current_span("qa_chain_initialization") as span:
            logger.info("Ініціалізую Qdrant та ланцюжок RetrievalQA...")
            # 1) embeddings
            _embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            # 2) клієнт та векторне сховище
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            span.set_attribute("qdrant.url", qdrant_url)
            client = QdrantClient(url=qdrant_url)
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name="documents",
                embedding=_embedding_model,
            )
            # 3) retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            # 4) LLM
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            # 5) chain
            _qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
            )
            logger.info("Qdrant та RetrievalQA готові.")
    return _qa_chain


@app.route('/health', methods=['GET'])
@swag_from({
    'summary': 'Перевірка стану сервісу',
    'description': 'Повертає статус "OK", якщо модель успішно завантажена.',
    'responses': {
        '200': {
            'description': 'Статус сервісу',
            'schema': {'properties': {'status': {'type': 'string', 'example': 'OK'}}}
        }
    }
})
def health_check():
    """Перевірка стану сервісу."""
    status = "OK"
    return jsonify({"status": status})


@app.route("/ask", methods=['POST'])
@swag_from({
    'summary': 'Поставити запитання до RAG-моделі',
    'description': 'Приймає запитання, знаходить релевантну інформацію в базі знань та генерує відповідь за допомогою LLM.',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'id': 'Question',
                'required': ['question'],
                'properties': {
                    'question': {
                        'type': 'string',
                        'description': 'Текст запитання до системи.',
                        'example': 'What is the API First principle?'
                    }
                }
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Успішна відповідь від RAG-моделі.',
            'schema': {
                'properties': {
                    'answer': {
                        'type': 'object',
                        'properties': {
                            'query': {'type': 'string'},
                            'result': {'type': 'string'}
                        }
                    }
                }
            }
        },
        '400': {
            'description': 'Неправильний запит, відсутнє поле "question".'
        }
    }
})
def ask_question():
    """
    Приймає запитання, знаходить релевантну інформацію та генерує відповідь.
    """
    # Отримуємо JSON з тіла запиту
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Invalid request. 'question' field is required."}), 400

    question = data['question']

    # Отримуємо trace_id, якщо він є, для логування
    current_span = trace.get_current_span()
    trace_id = current_span.get_span_context().trace_id

    logger.info(f"Отримано запитання: {question}")

    # Інкрементуємо лічильник запитів
    rag_queries_total.inc()

    # Кастомний трейсинг для основного логічного блоку
    with tracer.start_as_current_span("rag_chain_execution") as span:
        span.set_attribute("rag.question", question)

        start_time = time.time()

        # Lazy init + виклик ланцюжка
        qa_chain = get_qa_chain()
        # Використовуємо RAG chain для генерації відповіді
        answer = qa_chain.invoke({"query": question})

        # Записуємо час виконання в гістограму
        latency = time.time() - start_time
        rag_query_latency.observe(latency)

        span.set_attribute("rag.answer.result", answer.get('result', ''))
        span.set_attribute("rag.answer.length", len(answer.get('result', '')))

    logger.info(f"Згенерована відповідь: {answer}")
    return jsonify({"answer": answer})


# Цей блок дозволяє запускати сервер напряму для тестування
# python app/main.py
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
