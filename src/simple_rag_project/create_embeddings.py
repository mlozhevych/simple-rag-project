"""
Builds or updates a Qdrant vector index from **PDF** documents.

Key features:
• Loads all `*.pdf` files in `DATA_PATH` using **PyPDFLoader** (LangChain).
• Splits each page into overlapping text chunks via `RecursiveCharacterTextSplitter` for better semantic search.
• Generates local, free embeddings with `SentenceTransformer` (`all-MiniLM-L6-v2` by default).
• Inserts/updates vectors in a live Qdrant server (`host`/`port` via env vars).

### Environment variables (optional)
| Variable | Default | Purpose |
|----------|---------|---------|
| `QDRANT_HOST` | `"localhost"` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `COLLECTION_NAME` | `"documents"` | Target collection name |
| `DATA_PATH` | `"data/"` | Directory with PDFs |
| `EMBEDDING_MODEL` | `"sentence-transformers/all-MiniLM-L6-v2"` | Embedding model |
| `CHUNK_SIZE` | `1000` | Chunk size in characters |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |

Usage:
```bash
pip install qdrant-client sentence-transformers langchain-community pypdf tqdm
python create_embeddings.py
```
"""

import logging
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


# --- Tracing (OpenTelemetry) ---
def setup_telemetry():
    """Налаштовує OpenTelemetry для трейсингу."""
    service_name = "embedding-creator-job"
    exporter_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    resource = Resource(attributes={"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Експортер в консоль для локального дебагу
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Експортер в OpenTelemetry Collector
    if exporter_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=f"{exporter_endpoint}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info(f"Телеметрію налаштовано для '{service_name}' з експортером '{exporter_endpoint}'")

    trace.set_tracer_provider(provider)


setup_telemetry()
tracer = trace.get_tracer(__name__)


def load_and_split(path: str, chunk_size: int, chunk_overlap: int):
    """Load all PDFs and split them into chunks."""
    loader = DirectoryLoader(
        path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,  # one Document per page
        use_multithreading=True,
    )
    docs = loader.load()
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def build_payloads(docs) -> List[Dict[str, Any]]:
    """Add page text to metadata so it can be used later in search results."""
    payloads: List[Dict[str, Any]] = []
    for d in docs:
        meta = d.metadata.copy()
        # Store **entire** chunk text; truncate if you prefer (<‑‑ adjust here)
        meta["text"] = d.page_content
        payloads.append(meta)
    return payloads


def ensure_collection(client: QdrantClient, name: str, dim: int):
    """Create collection if it doesn't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20_000),
        )
        logger.info(f"▶️  Created collection '{name}' (dim={dim})")


def run_indexing_pipeline(
        host: str,
        port: int,
        collection: str,
        data_path: str,
        model_name: str,
        chunk_size: int,
        chunk_overlap: int
):
    """
    The core logic for creating and populating the Qdrant collection.
    This function is designed to be called from tests or other scripts.
    """
    logger.info(f"📂 Reading PDFs from: {Path(data_path).resolve()}")

    # with tracer.start_as_current_span("load_and_split_docs") as child_span:
    docs = load_and_split(data_path, chunk_size, chunk_overlap)
    # child_span.set_attribute("num_documents_after_split", len(docs))

    if not docs:
        logger.warning("⚠️  No PDF documents found. Exiting.")
        # child_span.set_attribute("result", "no_documents_found")
        return

    texts = [d.page_content for d in docs]
    payloads = build_payloads(docs)

    logger.info(f"🧠 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    # child_span.set_attribute("embedding.dimension", dim)

    logger.info("🔢 Encoding embeddings…")
    # with tracer.start_as_current_span("encode_embeddings"):
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    logger.info(f"🔌 Connecting to Qdrant → {host}:{port}")
    client = QdrantClient(host=host, port=port)
    ensure_collection(client, collection, dim)

    logger.info("⬆️  Upserting into Qdrant…")
    ids = [(uuid.uuid4().int >> 64) for _ in range(len(embeddings))]

    # with tracer.start_as_current_span("upsert_to_qdrant") as child_span:
    client.upsert(
        collection_name=collection,
        points=models.Batch(ids=ids, vectors=embeddings, payloads=payloads),
    )
    # child_span.set_attribute("num_vectors_upserted", len(embeddings))

    logger.info(f"✅ {len(embeddings)} chunks upserted into '{collection}'")
    # child_span.set_attribute("result", "success")
    # child_span.set_attribute("chunks_upserted", len(embeddings))
    num_chunks = len(embeddings)

    return num_chunks


def main():
    # Починаємо головний спан для всього процесу
    with tracer.start_as_current_span("create_embeddings_job") as span:
        # --- config from env or defaults ---
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        collection = os.getenv("COLLECTION_NAME", "documents")
        data_path = BASE_DIR / os.getenv("DATA_PATH", "data/")
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))

        # Додаємо атрибути до спану
        span.set_attribute("qdrant.host", host)
        span.set_attribute("qdrant.port", port)
        span.set_attribute("qdrant.collection_name", collection)
        span.set_attribute("embedding.model_name", model_name)
        span.set_attribute("embedding.chunk_size", chunk_size)
        span.set_attribute("embedding.chunk_overlap", chunk_overlap)

        run_indexing_pipeline(
            host=host,
            port=port,
            collection=collection,
            data_path=data_path,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, )


if __name__ == "__main__":
    main()
