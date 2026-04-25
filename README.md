# Legal MVP Starter

Starter skeleton for a Vietnam legal chatbot MVP using:
- FastAPI for HTTP API
- CrewAI Flow for orchestration
- Qdrant for retrieval
- OpenAI SDK for answer generation
- SentenceTransformers for dense embeddings in the hybrid retrieval pipeline

## Quickstart

```bash
cp .env.example .env
pip install -e .
uvicorn app.main:app --reload
```

Open:
- `GET /healthz`
- `POST /v1/chat`
- `POST /v1/search`

## Hybrid ingest for large CSV

```bash
python scripts/ingest_hybrid_qdrant.py \
  --content-csv data/content.csv \
  --metadata-csv data/metadata.csv \
  --relationships-csv data/relationships.csv \
  --dense-model-name sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  --collection legal_chunks_hybrid_v1 \
  --device cuda \
  --resume \
  --limit 100
```

The ingest pipeline:
- streams `content.csv` row-by-row, so the 4 GB source file is not loaded fully into RAM
- imports metadata and relationships into a SQLite sidecar for checkpoint/resume and sparse vocab stability
- builds dense vectors from Hugging Face and sparse BM25-style term vectors for Qdrant hybrid search
- upserts deterministic point IDs, so reruns do not create duplicates

Run a small dry run with `--limit 100` first to tune `--embed-batch-size` for your Colab GPU.

## Colab notes

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/agentic_legal
!pip install -e .
!python scripts/ingest_hybrid_qdrant.py --device cuda --resume --limit 100
```

## Seed sample data

```bash
python scripts/seed_qdrant.py
```

## Example search request

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "người lao động nghỉ việc không báo trước có phải bồi thường không?",
    "top_k": 5,
    "search_mode": "hybrid",
    "candidate_limit": 50,
    "filters": {
      "nganh": ["lao_dong"]
    }
  }'
```

## Notes

This is still not production-ready. Add:
- auth
- rate limiting
- observability
- persistent message store
- reranking
- legal evaluation set
