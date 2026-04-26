# Legal MVP Starter

Starter skeleton for a Vietnam legal chatbot MVP using:
- FastAPI for HTTP API
- CrewAI Flow for orchestration
- Qdrant for retrieval
- OpenAI SDK for answer generation
- SentenceTransformers for dense embeddings in the hybrid retrieval pipeline
- Qdrant native BM25 for sparse retrieval

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

## Hybrid ingest for large CSV into Qdrant

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
- imports metadata and relationships into a SQLite sidecar for checkpoint/resume
- builds dense vectors from Hugging Face and uses Qdrant native BM25 (`Qdrant/bm25`) for sparse search
- upserts deterministic point IDs, so reruns do not create duplicates

Run a small dry run with `--limit 100` first to tune `--embed-batch-size` for your Colab GPU.

## Kaggle export, local Qdrant import

Use this when Qdrant Cloud free storage is too small. Kaggle does GPU dense embedding and writes compressed point shards; your local machine imports them into a local Qdrant instance and lets Qdrant build BM25 sparse vectors from `context_text`.

Local Qdrant must be `>=1.15.3` for native BM25:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

On Kaggle:

```bash
python scripts/export_hybrid_artifacts.py \
  --content-csv data/content.csv \
  --metadata-csv data/metadata.csv \
  --relationships-csv data/relationships.csv \
  --output-dir data/artifacts/hybrid_qdrant \
  --device cuda \
  --embed-batch-size 64 \
  --resume
```

Download `data/artifacts/hybrid_qdrant` to your local machine, then import:

```bash
python scripts/import_hybrid_artifacts_to_qdrant.py \
  --artifact-dir data/artifacts/hybrid_qdrant \
  --qdrant-url http://localhost:6333 \
  --collection legal_chunks_hybrid_v1 \
  --batch-size 128
```

The artifact contains `manifest.json` plus `points-*.jsonl.gz`. Each point stores deterministic ID, dense vector, and payload; sparse BM25 is generated during local import from `payload.context_text`.

When `--resume` is used, export appends new shards after existing `points-*.jsonl.gz` files and rewrites `manifest.json`. Without `--resume`, the output directory must be empty to avoid accidentally mixing exports.

## Colab/Kaggle notes

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/agentic_legal
!pip install -e .
!python scripts/export_hybrid_artifacts.py --device cuda --resume --limit 100
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
