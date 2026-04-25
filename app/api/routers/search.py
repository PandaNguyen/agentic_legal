from fastapi import APIRouter, Depends
from app.api.deps import get_qdrant_service
from app.schemas.search import SearchRequest, SearchResponse
from app.services.retrieval.qdrant_service import QdrantService

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, svc: QdrantService = Depends(get_qdrant_service)):
    hits = await svc.search(
        query=req.query,
        top_k=req.top_k,
        search_mode=req.search_mode,
        candidate_limit=req.candidate_limit,
        filters=req.filters,
    )
    return SearchResponse(results=hits)
