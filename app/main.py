

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.cache.semantic_cache import SemanticCache
from app.config import settings
from app.models.schemas import (
    CacheDeleteResponse,
    CacheStats,
    QueryRequest,
    QueryResponse,
    SearchResult,
)
from app.services.clustering import ClusteringService
from app.services.embedder import EmbeddingService
from app.services.search import SearchOrchestrator
from app.services.vector_store import VectorStoreService


orchestrator: SearchOrchestrator = None
cache: SemanticCache = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, cache

    print("=" * 60)
    print("STARTUP: Loading semantic search system...")
    print("=" * 60)

    embedder = EmbeddingService()
    embedder.load_model()

    vector_store = VectorStoreService()
    vector_store.initialize()

    if vector_store.document_count == 0:
        print(
            "\nVector store is empty! Run the setup pipeline first:\n"
            "   python -m scripts.setup_data\n"
        )

    clustering = ClusteringService()
    try:
        clustering.load()
    except FileNotFoundError:
        print(
            "\nClustering models not found! Run the setup pipeline first:\n"
            "   python -m scripts.setup_data\n"
        )

    cache = SemanticCache(
        threshold=settings.cache_similarity_threshold,
        max_size=settings.cache_max_size,
        cluster_search_depth=settings.cache_cluster_search_depth,
    )

    orchestrator = SearchOrchestrator(
        embedder=embedder,
        vector_store=vector_store,
        clustering=clustering,
        cache=cache,
    )

    print("=" * 60)
    print("STARTUP COMPLETE")
    print(f"  Model:       {settings.embedding_model}")
    print(f"  Documents:   {vector_store.document_count}")
    print(f"  Clusters:    {settings.n_clusters}")
    print(f"  Cache τ:     {settings.cache_similarity_threshold}")
    print(f"  Cache max:   {settings.cache_max_size}")
    print("=" * 60)

    yield

    print("Shutting down semantic search system...")


app = FastAPI(
    title="Semantic Search System",
    description=(
        "A lightweight semantic search system for the 20 Newsgroups dataset "
        "with fuzzy clustering and a custom semantic cache layer. "
        "Built for the Trademarkia AI & ML Engineer assessment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)






@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Resources still loading.",
        )

    try:
        result = await orchestrator.query(request.query)

        return QueryResponse(
            query=result["query"],
            cache_hit=result["cache_hit"],
            matched_query=result["matched_query"],
            similarity_score=result["similarity_score"],
            result=[SearchResult(**r) for r in result["result"]],
            dominant_cluster=result["dominant_cluster"],
            cluster_distribution=result.get("cluster_distribution"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )


@app.get("/cache/stats", response_model=CacheStats)
async def get_cache_stats():
    if cache is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready.",
        )

    stats = cache.get_stats()
    return CacheStats(**stats)


@app.delete("/cache", response_model=CacheDeleteResponse)
async def clear_cache():
    if cache is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready.",
        )

    entries_removed = await cache.clear()
    return CacheDeleteResponse(
        message="Cache cleared successfully",
        entries_removed=entries_removed,
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "documents": (
            orchestrator.vector_store.document_count
            if orchestrator
            else 0
        ),
        "cache_entries": len(cache._entries) if cache else 0,
    }
