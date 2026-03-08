

from typing import Any, Dict, Optional

from app.cache.semantic_cache import SemanticCache
from app.services.clustering import ClusteringService
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.config import settings


class SearchOrchestrator:


    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: VectorStoreService,
        clustering: ClusteringService,
        cache: SemanticCache,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.clustering = clustering
        self.cache = cache

    async def query(self, query_text: str) -> Dict[str, Any]:

        
        query_embedding = self.embedder.embed_query(query_text)

        
        cluster_info = self.clustering.predict(query_embedding)
        dominant_cluster = cluster_info["dominant_cluster"]
        cluster_distribution = cluster_info["cluster_distribution"]

        
        cache_result = await self.cache.lookup(
            query_text=query_text,
            query_embedding=query_embedding,
            dominant_cluster=dominant_cluster,
            cluster_distribution=cluster_distribution,
        )

        if cache_result is not None:
            
            cached_entry, similarity_score = cache_result

            return {
                "query": query_text,
                "cache_hit": True,
                "matched_query": cached_entry.query_text,
                "similarity_score": round(similarity_score, 4),
                "result": cached_entry.result,
                "dominant_cluster": dominant_cluster,
                "cluster_distribution": cluster_distribution,
            }

        
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=settings.search_top_k,
        )

        
        formatted_results = [
            {
                "document": r["document"][:500],
                "category": r["category"],
                "similarity": r["similarity"],
                "cluster_distribution": r.get("cluster_distribution"),
            }
            for r in search_results
        ]


        await self.cache.store(
            query_text=query_text,
            query_embedding=query_embedding,
            result=formatted_results,
            dominant_cluster=dominant_cluster,
            cluster_distribution=cluster_distribution,
        )

        return {
            "query": query_text,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
            "result": formatted_results,
            "dominant_cluster": dominant_cluster,
            "cluster_distribution": cluster_distribution,
        }
