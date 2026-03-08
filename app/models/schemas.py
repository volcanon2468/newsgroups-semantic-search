

from typing import Dict, List, Optional

from pydantic import BaseModel, Field





class QueryRequest(BaseModel):


    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language search query",
        examples=["What are the latest developments in space exploration?"],
    )





class SearchResult(BaseModel):


    document: str = Field(..., description="Document text (possibly truncated)")
    category: str = Field(..., description="Original newsgroup category")
    similarity: float = Field(..., description="Cosine similarity to query")
    cluster_distribution: Optional[Dict[int, float]] = Field(
        None,
        description="Top cluster membership probabilities for this document",
    )


class QueryResponse(BaseModel):


    query: str = Field(..., description="The original query text")
    cache_hit: bool = Field(..., description="Whether the cache was used")
    matched_query: Optional[str] = Field(
        None, description="The cached query that matched (if cache hit)"
    )
    similarity_score: Optional[float] = Field(
        None, description="Cosine similarity to matched cached query"
    )
    result: List[SearchResult] = Field(
        ..., description="Top-K search results"
    )
    dominant_cluster: int = Field(
        ..., description="Primary cluster assignment for this query"
    )
    cluster_distribution: Optional[Dict[int, float]] = Field(
        None,
        description="Query's membership probabilities across top clusters",
    )


class CacheStats(BaseModel):


    total_entries: int = Field(..., description="Current cache size")
    hit_count: int = Field(..., description="Total cache hits since startup/reset")
    miss_count: int = Field(..., description="Total cache misses since startup/reset")
    hit_rate: float = Field(
        ..., description="Hit rate = hit_count / (hit_count + miss_count)"
    )
    cluster_distribution: Optional[Dict[int, int]] = Field(
        None,
        description="Number of cached entries per cluster",
    )


class CacheDeleteResponse(BaseModel):


    message: str = Field(default="Cache cleared successfully")
    entries_removed: int = Field(..., description="Number of entries that were flushed")
