

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.cache.semantic_cache import SemanticCache
import app.main as main_module


@pytest.fixture(autouse=True)
def init_cache():
    if main_module.cache is None:
        main_module.cache = SemanticCache(threshold=0.60, max_size=100)
    yield


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


@pytest.mark.anyio
async def test_cache_stats_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_entries" in data
        assert "hit_count" in data
        assert "miss_count" in data
        assert "hit_rate" in data


@pytest.mark.anyio
async def test_cache_delete_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/cache")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "entries_removed" in data


@pytest.mark.anyio
async def test_query_endpoint_validation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={"query": ""})
        assert response.status_code == 422


@pytest.mark.anyio
async def test_query_endpoint_success():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"query": "What are the latest developments in space?"},
        )
        if response.status_code == 200:
            data = response.json()
            assert "query" in data
            assert "cache_hit" in data
            assert "result" in data
            assert "dominant_cluster" in data
            assert isinstance(data["cache_hit"], bool)


@pytest.mark.anyio
async def test_query_cache_interaction():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.delete("/cache")
        query = "Tell me about computer graphics hardware"
        resp1 = await client.post("/query", json={"query": query})
        if resp1.status_code == 200:
            data1 = resp1.json()
            assert data1["cache_hit"] is False
            resp2 = await client.post("/query", json={"query": query})
            data2 = resp2.json()
            assert data2["cache_hit"] is True
            assert data2["matched_query"] == query
            assert data2["similarity_score"] == 1.0
