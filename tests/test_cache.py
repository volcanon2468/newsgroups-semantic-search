

import asyncio

import numpy as np
import pytest

from app.cache.semantic_cache import SemanticCache


@pytest.fixture
def cache():
    return SemanticCache(threshold=0.85, max_size=100, cluster_search_depth=2)


@pytest.fixture
def random_embedding():
    def _make(dim=384, seed=None):
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec
    return _make


def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSemanticCacheBasics:


    def test_empty_cache_returns_miss(self, cache, random_embedding):
        emb = random_embedding(seed=1)
        result = run_async(cache.lookup("test query", emb, dominant_cluster=0))
        assert result is None

    def test_empty_cache_stats(self, cache):
        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0
        assert stats["hit_rate"] == 0.0

    def test_store_and_exact_match_hit(self, cache, random_embedding):
        emb = random_embedding(seed=1)
        query = "What is machine learning?"

        run_async(cache.store(
            query_text=query,
            query_embedding=emb,
            result={"answer": "ML is..."},
            dominant_cluster=0,
            cluster_distribution={0: 0.9, 1: 0.1},
        ))

        result = run_async(cache.lookup(query, emb, dominant_cluster=0))
        assert result is not None
        entry, score = result
        assert score == 1.0
        assert entry.query_text == query
        assert entry.result == {"answer": "ML is..."}

    def test_store_increments_entry_count(self, cache, random_embedding):
        for i in range(5):
            emb = random_embedding(seed=i)
            run_async(cache.store(
                query_text=f"query {i}",
                query_embedding=emb,
                result=f"result {i}",
                dominant_cluster=i % 3,
                cluster_distribution={i % 3: 0.9},
            ))

        stats = cache.get_stats()
        assert stats["total_entries"] == 5


class TestSemanticSimilarity:


    def test_similar_embedding_is_hit(self, cache):
        
        dim = 384
        rng = np.random.RandomState(42)

        base = rng.randn(dim).astype(np.float32)
        base /= np.linalg.norm(base)

        noise = rng.randn(dim).astype(np.float32) * 0.02
        similar = base + noise
        similar /= np.linalg.norm(similar)

        sim = float(base @ similar)
        assert sim > 0.85, f"Test setup issue: similarity {sim} < 0.85"

        run_async(cache.store(
            query_text="base query",
            query_embedding=base,
            result="base result",
            dominant_cluster=0,
            cluster_distribution={0: 0.9},
        ))

        result = run_async(cache.lookup(
            "similar query", similar, dominant_cluster=0,
            cluster_distribution={0: 0.9},
        ))

        assert result is not None
        entry, score = result
        assert score >= 0.85
        assert entry.query_text == "base query"

    def test_dissimilar_embedding_is_miss(self, cache, random_embedding):
        
        emb1 = random_embedding(seed=1)
        emb2 = random_embedding(seed=999)

        sim = float(emb1 @ emb2)
        assert sim < 0.85, f"Test setup issue: similarity {sim} >= 0.85"

        run_async(cache.store(
            query_text="query 1",
            query_embedding=emb1,
            result="result 1",
            dominant_cluster=0,
            cluster_distribution={0: 0.9},
        ))

        result = run_async(cache.lookup(
            "query 2", emb2, dominant_cluster=0,
            cluster_distribution={0: 0.9},
        ))

        assert result is None


class TestClusterPartitioning:


    def test_lookup_only_searches_relevant_clusters(self, cache, random_embedding):
        
        dim = 384
        rng = np.random.RandomState(42)

        base = rng.randn(dim).astype(np.float32)
        base /= np.linalg.norm(base)

        run_async(cache.store(
            query_text="cluster 0 query",
            query_embedding=base.copy(),
            result="cluster 0 result",
            dominant_cluster=0,
            cluster_distribution={0: 0.95},
        ))

        result = run_async(cache.lookup(
            "cluster 5 query", base.copy(), dominant_cluster=5,
            cluster_distribution={5: 0.95},
        ))

        assert result is None

    def test_cross_cluster_search_with_depth(self, cache, random_embedding):
        
        dim = 384
        rng = np.random.RandomState(42)

        base = rng.randn(dim).astype(np.float32)
        base /= np.linalg.norm(base)

        run_async(cache.store(
            query_text="stored in cluster 0",
            query_embedding=base.copy(),
            result="result",
            dominant_cluster=0,
            cluster_distribution={0: 0.95},
        ))

        result = run_async(cache.lookup(
            "query", base.copy(), dominant_cluster=1,
            cluster_distribution={1: 0.55, 0: 0.40},
        ))

        assert result is not None


class TestLRUEviction:


    def test_eviction_at_capacity(self, random_embedding):
        
        cache = SemanticCache(threshold=0.85, max_size=3, cluster_search_depth=1)

        for i in range(3):
            emb = random_embedding(seed=i)
            run_async(cache.store(
                query_text=f"query {i}",
                query_embedding=emb,
                result=f"result {i}",
                dominant_cluster=0,
                cluster_distribution={0: 0.9},
            ))

        assert cache.get_stats()["total_entries"] == 3

        emb = random_embedding(seed=100)
        run_async(cache.store(
            query_text="query 3",
            query_embedding=emb,
            result="result 3",
            dominant_cluster=0,
            cluster_distribution={0: 0.9},
        ))

        assert cache.get_stats()["total_entries"] == 3

    def test_lru_order_updated_on_access(self, random_embedding):
        
        cache = SemanticCache(threshold=0.85, max_size=3, cluster_search_depth=1)

        embs = [random_embedding(seed=i) for i in range(3)]

        for i in range(3):
            run_async(cache.store(
                query_text=f"query {i}",
                query_embedding=embs[i],
                result=f"result {i}",
                dominant_cluster=0,
                cluster_distribution={0: 0.9},
            ))

        run_async(cache.lookup(
            "query 0", embs[0], dominant_cluster=0,
        ))

        emb_new = random_embedding(seed=100)
        run_async(cache.store(
            query_text="query new",
            query_embedding=emb_new,
            result="new",
            dominant_cluster=0,
            cluster_distribution={0: 0.9},
        ))

        result = run_async(cache.lookup(
            "query 0", embs[0], dominant_cluster=0,
        ))
        assert result is not None


class TestStatistics:


    def test_miss_increments_count(self, cache, random_embedding):
        emb = random_embedding(seed=1)
        run_async(cache.lookup("test", emb, dominant_cluster=0))
        stats = cache.get_stats()
        assert stats["miss_count"] == 1
        assert stats["hit_count"] == 0

    def test_hit_increments_count(self, cache, random_embedding):
        emb = random_embedding(seed=1)
        run_async(cache.store(
            "test", emb, "result", 0, {0: 0.9},
        ))

        run_async(cache.lookup("test", emb, dominant_cluster=0))

        stats = cache.get_stats()
        assert stats["hit_count"] == 1
        assert stats["hit_rate"] > 0

    def test_hit_rate_calculation(self, cache, random_embedding):
        emb = random_embedding(seed=1)
        run_async(cache.store(
            "test", emb, "result", 0, {0: 0.9},
        ))

        run_async(cache.lookup("test", emb, dominant_cluster=0))
        run_async(cache.lookup(
            "other", random_embedding(seed=999), dominant_cluster=0,
        ))

        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.5

    def test_cluster_distribution_in_stats(self, cache, random_embedding):
        for i in range(5):
            emb = random_embedding(seed=i)
            cluster = 0 if i < 3 else 1
            run_async(cache.store(
                f"q{i}", emb, f"r{i}", cluster, {cluster: 0.9},
            ))

        stats = cache.get_stats()
        assert stats["cluster_distribution"][0] == 3
        assert stats["cluster_distribution"][1] == 2


class TestCacheClear:


    def test_clear_removes_all_entries(self, cache, random_embedding):
        for i in range(5):
            emb = random_embedding(seed=i)
            run_async(cache.store(
                f"q{i}", emb, f"r{i}", 0, {0: 0.9},
            ))

        removed = run_async(cache.clear())
        assert removed == 5

        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0

    def test_clear_returns_count(self, cache, random_embedding):
        for i in range(3):
            emb = random_embedding(seed=i)
            run_async(cache.store(
                f"q{i}", emb, f"r{i}", 0, {0: 0.9},
            ))

        removed = run_async(cache.clear())
        assert removed == 3
