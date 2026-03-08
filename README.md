# Semantic Search System for 20 Newsgroups

A lightweight semantic search system with fuzzy clustering and a custom semantic cache layer, built for the **Trademarkia AI & ML Engineer** assessment.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI Service                            │
│  POST /query  │  GET /cache/stats  │  DELETE /cache  │  /health  │
└───────┬────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────┐    ┌─────────────────────────────┐
│ Search            │    │ Semantic Cache               │
│ Orchestrator      │◄──►│ (from-scratch, no Redis)     │
│                   │    │                              │
│ embed → cluster   │    │ • Cluster-partitioned lookup │
│ → cache → search  │    │ • Cosine similarity matching │
└───────┬───────────┘    │ • LRU eviction               │
        │                │ • O(N/K) per lookup          │
        ▼                └──────────────┬───────────────┘
┌───────────────┐    ┌────────────────┐ │
│ Embedder       │    │ Clustering     │ │
│ (MiniLM-L6-v2)│    │ (PCA + GMM)    │◄┘
│ 384-dim        │    │ Soft assigns   │
└───────┬────────┘    └────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│ ChromaDB (persistent vector store)     │
│ ~18K documents + metadata + clusters  │
└────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Embedding model** | `all-MiniLM-L6-v2` (384d) | Best speed/quality ratio for broad topics. 5× faster than mpnet, Docker image ~300MB lighter. 384d is sufficient for 20 coarse categories. |
| **Vector store** | ChromaDB | Zero-config persistence, native metadata filtering, in-process (no external server). Perfect for single-container deployment. |
| **Dim. reduction** | PCA → 50d | GMM on 384d is numerically degenerate (more parameters than data). PCA retains ~49% variance (expected for sentence-transformers), is deterministic, and makes GMM tractable. |
| **Clustering** | Gaussian Mixture Model | Produces calibrated probability distributions (not just membership degrees). `predict_proba()` gives true soft assignments per the spec. |
| **Cluster count** | Evidence-based (BIC + silhouette + NMI) | NOT anchored to 20. Real semantic structure ≠ label taxonomy. See `notebooks/cluster_analysis.ipynb` for the full justification. |
| **Cache architecture** | Custom OrderedDict + cluster partitioning | No Redis/external libs. Cluster index makes lookup O(N/K) instead of O(N). LRU eviction via OrderedDict. |
| **Cache threshold (τ)** | 0.60 (tunable) | Empirically determined via query pair analysis. MiniLM on short queries produces lower cosine sim than expected (0.53-0.73 for rephrasings). See notebook & `scripts/analyze_threshold.py`. |

## Project Structure

```
├── app/
│   ├── main.py                  # FastAPI app, endpoints, lifespan
│   ├── config.py                # Pydantic settings (single source of truth)
│   ├── models/
│   │   └── schemas.py           # Request/response Pydantic models
│   ├── services/
│   │   ├── embedder.py          # Sentence-transformer wrapper
│   │   ├── vector_store.py      # ChromaDB operations
│   │   ├── clustering.py        # PCA + GMM fuzzy clustering
│   │   └── search.py            # Query orchestration
│   ├── cache/
│   │   └── semantic_cache.py    # Custom semantic cache (★ core component)
│   └── utils/
│       └── preprocessing.py     # Text cleaning pipeline
├── scripts/
│   ├── setup_data.py            # One-time data pipeline
│   ├── analyze_threshold.py     # Cache threshold empirical analysis
│   └── test_endpoints.py        # Manual API integration tests
├── notebooks/
│   └── cluster_analysis.ipynb   # Evidence & visualizations
├── tests/
│   ├── test_cache.py            # Cache unit tests
│   └── test_api.py              # API integration tests
├── data/                        # Generated at runtime
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quickstart

### 1. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the setup pipeline

This downloads the dataset, cleans it, computes embeddings, stores them in ChromaDB, fits the clustering model, and saves all artifacts:

```bash
python -m scripts.setup_data
```

This takes ~5-10 minutes on CPU (mostly embedding 18K documents). It's idempotent — safe to re-run.

### 4. Start the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server loads the pre-computed models at startup (~3s) and is then ready for queries.

### 5. Test the endpoints

```bash
# Semantic search (first call = cache miss)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest developments in space exploration?"}'

# Same query rephrased (should be cache hit)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Recent news about NASA and space missions"}'

# Cache statistics
curl http://localhost:8000/cache/stats

# Clear cache
curl -X DELETE http://localhost:8000/cache
```

## Docker

### Build and run

```bash
docker build -t semantic-search .
docker run -p 8000:8000 semantic-search
```

### With docker-compose

```bash
docker-compose up --build
```

## API Reference

### `POST /query`

**Request:**
```json
{
  "query": "What are the latest developments in space exploration?"
}
```

**Response (cache miss):**
```json
{
  "query": "What are the latest developments in space exploration?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [
    {
      "document": "The shuttle mission STS-52 launched yesterday...",
      "category": "sci.space",
      "similarity": 0.7823,
      "cluster_distribution": {"3": 0.82, "7": 0.11}
    }
  ],
  "dominant_cluster": 3,
  "cluster_distribution": {"3": 0.85, "7": 0.09, "12": 0.03}
}
```

**Response (cache hit):**
```json
{
  "query": "Recent news about NASA and space missions",
  "cache_hit": true,
  "matched_query": "What are the latest developments in space exploration?",
  "similarity_score": 0.91,
  "result": [...],
  "dominant_cluster": 3,
  "cluster_distribution": {"3": 0.87, "7": 0.08}
}
```

### `GET /cache/stats`

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405,
  "cluster_distribution": {"0": 5, "3": 12, "7": 8, "15": 17}
}
```

### `DELETE /cache`

```json
{
  "message": "Cache cleared successfully",
  "entries_removed": 42
}
```

## The Semantic Cache — In Depth

The cache is the centerpiece of this system. Here's what makes it interesting:

### How it works

1. **Embed the query** → 384-dim vector
2. **Determine cluster membership** → GMM gives a probability distribution over K clusters
3. **Partition-based lookup** → Only search cached queries in the same cluster(s)
4. **Cosine similarity** → If max similarity ≥ τ → cache hit
5. **LRU eviction** → When full, evict least-recently-accessed entries

### Why cluster partitioning matters

Without partitioning: lookup is O(N) — compare against every cached query.
With partitioning: lookup is O(N/K) — only compare within the relevant cluster(s).

At 10,000 cached entries and K=18, that's comparing against ~550 entries instead of 10,000. The speedup scales linearly with cache size.

### The threshold exploration

The threshold τ is the most important parameter. Our analysis (see notebook and `scripts/analyze_threshold.py`) shows:

**Key empirical finding**: MiniLM-L6-v2 produces lower cosine similarities for short rephrased queries than you'd expect. Rephrasings of the same question score 0.53-0.73, while cross-topic queries score 0.05-0.40. This gap determines the viable threshold range.

| τ Range | Behavior | Use Case |
|---------|----------|----------|
| 0.80-0.95 | Near-exact matching only | Maximum safety, but cache barely fires |
| 0.65-0.80 | Catches close rephrasings | Safe but limited recall |
| **0.55-0.65** | **Catches semantic rephrasings** | **The useful operating range for short queries** |
| < 0.50 | Topic-level matching | High hit rate, but wrong results |

At τ=0.60 (our default), we catch rephrasings like:
- "What are the latest developments in space?" ↔ "Recent NASA space missions" (sim ~0.65)
- "What are the new developments in space exploration?" ↔ original (sim ~0.97)

While correctly missing cross-topic queries (sim < 0.40).

## Cluster Analysis Highlights

The analysis notebook (`notebooks/cluster_analysis.ipynb`) contains:

- **BIC/AIC/Silhouette curves** for K ∈ [5, 35]
- **Cluster × Category heatmap** showing how the 20 labels map to discovered clusters
- **Boundary document analysis** — the most ambiguous specimens with high entropy
- **Threshold exploration** — precision/recall curves at various τ values
- **2D visualization** — PCA projection comparing discovered clusters vs. original labels

## Testing

```bash
pytest tests/ -v
```

---

*Built for the Trademarkia AI & ML Engineer assessment.*
