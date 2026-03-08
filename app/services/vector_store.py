

import json
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np

from app.config import settings


class VectorStoreService:


    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None

    def initialize(self) -> None:
        print(f"[VectorStore] Initializing ChromaDB at: {self.persist_dir}")

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        count = self.collection.count()
        print(f"[VectorStore] Collection '{self.collection_name}' ready. Documents: {count}")

    @property
    def document_count(self) -> int:

        if self.collection is None:
            return 0
        return self.collection.count()

    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 500,
    ) -> None:

        assert self.collection is not None, "Collection not initialized"

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        n_docs = len(documents)
        print(f"[VectorStore] Adding {n_docs} documents in batches of {batch_size}...")

        for start in range(0, n_docs, batch_size):
            end = min(start + batch_size, n_docs)
            self.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=metadatas[start:end],
            )
            if (start // batch_size) % 10 == 0:
                print(f"  → Batch {start//batch_size + 1}: {end}/{n_docs} documents")

        print(f"[VectorStore] Successfully added {n_docs} documents.")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where_filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:

        assert self.collection is not None, "Collection not initialized"

        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if where_filter:
            query_params["where"] = where_filter

        results = self.collection.query(**query_params)

        
        formatted = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]


            cluster_dist = None
            if "cluster_distribution" in metadata:
                try:
                    cluster_dist = json.loads(metadata["cluster_distribution"])
                except (json.JSONDecodeError, TypeError):
                    cluster_dist = None

            formatted.append(
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "category": metadata.get("category", "unknown"),
                    "similarity": round(1.0 - results["distances"][0][i], 4),
                    "dominant_cluster": metadata.get("dominant_cluster", -1),
                    "cluster_distribution": cluster_dist,
                }
            )

        return formatted

    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str], List[Dict]]:

        assert self.collection is not None, "Collection not initialized"

        result = self.collection.get(
            include=["embeddings", "metadatas"],
        )

        embeddings = np.array(result["embeddings"], dtype=np.float32)
        return embeddings, result["ids"], result["metadatas"]

    def update_metadata(
        self,
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 500,
    ) -> None:

        assert self.collection is not None, "Collection not initialized"

        print(f"[VectorStore] Updating metadata for {len(ids)} documents...")

        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            self.collection.update(
                ids=ids[start:end],
                metadatas=metadatas[start:end],
            )

        print(f"[VectorStore] Metadata update complete.")
