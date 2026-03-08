

from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:


    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self.model: Optional[SentenceTransformer] = None
        self.dimension = settings.embedding_dim

    def load_model(self) -> None:

        if self.model is not None:
            return

        print(f"[Embedder] Loading model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Model loaded. Dimension: {self.dimension}")

    def embed_query(self, query: str) -> np.ndarray:

        assert self.model is not None, "Model not loaded. Call load_model() first."

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embedding, dtype=np.float32)

    def embed_documents(
        self,
        documents: List[str],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:

        assert self.model is not None, "Model not loaded. Call load_model() first."

        print(f"[Embedder] Encoding {len(documents)} documents (batch_size={batch_size})...")

        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )

        embeddings = np.array(embeddings, dtype=np.float32)
        print(f"[Embedder] Encoding complete. Shape: {embeddings.shape}")

        return embeddings
