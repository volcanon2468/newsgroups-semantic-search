



import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings


@dataclass
class CacheEntry:


    query_text: str
    embedding: np.ndarray
    result: Any
    dominant_cluster: int
    cluster_distribution: Dict[int, float]
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class SemanticCache:


    def __init__(
        self,
        threshold: Optional[float] = None,
        max_size: Optional[int] = None,
        cluster_search_depth: Optional[int] = None,
    ):
        self.threshold = threshold or settings.cache_similarity_threshold
        self.max_size = max_size or settings.cache_max_size
        self.cluster_search_depth = (
            cluster_search_depth or settings.cache_cluster_search_depth
        )

        
        
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

        
        self._cluster_index: Dict[int, List[str]] = {}

        
        
        self._cluster_matrices: Dict[int, Tuple[List[str], np.ndarray]] = {}

        
        self._hit_count: int = 0
        self._miss_count: int = 0

        
        self._lock = asyncio.Lock()

    

    async def lookup(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        dominant_cluster: int,
        cluster_distribution: Optional[Dict[int, float]] = None,
    ) -> Optional[Tuple[CacheEntry, float]]:

        async with self._lock:
            
            key = self._hash_query(query_text)
            if key in self._entries:
                entry = self._entries[key]
                entry.last_accessed = time.time()
                self._entries.move_to_end(key)
                self._hit_count += 1
                return entry, 1.0

            
            clusters_to_search = self._get_search_clusters(
                dominant_cluster, cluster_distribution
            )

            
            best_entry: Optional[CacheEntry] = None
            best_similarity: float = -1.0

            for cluster_id in clusters_to_search:
                if cluster_id not in self._cluster_index:
                    continue
                if not self._cluster_index[cluster_id]:
                    continue

                
                keys, matrix = self._get_cluster_matrix(cluster_id)
                if matrix is None or len(keys) == 0:
                    continue

                
                similarities = matrix @ query_embedding

                max_idx = np.argmax(similarities)
                max_sim = float(similarities[max_idx])

                if max_sim > best_similarity:
                    best_similarity = max_sim
                    best_entry = self._entries.get(keys[max_idx])

            
            if best_entry is not None and best_similarity >= self.threshold:
                
                best_key = self._hash_query(best_entry.query_text)
                best_entry.last_accessed = time.time()
                self._entries.move_to_end(best_key)
                self._hit_count += 1
                return best_entry, best_similarity

            
            self._miss_count += 1
            return None

    async def store(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        result: Any,
        dominant_cluster: int,
        cluster_distribution: Dict[int, float],
    ) -> None:

        async with self._lock:
            key = self._hash_query(query_text)

            
            if key in self._entries:
                self._entries[key].result = result
                self._entries[key].last_accessed = time.time()
                self._entries.move_to_end(key)
                return

            
            if len(self._entries) >= self.max_size:
                self._evict_lru()

            
            entry = CacheEntry(
                query_text=query_text,
                embedding=query_embedding.copy(),
                result=result,
                dominant_cluster=dominant_cluster,
                cluster_distribution=cluster_distribution,
            )

            self._entries[key] = entry

            
            if dominant_cluster not in self._cluster_index:
                self._cluster_index[dominant_cluster] = []
            self._cluster_index[dominant_cluster].append(key)

            
            self._cluster_matrices.pop(dominant_cluster, None)

    async def clear(self) -> int:

        async with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._cluster_index.clear()
            self._cluster_matrices.clear()
            self._hit_count = 0
            self._miss_count = 0
            return count

    def get_stats(self) -> Dict[str, Any]:

        total_requests = self._hit_count + self._miss_count
        hit_rate = (
            self._hit_count / total_requests if total_requests > 0 else 0.0
        )

        
        cluster_dist = {
            int(k): len(v) for k, v in self._cluster_index.items() if v
        }

        return {
            "total_entries": len(self._entries),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(hit_rate, 4),
            "cluster_distribution": cluster_dist,
        }



    @staticmethod
    def _hash_query(query_text: str) -> str:
        
        return hashlib.sha256(query_text.strip().lower().encode("utf-8")).hexdigest()

    def _get_search_clusters(
        self,
        dominant_cluster: int,
        cluster_distribution: Optional[Dict[int, float]] = None,
    ) -> List[int]:
        
        if cluster_distribution and len(cluster_distribution) > 1:
            
            sorted_clusters = sorted(
                cluster_distribution.items(), key=lambda x: x[1], reverse=True
            )
            return [
                int(c) for c, _ in sorted_clusters[: self.cluster_search_depth]
            ]
        return [dominant_cluster]

    def _get_cluster_matrix(
        self, cluster_id: int
    ) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
        
        if cluster_id in self._cluster_matrices:
            return self._cluster_matrices[cluster_id]

        keys = self._cluster_index.get(cluster_id, [])
        if not keys:
            return None, None

        
        valid_keys = [k for k in keys if k in self._entries]

        if not valid_keys:
            self._cluster_index[cluster_id] = []
            return None, None

        
        matrix = np.vstack(
            [self._entries[k].embedding for k in valid_keys]
        )

        
        self._cluster_matrices[cluster_id] = (valid_keys, matrix)

        
        self._cluster_index[cluster_id] = valid_keys

        return valid_keys, matrix

    def _evict_lru(self) -> None:

        if not self._entries:
            return

        
        key, entry = self._entries.popitem(last=False)

        
        cluster_id = entry.dominant_cluster
        if cluster_id in self._cluster_index:
            try:
                self._cluster_index[cluster_id].remove(key)
            except ValueError:
                pass


            self._cluster_matrices.pop(cluster_id, None)
