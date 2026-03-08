

from pathlib import Path
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parent.parent



class Settings(BaseSettings):


    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    chroma_persist_dir: str = str(PROJECT_ROOT / "data" / "chromadb")
    chroma_collection_name: str = "newsgroups"

    pca_components: int = 50

    n_clusters: int = 18

    gmm_covariance_type: str = "full"

    cache_similarity_threshold: float = 0.60

    cache_max_size: int = 10000

    cache_cluster_search_depth: int = 2

    search_top_k: int = 5

    models_dir: str = str(PROJECT_ROOT / "data" / "models")
    plots_dir: str = str(PROJECT_ROOT / "data" / "plots")

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_file": str(PROJECT_ROOT / ".env"), "extra": "ignore"}


settings = Settings()

