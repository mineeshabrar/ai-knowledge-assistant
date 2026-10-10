import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    min_similarity: float = float(os.getenv("MIN_SIMILARITY", "0.30"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_MB", "15")) * 1024 * 1024

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("RETRIEVAL_TOP_K must be at least 1")
        if not -1 <= self.min_similarity <= 1:
            raise ValueError("MIN_SIMILARITY must be between -1 and 1")
        if self.chunk_size < 1 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Chunk overlap must be non-negative and smaller than chunk size")
        if self.max_upload_bytes < 1:
            raise ValueError("MAX_UPLOAD_MB must be at least 1")


settings = Settings()
