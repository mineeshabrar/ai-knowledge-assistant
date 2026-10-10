import json
import os
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings
from app.documents import TextChunk


class VectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.settings.data_dir / "knowledge.faiss"
        self.metadata_path = self.settings.data_dir / "chunks.json"
        self.documents_path = self.settings.data_dir / "documents.json"
        self._lock = threading.RLock()
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._chunks: list[dict[str, Any]] = []
        self._documents: dict[str, dict[str, Any]] = {}
        self._load()

    def _embedding_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.settings.embedding_model)
        return self._model

    def _load(self) -> None:
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
        if self.metadata_path.exists():
            self._chunks = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.documents_path.exists():
            self._documents = json.loads(
                self.documents_path.read_text(encoding="utf-8")
            )
        if self._index is None and (self._chunks or self._documents):
            raise RuntimeError("Knowledge metadata exists without its FAISS index")
        if self._index is not None and self._index.ntotal != len(self._chunks):
            raise RuntimeError("FAISS index and chunk metadata are out of sync")

    def add_document(
        self, filename: str, chunks: list[TextChunk]
    ) -> dict[str, Any]:
        if not chunks:
            raise ValueError("No readable text was found in the document")

        with self._lock:
            vectors = self._embedding_model().encode(
                [chunk.text for chunk in chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
                raise RuntimeError("Embedding model returned an invalid vector shape")

            if self._index is None:
                self._index = faiss.IndexFlatIP(vectors.shape[1])
            elif self._index.d != vectors.shape[1]:
                raise RuntimeError("Embedding dimensions do not match the saved index")

            document_id = str(uuid4())
            document = {
                "id": document_id,
                "filename": filename,
                "chunk_count": len(chunks),
            }
            entries = [
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                }
                for chunk in chunks
            ]
            self._index.add(vectors)
            self._chunks.extend(entries)
            self._documents[document_id] = document
            self._persist()
            return document

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            vector = self._embedding_model().encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            )
            scores, indices = self._index.search(
                np.asarray(vector, dtype=np.float32), top_k
            )
            results = []
            for score, index in zip(scores[0], indices[0]):
                if index < 0:
                    continue
                results.append({**self._chunks[int(index)], "score": float(score)})
            return results

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(
                self._documents.values(), key=lambda document: document["filename"].lower()
            )

    def _persist(self) -> None:
        assert self._index is not None
        self._atomic_write_json(self.metadata_path, self._chunks)
        self._atomic_write_json(self.documents_path, self._documents)
        temporary_index = self.index_path.with_suffix(".faiss.tmp")
        faiss.write_index(self._index, str(temporary_index))
        os.replace(temporary_index, self.index_path)

    @staticmethod
    def _atomic_write_json(path: Path, value: Any) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_path, path)
