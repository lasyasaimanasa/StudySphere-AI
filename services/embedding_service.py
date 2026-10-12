"""Create and search a FAISS index of StudySphere-AI document chunks.

The first EmbeddingService() initialization downloads the sentence-transformer
model if it is not already cached. An internet connection is needed the first time.
"""

import json
import os
import sqlite3
import unicodedata
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from config import Config


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIMENSION = 384

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(PROJECT_ROOT, "data", "vector_store")
INDEX_PATH = os.path.join(VECTOR_DIR, "chunks.faiss")
METADATA_PATH = os.path.join(VECTOR_DIR, "chunks_metadata.json")


class EmbeddingService:
    """Build, persist, and query semantic embeddings for stored document chunks."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        weak_match_threshold: float = 0.3,
        vector_dir: Optional[str] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        model: Optional[Any] = None,
        project_root: Optional[str] = None,
    ) -> None:
        self.project_root = project_root or PROJECT_ROOT
        self.vector_dir = vector_dir or os.path.join(self.project_root, "data", "vector_store")
        self.index_path = index_path or os.path.join(self.vector_dir, "chunks.faiss")
        self.metadata_path = metadata_path or os.path.join(self.vector_dir, "chunks_metadata.json")
        self.dimension = MODEL_DIMENSION
        self.weak_match_threshold = weak_match_threshold

        os.makedirs(self.vector_dir, exist_ok=True)

        if model is not None:
            self.model = model
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []
        self.load_index()

    def load_index(self) -> None:
        """Load a previously saved FAISS index and its matching metadata."""
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            return

        try:
            loaded_index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as metadata_file:
                loaded_metadata = json.load(metadata_file)

            if loaded_index.d != self.dimension:
                raise ValueError(
                    f"Saved index dimension {loaded_index.d} does not match "
                    f"expected dimension {self.dimension}."
                )

            if loaded_index.ntotal != len(loaded_metadata):
                raise ValueError(
                    f"FAISS index and metadata have different lengths: "
                    f"{loaded_index.ntotal} vectors vs {len(loaded_metadata)} metadata items."
                )

            self.index = loaded_index
            self.metadata = loaded_metadata
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
            # A corrupt or incompatible index resets to an empty synchronized state
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []

    def save_index(self) -> None:
        """Persist the FAISS index and metadata atomically to disk."""
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Cannot save unsynchronized index: FAISS ntotal ({self.index.ntotal}) "
                f"does not match metadata count ({len(self.metadata)})."
            )

        os.makedirs(self.vector_dir, exist_ok=True)
        temporary_index_path = self.index_path + ".tmp"
        temporary_metadata_path = self.metadata_path + ".tmp"

        try:
            faiss.write_index(self.index, temporary_index_path)
            with open(temporary_metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(self.metadata, metadata_file, ensure_ascii=False, indent=2)

            os.replace(temporary_index_path, self.index_path)
            os.replace(temporary_metadata_path, self.metadata_path)
        except Exception:
            # Clean up temporary files on failure so no inconsistent files remain
            if os.path.exists(temporary_index_path):
                try:
                    os.remove(temporary_index_path)
                except OSError:
                    pass
            if os.path.exists(temporary_metadata_path):
                try:
                    os.remove(temporary_metadata_path)
                except OSError:
                    pass
            raise

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """Ensure vectors are float32 and L2-normalized for cosine similarity via inner product."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    def _fetch_chunks_from_db(
        self,
        material_id: Optional[int] = None,
        database_path: Optional[str] = None
    ) -> List[sqlite3.Row]:
        """Fetch non-empty document chunks from SQLite database."""
        db_path = database_path or Config.DATABASE_PATH
        if not os.path.isabs(db_path):
            db_path = os.path.join(self.project_root, db_path)

        if not os.path.exists(db_path):
            return []

        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        try:
            chunk_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(document_chunks)")
            }
            if {"location_type", "location_value"}.issubset(chunk_columns):
                location_select = (
                    "COALESCE(dc.location_type, 'page') AS location_type, "
                    "COALESCE(dc.location_value, CAST(dc.page_number AS TEXT)) "
                    "AS location_value,"
                )
            else:
                # Keep compatibility with older temporary databases and callers
                # that have not run init_db() yet.
                location_select = (
                    "'page' AS location_type, "
                    "CAST(dc.page_number AS TEXT) AS location_value,"
                )

            if material_id is not None:
                return connection.execute(
                    f"""
                    SELECT
                        dc.id AS chunk_id,
                        dc.material_id,
                        dc.chunk_index,
                        dc.content,
                        dc.source_type,
                        COALESCE(dc.source_file, m.filename, '') AS source_file,
                        dc.page_number,
                        {location_select}
                        m.filename
                    FROM document_chunks AS dc
                    LEFT JOIN materials AS m ON m.id = dc.material_id
                    WHERE dc.material_id = ? AND TRIM(dc.content) != ''
                    ORDER BY dc.chunk_index, dc.id
                    """,
                    (material_id,),
                ).fetchall()
            else:
                return connection.execute(
                    f"""
                    SELECT
                        dc.id AS chunk_id,
                        dc.material_id,
                        dc.chunk_index,
                        dc.content,
                        dc.source_type,
                        COALESCE(dc.source_file, m.filename, '') AS source_file,
                        dc.page_number,
                        {location_select}
                        m.filename
                    FROM document_chunks AS dc
                    LEFT JOIN materials AS m ON m.id = dc.material_id
                    WHERE TRIM(dc.content) != ''
                    ORDER BY dc.material_id, dc.chunk_index, dc.id
                    """
                ).fetchall()
        finally:
            connection.close()

    def index_chunks(self, material_id: int, chunks: List[Dict[str, Any]]) -> int:
        """Incrementally index or re-index chunks for a specific material.

        Preserves existing vectors for other materials and avoids duplicate vectors
        if this material was previously indexed.

        Args:
            material_id: Unique identifier of the material.
            chunks: List of chunk dictionaries containing 'content' and optional metadata.

        Returns:
            Number of newly indexed chunks for this material.
        """
        valid_chunks = [
            c for c in chunks
            if c.get("content") and str(c.get("content")).strip()
        ]

        # Extract existing vectors for all materials EXCEPT the target material_id
        if self.index.ntotal > 0 and len(self.metadata) == self.index.ntotal:
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            kept_indices = [
                i for i, meta in enumerate(self.metadata)
                if meta.get("material_id") != material_id
            ]
            kept_vectors = all_vectors[kept_indices]
            kept_metadata = [self.metadata[i] for i in kept_indices]
        else:
            kept_vectors = np.empty((0, self.dimension), dtype=np.float32)
            kept_metadata = []

        new_metadata: List[Dict[str, Any]] = []
        if valid_chunks:
            texts = [str(c["content"]) for c in valid_chunks]
            raw_vectors = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 10,
            )
            new_vectors = self._normalize_vectors(raw_vectors)

            for c in valid_chunks:
                page_number = c.get("page_number") or c.get("page", 1)
                location_value = c.get("location_value")
                if location_value is None:
                    location_value = str(page_number)
                new_metadata.append(
                    {
                        "chunk_id": c.get("chunk_id"),
                        "material_id": material_id,
                        "chunk_index": c.get("chunk_index", 0),
                        "source_type": c.get("source_type", "textbook"),
                        "source_file": c.get("source_file") or c.get("filename", ""),
                        "page_number": page_number,
                        "location_type": c.get("location_type") or "page",
                        "location_value": location_value,
                        "content": str(c["content"]),
                    }
                )
        else:
            new_vectors = np.empty((0, self.dimension), dtype=np.float32)

        # Assemble new unified index
        updated_index = faiss.IndexFlatIP(self.dimension)
        if len(kept_vectors) > 0:
            updated_index.add(kept_vectors)
        if len(new_vectors) > 0:
            updated_index.add(new_vectors)

        self.index = updated_index
        self.metadata = kept_metadata + new_metadata
        self.save_index()
        return len(new_metadata)

    def index_material(
        self,
        material_id: int,
        database_path: Optional[str] = None
    ) -> int:
        """Incrementally index all non-empty chunks for a specific material from SQLite.

        Args:
            material_id: ID of the material to index.
            database_path: Optional override path to the SQLite database.

        Returns:
            Number of indexed chunks for this material.
        """
        rows = self._fetch_chunks_from_db(material_id=material_id, database_path=database_path)
        chunks = [dict(row) for row in rows]
        return self.index_chunks(material_id=material_id, chunks=chunks)

    def remove_material(self, material_id: int) -> int:
        """Remove all indexed chunks for a specific material from index and metadata.

        Args:
            material_id: ID of the material whose chunks should be removed.

        Returns:
            Number of removed chunks.
        """
        if self.index.ntotal == 0 or len(self.metadata) == 0:
            return 0

        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
        kept_indices = [
            i for i, meta in enumerate(self.metadata)
            if meta.get("material_id") != material_id
        ]
        removed_count = self.index.ntotal - len(kept_indices)
        if removed_count == 0:
            return 0

        kept_vectors = all_vectors[kept_indices]
        kept_metadata = [self.metadata[i] for i in kept_indices]

        updated_index = faiss.IndexFlatIP(self.dimension)
        if len(kept_vectors) > 0:
            updated_index.add(kept_vectors)

        self.index = updated_index
        self.metadata = kept_metadata
        self.save_index()
        return removed_count

    def build_index(
        self,
        material_id: Optional[int] = None,
        database_path: Optional[str] = None
    ) -> int:
        """Rebuild or update the vector index.

        If material_id is provided, safely updates only that material's vectors
        without erasing any other materials.
        If material_id is None, rebuilds the entire index from scratch from SQLite.

        Args:
            material_id: If provided, only index chunks from this specific material.
            database_path: Optional override path to the SQLite database.

        Returns:
            Number of indexed chunks.
        """
        if material_id is not None:
            return self.index_material(material_id=material_id, database_path=database_path)

        rows = self._fetch_chunks_from_db(material_id=None, database_path=database_path)
        new_index = faiss.IndexFlatIP(self.dimension)
        new_metadata: List[Dict[str, Any]] = []

        if rows:
            texts = [row["content"] for row in rows]
            raw_vectors = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 10,
            )
            vectors = self._normalize_vectors(raw_vectors)
            new_index.add(vectors)

            for row in rows:
                new_metadata.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "material_id": row["material_id"],
                        "chunk_index": row["chunk_index"],
                        "source_type": row["source_type"],
                        "source_file": row["source_file"] or row["filename"] or "",
                        "page_number": row["page_number"],
                        "location_type": row["location_type"],
                        "location_value": row["location_value"],
                        "content": row["content"],
                    }
                )

        self.index = new_index
        self.metadata = new_metadata
        self.save_index()
        return len(self.metadata)

    def search(
        self,
        query: str,
        top_k: int = 5,
        material_id: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Return the most relevant chunks for a natural-language query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            material_id: If provided, only search within this material.
            min_score: Minimum similarity score threshold (0.0 to 1.0).

        Returns:
            List of results with chunk_id, material_id, chunk_index, source_file,
            page_number, content, and similarity score.
        """
        if not query or not isinstance(query, str) or not query.strip():
            return []

        if top_k is None or not isinstance(top_k, int) or top_k <= 0:
            return []

        if self.index is None or self.index.ntotal == 0 or len(self.metadata) == 0:
            return []

        raw_query_vector = self.model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_vector = self._normalize_vectors(raw_query_vector)

        # Fix material-filtered search candidate retrieval:
        # If filtering by material_id, search the entire index so candidates belonging
        # to this material are not starved by a small global top_k.
        total_vectors = self.index.ntotal
        if material_id is not None or total_vectors <= 1000:
            candidate_depth = total_vectors
        else:
            candidate_depth = min(total_vectors, max(top_k * 5, 20))

        threshold = min_score if min_score is not None else self.weak_match_threshold

        while True:
            scores, indices = self.index.search(query_vector, candidate_depth)
            results: List[Dict[str, Any]] = []
            seen_chunk_ids = set()
            seen_passages = set()

            # Re-evaluate the ranked prefix after each expansion. This preserves
            # FAISS order even if equal-score candidates move at the search boundary.
            for score, index_position in zip(scores[0], indices[0]):
                if index_position < 0 or index_position >= len(self.metadata):
                    continue

                metadata = self.metadata[int(index_position)]
                if not isinstance(metadata, dict):
                    continue

                if material_id is not None and metadata.get("material_id") != material_id:
                    continue

                score_float = float(score)
                if score_float < threshold:
                    continue

                chunk_id = metadata.get("chunk_id")
                chunk_key = (metadata.get("material_id"), chunk_id)
                if chunk_id is not None and chunk_key in seen_chunk_ids:
                    continue

                content = metadata.get("content")
                if isinstance(content, str) and content.strip():
                    normalized_content = " ".join(
                        unicodedata.normalize("NFC", content).split()
                    )
                    passage_key = (metadata.get("material_id"), normalized_content)
                    if passage_key in seen_passages:
                        continue
                    seen_passages.add(passage_key)
                elif chunk_id is None:
                    fallback_key = (
                        metadata.get("material_id"),
                        metadata.get("chunk_index"),
                        content,
                    )
                    if fallback_key in seen_passages:
                        continue
                    seen_passages.add(fallback_key)

                if chunk_id is not None:
                    seen_chunk_ids.add(chunk_key)

                result = dict(metadata)
                result["location_type"] = result.get("location_type") or "page"
                if result.get("location_value") is None:
                    page_number = result.get("page_number")
                    if page_number is not None:
                        result["location_value"] = str(page_number)
                result["score"] = score_float
                results.append(result)

                if len(results) >= top_k:
                    break

            if len(results) >= top_k or candidate_depth >= total_vectors:
                return results

            candidate_depth = min(total_vectors, max(candidate_depth + 1, candidate_depth * 2))

    def set_weak_match_threshold(self, threshold: float) -> None:
        """Update the weak match threshold for search results."""
        if 0.0 <= threshold <= 1.0:
            self.weak_match_threshold = threshold
