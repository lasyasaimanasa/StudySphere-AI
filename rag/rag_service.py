"""RAG service for source-grounded answer generation."""

import json
import re
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from ai.llm_provider import get_llm_provider, LLMProviderError, LLMConfigurationError
from database import get_db_connection
from rag.prompts import SYSTEM_PROMPT, build_user_prompt, get_source_location


_BRACKETED_REFERENCE = re.compile(r"\[([^\]\r\n]+)\]")
_TYPED_REFERENCE = re.compile(r"^(.*),\s*(page|slide|timestamp)\s+(.+?)\s*$", re.IGNORECASE)
_TIMESTAMP = re.compile(r"^\d{2,}:[0-5]\d:[0-5]\d$")
_INSUFFICIENT_ANSWER = (
    "The uploaded materials do not contain enough relevant information to answer this question."
)


class RAGService:
    """Service for retrieval-augmented generation with source grounding."""

    def __init__(self, embedding_service=None, database_path=None):
        """Initialize RAG service.

        Args:
            embedding_service: Optional EmbeddingService instance for retrieval.
                              If None, must be set before calling generate_answer.
        """
        self.embedding_service = embedding_service
        self.database_path = database_path

    def set_embedding_service(self, embedding_service):
        """Set the embedding service for retrieval."""
        self.embedding_service = embedding_service

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        material_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate a source-grounded answer for a query.

        Args:
            query: The student's question.
            retrieved_chunks: Optional pre-retrieved chunks. If None, performs retrieval.
            top_k: Number of chunks to retrieve if retrieved_chunks is None.
            material_id: Optional material ID to filter retrieval.

        Returns:
            Dictionary with:
                - answer: Generated answer text
                - citations: List of unique citations used
                - sources: List of source metadata from retrieved chunks
                - insufficient_context: Boolean indicating if context was insufficient
                - error: Optional error message if generation failed
        """
        # Step 1: Retrieve chunks if not provided
        if retrieved_chunks is None:
            if self.embedding_service is None:
                return {
                    "answer": None,
                    "citations": [],
                    "sources": [],
                    "insufficient_context": False,
                    "error": "Embedding service not configured for retrieval"
                }

            try:
                retrieved_chunks = self.embedding_service.search(
                    query=query,
                    top_k=top_k,
                    material_id=material_id
                )
            except Exception as e:
                return {
                    "answer": None,
                    "citations": [],
                    "sources": [],
                    "insufficient_context": False,
                    "error": f"Retrieval failed: {str(e)}"
                }

        # Retrieval metadata must contain both actual passage text and a usable
        # source locator. A URL or passage without a locator is never evidence.
        try:
            retrieved_chunks = self._validate_stored_chunks(
                retrieved_chunks, requested_material_id=material_id
            )
        except (sqlite3.Error, OSError, ValueError) as exc:
            return {
                "answer": None,
                "citations": [],
                "sources": [],
                "insufficient_context": False,
                "error": f"Evidence validation failed: {exc}",
            }
        if not retrieved_chunks:
            return self._insufficient_result()

        # Step 3: Generate answer using LLM
        try:
            llm_provider = get_llm_provider()
        except LLMConfigurationError as e:
            return {
                "answer": None,
                "citations": [],
                "sources": retrieved_chunks,
                "insufficient_context": False,
                "error": f"LLM configuration error: {str(e)}"
            }

        try:
            # Build prompt
            user_prompt = build_user_prompt(query, retrieved_chunks)

            # Generate answer
            output = llm_provider.generate(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT
            )
            answer, citations = self._validated_grounded_answer(output, retrieved_chunks)
            if answer is None:
                return self._insufficient_result(retrieved_chunks)

            return {
                "answer": answer,
                "citations": citations,
                "sources": retrieved_chunks,
                "insufficient_context": False,
                "error": None
            }

        except LLMProviderError as e:
            return {
                "answer": None,
                "citations": [],
                "sources": retrieved_chunks,
                "insufficient_context": False,
                "error": f"Answer generation failed: {str(e)}"
            }
        except Exception as e:
            return {
                "answer": None,
                "citations": [],
                "sources": retrieved_chunks,
                "insufficient_context": False,
                "error": f"Unexpected error: {str(e)}"
            }

    @staticmethod
    def _insufficient_result(sources=None):
        return {
            "answer": _INSUFFICIENT_ANSWER,
            "citations": [],
            "sources": sources or [],
            "insufficient_context": True,
            "error": None,
        }

    @staticmethod
    def _usable_chunks(chunks):
        """Keep passages with content and a valid, explicit citation locator."""
        if not isinstance(chunks, list):
            return []
        usable = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            content = chunk.get("content")
            source_file = chunk.get("source_file")
            page_number = chunk.get("page_number")
            source_type = chunk.get("source_type")
            chunk_id = chunk.get("chunk_id")
            material_id = chunk.get("material_id")
            chunk_index = chunk.get("chunk_index")
            if (isinstance(chunk_id, bool) or not isinstance(chunk_id, int) or chunk_id <= 0
                    or isinstance(material_id, bool) or not isinstance(material_id, int) or material_id <= 0
                    or isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0):
                continue
            raw_type = chunk.get("location_type")
            location_type, location_value = get_source_location(chunk)
            if not isinstance(content, str) or not content.strip():
                continue
            if not isinstance(source_file, str) or not source_file.strip():
                continue
            if (isinstance(page_number, bool) or not isinstance(page_number, int)
                    or page_number < 1 or not isinstance(source_type, str) or not source_type.strip()):
                continue
            if raw_type not in (None, "", "page", "slide", "timestamp"):
                continue
            if location_value is None or not str(location_value).strip():
                continue
            locator = str(location_value).strip()
            if location_type in {"page", "slide"}:
                if not locator.isdigit() or int(locator) < 1:
                    continue
            elif location_type == "timestamp":
                if not _TIMESTAMP.fullmatch(locator):
                    continue
            else:
                continue
            normalized = dict(chunk)
            normalized["source_file"] = source_file.strip()
            normalized["location_type"] = location_type
            normalized["location_value"] = locator
            usable.append(normalized)
        return usable

    def _database_connection(self):
        if self.database_path:
            connection = sqlite3.connect(self.database_path)
            connection.row_factory = sqlite3.Row
            return connection
        return get_db_connection()

    def _validate_stored_chunks(self, chunks, requested_material_id=None):
        """Return only indexed passages that exactly match their stored chunk row.

        FAISS metadata is a retrieval cache, not the authoritative source record.
        Missing IDs, deleted rows, stale content, material mismatches, and locator
        mismatches are excluded before any passage reaches the answer model.
        """
        candidates = self._usable_chunks(chunks)
        if not candidates:
            return []

        # Enforce scope here as well as in EmbeddingService so a defective or
        # mocked retrieval implementation cannot widen a selected-material query.
        candidates = [
            chunk for chunk in candidates
            if requested_material_id is None or chunk["material_id"] == requested_material_id
        ]
        if not candidates:
            return []

        rows_by_id = {}
        connection = self._database_connection()
        try:
            chunk_ids = list(dict.fromkeys(chunk["chunk_id"] for chunk in candidates))
            for start in range(0, len(chunk_ids), 500):
                batch = chunk_ids[start:start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"""SELECT dc.id AS chunk_id, dc.material_id, dc.chunk_index,
                               dc.content, dc.source_type,
                               CASE WHEN dc.source_file IS NOT NULL AND dc.source_file <> ''
                                    THEN dc.source_file ELSE COALESCE(m.filename, '') END AS source_file,
                               dc.page_number,
                               CASE WHEN COALESCE(dc.location_type, '') = '' THEN 'page'
                                    ELSE dc.location_type END AS location_type,
                               COALESCE(dc.location_value, CAST(dc.page_number AS TEXT))
                                   AS location_value,
                               m.id AS joined_material_id
                        FROM document_chunks AS dc
                        JOIN materials AS m ON m.id = dc.material_id
                        WHERE dc.id IN ({placeholders})""",
                    batch,
                ).fetchall()
                rows_by_id.update({int(row["chunk_id"]): row for row in rows})
        finally:
            connection.close()

        validated = []
        for chunk in candidates:
            row = rows_by_id.get(chunk["chunk_id"])
            if row is None:
                continue
            row_location_value = row["location_value"]
            if row_location_value is None:
                continue
            matches = (
                int(row["joined_material_id"]) == chunk["material_id"]
                and int(row["material_id"]) == chunk["material_id"]
                and int(row["chunk_index"]) == chunk["chunk_index"]
                and row["content"] == chunk["content"]
                and row["source_type"] == chunk.get("source_type")
                and row["source_file"] == chunk["source_file"]
                and int(row["page_number"]) == chunk["page_number"]
                and row["location_type"] == chunk["location_type"]
                and str(row_location_value) == chunk["location_value"]
            )
            if matches:
                validated.append(chunk)
        return validated

    def _validated_grounded_answer(self, output, retrieved_chunks):
        """Parse the structured model result and bind every claim to exact sources."""
        if not isinstance(output, str):
            return None, []
        try:
            data = json.loads(output.strip())
        except (TypeError, json.JSONDecodeError):
            return None, []
        if not isinstance(data, dict) or set(data) != {"supported", "claims"}:
            return None, []
        if not isinstance(data["supported"], bool) or not isinstance(data["claims"], list):
            return None, []
        if not data["supported"]:
            return (None, []) if not data["claims"] else (None, [])
        if not data["claims"]:
            return None, []

        source_by_id = {f"S{index}": chunk for index, chunk in enumerate(retrieved_chunks, 1)}
        rendered_claims = []
        cited_chunks = []
        seen_citations = set()
        for claim in data["claims"]:
            if not isinstance(claim, dict) or set(claim) != {"text", "source_ids"}:
                return None, []
            text = claim["text"]
            source_ids = claim["source_ids"]
            if (not isinstance(text, str) or not text.strip() or "\n" in text
                    or "\r" in text or _BRACKETED_REFERENCE.search(text)):
                return None, []
            if (not isinstance(source_ids, list) or not source_ids
                    or any(not isinstance(source_id, str) or source_id not in source_by_id
                           for source_id in source_ids)):
                return None, []
            citations_for_claim = []
            for source_id in dict.fromkeys(source_ids):
                chunk = source_by_id[source_id]
                citation_key = self._reference_key(
                    chunk["source_file"], chunk["location_type"], chunk["location_value"]
                )
                citation = self._format_chunk_citation(chunk)
                citations_for_claim.append(citation)
                if citation_key not in seen_citations:
                    seen_citations.add(citation_key)
                    cited_chunks.append(chunk)
            rendered_claims.append(f"{text.strip()} " + " ".join(citations_for_claim))

        return "\n".join(rendered_claims), self._extract_citations(
            " ".join(rendered_claims), cited_chunks
        )

    @staticmethod
    def _format_chunk_citation(chunk):
        return f"[{chunk['source_file']}, {chunk['location_type']} {chunk['location_value']}]"

    def _extract_citations(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract unique citations from the answer based on retrieved chunks.

        Args:
            answer: Generated answer text.
            retrieved_chunks: Retrieved chunks with source metadata.

        Returns:
            List of unique citation dictionaries with filename and page_number.
        """
        references = set()
        for match in _BRACKETED_REFERENCE.finditer(answer or ""):
            parsed = _TYPED_REFERENCE.fullmatch(match.group(1).strip())
            if parsed:
                references.add(self._reference_key(
                    parsed.group(1).strip(), parsed.group(2), parsed.group(3).strip()
                ))

        citations = []
        seen = set()
        for chunk in retrieved_chunks:
            source_file = chunk.get("source_file")
            if not source_file:
                continue
            location_type, location_value = get_source_location(chunk)
            if location_value is None:
                continue
            citation_key = self._reference_key(source_file, location_type, location_value)
            if citation_key not in references or citation_key in seen:
                continue

            seen.add(citation_key)
            citation = {
                "source_file": source_file,
                "page_number": (
                    chunk.get("page_number")
                    if chunk.get("page_number") is not None
                    else location_value
                ),
            }
            if location_type in {"slide", "timestamp"}:
                citation["location_type"] = location_type
                citation["location_value"] = str(location_value)
            citations.append(citation)

        return citations

    @staticmethod
    def _reference_key(source_file: Any, location_type: str, location_value: Any):
        return (
            str(source_file).strip().casefold(),
            location_type.strip().casefold(),
            str(location_value).strip().casefold(),
        )

    def _remove_unretrieved_citations(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        """Remove typed references that do not match retrieved source metadata."""
        allowed_references = set()
        for chunk in retrieved_chunks:
            source_file = chunk.get("source_file")
            location_type, location_value = get_source_location(chunk)
            if source_file and location_value is not None:
                allowed_references.add(
                    self._reference_key(source_file, location_type, location_value)
                )

        def retain_if_retrieved(match):
            parsed = _TYPED_REFERENCE.fullmatch(match.group(1).strip())
            if not parsed:
                return match.group(0)
            key = self._reference_key(
                parsed.group(1).strip(), parsed.group(2), parsed.group(3).strip()
            )
            return match.group(0) if key in allowed_references else ""

        return _BRACKETED_REFERENCE.sub(retain_if_retrieved, answer or "")
