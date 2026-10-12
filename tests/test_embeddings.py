import unittest
import os
import sys
import tempfile
import sqlite3
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestChunkMetadataPreservation(unittest.TestCase):
    """Test that chunk metadata is preserved correctly."""

    def test_chunk_structure(self):
        """Test that chunks have required metadata fields."""
        from ingestion.chunker import Chunker

        chunker = Chunker(chunk_size=100, overlap=20)
        pages = [
            {"page": 1, "text": "Test text for page one. " * 10},
            {"page": 2, "text": "Test text for page two. " * 10}
        ]

        chunks = chunker.chunk_pages(pages, source_file="test.pdf", source_type="textbook")

        self.assertGreater(len(chunks), 0)

        # Check that all chunks have required fields
        for chunk in chunks:
            self.assertIn('chunk_id', chunk)
            self.assertIn('content', chunk)
            self.assertIn('source_type', chunk)
            self.assertIn('source_file', chunk)
            self.assertIn('page', chunk)
            self.assertIn('chunk_index', chunk)

        # Check that chunk_index is globally sequential
        chunk_indices = [chunk['chunk_index'] for chunk in chunks]
        self.assertEqual(len(chunk_indices), len(set(chunk_indices)))
        self.assertEqual(sorted(chunk_indices), list(range(len(chunks))))

        # Check that page numbers are preserved
        page_numbers = [chunk['page'] for chunk in chunks]
        self.assertIn(1, page_numbers)
        self.assertIn(2, page_numbers)


class MockEmbedder:
    """Deterministic mock embedder for fast, isolated vector unit tests."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            # Create a deterministic normalized vector based on string hash
            vec = np.zeros(self.dimension, dtype=np.float32)
            for idx, ch in enumerate(text[: self.dimension]):
                vec[idx] = (ord(ch) % 100) / 100.0

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            else:
                vec[0] = 1.0
            vectors.append(vec)

        arr = np.array(vectors, dtype=np.float32)
        return arr


class RankedCandidateIndex:
    """Small deterministic search double for candidate-depth unit tests."""

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.ntotal = len(self.scores)
        self.depths = []

    def search(self, query_vector, candidate_depth):
        self.depths.append(candidate_depth)
        indices = np.arange(candidate_depth, dtype=np.int64).reshape(1, -1)
        scores = self.scores[:candidate_depth].reshape(1, -1)
        return scores, indices


class TestEmbeddingService(unittest.TestCase):
    """Automated tests for EmbeddingService search, indexing, persistence, and synchronization."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vector_dir = os.path.join(self.temp_dir.name, "vector_store")
        self.index_path = os.path.join(self.vector_dir, "test_chunks.faiss")
        self.metadata_path = os.path.join(self.vector_dir, "test_metadata.json")
        self.mock_model = MockEmbedder(dimension=384)

        from services.embedding_service import EmbeddingService

        self.service = EmbeddingService(
            vector_dir=self.vector_dir,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            model=self.mock_model,
            project_root=self.temp_dir.name,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_query_and_edge_cases(self):
        """Test safe handling of empty, whitespace, and non-string queries."""
        # Empty and invalid queries
        self.assertEqual(self.service.search(""), [])
        self.assertEqual(self.service.search("   "), [])
        self.assertEqual(self.service.search(None), [])
        self.assertEqual(self.service.search(123), [])

        # Invalid top_k values
        self.assertEqual(self.service.search("test", top_k=0), [])
        self.assertEqual(self.service.search("test", top_k=-1), [])
        self.assertEqual(self.service.search("test", top_k=None), [])
        self.assertEqual(self.service.search("test", top_k="invalid"), [])

        # Search on empty index
        self.assertEqual(self.service.search("test", top_k=5), [])

    def test_metadata_retention(self):
        """Test that chunk_id, material_id, chunk_index, source_file, page_number, content, and score are preserved."""
        chunks = [
            {
                "chunk_id": 101,
                "material_id": 1,
                "chunk_index": 0,
                "source_type": "textbook",
                "source_file": "dbms_chapter1.pdf",
                "page_number": 5,
                "content": "A relational database stores data in tabular form with rows and columns.",
            }
        ]
        self.service.index_chunks(material_id=1, chunks=chunks)

        results = self.service.search("relational database", top_k=1, min_score=0.0)
        self.assertEqual(len(results), 1)

        result = results[0]
        self.assertEqual(result["chunk_id"], 101)
        self.assertEqual(result["material_id"], 1)
        self.assertEqual(result["chunk_index"], 0)
        self.assertEqual(result["source_file"], "dbms_chapter1.pdf")
        self.assertEqual(result["page_number"], 5)
        self.assertEqual(result["content"], "A relational database stores data in tabular form with rows and columns.")
        self.assertIn("score", result)
        self.assertIsInstance(result["score"], float)

    def test_search_adds_page_location_fallback_to_legacy_metadata(self):
        self.service.index = RankedCandidateIndex([0.92])
        self.service.metadata = [{
            "chunk_id": 77,
            "material_id": 5,
            "chunk_index": 2,
            "source_file": "legacy.pdf",
            "page_number": 7,
            "content": "Legacy PDF passage",
        }]

        results = self.service.search("legacy", top_k=1, min_score=0.0)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["page_number"], 7)
        self.assertEqual(results[0]["location_type"], "page")
        self.assertEqual(results[0]["location_value"], "7")
        self.assertEqual(results[0]["source_file"], "legacy.pdf")

    def test_timestamp_location_survives_indexing_and_search(self):
        self.service.index_chunks(material_id=1, chunks=[{
            "chunk_id": 801,
            "chunk_index": 0,
            "source_type": "transcript",
            "source_file": "lecture_transcript.txt",
            "page_number": 1,
            "location_type": "timestamp",
            "location_value": "00:01:25",
            "content": "A DBMS organizes stores and retrieves data.",
        }])

        results = self.service.search("DBMS retrieves data", top_k=1, min_score=0.0)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["page_number"], 1)
        self.assertEqual(results[0]["location_type"], "timestamp")
        self.assertEqual(results[0]["location_value"], "00:01:25")
        self.assertEqual(results[0]["source_file"], "lecture_transcript.txt")

    def test_search_suppresses_nfc_and_whitespace_normalized_duplicates(self):
        self.service.index = RankedCandidateIndex([0.95, 0.90, 0.85])
        self.service.metadata = [
            {
                "chunk_id": 10, "material_id": 1, "chunk_index": 0,
                "content": "Caf\u00e9\t  database passage", "source_file": "first.pdf",
                "page_number": 2,
            },
            {
                "chunk_id": 11, "material_id": 1, "chunk_index": 1,
                "content": "Cafe\u0301 database\n passage", "source_file": "duplicate.pdf",
                "page_number": 8,
            },
            {
                "chunk_id": 12, "material_id": 1, "chunk_index": 2,
                "content": "Different passage", "source_file": "other.pdf",
                "page_number": 9,
            },
        ]

        results = self.service.search("query", top_k=3, min_score=0.0)

        self.assertEqual([result["chunk_id"] for result in results], [10, 12])

    def test_search_keeps_identical_text_from_different_materials(self):
        self.service.index = RankedCandidateIndex([0.95, 0.90, 0.85])
        self.service.metadata = [
            {"chunk_id": 20, "material_id": 1, "content": "Exact same passage"},
            {"chunk_id": 21, "material_id": 2, "content": "Exact same passage"},
            {"chunk_id": 22, "material_id": 1, "content": "Exact same passage!"},
        ]

        results = self.service.search("query", top_k=3, min_score=0.0)

        self.assertEqual([result["chunk_id"] for result in results], [20, 21, 22])

    def test_search_progressively_backfills_duplicates_and_below_threshold(self):
        metadata = []
        for index in range(1001):
            if index < 19:
                content = "Repeated passage"
            elif index == 19:
                content = "Below threshold passage"
            else:
                content = f"Unique passage {index}"
            metadata.append({
                "chunk_id": index,
                "material_id": 1,
                "chunk_index": index,
                "content": content,
            })
        scores = [0.99 - index * 0.001 for index in range(1001)]
        scores[19] = 0.1
        self.service.index = RankedCandidateIndex(scores)
        self.service.metadata = metadata

        results = self.service.search("query", top_k=3, min_score=0.5)

        self.assertEqual([result["chunk_id"] for result in results], [0, 20, 21])
        self.assertEqual(self.service.index.depths, [20, 40])
        self.assertEqual(results[0]["score"], float(np.float32(scores[0])))

    def test_search_expands_to_all_candidates_when_unique_results_are_exhausted(self):
        self.service.index = RankedCandidateIndex([0.9] * 1001)
        self.service.metadata = [
            {"chunk_id": index, "material_id": 3, "content": "Only unique passage"}
            for index in range(1001)
        ]

        results = self.service.search("query", top_k=2, min_score=0.0)

        self.assertEqual([result["chunk_id"] for result in results], [0])
        self.assertEqual(self.service.index.depths, [20, 40, 80, 160, 320, 640, 1001])
        self.assertEqual(len(self.service.index.depths), len(set(self.service.index.depths)))

    def test_duplicate_search_result_keeps_complete_citation_metadata_and_original_score(self):
        self.service.index = RankedCandidateIndex([0.95, 0.90, 0.85])
        retained_metadata = {
            "chunk_id": 30,
            "material_id": 4,
            "chunk_index": 6,
            "source_type": "textbook",
            "source_file": "dbms.pdf",
            "page_number": 14,
            "content": "A passage with   repeated whitespace.",
            "extra_source_field": "preserved",
        }
        self.service.metadata = [
            retained_metadata,
            {
                "chunk_id": 31, "material_id": 4, "chunk_index": 7,
                "source_type": "textbook", "source_file": "duplicate.pdf",
                "page_number": 99, "content": "A passage with repeated whitespace.",
            },
            {"chunk_id": 32, "material_id": 4, "content": "Another unique passage"},
        ]

        results = self.service.search("query", top_k=1, min_score=0.0)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], 30)
        self.assertEqual(results[0]["source_file"], "dbms.pdf")
        self.assertEqual(results[0]["page_number"], 14)
        self.assertEqual(results[0]["chunk_index"], 6)
        self.assertEqual(results[0]["extra_source_field"], "preserved")
        self.assertEqual(results[0]["score"], float(np.float32(0.95)))

    def test_material_filtering_not_starved_by_top_k(self):
        """Test that material-filtered search returns material chunks even if another material has globally higher scores."""
        # Chunks for Material 1 (will match query "apple query" very closely)
        chunks_mat1 = [
            {"chunk_id": 1, "content": "apple query unique text one"},
            {"chunk_id": 2, "content": "apple query unique text two"},
            {"chunk_id": 3, "content": "apple query unique text three"},
            {"chunk_id": 4, "content": "apple query unique text four"},
            {"chunk_id": 5, "content": "apple query unique text five"},
        ]
        # Chunks for Material 2 (still matches, but slightly different text)
        chunks_mat2 = [
            {"chunk_id": 10, "content": "banana different document content ten"},
            {"chunk_id": 11, "content": "banana different document content eleven"},
        ]

        self.service.index_chunks(material_id=1, chunks=chunks_mat1)
        self.service.index_chunks(material_id=2, chunks=chunks_mat2)

        self.assertEqual(self.service.index.ntotal, 7)

        # Searching with material_id=2 with top_k=2 should NOT return empty,
        # even though Material 1 chunks dominate the global top 5
        results_mat2 = self.service.search("apple query", top_k=2, material_id=2, min_score=0.0)
        self.assertEqual(len(results_mat2), 2)
        for r in results_mat2:
            self.assertEqual(r["material_id"], 2)

    def test_indexing_multiple_materials_preserves_all(self):
        """Test that indexing one material does not erase vectors belonging to other materials."""
        chunks_m1 = [{"chunk_id": 1, "content": "Material one chunk text"}]
        chunks_m2 = [{"chunk_id": 2, "content": "Material two chunk text"}]
        chunks_m3 = [{"chunk_id": 3, "content": "Material three chunk text"}]

        self.service.index_chunks(material_id=1, chunks=chunks_m1)
        self.assertEqual(self.service.index.ntotal, 1)

        # Indexing material 2 must preserve material 1
        self.service.index_chunks(material_id=2, chunks=chunks_m2)
        self.assertEqual(self.service.index.ntotal, 2)

        # Indexing material 3 must preserve materials 1 and 2
        self.service.index_chunks(material_id=3, chunks=chunks_m3)
        self.assertEqual(self.service.index.ntotal, 3)

        materials_in_meta = {m["material_id"] for m in self.service.metadata}
        self.assertEqual(materials_in_meta, {1, 2, 3})

    def test_repeated_indexing_no_duplicates(self):
        """Test that re-indexing the same material replaces vectors rather than creating duplicates."""
        initial_chunks = [
            {"chunk_id": 1, "content": "First version chunk A"},
            {"chunk_id": 2, "content": "First version chunk B"},
        ]
        self.service.index_chunks(material_id=1, chunks=initial_chunks)
        self.assertEqual(self.service.index.ntotal, 2)
        self.assertEqual(len(self.service.metadata), 2)

        # Re-index material 1 with updated/same chunks
        updated_chunks = [
            {"chunk_id": 1, "content": "Updated version chunk A"},
            {"chunk_id": 2, "content": "Updated version chunk B"},
            {"chunk_id": 3, "content": "New chunk C"},
        ]
        self.service.index_chunks(material_id=1, chunks=updated_chunks)

        # Should have exactly 3 chunks, NOT 2 + 3 = 5 chunks
        self.assertEqual(self.service.index.ntotal, 3)
        self.assertEqual(len(self.service.metadata), 3)

        contents = [m["content"] for m in self.service.metadata]
        self.assertIn("Updated version chunk A", contents)
        self.assertNotIn("First version chunk A", contents)

    def test_remove_material(self):
        """Test removing a material's vectors and metadata."""
        self.service.index_chunks(material_id=1, chunks=[{"chunk_id": 1, "content": "M1 chunk"}])
        self.service.index_chunks(material_id=2, chunks=[{"chunk_id": 2, "content": "M2 chunk"}])
        self.assertEqual(self.service.index.ntotal, 2)

        removed = self.service.remove_material(material_id=1)
        self.assertEqual(removed, 1)
        self.assertEqual(self.service.index.ntotal, 1)
        self.assertEqual(self.service.metadata[0]["material_id"], 2)

    def test_search_relevance_and_threshold(self):
        """Test search ordering and weak match score threshold filtering."""
        target_content = "Specialized quantum algorithms for optimization"
        unrelated_content = "Historical analysis of 18th century pottery"

        self.service.index_chunks(
            material_id=1,
            chunks=[
                {"chunk_id": 1, "content": unrelated_content},
                {"chunk_id": 2, "content": target_content},
            ],
        )

        # Query matching target_content exactly should rank chunk 2 first with score ~1.0
        results = self.service.search(target_content, top_k=2, min_score=0.0)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], 2)
        self.assertGreater(results[0]["score"], results[1]["score"])

        # High threshold filtering
        high_threshold_results = self.service.search(target_content, top_k=2, min_score=0.99)
        self.assertTrue(all(r["score"] >= 0.99 for r in high_threshold_results))

        # Update weak match threshold
        self.service.set_weak_match_threshold(0.85)
        self.assertEqual(self.service.weak_match_threshold, 0.85)

    def test_index_persistence_and_reload(self):
        """Test that FAISS index and metadata are synchronized and reload accurately from disk."""
        chunks = [
            {"chunk_id": 10, "content": "Persistent chunk one", "page_number": 1},
            {"chunk_id": 20, "content": "Persistent chunk two", "page_number": 2},
        ]
        self.service.index_chunks(material_id=5, chunks=chunks)

        self.assertTrue(os.path.exists(self.index_path))
        self.assertTrue(os.path.exists(self.metadata_path))

        # Create new service pointing to the same files
        from services.embedding_service import EmbeddingService

        reloaded_service = EmbeddingService(
            vector_dir=self.vector_dir,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            model=self.mock_model,
        )

        self.assertEqual(reloaded_service.index.ntotal, 2)
        self.assertEqual(len(reloaded_service.metadata), 2)
        self.assertEqual(reloaded_service.metadata[0]["chunk_id"], 10)
        self.assertEqual(reloaded_service.metadata[1]["chunk_id"], 20)

    def test_unsynchronized_save_raises(self):
        """Test that save_index() raises an error if index and metadata lengths do not match."""
        import faiss

        self.service.metadata = [{"chunk_id": 1}]
        self.service.index = faiss.IndexFlatIP(384)  # 0 vectors vs 1 metadata item

        with self.assertRaises(ValueError):
            self.service.save_index()

    def test_corrupt_index_graceful_recovery(self):
        """Test that a corrupt metadata file safely resets to an empty synchronized state on reload."""
        os.makedirs(self.vector_dir, exist_ok=True)
        # Write corrupt JSON
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            f.write("corrupted { invalid json content")

        import faiss
        idx = faiss.IndexFlatIP(384)
        faiss.write_index(idx, self.index_path)

        from services.embedding_service import EmbeddingService

        recovered_service = EmbeddingService(
            vector_dir=self.vector_dir,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            model=self.mock_model,
        )

        self.assertEqual(recovered_service.index.ntotal, 0)
        self.assertEqual(len(recovered_service.metadata), 0)

    def test_sqlite_material_indexing_flow(self):
        """Test index_material and build_index reading from a temporary SQLite database."""
        db_path = os.path.join(self.temp_dir.name, "test_db.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE materials (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE document_chunks (
                id INTEGER PRIMARY KEY,
                material_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_file TEXT,
                page_number INTEGER NOT NULL
            )
            """
        )
        conn.execute("INSERT INTO materials (id, filename) VALUES (1, 'intro.pdf'), (2, 'adv.pdf')")
        conn.execute(
            """
            INSERT INTO document_chunks (id, material_id, chunk_index, content, source_type, source_file, page_number)
            VALUES
                (1, 1, 0, 'Introduction to computer science', 'textbook', 'intro.pdf', 1),
                (2, 1, 1, 'Data structures and algorithms', 'textbook', 'intro.pdf', 2),
                (3, 2, 0, 'Advanced operating systems', 'textbook', 'adv.pdf', 1)
            """
        )
        conn.commit()
        conn.close()

        # Index only material 1
        count1 = self.service.index_material(material_id=1, database_path=db_path)
        self.assertEqual(count1, 2)
        self.assertEqual(self.service.index.ntotal, 2)

        # Index material 2 (incremental)
        count2 = self.service.index_material(material_id=2, database_path=db_path)
        self.assertEqual(count2, 1)
        self.assertEqual(self.service.index.ntotal, 3)

        # Full rebuild
        total = self.service.build_index(material_id=None, database_path=db_path)
        self.assertEqual(total, 3)
        self.assertEqual(self.service.index.ntotal, 3)


if __name__ == '__main__':
    unittest.main()
