import unittest
import os
import sys
import tempfile
import sqlite3
from unittest.mock import MagicMock
from reportlab.pdfgen import canvas

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.material_service import MaterialService
from services.embedding_service import EmbeddingService
from tests.test_embeddings import MockEmbedder


class TestUploadSemanticIndexing(unittest.TestCase):
    """Test automated semantic indexing during PDF upload and error recovery."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.upload_dir = os.path.join(self.temp_dir.name, "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)

        self.db_path = os.path.join(self.temp_dir.name, "test_study.db")
        self._init_test_database(self.db_path)

        self.vector_dir = os.path.join(self.temp_dir.name, "vector_store")
        self.index_path = os.path.join(self.vector_dir, "test_chunks.faiss")
        self.metadata_path = os.path.join(self.vector_dir, "test_metadata.json")

        self.mock_embedder = MockEmbedder(dimension=384)
        self.embedding_service = EmbeddingService(
            vector_dir=self.vector_dir,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            model=self.mock_embedder,
            project_root=self.temp_dir.name,
        )

        self.material_service = MaterialService(
            upload_folder=self.upload_dir,
            embedding_service=self.embedding_service,
            database_path=self.db_path,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _init_test_database(self, db_path: str):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                page_count INTEGER,
                processing_status TEXT DEFAULT 'pending',
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                material_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_file TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute("INSERT INTO users (id, name, email, password_hash) VALUES (1, 'Test User', 'test@example.com', 'hash')")
        conn.commit()
        conn.close()

    def _create_sample_pdf(self, filename: str, pages_text: list) -> str:
        pdf_path = os.path.join(self.upload_dir, filename)
        c = canvas.Canvas(pdf_path)
        for text in pages_text:
            c.drawString(72, 700, text)
            c.showPage()
        c.save()
        return pdf_path

    def test_successful_upload_indexing(self):
        """Test that successfully processing a PDF automatically indexes its chunks in EmbeddingService."""
        pdf_path = self._create_sample_pdf(
            "sample_lecture.pdf",
            [
                "Lecture 1: Introduction to relational database architectures and ACID transactions.",
                "Lecture 2: Normalization theory, third normal form, and Boyce Codd normal form."
            ]
        )

        material_id, error = self.material_service.process_pdf(pdf_path, "sample_lecture.pdf", user_id=1)

        self.assertIsNone(error)
        self.assertIsNotNone(material_id)

        # Check SQLite database status
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        mat = conn.execute("SELECT * FROM materials WHERE id = ?", (material_id,)).fetchone()
        chunks = conn.execute("SELECT * FROM document_chunks WHERE material_id = ?", (material_id,)).fetchall()
        conn.close()

        self.assertEqual(mat["processing_status"], "processed")
        self.assertEqual(len(chunks), 2)

        # Verify FAISS index has automatically received the chunks
        self.assertEqual(self.embedding_service.index.ntotal, 2)
        self.assertEqual(len(self.embedding_service.metadata), 2)

        # Verify search retrieves the new chunks with correct metadata
        search_res = self.embedding_service.search("relational database architectures", top_k=2, min_score=0.0)
        self.assertGreater(len(search_res), 0)
        self.assertEqual(search_res[0]["material_id"], material_id)
        self.assertEqual(search_res[0]["source_file"], "sample_lecture.pdf")

    def test_indexing_failure_handling(self):
        """Test that if semantic indexing fails, chunks are preserved in SQLite and status is indexing_failed."""
        mock_failing_embedder = MagicMock()
        mock_failing_embedder.index_material.side_effect = RuntimeError("FAISS memory allocation failure")

        failing_material_service = MaterialService(
            upload_folder=self.upload_dir,
            embedding_service=mock_failing_embedder,
            database_path=self.db_path,
        )

        pdf_path = self._create_sample_pdf(
            "faulty_doc.pdf",
            ["Sample document content for failure simulation."]
        )

        material_id, error = failing_material_service.process_pdf(pdf_path, "faulty_doc.pdf", user_id=1)

        # Error must be reported
        self.assertIsNotNone(error)
        self.assertIn("semantic indexing failed", error)
        self.assertIsNotNone(material_id)

        # Check SQLite state: chunks committed, status is indexing_failed (NOT processed)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        mat = conn.execute("SELECT * FROM materials WHERE id = ?", (material_id,)).fetchone()
        chunks = conn.execute("SELECT * FROM document_chunks WHERE material_id = ?", (material_id,)).fetchall()
        conn.close()

        self.assertEqual(mat["processing_status"], "indexing_failed")
        self.assertEqual(len(chunks), 1)

    def test_retry_indexing(self):
        """Test retrying indexing for a material with indexing_failed status."""
        pdf_path = self._create_sample_pdf(
            "retry_doc.pdf",
            ["Content to be indexed during retry step."]
        )

        # 1. First trigger a simulated failure
        mock_service = MagicMock()
        mock_service.index_material.side_effect = RuntimeError("Temporary network timeout")

        failing_material_service = MaterialService(
            upload_folder=self.upload_dir,
            embedding_service=mock_service,
            database_path=self.db_path,
        )
        material_id, error = failing_material_service.process_pdf(pdf_path, "retry_doc.pdf", user_id=1)
        self.assertIsNotNone(error)

        # 2. Attach working embedding service and retry
        failing_material_service._embedding_service = self.embedding_service
        success, retry_error = failing_material_service.retry_indexing(material_id)

        self.assertTrue(success)
        self.assertIsNone(retry_error)

        # 3. Check status is now processed and vector store contains chunk
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        mat = conn.execute("SELECT * FROM materials WHERE id = ?", (material_id,)).fetchone()
        conn.close()

        self.assertEqual(mat["processing_status"], "processed")
        self.assertEqual(self.embedding_service.index.ntotal, 1)
        self.assertEqual(self.embedding_service.metadata[0]["material_id"], material_id)

    def test_repeated_indexing_without_duplicates(self):
        """Test that repeatedly indexing uploaded materials does not create duplicate vectors."""
        pdf1_path = self._create_sample_pdf("doc1.pdf", ["Doc 1 Page 1", "Doc 1 Page 2"])
        pdf2_path = self._create_sample_pdf("doc2.pdf", ["Doc 2 Page 1"])

        # Upload Doc 1
        mat1_id, _ = self.material_service.process_pdf(pdf1_path, "doc1.pdf", user_id=1)
        self.assertEqual(self.embedding_service.index.ntotal, 2)

        # Re-index Doc 1 via retry_indexing
        self.material_service.retry_indexing(mat1_id)
        self.assertEqual(self.embedding_service.index.ntotal, 2)  # Still 2, no duplicates!

        # Upload Doc 2
        mat2_id, _ = self.material_service.process_pdf(pdf2_path, "doc2.pdf", user_id=1)
        self.assertEqual(self.embedding_service.index.ntotal, 3)  # 2 + 1 = 3

        # Re-index Doc 1 again
        self.material_service.retry_indexing(mat1_id)
        # Vector count must remain 3 (Doc 2 is not lost, Doc 1 is not duplicated)
        self.assertEqual(self.embedding_service.index.ntotal, 3)

        mat_ids_in_vector = [m["material_id"] for m in self.embedding_service.metadata]
        self.assertEqual(mat_ids_in_vector.count(mat1_id), 2)
        self.assertEqual(mat_ids_in_vector.count(mat2_id), 1)

    def test_embedding_service_instance_reused(self):
        """Test that a shared EmbeddingService instance is reused and not reinstantiated."""
        shared_service = self.embedding_service
        service = MaterialService(
            upload_folder=self.upload_dir,
            embedding_service=shared_service,
            database_path=self.db_path,
        )

        pdf_path = self._create_sample_pdf("reuse_test.pdf", ["Page one text."])
        service.process_pdf(pdf_path, "reuse_test.pdf", user_id=1)

        # Verify embedding_service reference remains identical
        self.assertIs(service.embedding_service, shared_service)


if __name__ == '__main__':
    unittest.main()
