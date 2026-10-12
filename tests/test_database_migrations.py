"""Isolated tests for additive database migrations and legacy page locations."""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import database
from services.embedding_service import EmbeddingService


class TestDocumentChunkLocationMigration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, "legacy.sqlite")
        connection = sqlite3.connect(self.database_path)
        connection.executescript(
            """
            CREATE TABLE materials (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL
            );
            CREATE TABLE document_chunks (
                id INTEGER PRIMARY KEY,
                material_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_file TEXT NOT NULL,
                page_number INTEGER NOT NULL
            );
            INSERT INTO materials (id, filename, user_id, file_type, file_path)
            VALUES (41, 'legacy.pdf', 1, 'pdf', 'legacy.pdf');
            INSERT INTO document_chunks
                (id, material_id, chunk_index, content, source_type, source_file, page_number)
            VALUES (71, 41, 0, 'Legacy page content', 'textbook', 'legacy.pdf', 7);
            """
        )
        connection.commit()
        connection.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _initialize_database(self):
        with patch.object(database.Config, "DATABASE_PATH", self.database_path):
            database.init_db()

    def test_migration_adds_locator_columns_and_preserves_required_page_column(self):
        self._initialize_database()

        connection = sqlite3.connect(self.database_path)
        columns = {row[1]: row for row in connection.execute("PRAGMA table_info(document_chunks)")}
        row = connection.execute(
            "SELECT id, material_id, chunk_index, content, page_number, "
            "location_type, location_value FROM document_chunks WHERE id = 71"
        ).fetchone()
        connection.close()

        self.assertIn("location_type", columns)
        self.assertIn("location_value", columns)
        self.assertEqual(columns["page_number"][2].upper(), "INTEGER")
        self.assertEqual(columns["page_number"][3], 1)
        self.assertEqual(row, (71, 41, 0, "Legacy page content", 7, "page", None))

    def test_repeated_initialization_is_idempotent_and_preserves_legacy_row(self):
        self._initialize_database()
        self._initialize_database()

        connection = sqlite3.connect(self.database_path)
        columns = [row[1] for row in connection.execute("PRAGMA table_info(document_chunks)")]
        rows = connection.execute(
            "SELECT id, content, page_number, location_type, location_value "
            "FROM document_chunks"
        ).fetchall()
        connection.close()

        self.assertEqual(columns.count("location_type"), 1)
        self.assertEqual(columns.count("location_value"), 1)
        self.assertEqual(rows, [(71, "Legacy page content", 7, "page", None)])

    def test_legacy_chunk_fetch_falls_back_to_page_locator(self):
        self._initialize_database()
        service = EmbeddingService.__new__(EmbeddingService)
        service.project_root = self.temp_dir.name

        rows = service._fetch_chunks_from_db(material_id=41, database_path=self.database_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["page_number"], 7)
        self.assertEqual(rows[0]["location_type"], "page")
        self.assertEqual(rows[0]["location_value"], "7")

    def test_assessment_and_youtube_migrations_are_additive_and_idempotent(self):
        self._initialize_database()
        self._initialize_database()

        connection = sqlite3.connect(self.database_path)
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        indexes = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )}
        chunk = connection.execute(
            "SELECT content, page_number FROM document_chunks WHERE id = 71"
        ).fetchone()
        lecture_columns = {row[1] for row in connection.execute(
            "PRAGMA table_info(youtube_lectures)"
        )}
        connection.close()

        self.assertTrue({
            "assessments", "assessment_sources", "assessment_questions",
            "assessment_attempts", "assessment_responses",
            "youtube_lectures",
        }.issubset(tables))
        self.assertTrue({
            "idx_assessment_questions_assessment",
            "idx_assessment_attempts_completed",
            "idx_assessment_responses_question",
            "idx_youtube_lectures_transcript",
        }.issubset(indexes))
        self.assertTrue({
            "video_id", "canonical_url", "display_title",
            "transcript_material_id", "created_at",
        }.issubset(lecture_columns))
        self.assertEqual(chunk, ("Legacy page content", 7))


if __name__ == "__main__":
    unittest.main()
