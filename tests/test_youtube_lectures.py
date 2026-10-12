"""Offline persistence and API tests for saved YouTube lecture associations."""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import database

with patch.object(database, "init_db"):
    import app as app_module

from app import app
from services.youtube_lecture_service import (
    YouTubeLectureError,
    YouTubeLectureService,
    canonicalize_youtube_url,
)


VIDEO_ID = "dQw4w9WgXcQ"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


class TestYouTubeLectureService(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, "lectures.sqlite")
        with patch.object(database.Config, "DATABASE_PATH", self.database_path):
            database.init_db()
        self.service = YouTubeLectureService(self.database_path)
        self._add_material(11, "transcript", "processed", timestamped=True)
        self._add_material(12, "transcript", "indexing_failed", timestamped=True)
        self._add_material(13, "pdf", "processed", timestamped=True)
        self._add_material(14, "transcript", "processed", timestamped=False)

        self.previous_service = app_module.youtube_lecture_service
        self.previous_materials = app_module.material_service.get_user_materials
        app_module.youtube_lecture_service = self.service
        app_module.material_service.get_user_materials = lambda user_id: [
            {"id": 11, "filename": "lecture.txt", "file_type": "transcript", "page_count": 1}
        ]
        app.config["TESTING"] = True
        self.client = app.test_client()

    def tearDown(self):
        app_module.youtube_lecture_service = self.previous_service
        app_module.material_service.get_user_materials = self.previous_materials
        self.temp_dir.cleanup()

    def _add_material(self, material_id, file_type, status, timestamped):
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            """INSERT INTO materials
               (id, user_id, filename, file_type, file_path, processing_status)
               VALUES (?, 1, ?, ?, ?, ?)""",
            (material_id, f"material-{material_id}.txt", file_type, "not-created", status),
        )
        if timestamped:
            connection.execute(
                """INSERT INTO document_chunks
                   (material_id, chunk_index, content, source_type, source_file,
                    page_number, location_type, location_value)
                   VALUES (?, 0, 'timestamped lesson content', 'transcript', ?, 0,
                           'timestamp', '00:01:25')""",
                (material_id, f"material-{material_id}.txt"),
            )
        else:
            connection.execute(
                """INSERT INTO document_chunks
                   (material_id, chunk_index, content, source_type, source_file,
                    page_number, location_type, location_value)
                   VALUES (?, 0, 'untimestamped content', 'transcript', ?, 1,
                           'page', '1')""",
                (material_id, f"material-{material_id}.txt"),
            )
        connection.commit()
        connection.close()

    def test_save_persists_canonical_identity_title_transcript_and_created_time(self):
        saved = self.service.save_lecture(
            f"https://youtu.be/{VIDEO_ID}?si=share", 11, "Linear Algebra Lecture"
        )

        self.assertEqual(saved["video_id"], VIDEO_ID)
        self.assertEqual(saved["canonical_url"], VIDEO_URL)
        self.assertEqual(saved["display_title"], "Linear Algebra Lecture")
        self.assertEqual(saved["transcript_material_id"], 11)
        self.assertTrue(saved["created_at"])
        self.assertTrue(saved["transcript_available"])

    def test_duplicate_video_updates_association_without_creating_duplicate(self):
        first = self.service.save_lecture(VIDEO_URL, 11, "Lecture")
        second = self.service.save_lecture(
            f"https://www.youtube.com/shorts/{VIDEO_ID}", 11, "Updated title"
        )
        connection = sqlite3.connect(self.database_path)
        count = connection.execute("SELECT COUNT(*) FROM youtube_lectures").fetchone()[0]
        connection.close()

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(second["display_title"], "Updated title")
        self.assertEqual(count, 1)

    def test_rejects_missing_non_transcript_unindexed_and_untimestamped_materials(self):
        for material_id, expected in (
            (999, "not found"),
            (13, "not a transcript"),
            (12, "finished indexing"),
            (14, "timestamped chunks"),
        ):
            with self.subTest(material_id=material_id), self.assertRaises(YouTubeLectureError) as caught:
                self.service.save_lecture(VIDEO_URL, material_id)
            self.assertIn(expected, str(caught.exception))

    def test_unavailable_deleted_transcript_is_reported_and_can_be_replaced(self):
        saved = self.service.save_lecture(VIDEO_URL, 11)
        connection = sqlite3.connect(self.database_path)
        connection.execute("DELETE FROM materials WHERE id = 11")
        connection.commit()
        connection.close()

        unavailable = self.service.list_lectures()[0]
        self.assertEqual(unavailable["id"], saved["id"])
        self.assertFalse(unavailable["transcript_available"])
        self.assertIsNone(unavailable["transcript_material_id"])

        self._add_material(15, "transcript", "processed", timestamped=True)
        replacement = self.service.save_lecture(VIDEO_URL, 15)
        self.assertEqual(replacement["id"], saved["id"])
        self.assertTrue(replacement["transcript_available"])

    def test_api_save_list_and_tutor_restore_saved_transcript(self):
        response = self.client.post("/api/youtube-lectures", json={
            "url": VIDEO_URL,
            "transcript_material_id": 11,
            "display_title": "Lecture title",
        })
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["lecture"]["transcript_material_id"], 11)

        listed = self.client.get("/api/youtube-lectures")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.get_json()["lectures"][0]["video_id"], VIDEO_ID)

        tutor = self.client.get("/tutor")
        self.assertEqual(tutor.status_code, 200)
        self.assertIn(b'data-transcript-id="11"', tutor.data)
        self.assertIn(b'data-transcript-available="true"', tutor.data)
        self.assertIn(b"Lecture title", tutor.data)

    def test_api_rejects_invalid_url_and_transcript_references(self):
        invalid_url = self.client.post("/api/youtube-lectures", json={
            "url": "https://example.com/watch?v=" + VIDEO_ID,
            "transcript_material_id": 11,
        })
        self.assertEqual(invalid_url.status_code, 400)

        missing_transcript = self.client.post("/api/youtube-lectures", json={
            "url": VIDEO_URL,
            "transcript_material_id": 999,
        })
        self.assertEqual(missing_transcript.status_code, 404)

        malformed_id = self.client.post("/api/youtube-lectures", json={
            "url": VIDEO_URL,
            "transcript_material_id": True,
        })
        self.assertEqual(malformed_id.status_code, 400)

    def test_backend_url_canonicalization_accepts_supported_shapes(self):
        for url in (
            VIDEO_URL,
            f"https://youtu.be/{VIDEO_ID}",
            f"https://m.youtube.com/watch?v={VIDEO_ID}&list=PL123",
            f"https://www.youtube.com/shorts/{VIDEO_ID}",
            f"https://www.youtube.com/embed/{VIDEO_ID}",
            f"https://www.youtube.com/live/{VIDEO_ID}",
        ):
            with self.subTest(url=url):
                self.assertEqual(canonicalize_youtube_url(url), (VIDEO_ID, VIDEO_URL))


if __name__ == "__main__":
    unittest.main()
