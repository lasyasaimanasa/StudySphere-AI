import os
import tempfile
import unittest

from ingestion.chunker import Chunker
from ingestion.transcript_processor import TranscriptProcessor


class TestTranscriptIngestion(unittest.TestCase):
    def _extract(self, content):
        temporary = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        )
        try:
            temporary.write(content)
            temporary.close()
            return TranscriptProcessor(temporary.name).extract_sections()
        finally:
            try:
                os.unlink(temporary.name)
            except FileNotFoundError:
                pass

    def test_parses_timestamped_lines_and_normalizes_body_whitespace(self):
        sections = self._extract(
            "[00:00:12]  Introduction\t to database systems.\n"
            "\n[01:02:03] A DBMS stores and retrieves data.\n"
        )

        self.assertEqual(sections, [
            {
                "page": 1,
                "text": "Introduction to database systems.",
                "location_type": "timestamp",
                "location_value": "00:00:12",
            },
            {
                "page": 1,
                "text": "A DBMS stores and retrieves data.",
                "location_type": "timestamp",
                "location_value": "01:02:03",
            },
        ])

    def test_rejects_malformed_lines_invalid_clock_values_and_unordered_timestamps(self):
        invalid = [
            "Introduction without timestamp",
            "[00:60:00] invalid minute",
            "[00:00:60] invalid second",
            "[00:00:01] \n",
            "[00:00:10] later\n[00:00:09] earlier",
        ]
        for content in invalid:
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "transcript|Transcript"):
                self._extract(content)

    def test_rejects_empty_transcript(self):
        with self.assertRaisesRegex(ValueError, "no timestamped text"):
            self._extract("\n   \n")

    def test_chunker_preserves_timestamp_on_every_split_chunk(self):
        sections = self._extract(
            "[00:01:25] This timestamped section contains enough words to be split "
            "across multiple chunks while keeping its original time locator."
        )
        chunker = Chunker(chunk_size=24, overlap=5)
        chunks = chunker.chunk_pages(
            sections, source_file="lecture_transcript.txt", source_type="transcript"
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["page"] == 1 for chunk in chunks))
        self.assertTrue(all(chunk["location_type"] == "timestamp" for chunk in chunks))
        self.assertTrue(all(chunk["location_value"] == "00:01:25" for chunk in chunks))
        self.assertTrue(all(chunk["source_type"] == "transcript" for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
