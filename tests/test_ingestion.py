import unittest
import os
import sys
import tempfile
import importlib.util

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import chunker directly to avoid PyMuPDF DLL issues in test environment
spec = importlib.util.spec_from_file_location("chunker", os.path.join(os.path.dirname(__file__), '..', 'ingestion', 'chunker.py'))
chunker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunker_module)
Chunker = chunker_module.Chunker


class TestChunker(unittest.TestCase):
    """Test cases for the Chunker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.chunker = Chunker(chunk_size=100, overlap=20)

    def test_split_text_basic(self):
        """Test basic text splitting."""
        text = "This is a test. This is only a test. " * 10
        chunks = self.chunker._split_text(text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_split_text_short(self):
        """Test splitting of short text (smaller than chunk size)."""
        text = "Short text"
        chunks = self.chunker._split_text(text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_split_text_empty(self):
        """Test splitting of empty text."""
        text = ""
        chunks = self.chunker._split_text(text)

        self.assertEqual(len(chunks), 0)

    def test_chunk_pages_preserves_metadata(self):
        """Test that chunking preserves page metadata."""
        pages = [
            {"page": 1, "text": "This is page one text. " * 20},
            {"page": 2, "text": "This is page two text. " * 20}
        ]

        chunks = self.chunker.chunk_pages(pages, source_file="test.pdf", source_type="textbook")

        self.assertGreater(len(chunks), 0)

        # Check that all chunks have required metadata
        for chunk in chunks:
            self.assertIn('chunk_id', chunk)
            self.assertIn('content', chunk)
            self.assertIn('source_type', chunk)
            self.assertIn('source_file', chunk)
            self.assertIn('page', chunk)
            self.assertIn('chunk_index', chunk)

        # Check that page numbers are preserved
        page_numbers = [chunk['page'] for chunk in chunks]
        self.assertIn(1, page_numbers)
        self.assertIn(2, page_numbers)

        # Check that chunk_index is unique and sequential
        chunk_indices = [chunk['chunk_index'] for chunk in chunks]
        self.assertEqual(len(chunk_indices), len(set(chunk_indices)))
        self.assertEqual(sorted(chunk_indices), list(range(len(chunks))))

    def test_chunk_size_configuration(self):
        """Test that chunk size can be configured."""
        text = "Test " * 100

        self.chunker.set_chunk_size(50)
        chunks = self.chunker._split_text(text)

        # With smaller chunk size, should have more chunks
        self.assertGreater(len(chunks), 1)

    def test_overlap_configuration(self):
        """Test that overlap can be configured."""
        text = "Test " * 100

        self.chunker.set_overlap(10)
        chunks = self.chunker._split_text(text)

        self.assertGreater(len(chunks), 0)


if __name__ == '__main__':
    unittest.main()
