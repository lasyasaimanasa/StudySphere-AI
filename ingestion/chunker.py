from typing import List, Dict, Optional
import uuid


class Chunker:
    """Split text into chunks while preserving source metadata."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize chunker with configuration.

        Args:
            chunk_size: Target size of each chunk in characters.
            overlap: Number of characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_pages(self, pages: List[Dict[str, any]], source_file: str, source_type: str = "textbook") -> List[Dict[str, any]]:
        """
        Split page text into chunks with preserved metadata.

        Args:
            pages: List of page dictionaries with 'page' and 'text' keys.
            source_file: Name of the source file.
            source_type: Type of source material (e.g., 'textbook', 'slides').

        Returns:
            List of chunk dictionaries with metadata.
        """
        chunks = []
        global_chunk_index = 0

        for page_data in pages:
            page_num = page_data["page"]
            text = page_data["text"]

            # Split text into chunks
            page_chunks = self._split_text(text)

            for chunk_text in page_chunks:
                chunk = {
                    "chunk_id": str(uuid.uuid4()),
                    "content": chunk_text,
                    "source_type": source_type,
                    "source_file": source_file,
                    "page": page_num,
                    "chunk_index": global_chunk_index
                }
                # Typed source locators are optional so existing PDF and slide
                # chunk shapes remain unchanged. Transcript locators must follow
                # every split chunk from their timestamped section.
                if "location_type" in page_data:
                    chunk["location_type"] = page_data["location_type"]
                if "location_value" in page_data:
                    chunk["location_value"] = page_data["location_value"]
                chunks.append(chunk)
                global_chunk_index += 1

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at a sentence boundary or natural break
            if end < len(text):
                # Look for sentence endings in the last 100 characters
                break_candidates = ['.', '!', '?', '\n', ';']
                best_break = end

                for i in range(min(100, self.chunk_size)):
                    pos = end - i - 1
                    if pos >= start and text[pos] in break_candidates:
                        best_break = pos + 1
                        break

                end = best_break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.overlap if end > self.overlap else end

            # Prevent infinite loop
            if start >= len(text) or end >= len(text):
                break

        return chunks

    def set_chunk_size(self, chunk_size: int):
        """Update chunk size."""
        self.chunk_size = chunk_size

    def set_overlap(self, overlap: int):
        """Update overlap size."""
        self.overlap = overlap
