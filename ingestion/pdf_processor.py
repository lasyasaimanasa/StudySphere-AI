import pypdf
from typing import List, Dict, Optional, Tuple


class PDFProcessor:
    """Process PDF files and extract text with page metadata."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_text_by_page(self) -> List[Dict[str, any]]:
        """
        Extract text from PDF page by page.

        Returns:
            List of dictionaries with page number and text content.
        """
        try:
            with open(self.file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                page_count = len(reader.pages)
                return self._extract_text_from_reader(reader, page_count)

        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    def validate_and_extract_text_by_page(
        self,
    ) -> Tuple[bool, int, List[Dict[str, any]]]:
        """Validate, count, and extract pages using one open PDF reader.

        The file stays open while the reader accesses its pages. Invalid PDFs
        retain the ``validate_pdf`` behavior and return ``False``; extraction
        errors retain ``extract_text_by_page``'s ValueError wrapping.
        """
        try:
            file = open(self.file_path, 'rb')
        except Exception:
            # Opening failures are treated as invalid PDFs, as in validate_pdf().
            return False, 0, []

        with file:
            try:
                reader = pypdf.PdfReader(file)
                page_count = len(reader.pages)
            except Exception:
                # Reader parsing failures are treated as invalid PDFs, as in
                # validate_pdf().
                return False, 0, []

            if page_count == 0:
                return True, 0, []

            try:
                pages = self._extract_text_from_reader(reader, page_count)
            except Exception as e:
                raise ValueError(f"Failed to extract text from PDF: {str(e)}")

            return True, page_count, pages

    def _extract_text_from_reader(
        self,
        reader: pypdf.PdfReader,
        page_count: int,
    ) -> List[Dict[str, any]]:
        """Extract and clean nonempty pages from an already-open reader."""
        pages = []
        warnings = []

        for page_num in range(page_count):
            page = reader.pages[page_num]
            text = page.extract_text()

            # Clean whitespace
            text = self._clean_text(text)

            # Skip completely empty pages
            if text.strip():
                pages.append({
                    "page": page_num + 1,  # 1-indexed page number
                    "text": text
                })
            else:
                warnings.append(f"Page {page_num + 1} is empty or contains no extractable text")

        if warnings:
            print(f"PDF extraction warnings: {'; '.join(warnings)}")

        return pages

    def get_page_count(self) -> int:
        """
        Get the total number of pages in the PDF.

        Returns:
            Number of pages in the PDF.
        """
        try:
            with open(self.file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                return len(reader.pages)
        except Exception as e:
            raise ValueError(f"Failed to read PDF: {str(e)}")

    def get_extraction_diagnostics(self) -> Dict[str, any]:
        """
        Get diagnostic information about PDF extraction quality.

        Returns:
            Dictionary with diagnostic information.
        """
        try:
            with open(self.file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                page_count = len(reader.pages)

                total_chars = 0
                empty_pages = 0
                short_pages = 0
                pages_with_text = 0

                for page_num in range(page_count):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    text = self._clean_text(text)

                    char_count = len(text)
                    total_chars += char_count

                    if char_count == 0:
                        empty_pages += 1
                    elif char_count < 50:
                        short_pages += 1
                    else:
                        pages_with_text += 1

                return {
                    "page_count": page_count,
                    "pages_with_text": pages_with_text,
                    "empty_pages": empty_pages,
                    "short_pages": short_pages,
                    "total_characters": total_chars,
                    "avg_chars_per_page": total_chars / page_count if page_count > 0 else 0,
                    "extraction_quality": "good" if pages_with_text == page_count else (
                        "poor" if pages_with_text == 0 else "mixed"
                    )
                }
        except Exception as e:
            return {
                "error": str(e),
                "extraction_quality": "failed"
            }

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by normalizing whitespace.

        Args:
            text: Raw text from PDF.

        Returns:
            Cleaned text.
        """
        # Replace multiple whitespace with single space
        text = ' '.join(text.split())
        return text.strip()

    def validate_pdf(self) -> bool:
        """
        Validate that the file is a valid PDF.

        Returns:
            True if valid PDF, False otherwise.
        """
        try:
            with open(self.file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                _ = len(reader.pages)
            return True
        except Exception:
            return False
