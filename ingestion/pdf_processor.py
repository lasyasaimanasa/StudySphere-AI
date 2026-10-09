import pypdf
from typing import List, Dict, Optional


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
        pages = []
        try:
            with open(self.file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                page_count = len(reader.pages)

                for page_num in range(page_count):
                    page = reader.pages[page_num]
                    text = page.extract_text()

                    # Clean whitespace
                    text = self._clean_text(text)

                    # Skip completely empty pages
                    if text.strip():
                        pages.append({
                            "page": page_num + 1,  # 1-indexed page numbers
                            "text": text
                        })

            return pages

        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

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
