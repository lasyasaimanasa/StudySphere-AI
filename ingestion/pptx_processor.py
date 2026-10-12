"""Extract text from PowerPoint slides while preserving their slide numbers."""

from typing import Any, Dict, List

from pptx import Presentation


class PPTXProcessor:
    """Process PPTX files and extract nonempty slide text with source metadata."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.slide_count = 0

    def extract_text_by_slide(self) -> List[Dict[str, Any]]:
        """Return nonempty slides in order using the existing chunker input shape.

        Each result uses ``page`` for the original, one-based slide number because
        ``Chunker.chunk_pages()`` preserves that field as the chunk's source
        locator. Pass ``source_type="slides"`` to the chunker to distinguish it
        from PDF page content.
        """
        try:
            presentation = Presentation(self.file_path)
            self.slide_count = len(presentation.slides)
            slides = []

            for slide_number, slide in enumerate(presentation.slides, start=1):
                paragraphs = []
                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False):
                        frame_paragraphs = [
                            self._clean_paragraph(paragraph.text)
                            for paragraph in shape.text_frame.paragraphs
                        ]
                        frame_text = " ".join(
                            paragraph for paragraph in frame_paragraphs if paragraph
                        )
                        if frame_text:
                            paragraphs.append(frame_text)

                    if getattr(shape, "has_table", False):
                        for row in shape.table.rows:
                            for cell in row.cells:
                                cell_paragraphs = [
                                    self._clean_paragraph(paragraph.text)
                                    for paragraph in cell.text_frame.paragraphs
                                ]
                                cell_text = " ".join(
                                    paragraph for paragraph in cell_paragraphs if paragraph
                                )
                                if cell_text:
                                    paragraphs.append(cell_text)

                text = "\n".join(paragraphs)
                if text:
                    slides.append({"page": slide_number, "text": text})

            return slides
        except Exception as exc:
            raise ValueError(
                f"Failed to extract text from PowerPoint: {exc}"
            ) from exc

    @staticmethod
    def _clean_paragraph(text: str) -> str:
        """Collapse unnecessary whitespace while retaining paragraph boundaries."""
        return " ".join((text or "").split())
