"""Offline PPTX text extraction tests using presentations created in memory."""

import os
import tempfile
import unittest

from pptx import Presentation
from pptx.util import Inches

from ingestion.chunker import Chunker
from ingestion.pptx_processor import PPTXProcessor


class TestPPTXProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pptx_path = os.path.join(self.temp_dir.name, "generated.pptx")

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _add_text(slide, text):
        shape = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(6), Inches(1))
        shape.text_frame.text = text
        return shape

    def test_extracts_multiple_slides_in_order_with_one_based_numbers(self):
        presentation = Presentation()
        first = presentation.slides.add_slide(presentation.slide_layouts[6])
        self._add_text(first, "  First slide\ncontent  ")
        second = presentation.slides.add_slide(presentation.slide_layouts[6])
        self._add_text(second, "Second slide content")
        presentation.save(self.pptx_path)

        sections = PPTXProcessor(self.pptx_path).extract_text_by_slide()

        self.assertEqual(sections, [
            {"page": 1, "text": "First slide content"},
            {"page": 2, "text": "Second slide content"},
        ])

    def test_extracts_table_cells_and_placeholder_text(self):
        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = "Placeholder title"
        body = slide.placeholders[1]
        body.text = "Placeholder body"
        table = slide.shapes.add_table(
            2, 2, Inches(1), Inches(2), Inches(5), Inches(1.5)
        ).table
        table.cell(0, 0).text = "Cell one"
        table.cell(0, 1).text = "Cell two"
        table.cell(1, 0).text = "Cell three"
        table.cell(1, 1).text = "Cell four"
        presentation.save(self.pptx_path)

        sections = PPTXProcessor(self.pptx_path).extract_text_by_slide()

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["page"], 1)
        self.assertEqual(
            sections[0]["text"].splitlines(),
            [
                "Placeholder title",
                "Placeholder body",
                "Cell one",
                "Cell two",
                "Cell three",
                "Cell four",
            ],
        )

    def test_empty_slide_is_skipped_without_shifting_following_slide_number(self):
        presentation = Presentation()
        presentation.slides.add_slide(presentation.slide_layouts[6])
        third_slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        self._add_text(third_slide, "Text on slide two")
        presentation.save(self.pptx_path)

        sections = PPTXProcessor(self.pptx_path).extract_text_by_slide()

        self.assertEqual(sections, [{"page": 2, "text": "Text on slide two"}])

    def test_presentation_with_no_extractable_text_returns_empty_list(self):
        presentation = Presentation()
        presentation.slides.add_slide(presentation.slide_layouts[6])
        presentation.slides.add_slide(presentation.slide_layouts[6])
        presentation.save(self.pptx_path)

        self.assertEqual(PPTXProcessor(self.pptx_path).extract_text_by_slide(), [])

    def test_invalid_and_corrupt_inputs_raise_controlled_value_error(self):
        with self.assertRaisesRegex(ValueError, "Failed to extract text from PowerPoint"):
            PPTXProcessor(os.path.join(self.temp_dir.name, "missing.pptx")).extract_text_by_slide()

        with open(self.pptx_path, "wb") as corrupt_file:
            corrupt_file.write(b"not a presentation")
        with self.assertRaisesRegex(ValueError, "Failed to extract text from PowerPoint"):
            PPTXProcessor(self.pptx_path).extract_text_by_slide()

    def test_output_is_accepted_by_existing_chunker_with_slide_locator(self):
        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        self._add_text(slide, "Slide locator is retained through chunking.")
        presentation.save(self.pptx_path)

        sections = PPTXProcessor(self.pptx_path).extract_text_by_slide()
        chunks = Chunker(chunk_size=200, overlap=20).chunk_pages(
            sections, source_file="generated.pptx", source_type="slides"
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["page"], 1)
        self.assertEqual(chunks[0]["source_type"], "slides")
        self.assertEqual(chunks[0]["source_file"], "generated.pptx")
        self.assertIn("Slide locator", chunks[0]["content"])


if __name__ == "__main__":
    unittest.main()
