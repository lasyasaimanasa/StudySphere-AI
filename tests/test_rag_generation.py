import json
import unittest
import os
import sys
import sqlite3
import tempfile
import urllib.error
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.rag_service import RAGService
from rag.prompts import build_user_prompt
from ai.llm_provider import (
    LLMConfigurationError,
    LLMProviderError,
    OllamaProvider,
    OpenAIProvider,
    get_llm_provider,
)


class MockEmbeddingService:
    """Mock embedding service for testing."""

    def __init__(self):
        self.index = MagicMock()
        self.index.ntotal = 2

    def search(self, query, top_k=5, material_id=None, min_score=None):
        """Return mock search results."""
        return [
            {
                "chunk_id": 1,
                "material_id": 1,
                "chunk_index": 0,
                "source_file": "test_doc.pdf",
                "source_type": "textbook",
                "page_number": 5,
                "location_type": "page",
                "location_value": "5",
                "content": "The relational model organizes data into tables with rows and columns.",
                "score": 0.85
            },
            {
                "chunk_id": 2,
                "material_id": 1,
                "chunk_index": 1,
                "source_file": "test_doc.pdf",
                "source_type": "textbook",
                "page_number": 7,
                "location_type": "page",
                "location_value": "7",
                "content": "Primary keys uniquely identify each row in a table.",
                "score": 0.78
            }
        ]


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, should_fail=False, should_timeout=False):
        self.should_fail = should_fail
        self.should_timeout = should_timeout

    def generate(self, prompt, system_prompt=None):
        if self.should_fail:
            raise LLMProviderError("Mock LLM failure")
        if self.should_timeout:
            raise LLMProviderError("Request timeout")
        return json.dumps({
            "supported": True,
            "claims": [
                {"text": "The relational model organizes data into tables with rows and columns.", "source_ids": ["S1"]},
                {"text": "Primary keys uniquely identify each row in a table.", "source_ids": ["S2"]},
            ],
        })


class TestRAGGeneration(unittest.TestCase):
    """Test suite for RAG answer generation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, "rag.sqlite")
        connection = sqlite3.connect(self.database_path)
        connection.executescript("""
            CREATE TABLE materials (
                id INTEGER PRIMARY KEY, filename TEXT NOT NULL, file_type TEXT NOT NULL
            );
            CREATE TABLE document_chunks (
                id INTEGER PRIMARY KEY, material_id INTEGER NOT NULL, chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL, source_type TEXT NOT NULL, source_file TEXT NOT NULL,
                page_number INTEGER NOT NULL, location_type TEXT NOT NULL, location_value TEXT
            );
            CREATE TABLE youtube_lectures (
                id INTEGER PRIMARY KEY, video_id TEXT NOT NULL, display_title TEXT,
                transcript_material_id INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        connection.commit()
        connection.close()
        self._add_stored_chunk(
            1, 1, 0, "The relational model organizes data into tables with rows and columns.",
            "test_doc.pdf", "pdf", 5, "page", "5", "textbook",
        )
        self._add_stored_chunk(
            2, 1, 1, "Primary keys uniquely identify each row in a table.",
            "test_doc.pdf", "pdf", 7, "page", "7", "textbook",
        )
        self.mock_embedding_service = MockEmbeddingService()
        self.rag_service = RAGService(
            embedding_service=self.mock_embedding_service, database_path=self.database_path
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _add_stored_chunk(self, chunk_id, material_id, chunk_index, content, filename,
                          file_type, page_number, location_type, location_value, source_type):
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            "INSERT OR IGNORE INTO materials (id, filename, file_type) VALUES (?, ?, ?)",
            (material_id, filename, file_type),
        )
        connection.execute(
            """INSERT OR REPLACE INTO document_chunks
               (id, material_id, chunk_index, content, source_type, source_file,
                page_number, location_type, location_value)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (chunk_id, material_id, chunk_index, content, source_type, filename,
             page_number, location_type, location_value),
        )
        connection.commit()
        connection.close()

    def test_successful_answer_generation(self):
        """Test successful answer generation with citations."""
        mock_llm = MockLLMProvider()

        with patch('rag.rag_service.get_llm_provider', return_value=mock_llm):
            result = self.rag_service.generate_answer(
                query="What is the relational model?",
                top_k=5
            )

            self.assertIsNotNone(result)
            self.assertIsNone(result.get("error"))
            self.assertIsNotNone(result.get("answer"))
            self.assertFalse(result.get("insufficient_context"))
            self.assertGreater(len(result.get("citations", [])), 0)
            self.assertEqual(len(result.get("sources", [])), 2)

    def test_answer_with_pre_retrieved_chunks(self):
        """Test answer generation when chunks are pre-retrieved."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "supported": True,
            "claims": [{"text": "Custom content is available.", "source_ids": ["S1"]}],
        })
        pre_retrieved = [
            {
                "chunk_id": 77,
                "material_id": 2,
                "chunk_index": 0,
                "source_type": "textbook",
                "source_file": "custom.pdf",
                "page_number": 10,
                "location_type": "page",
                "location_value": "10",
                "content": "Custom content for testing",
                "score": 0.9
            }
        ]
        self._add_stored_chunk(
            77, 2, 0, "Custom content for testing", "custom.pdf", "pdf", 10,
            "page", "10", "textbook",
        )

        with patch('rag.rag_service.get_llm_provider', return_value=mock_llm):
            result = self.rag_service.generate_answer(
                query="Test query",
                retrieved_chunks=pre_retrieved
            )

            self.assertIsNotNone(result.get("answer"))
            self.assertEqual(len(result.get("sources", [])), 1)
            self.assertEqual(result["sources"][0]["source_file"], "custom.pdf")

    def test_insufficient_context_empty_retrieval(self):
        """Test response when no chunks are retrieved."""
        rag_service = RAGService(embedding_service=None)

        with patch('rag.rag_service.get_llm_provider', return_value=MockLLMProvider()):
            result = rag_service.generate_answer(
                query="What is irrelevant topic?",
                retrieved_chunks=[]
            )

            self.assertTrue(result.get("insufficient_context"))
            self.assertIn("do not contain enough relevant information", result.get("answer", ""))

    def test_llm_configuration_error(self):
        """Test handling of missing LLM configuration."""
        with patch('rag.rag_service.get_llm_provider', side_effect=LLMConfigurationError("API key missing")):
            result = self.rag_service.generate_answer(query="Test query")

            self.assertIsNotNone(result.get("error"))
            self.assertIn("configuration error", result["error"].lower())
            self.assertIsNone(result.get("answer"))

    def test_llm_provider_error(self):
        """Test handling of LLM provider failures."""
        mock_llm = MockLLMProvider(should_fail=True)

        with patch('rag.rag_service.get_llm_provider', return_value=mock_llm):
            result = self.rag_service.generate_answer(query="Test query")

            self.assertIsNotNone(result.get("error"))
            self.assertIn("generation failed", result["error"].lower())
            self.assertIsNone(result.get("answer"))

    def test_retrieval_failure(self):
        """Test handling of retrieval failures."""
        failing_service = MagicMock()
        failing_service.search.side_effect = Exception("Search failed")

        rag_service = RAGService(embedding_service=failing_service)

        result = rag_service.generate_answer(query="Test query")

        self.assertIsNotNone(result.get("error"))
        self.assertIn("retrieval failed", result["error"].lower())

    def test_citation_extraction(self):
        """Test citation extraction from answer."""
        mock_llm = MockLLMProvider()

        with patch('rag.rag_service.get_llm_provider', return_value=mock_llm):
            result = self.rag_service.generate_answer(query="Test query")

            citations = result.get("citations", [])
            self.assertGreater(len(citations), 0)

            # Check citation structure
            for citation in citations:
                self.assertIn("source_file", citation)
                self.assertIn("page_number", citation)

    def test_citation_extraction_returns_only_cited_sources(self):
        chunks = [
            {"source_file": "notes (v1)+.pdf", "page_number": 3},
            {"source_file": "uncited.pdf", "page_number": 4},
        ]

        citations = self.rag_service._extract_citations(
            "Supported by [notes (v1)+.pdf, page 3].", chunks
        )

        self.assertEqual(citations, [
            {"source_file": "notes (v1)+.pdf", "page_number": 3}
        ])

    def test_citation_extraction_deduplicates_same_source_and_page(self):
        chunk = {"source_file": "notes.pdf", "page_number": 3}
        citations = self.rag_service._extract_citations(
            "See [notes.pdf, page 3] and again [notes.pdf, page 3].",
            [chunk, dict(chunk)],
        )

        self.assertEqual(citations, [chunk])

    def test_citation_extraction_returns_empty_when_answer_has_no_citations(self):
        citations = self.rag_service._extract_citations(
            "The answer contains no source reference.",
            [{"source_file": "notes.pdf", "page_number": 3}],
        )

        self.assertEqual(citations, [])

    def test_slide_citations_are_prompted_and_validated_against_retrieved_sources(self):
        chunk = {
            "chunk_id": 15,
            "material_id": 8,
            "chunk_index": 2,
            "source_type": "slides",
            "source_file": "lecture_slides.pptx",
            "page_number": 3,
            "location_type": "slide",
            "location_value": "3",
            "content": "Photosynthesis converts light into chemical energy.",
        }
        self._add_stored_chunk(
            15, 8, 2, chunk["content"], "lecture_slides.pptx", "pptx", 3,
            "slide", "3", "slides",
        )
        prompt = build_user_prompt("What does photosynthesis do?", [chunk])
        self.assertIn("[lecture_slides.pptx, slide 3]", prompt)

        provider = MagicMock()
        provider.generate.return_value = json.dumps({
            "supported": True,
            "claims": [{
                "text": "Photosynthesis converts light into chemical energy.",
                "source_ids": ["S1"],
            }],
        })
        with patch('rag.rag_service.get_llm_provider', return_value=provider):
            result = self.rag_service.generate_answer(
                query="What does photosynthesis do?",
                retrieved_chunks=[chunk],
            )

        self.assertEqual(result["citations"], [{
            "source_file": "lecture_slides.pptx",
            "page_number": 3,
            "location_type": "slide",
            "location_value": "3",
        }])
        self.assertIn("[lecture_slides.pptx, slide 3]", result["answer"])

    def test_legacy_pdf_prompt_and_citation_shape_remain_page_based(self):
        chunk = {"source_file": "DATABASE_MANAGEMENT_SYSTEMS.pdf", "page_number": 6}

        self.assertIn(
            "[DATABASE_MANAGEMENT_SYSTEMS.pdf, page 6]",
            build_user_prompt("Question?", [chunk]),
        )
        self.assertEqual(
            self.rag_service._extract_citations(
                "Supported by [DATABASE_MANAGEMENT_SYSTEMS.pdf, page 6].", [chunk]
            ),
            [{"source_file": "DATABASE_MANAGEMENT_SYSTEMS.pdf", "page_number": 6}],
        )

    def test_transcript_timestamp_citations_are_prompted_and_validated_exactly(self):
        chunk = {
            "chunk_id": 21,
            "material_id": 9,
            "chunk_index": 0,
            "source_type": "transcript",
            "source_file": "lecture_transcript.txt",
            "page_number": 1,
            "location_type": "timestamp",
            "location_value": "00:01:25",
            "content": "A DBMS organizes, stores, and retrieves data.",
        }
        self._add_stored_chunk(
            21, 9, 0, chunk["content"], "lecture_transcript.txt", "transcript", 1,
            "timestamp", "00:01:25", "transcript",
        )
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            """INSERT INTO youtube_lectures
               (video_id, display_title, transcript_material_id)
               VALUES ('video1234567', 'Database Systems Lecture', 9)"""
        )
        connection.commit()
        connection.close()
        self.assertIn(
            "[lecture_transcript.txt, timestamp 00:01:25]",
            build_user_prompt("What does a DBMS do?", [chunk]),
        )

        provider = MagicMock()
        provider.generate.return_value = json.dumps({
            "supported": True,
            "claims": [{
                "text": "A DBMS organizes, stores, and retrieves data.",
                "source_ids": ["S1"],
            }],
        })
        with patch('rag.rag_service.get_llm_provider', return_value=provider):
            result = self.rag_service.generate_answer(
                query="What does a DBMS do?", retrieved_chunks=[chunk]
            )

        self.assertEqual(result["citations"], [{
            "source_file": "lecture_transcript.txt",
            "video_title": "Database Systems Lecture",
            "page_number": 1,
            "location_type": "timestamp",
            "location_value": "00:01:25",
        }])
        self.assertIn("[Database Systems Lecture, timestamp 00:01:25]", result["answer"])
        self.assertIn(
            "[Database Systems Lecture, timestamp 00:01:25]",
            provider.generate.call_args.kwargs["prompt"],
        )
    def test_material_filter_in_retrieval(self):
        """Test that material_id is passed to retrieval."""
        mock_llm = MockLLMProvider()
        self.mock_embedding_service.calls = []
        self.mock_embedding_service.search = MagicMock(
            side_effect=self.mock_embedding_service.search
        )

        with patch('rag.rag_service.get_llm_provider', return_value=mock_llm):
            result = self.rag_service.generate_answer(
                query="Test query",
                material_id=1
            )

            self.mock_embedding_service.search.assert_called_once_with(
                query="Test query", top_k=5, material_id=1
            )
            self.assertIsNotNone(result.get("answer"))

    def test_irrelevant_but_nonempty_retrieval_can_abstain(self):
        irrelevant_passages = self.mock_embedding_service.search("x")
        self.assertTrue(irrelevant_passages)
        provider = MagicMock()
        provider.generate.return_value = json.dumps({"supported": False, "claims": []})
        with patch('rag.rag_service.get_llm_provider', return_value=provider):
            result = self.rag_service.generate_answer(
                "Explain black holes", retrieved_chunks=irrelevant_passages
            )
        self.assertTrue(result["insufficient_context"])
        self.assertIn("do not contain enough relevant information", result["answer"])
        self.assertEqual(result["citations"], [])
        provider.generate.assert_called_once()

    def test_stale_missing_and_mismatched_retrieval_metadata_is_not_evidence(self):
        mutations = {
            "missing_chunk": lambda chunk: chunk.update(chunk_id=999),
            "wrong_material": lambda chunk: chunk.update(material_id=99),
            "wrong_chunk_index": lambda chunk: chunk.update(chunk_index=99),
            "wrong_content": lambda chunk: chunk.update(content="Stale indexed passage."),
            "wrong_source_file": lambda chunk: chunk.update(source_file="other.pdf"),
            "wrong_locator": lambda chunk: chunk.update(location_value="999"),
        }
        for name, mutate in mutations.items():
            candidate = dict(self.mock_embedding_service.search("query")[0])
            mutate(candidate)
            provider = MagicMock()
            provider.generate.return_value = json.dumps({
                "supported": True,
                "claims": [{"text": "Unverified claim.", "source_ids": ["S1"]}],
            })
            with self.subTest(metadata=name), patch(
                'rag.rag_service.get_llm_provider', return_value=provider
            ):
                result = self.rag_service.generate_answer(
                    "Question?", retrieved_chunks=[candidate]
                )
            self.assertTrue(result["insufficient_context"])
            self.assertEqual(result["citations"], [])
            provider.generate.assert_not_called()

    def test_missing_evidence_does_not_call_llm(self):
        provider_factory = patch('rag.rag_service.get_llm_provider')
        with provider_factory as provider:
            result = self.rag_service.generate_answer("Question?", retrieved_chunks=[])
        provider.assert_not_called()
        self.assertTrue(result["insufficient_context"])

    def test_rejects_malformed_model_output(self):
        provider = MagicMock()
        provider.generate.return_value = "Not JSON"
        with patch('rag.rag_service.get_llm_provider', return_value=provider):
            result = self.rag_service.generate_answer("Question?")
        self.assertTrue(result["insufficient_context"])
        self.assertIn("do not contain enough relevant information", result["answer"])

    def test_rejects_missing_or_invalid_citation_ids(self):
        outputs = [
            json.dumps({"supported": True, "claims": [{"text": "Claim", "source_ids": []}]}),
            json.dumps({"supported": True, "claims": [{"text": "Claim", "source_ids": ["S99"]}]}),
            json.dumps({"supported": True, "claims": [{"text": "Claim"}]}),
        ]
        for output in outputs:
            provider = MagicMock()
            provider.generate.return_value = output
            with self.subTest(output=output), patch(
                'rag.rag_service.get_llm_provider', return_value=provider
            ):
                result = self.rag_service.generate_answer("Question?")
            self.assertTrue(result["insufficient_context"])
            self.assertEqual(result["citations"], [])

    def test_answer_claim_citations_are_built_from_exact_retrieved_locators(self):
        chunk = {
            "chunk_id": 31, "material_id": 3, "chunk_index": 0,
            "source_type": "transcript", "source_file": "lecture.txt",
            "page_number": 1, "location_type": "timestamp",
            "location_value": "00:01:25", "content": "Evidence.",
        }
        self._add_stored_chunk(
            31, 3, 0, "Evidence.", "lecture.txt", "transcript", 1,
            "timestamp", "00:01:25", "transcript",
        )
        provider = MagicMock()
        provider.generate.return_value = json.dumps({
            "supported": True,
            "claims": [{"text": "Supported claim.", "source_ids": ["S1"]}],
        })
        with patch('rag.rag_service.get_llm_provider', return_value=provider):
            result = self.rag_service.generate_answer("Question?", retrieved_chunks=[chunk])
        self.assertEqual(result["answer"], "Supported claim. [lecture.txt, timestamp 00:01:25]")
        self.assertEqual(result["citations"][0]["location_value"], "00:01:25")

    def test_set_embedding_service(self):
        """Test setting embedding service after initialization."""
        rag_service = RAGService(embedding_service=None, database_path=self.database_path)
        self.assertIsNone(rag_service.embedding_service)

        rag_service.set_embedding_service(self.mock_embedding_service)
        self.assertIsNotNone(rag_service.embedding_service)

        mock_llm = MockLLMProvider()
        with patch('rag.rag_service.get_llm_provider', return_value=mock_llm):
            result = rag_service.generate_answer(query="Test query")
            self.assertIsNotNone(result.get("answer"))


class TestOpenAIProvider(unittest.TestCase):
    """Test suite for OpenAI provider."""

    def test_provider_requires_api_key(self):
        """Test that provider fails without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with patch('ai.llm_provider.Config.OPENAI_API_KEY', None):
                with self.assertRaises(LLMConfigurationError):
                    OpenAIProvider()

    def test_provider_with_mock_api_key(self):
        """Test provider initialization with API key."""
        with patch('ai.llm_provider.Config.OPENAI_API_KEY', 'test-key'):
            with patch('openai.OpenAI'):
                provider = OpenAIProvider()
                self.assertIsNotNone(provider)

    def test_generate_calls_openai(self):
        """Test that generate calls OpenAI API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response

        with patch('ai.llm_provider.Config.OPENAI_API_KEY', 'test-key'):
            with patch('openai.OpenAI', return_value=mock_client):
                provider = OpenAIProvider()
                result = provider.generate("Test prompt")

                self.assertEqual(result, "Test response")
                mock_client.chat.completions.create.assert_called_once()

    def test_generate_with_system_prompt(self):
        """Test generate with system prompt."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response

        with patch('ai.llm_provider.Config.OPENAI_API_KEY', 'test-key'):
            with patch('openai.OpenAI', return_value=mock_client):
                provider = OpenAIProvider()
                provider.generate("Test prompt", system_prompt="System instruction")

                call_args = mock_client.chat.completions.create.call_args
                messages = call_args[1]['messages']
                self.assertEqual(len(messages), 2)
                self.assertEqual(messages[0]['role'], 'system')
                self.assertEqual(messages[1]['role'], 'user')


class TestOllamaProvider(unittest.TestCase):
    """Ollama provider tests with all HTTP requests mocked."""

    def test_generate_sends_chat_messages_and_returns_content(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "message": {"content": "Grounded answer. [dbms.pdf, page 4]"}
        }).encode("utf-8")

        provider = OllamaProvider(
            model="qwen2.5:1.5b",
            base_url="http://localhost:11434/",
            timeout=75,
        )
        with patch(
            'ai.llm_provider.urllib.request.urlopen',
            return_value=response,
        ) as mocked_urlopen:
            answer = provider.generate(
                "Question and retrieved passage",
                system_prompt="Use only the supplied passages.",
            )

        self.assertEqual(answer, "Grounded answer. [dbms.pdf, page 4]")
        mocked_urlopen.assert_called_once()
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(mocked_urlopen.call_args.kwargs['timeout'], 75)
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body['model'], "qwen2.5:1.5b")
        self.assertFalse(body['stream'])
        self.assertEqual(body['messages'], [
            {"role": "system", "content": "Use only the supplied passages."},
            {"role": "user", "content": "Question and retrieved passage"},
        ])

    def test_ollama_provider_selection_does_not_require_openai_key(self):
        with patch('ai.llm_provider.Config.LLM_PROVIDER', 'ollama'), \
                patch('ai.llm_provider.Config.OPENAI_API_KEY', None):
            provider = get_llm_provider()

        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.model, "qwen2.5:1.5b")

    def test_openai_selection_without_key_has_clear_configuration_error(self):
        with patch('ai.llm_provider.Config.LLM_PROVIDER', 'openai'), \
                patch('ai.llm_provider.Config.OPENAI_API_KEY', None):
            with self.assertRaisesRegex(
                LLMConfigurationError,
                'OpenAI API key not configured',
            ):
                get_llm_provider()

    def test_timeout_and_connection_errors_are_reported_as_provider_errors(self):
        failures = (
            TimeoutError("request timed out"),
            urllib.error.URLError(ConnectionRefusedError("connection refused")),
        )

        for failure in failures:
            with self.subTest(failure=type(failure).__name__), patch(
                'ai.llm_provider.urllib.request.urlopen',
                side_effect=failure,
            ):
                with self.assertRaisesRegex(
                    LLMProviderError,
                    'Could not reach the local Ollama service',
                ):
                    OllamaProvider().generate("Test prompt")

    def test_ollama_generation_error_is_reported_as_provider_error(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "error": "model is unavailable"
        }).encode("utf-8")

        with patch('ai.llm_provider.urllib.request.urlopen', return_value=response):
            with self.assertRaisesRegex(LLMProviderError, 'model is unavailable'):
                OllamaProvider().generate("Test prompt")


if __name__ == '__main__':
    unittest.main()
