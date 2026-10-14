import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Prevent app.py's import-time database initialization from touching the
# configured project database during route tests.
import database

with patch.object(database, 'init_db'):
    import app as app_module

from app import app
from rag.rag_service import RAGService
from services.youtube_lecture_service import YouTubeLectureService


class TestAnswerAPI(unittest.TestCase):
    """Tests for POST /api/answer using a mocked RAG service."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, 'answer_api.sqlite')
        with patch.object(database.Config, 'DATABASE_PATH', self.database_path):
            database.init_db()
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.previous_youtube_service = app_module.youtube_lecture_service
        self.rag_service = MagicMock()
        self.rag_service.generate_answer.return_value = {
            'answer': 'A primary key uniquely identifies each row. [dbms.pdf, page 4]',
            'citations': [{'source_file': 'dbms.pdf', 'page_number': 4}],
            'sources': [
                {
                    'chunk_id': 7,
                    'material_id': 42,
                    'chunk_index': 2,
                    'source_file': 'dbms.pdf',
                    'page_number': 4,
                    'content': 'A primary key uniquely identifies each row.',
                    'score': 0.876543,
                }
            ],
            'insufficient_context': False,
            'error': None,
        }

    def tearDown(self):
        app_module.youtube_lecture_service = self.previous_youtube_service
        self.temp_dir.cleanup()

    def _add_chunk(self, chunk_id, material_id, filename, file_type, content,
                   page_number, location_type='page', location_value=None,
                   source_type='textbook', chunk_index=0):
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            """INSERT OR IGNORE INTO materials
               (id, user_id, filename, file_type, file_path, processing_status)
               VALUES (?, 1, ?, ?, ?, 'processed')""",
            (material_id, filename, file_type, filename),
        )
        connection.execute(
            """INSERT OR REPLACE INTO document_chunks
               (id, material_id, chunk_index, content, source_type, source_file,
                page_number, location_type, location_value)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (chunk_id, material_id, chunk_index, content, source_type, filename,
             page_number, location_type, str(location_value if location_value is not None else page_number)),
        )
        connection.commit()
        connection.close()

    def test_successful_answer_includes_answer_citations_sources_and_material(self):
        connection = MagicMock()
        connection.cursor.return_value.fetchone.return_value = (42, 'dbms.pdf')

        with patch.object(app_module, 'get_rag_service', return_value=self.rag_service), \
                patch.object(app_module.material_service, '_get_connection', return_value=connection):
            response = self.client.post(
                '/api/answer',
                json={'query': 'What is a primary key?', 'top_k': 3, 'material_id': '42'},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['query'], 'What is a primary key?')
        self.assertEqual(data['answer'], self.rag_service.generate_answer.return_value['answer'])
        self.assertEqual(data['citations'], [{'source_file': 'dbms.pdf', 'page_number': 4}])
        self.assertEqual(data['sources'], [{
            'source_file': 'dbms.pdf',
            'page_number': 4,
            'location_type': 'page',
            'location_value': '4',
            'similarity_score': 0.8765,
        }])
        self.assertEqual(data['results'], [{
            'chunk_id': 7,
            'material_id': 42,
            'chunk_index': 2,
            'source_file': 'dbms.pdf',
            'page_number': 4,
            'location_type': 'page',
            'location_value': '4',
            'content': 'A primary key uniquely identifies each row.',
            'similarity_score': 0.8765,
        }])
        self.assertFalse(data['insufficient_context'])
        self.assertEqual(data['material_id'], 42)
        self.rag_service.generate_answer.assert_called_once_with(
            query='What is a primary key?', top_k=3, material_id=42
        )

    def test_rag_provider_answer_preserves_api_response_contract(self):
        embedding_service = MagicMock()
        embedding_service.search.return_value = [{
            'chunk_id': 7,
            'material_id': 42,
            'chunk_index': 2,
            'source_type': 'textbook',
            'source_file': 'dbms.pdf',
            'page_number': 4,
            'content': 'A primary key uniquely identifies each row.',
            'score': 0.876543,
        }]
        self._add_chunk(7, 42, 'dbms.pdf', 'pdf',
                        'A primary key uniquely identifies each row.', 4, chunk_index=2)
        rag_service = RAGService(
            embedding_service=embedding_service, database_path=self.database_path
        )
        provider = MagicMock()
        provider.generate.return_value = json.dumps({
            'supported': True,
            'claims': [{
                'text': 'A primary key uniquely identifies each row.',
                'source_ids': ['S1'],
            }],
        })

        with patch.object(app_module, 'get_rag_service', return_value=rag_service), \
                patch('rag.rag_service.get_llm_provider', return_value=provider):
            response = self.client.post(
                '/api/answer',
                json={'query': 'What is a primary key?', 'top_k': 3},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['answer'], 'A primary key uniquely identifies each row. [dbms.pdf, page 4]')
        self.assertEqual(data['citations'], [
            {'source_file': 'dbms.pdf', 'page_number': 4}
        ])
        self.assertEqual(data['sources'], [{
            'source_file': 'dbms.pdf',
            'page_number': 4,
            'location_type': 'page',
            'location_value': '4',
            'similarity_score': 0.8765,
        }])
        self.assertEqual(data['results'][0]['content'], 'A primary key uniquely identifies each row.')
        self.assertFalse(data['insufficient_context'])
        self.assertIsNone(data['material_id'])
        provider.generate.assert_called_once()
        self.assertIn('A primary key uniquely identifies each row.', provider.generate.call_args.kwargs['prompt'])
        self.assertIn('[dbms.pdf, page 4]', provider.generate.call_args.kwargs['prompt'])

    def test_pptx_answer_includes_slide_citation_and_typed_api_metadata(self):
        embedding_service = MagicMock()
        embedding_service.search.return_value = [{
            'chunk_id': 15,
            'material_id': 8,
            'chunk_index': 2,
            'source_type': 'slides',
            'source_file': 'lecture_slides.pptx',
            'page_number': 3,
            'location_type': 'slide',
            'location_value': '3',
            'content': 'Photosynthesis converts light into chemical energy.',
            'score': 0.93,
        }]
        self._add_chunk(15, 8, 'lecture_slides.pptx', 'pptx',
                        'Photosynthesis converts light into chemical energy.', 3,
                        'slide', '3', 'slides', 2)
        rag_service = RAGService(
            embedding_service=embedding_service, database_path=self.database_path
        )
        provider = MagicMock()
        provider.generate.return_value = json.dumps({
            'supported': True,
            'claims': [{
                'text': 'Photosynthesis converts light into chemical energy.',
                'source_ids': ['S1'],
            }],
        })

        with patch.object(app_module, 'get_rag_service', return_value=rag_service), \
                patch('rag.rag_service.get_llm_provider', return_value=provider):
            response = self.client.post(
                '/api/answer', json={'query': 'What does photosynthesis do?'}
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['answer'],
                         'Photosynthesis converts light into chemical energy. [lecture_slides.pptx, slide 3]')
        self.assertEqual(data['citations'], [{
            'source_file': 'lecture_slides.pptx',
            'page_number': 3,
            'location_type': 'slide',
            'location_value': '3',
        }])
        self.assertEqual(data['sources'][0]['location_type'], 'slide')
        self.assertEqual(data['sources'][0]['location_value'], '3')
        self.assertEqual(data['results'][0]['page_number'], 3)
        self.assertEqual(data['results'][0]['location_type'], 'slide')
        self.assertEqual(data['results'][0]['location_value'], '3')
        self.assertNotIn('invented.pptx', data['answer'])

    def test_transcript_answer_returns_validated_timestamp_metadata(self):
        embedding_service = MagicMock()
        embedding_service.search.return_value = [{
            'chunk_id': 21,
            'material_id': 9,
            'chunk_index': 0,
            'source_type': 'transcript',
            'source_file': 'lecture_transcript.txt',
            'page_number': 1,
            'location_type': 'timestamp',
            'location_value': '00:01:25',
            'content': 'A DBMS organizes, stores, and retrieves data.',
            'score': 0.91,
        }]
        self._add_chunk(21, 9, 'lecture_transcript.txt', 'transcript',
                        'A DBMS organizes, stores, and retrieves data.', 1,
                        'timestamp', '00:01:25', 'transcript')
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            """INSERT INTO youtube_lectures
               (video_id, canonical_url, display_title, transcript_material_id)
               VALUES (?, ?, ?, ?)""",
            ('dQw4w9WgXcQ', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
             'Database Systems Lecture', 9),
        )
        connection.commit()
        connection.close()
        rag_service = RAGService(
            embedding_service=embedding_service, database_path=self.database_path
        )
        provider = MagicMock()
        provider.generate.return_value = json.dumps({
            'supported': True,
            'claims': [{
                'text': 'A DBMS organizes, stores, and retrieves data.',
                'source_ids': ['S1'],
            }],
        })

        with patch.object(app_module, 'get_rag_service', return_value=rag_service), \
                patch('rag.rag_service.get_llm_provider', return_value=provider):
            response = self.client.post(
                '/api/answer', json={'query': 'What does a DBMS do?'}
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        expected = {
            'source_file': 'lecture_transcript.txt',
            'video_title': 'Database Systems Lecture',
            'page_number': 1,
            'location_type': 'timestamp',
            'location_value': '00:01:25',
        }
        self.assertEqual(data['citations'], [expected])
        self.assertIn(
            '[Database Systems Lecture, timestamp 00:01:25]',
            data['answer'],
        )
        self.assertEqual(data['sources'][0]['video_title'], 'Database Systems Lecture')
        self.assertEqual(data['sources'][0]['location_type'], 'timestamp')
        self.assertEqual(data['sources'][0]['location_value'], '00:01:25')
        self.assertEqual(data['results'][0]['page_number'], 1)
        self.assertEqual(data['results'][0]['location_type'], 'timestamp')
        self.assertEqual(data['results'][0]['location_value'], '00:01:25')
        self.assertNotIn('00:03:99', data['answer'])

    def test_saved_youtube_lecture_scopes_answer_api_to_associated_transcript(self):
        video_id = 'dQw4w9WgXcQ'
        transcript_id = 31
        transcript = {
            'chunk_id': 310, 'material_id': transcript_id, 'chunk_index': 0,
            'source_type': 'transcript', 'source_file': 'authorized_lecture.txt',
            'page_number': 1, 'location_type': 'timestamp',
            'location_value': '00:02:10',
            'content': 'The lecture defines eigenvectors as directions unchanged by a transformation.',
            'score': 0.92,
        }
        unrelated = {
            'chunk_id': 420, 'material_id': 42, 'chunk_index': 0,
            'source_type': 'textbook', 'source_file': 'other_course.pdf',
            'page_number': 9, 'location_type': 'page', 'location_value': '9',
            'content': 'An unrelated passage.', 'score': 0.8,
        }
        self._add_chunk(
            310, transcript_id, 'authorized_lecture.txt', 'transcript',
            transcript['content'], 1, 'timestamp', '00:02:10', 'transcript',
        )
        self._add_chunk(420, 42, 'other_course.pdf', 'pdf', unrelated['content'], 9)

        youtube_service = YouTubeLectureService(self.database_path)
        app_module.youtube_lecture_service = youtube_service
        saved = youtube_service.save_lecture(
            f'https://www.youtube.com/watch?v={video_id}', transcript_id, 'Linear algebra lecture'
        )
        self.assertTrue(saved['transcript_available'])

        class RetrievalStub:
            def __init__(self):
                self.calls = []

            def search(self, query, top_k=5, material_id=None):
                self.calls.append((query, top_k, material_id))
                # Deliberately return a broad candidate set; RAG re-enforces scope.
                return [transcript, unrelated]

        retrieval = RetrievalStub()
        rag_service = RAGService(retrieval, database_path=self.database_path)
        provider = MagicMock()
        provider.generate.return_value = json.dumps({
            'supported': True,
            'claims': [{
                'text': 'An eigenvector is a direction unchanged by a transformation.',
                'source_ids': ['S1'],
            }],
        })
        def open_material_connection():
            connection = sqlite3.connect(self.database_path)
            connection.row_factory = sqlite3.Row
            return connection
        with patch.object(app_module, 'get_rag_service', return_value=rag_service), \
                patch.object(app_module.material_service, '_get_connection',
                             side_effect=open_material_connection), \
                patch('rag.rag_service.get_llm_provider', return_value=provider):
            response = self.client.post('/api/answer', json={
                'query': 'What is an eigenvector in this lecture?',
                'material_id': saved['transcript_material_id'],
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(retrieval.calls[0][2], transcript_id)
        data = response.get_json()
        self.assertEqual(data['material_id'], transcript_id)
        self.assertEqual(len(data['results']), 1)
        self.assertEqual(data['results'][0]['source_file'], 'authorized_lecture.txt')
        self.assertEqual(data['citations'][0]['location_value'], '00:02:10')

    def test_unavailable_saved_transcript_is_rejected_as_selected_evidence(self):
        youtube_service = YouTubeLectureService(self.database_path)
        app_module.youtube_lecture_service = youtube_service
        self._add_chunk(
            311, 32, 'removed_lecture.txt', 'transcript', 'Removed lecture content.',
            1, 'timestamp', '00:00:30', 'transcript',
        )
        saved = youtube_service.save_lecture(
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 32
        )
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute('DELETE FROM materials WHERE id = 32')
            connection.commit()
        finally:
            connection.close()
        self.assertFalse(youtube_service.list_lectures()[0]['transcript_available'])

        retrieval = MagicMock()
        rag_service = RAGService(retrieval, database_path=self.database_path)
        def open_material_connection():
            connection = sqlite3.connect(self.database_path)
            connection.row_factory = sqlite3.Row
            return connection
        with patch.object(app_module, 'get_rag_service', return_value=rag_service), \
                patch.object(app_module.material_service, '_get_connection',
                             side_effect=open_material_connection):
            response = self.client.post('/api/answer', json={
                'query': 'Summarize the saved lecture',
                'material_id': saved['transcript_material_id'],
            })
        self.assertEqual(response.status_code, 404)
        retrieval.search.assert_not_called()

    def test_missing_invalid_and_blank_questions_return_400(self):
        invalid_payloads = [
            {},
            {'query': ''},
            {'query': '   '},
            {'query': 123},
        ]

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post('/api/answer', json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn('Query parameter is required', response.get_json()['error'])

        self.rag_service.generate_answer.assert_not_called()

    def test_invalid_top_k_values_return_400(self):
        for top_k in (0, -1, 51, 'not-an-integer'):
            with self.subTest(top_k=top_k):
                response = self.client.post(
                    '/api/answer', json={'query': 'Valid question', 'top_k': top_k}
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn('top_k', response.get_json()['error'])

        self.rag_service.generate_answer.assert_not_called()

    def test_nonexistent_material_id_returns_404(self):
        connection = MagicMock()
        connection.cursor.return_value.fetchone.return_value = None

        with patch.object(app_module.material_service, '_get_connection', return_value=connection), \
                patch.object(app_module, 'get_rag_service', return_value=self.rag_service):
            response = self.client.post(
                '/api/answer', json={'query': 'Valid question', 'material_id': 999999}
            )

        self.assertEqual(response.status_code, 404)
        self.assertIn('not found', response.get_json()['error'])
        self.rag_service.generate_answer.assert_not_called()

    def test_insufficient_context_returns_empty_results(self):
        self.rag_service.generate_answer.return_value = {
            'answer': 'The materials do not contain enough information.',
            'citations': [],
            'sources': [],
            'insufficient_context': True,
            'error': None,
        }

        with patch.object(app_module, 'get_rag_service', return_value=self.rag_service):
            response = self.client.post('/api/answer', json={'query': 'Uncovered question'})

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['insufficient_context'])
        self.assertEqual(data['answer'], 'The materials do not contain enough information.')
        self.assertEqual(data['results'], [])

    def test_provider_error_retains_retrieved_results_without_exposing_details(self):
        self.rag_service.generate_answer.return_value = {
            'answer': None,
            'citations': [],
            'sources': [{
                'chunk_id': 11,
                'material_id': 42,
                'chunk_index': 0,
                'source_file': 'dbms.pdf',
                'page_number': 4,
                'content': 'Retrieved passage remains available.',
                'score': 0.91234,
            }],
            'insufficient_context': False,
            'error': 'Provider exception included sensitive internal details',
        }

        with patch.object(app_module, 'get_rag_service', return_value=self.rag_service):
            response = self.client.post('/api/answer', json={'query': 'Valid question'})

        self.assertEqual(response.status_code, 500)
        data = response.get_json()
        self.assertEqual(data['query'], 'Valid question')
        self.assertIsNone(data['answer'])
        self.assertEqual(data['citations'], [])
        self.assertEqual(data['sources'], [])
        self.assertFalse(data['insufficient_context'])
        self.assertEqual(data['error'], 'Answer generation is unavailable right now. Please try again later.')
        self.assertNotIn('sensitive internal details', data['error'])
        self.assertEqual(data['results'], [{
            'chunk_id': 11,
            'material_id': 42,
            'chunk_index': 0,
            'source_file': 'dbms.pdf',
            'page_number': 4,
            'location_type': 'page',
            'location_value': '4',
            'content': 'Retrieved passage remains available.',
            'similarity_score': 0.9123,
        }])


if __name__ == '__main__':
    unittest.main()
