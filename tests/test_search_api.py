import unittest
import os
import sys
import json
from unittest.mock import MagicMock
import sqlite3
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import database

with patch.object(database, 'init_db'):
    import app as app_module
from app import app


class TestSearchAPI(unittest.TestCase):
    """Test suite for POST /api/search endpoint and AI Tutor backend integration."""

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.mock_service = MagicMock()
        self.mock_service.index.ntotal = 5
        app_module.embedding_service = self.mock_service
        self.previous_youtube_service = app_module.youtube_lecture_service
        app_module.youtube_lecture_service = MagicMock()
        app_module.youtube_lecture_service.list_lectures.return_value = []

    def tearDown(self):
        app_module.embedding_service = None
        app_module.youtube_lecture_service = self.previous_youtube_service

    def test_search_api_requires_json(self):
        """Test that requests without JSON body return 400 Bad Request."""
        res = self.client.post('/api/search', data='plain text query', content_type='text/plain')
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn('error', data)
        self.assertIn('application/json', data['error'])

    def test_search_api_missing_or_blank_query(self):
        """Test that empty or blank queries return 400 with a clear error message."""
        # Empty payload
        res1 = self.client.post('/api/search', json={})
        self.assertEqual(res1.status_code, 400)
        self.assertIn('Query parameter is required', res1.get_json()['error'])

        # Empty string
        res2 = self.client.post('/api/search', json={'query': ''})
        self.assertEqual(res2.status_code, 400)

        # Whitespace-only string
        res3 = self.client.post('/api/search', json={'query': '    '})
        self.assertEqual(res3.status_code, 400)

        # Non-string query
        res4 = self.client.post('/api/search', json={'query': 12345})
        self.assertEqual(res4.status_code, 400)

    def test_search_api_invalid_top_k(self):
        """Test that invalid top_k parameter values return 400."""
        # Zero
        res1 = self.client.post('/api/search', json={'query': 'databases', 'top_k': 0})
        self.assertEqual(res1.status_code, 400)
        self.assertIn('top_k', res1.get_json()['error'])

        # Negative
        res2 = self.client.post('/api/search', json={'query': 'databases', 'top_k': -5})
        self.assertEqual(res2.status_code, 400)

        # Too large
        res3 = self.client.post('/api/search', json={'query': 'databases', 'top_k': 100})
        self.assertEqual(res3.status_code, 400)

        # Non-integer
        res4 = self.client.post('/api/search', json={'query': 'databases', 'top_k': 'abc'})
        self.assertEqual(res4.status_code, 400)

    def test_search_api_invalid_material_id(self):
        """Test that invalid or non-existent material IDs return appropriate 400 / 404 errors."""
        # Negative material ID
        res1 = self.client.post('/api/search', json={'query': 'databases', 'material_id': -1})
        self.assertEqual(res1.status_code, 400)

        # Non-integer material ID
        res2 = self.client.post('/api/search', json={'query': 'databases', 'material_id': 'invalid'})
        self.assertEqual(res2.status_code, 400)

        # Non-existent material ID
        res3 = self.client.post('/api/search', json={'query': 'databases', 'material_id': 999999})
        self.assertEqual(res3.status_code, 404)
        self.assertIn('not found', res3.get_json()['error'])

    def test_search_api_successful_results_and_metadata(self):
        """Test that valid search returns 200 with all metadata keys and formatted scores."""
        self.mock_service.search.return_value = [
            {
                'chunk_id': 10,
                'material_id': 1,
                'chunk_index': 0,
                'source_file': 'dbms_unit1.pdf',
                'page_number': 4,
                'location_type': 'page',
                'location_value': '4',
                'content': 'A database management system facilitates data definition and querying.',
                'score': 0.876543
            }
        ]

        payload = {'query': 'what is dbms', 'top_k': 3}
        res = self.client.post('/api/search', json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data['query'], 'what is dbms')
        self.assertEqual(data['total_results'], 1)
        self.assertEqual(len(data['results']), 1)

        result_item = data['results'][0]
        self.assertEqual(result_item['chunk_id'], 10)
        self.assertEqual(result_item['material_id'], 1)
        self.assertEqual(result_item['chunk_index'], 0)
        self.assertEqual(result_item['source_file'], 'dbms_unit1.pdf')
        self.assertEqual(result_item['page_number'], 4)
        self.assertEqual(result_item['location_type'], 'page')
        self.assertEqual(result_item['location_value'], '4')
        self.assertEqual(result_item['content'], 'A database management system facilitates data definition and querying.')
        self.assertEqual(result_item['similarity_score'], 0.8765)

    def test_search_api_returns_slide_locator_metadata(self):
        self.mock_service.search.return_value = [{
            'chunk_id': 12,
            'material_id': 5,
            'chunk_index': 1,
            'source_file': 'lecture_slides.pptx',
            'page_number': 3,
            'location_type': 'slide',
            'location_value': '3',
            'content': 'Photosynthesis uses light energy.',
            'score': 0.91,
        }]

        response = self.client.post('/api/search', json={'query': 'photosynthesis'})

        self.assertEqual(response.status_code, 200)
        result = response.get_json()['results'][0]
        self.assertEqual(result['source_file'], 'lecture_slides.pptx')
        self.assertEqual(result['page_number'], 3)
        self.assertEqual(result['location_type'], 'slide')
        self.assertEqual(result['location_value'], '3')

    def test_search_api_returns_transcript_timestamp_metadata(self):
        self.mock_service.search.return_value = [{
            'chunk_id': 17,
            'material_id': 6,
            'chunk_index': 2,
            'source_file': 'lecture_transcript.txt',
            'page_number': 1,
            'location_type': 'timestamp',
            'location_value': '00:01:25',
            'content': 'A DBMS stores and retrieves data.',
            'score': 0.89,
        }]

        response = self.client.post('/api/search', json={'query': 'DBMS retrieves data'})

        self.assertEqual(response.status_code, 200)
        result = response.get_json()['results'][0]
        self.assertEqual(result['source_file'], 'lecture_transcript.txt')
        self.assertEqual(result['page_number'], 1)
        self.assertEqual(result['location_type'], 'timestamp')
        self.assertEqual(result['location_value'], '00:01:25')

    def test_search_api_empty_matches(self):
        """Test that queries with no relevant matches return 200 with an empty results list."""
        self.mock_service.search.return_value = []

        res = self.client.post('/api/search', json={'query': 'unrelated astronomy question'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data['total_results'], 0)
        self.assertEqual(data['results'], [])

    def test_search_api_unavailable_index(self):
        """Test search behavior when vector index is empty or unavailable."""
        self.mock_service.index.ntotal = 0

        res = self.client.post('/api/search', json={'query': 'valid query'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['total_results'], 0)
        self.assertIn('No indexed study materials', data.get('message', ''))

    def test_search_api_internal_error_handling(self):
        """Test that internal search exceptions return 500 with a clean error message."""
        self.mock_service.search.side_effect = RuntimeError("FAISS computation error")

        res = self.client.post('/api/search', json={'query': 'valid query'})
        self.assertEqual(res.status_code, 500)
        data = res.get_json()
        self.assertIn('error', data)
        self.assertIn('Search execution failed', data['error'])

    def test_tutor_page_loads_with_interface(self):
        """Test that GET /tutor renders successfully with the AI Tutor interface."""
        res = self.client.get('/tutor')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn('AI Tutor', html)
        self.assertIn('chatInput', html)
        self.assertIn('materialFilter', html)
        self.assertIn('chatMessages', html)
        self.assertIn('youtubeLectureUrl', html)
        self.assertIn('youtubeTranscriptMaterial', html)
        self.assertIn('youtubePlayerLink', html)
        self.assertIn(b'does not fetch captions or download video/audio', res.data)

    def test_tutor_labels_supported_material_locators(self):
        materials = [
            {'id': 3, 'filename': 'lecture_slides.pptx', 'file_type': 'pptx', 'page_count': 8},
            {'id': 4, 'filename': 'textbook.pdf', 'file_type': 'pdf', 'page_count': 20},
            {'id': 5, 'filename': 'lecture_transcript.txt', 'file_type': 'transcript', 'page_count': 12},
        ]
        with patch.object(app_module.material_service, 'get_user_materials', return_value=materials):
            response = self.client.get('/tutor')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'lecture_slides.pptx (8 slides)', response.data)
        self.assertIn(b'textbook.pdf (20 pages)', response.data)
        self.assertIn(b'lecture_transcript.txt (12 transcript segments)', response.data)
        self.assertIn(b'PDF page citations', response.data)
        self.assertIn(b'PowerPoint slide citations', response.data)
        self.assertIn(b'Transcript timestamp citations', response.data)


if __name__ == '__main__':
    unittest.main()
