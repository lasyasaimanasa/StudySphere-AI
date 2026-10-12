import importlib
import os
import unittest
from unittest.mock import patch

import config


class TestConfiguration(unittest.TestCase):
    def setUp(self):
        self.original_timeout = os.environ.get('LLM_TIMEOUT')

    def tearDown(self):
        if self.original_timeout is None:
            os.environ.pop('LLM_TIMEOUT', None)
        else:
            os.environ['LLM_TIMEOUT'] = self.original_timeout
        with patch('dotenv.load_dotenv'):
            importlib.reload(config)

    def _load_timeout(self, value):
        with patch('dotenv.load_dotenv'):
            if value is None:
                os.environ.pop('LLM_TIMEOUT', None)
            else:
                os.environ['LLM_TIMEOUT'] = value
            return importlib.reload(config).Config.LLM_TIMEOUT

    def test_llm_timeout_defaults_to_30_when_unset(self):
        self.assertEqual(self._load_timeout(None), 30)

    def test_ollama_is_default_and_does_not_load_openai_key(self):
        with patch('dotenv.load_dotenv'), patch.dict(
            os.environ,
            {
                'OPENAI_API_KEY': 'test-key-that-must-not-be-loaded',
            },
            clear=True,
        ):
            loaded_config = importlib.reload(config).Config

        self.assertEqual(loaded_config.LLM_PROVIDER, 'ollama')
        self.assertEqual(loaded_config.OLLAMA_BASE_URL, 'http://localhost:11434')
        self.assertEqual(loaded_config.OLLAMA_MODEL, 'qwen2.5:1.5b')
        self.assertEqual(loaded_config.OLLAMA_TIMEOUT, 120)
        self.assertIsNone(loaded_config.OPENAI_API_KEY)

    def test_ollama_timeout_is_configurable(self):
        with patch('dotenv.load_dotenv'), patch.dict(
            os.environ, {'OLLAMA_TIMEOUT': '75'}, clear=True
        ):
            loaded_config = importlib.reload(config).Config

        self.assertEqual(loaded_config.OLLAMA_TIMEOUT, 75)

    def test_ollama_timeout_rejects_non_positive_or_non_integer_values(self):
        for timeout in ('slow', '0', '-5'):
            with self.subTest(timeout=timeout), patch('dotenv.load_dotenv'), patch.dict(
                os.environ, {'OLLAMA_TIMEOUT': timeout}, clear=True
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    'OLLAMA_TIMEOUT must be a positive integer',
                ):
                    importlib.reload(config)

    def test_llm_timeout_accepts_positive_integer(self):
        self.assertEqual(self._load_timeout('45'), 45)

    def test_llm_timeout_rejects_non_integer(self):
        with self.assertRaisesRegex(ValueError, 'LLM_TIMEOUT must be a positive integer'):
            self._load_timeout('fast')

    def test_llm_timeout_rejects_zero(self):
        with self.assertRaisesRegex(ValueError, 'LLM_TIMEOUT must be a positive integer'):
            self._load_timeout('0')

    def test_llm_timeout_rejects_negative_value(self):
        with self.assertRaisesRegex(ValueError, 'LLM_TIMEOUT must be a positive integer'):
            self._load_timeout('-5')

    def test_flask_app_uses_configured_secret_key(self):
        import database

        with patch.object(database, 'init_db'):
            import app as app_module

        self.assertTrue(
            app_module.app.secret_key == app_module.Config.SECRET_KEY,
            'Flask app should use Config.SECRET_KEY',
        )


if __name__ == '__main__':
    unittest.main()
