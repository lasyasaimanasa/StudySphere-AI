import os
from dotenv import load_dotenv

load_dotenv()


def _get_positive_timeout(setting_name, default):
    raw_timeout = os.environ.get(setting_name)
    if raw_timeout is None:
        return default

    try:
        timeout = int(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'{setting_name} must be a positive integer number of seconds.'
        ) from exc

    if timeout <= 0:
        raise ValueError(f'{setting_name} must be a positive integer number of seconds.')
    return timeout


def _get_llm_timeout():
    return _get_positive_timeout('LLM_TIMEOUT', 30)


def _get_ollama_timeout():
    return _get_positive_timeout('OLLAMA_TIMEOUT', 120)


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DATABASE_PATH = os.environ.get('DATABASE_PATH') or 'data/studysphere.db'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB maximum request size

    # LLM Configuration
    LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'ollama').strip().lower()
    # Ollama is local and does not use OpenAI credentials. Avoid retaining the
    # credential in application configuration unless OpenAI was selected.
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') if LLM_PROVIDER == 'openai' else None
    LLM_MODEL = os.environ.get('LLM_MODEL', 'gpt-4o-mini')
    LLM_TIMEOUT = _get_llm_timeout()  # seconds
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')
    OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5:1.5b')
    OLLAMA_TIMEOUT = _get_ollama_timeout()  # seconds
