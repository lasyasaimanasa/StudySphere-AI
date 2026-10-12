# StudySphere AI

StudySphere AI is a local study assistant that answers questions using passages retrieved from uploaded course PDFs. It is intended to make long, text-based study materials easier to search and discuss while showing the source file and page for supporting passages.

## Current features

- Uploads PDF files up to 50 MB.
- Extracts text page by page with `pypdf`; pages without extractable text are skipped.
- Splits extracted text into overlapping chunks while retaining file, page, and chunk metadata.
- Generates 384-dimensional text embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Stores vectors in a FAISS inner-product index and stores the matching passage metadata in JSON.
- Stores users, materials, processing status, and document chunks in SQLite.
- Retrieves relevant passages, optionally filtered to one material. Search suppresses normalized exact duplicate passages within a material and expands its candidate search when needed to fill results.
- Generates grounded answers using local Ollama by default, with the `qwen2.5:1.5b` model configured by default.
- Shows retrieved passages and page-level citations in the Tutor. The answer prompt requests citations in the form `[filename, page X]`.

## Architecture and RAG workflow

```text
PDF upload
   │
   ▼
PDFProcessor ── extracts text and page numbers
   │
   ▼
Chunker ── makes 1,000-character chunks with 200-character overlap
   │
   ├──► SQLite: material record and document chunks
   │
   ▼
EmbeddingService ── Sentence Transformers embeddings
   │
   ├──► FAISS: data/vector_store/chunks.faiss
   └──► JSON:  data/vector_store/chunks_metadata.json

Tutor question
   │
   ▼
EmbeddingService.search ── ranks passages and returns source metadata
   │
   ▼
RAGService ── builds a prompt from the question and retrieved passages
   │
   ▼
Configured LLM provider ── Ollama locally by default, or optional OpenAI
   │
   ▼
Tutor answer, retrieved sources, and citations
```

SQLite is the source for material records and chunks. FAISS stores vectors; its parallel JSON metadata file maps vector positions to chunk IDs, material IDs, passage text, and citation fields. The embedding service loads this index lazily when retrieval or indexing is first needed.

## Prerequisites

- Windows 10 or 11, 64-bit.
- 64-bit Python compatible with the pinned packages in `requirements.txt` (Python 3.10 is recommended for this dependency set).
- Ollama for Windows for local answer generation.
- The Ollama model `qwen2.5:1.5b`.
- Internet access during dependency installation and first use if the Sentence Transformers embedding model is not already cached. Ollama answer generation itself uses the local Ollama service.

## Setup on Windows

Open PowerShell in the project directory. If the project’s `venv` does not already exist, create it:

```powershell
py -3.10 -m venv venv
```

Activate the environment and install the pinned project dependencies:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install Ollama for Windows, then download the configured model:

```powershell
ollama pull qwen2.5:1.5b
ollama list
```

Ollama normally runs as a local service after installation. If it is not running, start it in a separate PowerShell window with `ollama serve` and leave that window open.

Create `.env` from the example only if you do not already have one:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

The example selects Ollama and `qwen2.5:1.5b` by default. Review `.env` locally if you need to change settings. Set a private, random `SECRET_KEY` before exposing the development server beyond your own machine; do not commit `.env` or API credentials.

Start Flask:

```powershell
python app.py
```

Open <http://127.0.0.1:5000/> in your browser. `app.py` starts Flask in debug mode, so use it for local development only.

## Use the application

1. Open **Materials** and upload a text-based PDF. The app extracts its text, saves chunks and metadata in SQLite, and indexes the chunks for retrieval.
2. Open **Tutor**, choose **All Uploaded Materials** or a specific material from the filter, and enter a question.
3. The Tutor retrieves up to four relevant passages, sends those passages with your question to the configured answer provider, and displays the answer and source information. If retrieval finds no usable passages, the app reports that the uploaded materials do not contain enough relevant information.

Large PDFs can take several minutes to process. The upload page accepts PDF files up to 50 MB.

## Configuration

Settings are read from environment variables and, when present, `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | Selects `ollama` or `openai`. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama service URL. |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama model used for answers. |
| `OLLAMA_TIMEOUT` | `120` | Ollama request timeout in seconds; must be positive. |
| `LLM_MODEL` | `gpt-4o-mini` | Model name used by the optional OpenAI provider. |
| `LLM_TIMEOUT` | `30` | OpenAI request timeout in seconds; must be positive. |
| `OPENAI_API_KEY` | unset | Required only when `LLM_PROVIDER=openai`. |
| `DATABASE_PATH` | `data/studysphere.db` | SQLite database path; relative paths resolve from the project directory. |
| `SECRET_KEY` | development fallback | Flask session signing key; set a private random value for deployment. |

Ollama is the default and does not need an OpenAI API key. To use OpenAI instead, set `LLM_PROVIDER=openai`, provide `OPENAI_API_KEY`, and optionally set `LLM_MODEL` and `LLM_TIMEOUT`. OpenAI requests require network access and an OpenAI account. The OpenAI Python client is included in `requirements.txt`.

## Tests

The test suite covers PDF text chunking, embedding and FAISS behavior, SQLite material indexing, upload handling, search and answer APIs, citation extraction, and LLM provider configuration. It uses offline mocks for external model/provider behavior.

Run the complete suite from PowerShell at the project root:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The current suite contains 82 passing tests.

## Known limitations

- PDF is the only implemented upload format. PowerPoint and video are shown as coming soon in the interface; they are not ingested. There is no OCR for image-only or scanned PDF pages.
- Chunk size and overlap are fixed in the material service. There is no adaptive chunking.
- The app currently uses a placeholder user ID rather than authentication or multi-user access controls.
- Answers depend on retrieved text and the configured LLM; a generated citation is useful only when it matches the supplied passage metadata.
- Materials 6–8 are stale duplicate records with `processing` status. Material 9 contains the identical ordered content and is already indexed. These records are currently left unchanged; retrying or re-uploading them can add duplicate content under separate material IDs.
- There is no automatic startup retry for stale processing records. The project includes a manual per-material retry route, which should not be used for these duplicate records without an intentional recovery plan.
- The development server runs with Flask debug mode enabled and is not configured as a production deployment.

## Project structure

```text
StudySphere-AI/
├── ai/                  # Ollama and OpenAI provider implementations
├── data/                # SQLite database and vector index artifacts (local)
├── ingestion/           # PDF text extraction and chunking
├── rag/                 # Retrieval-augmented answer generation and prompts
├── services/            # Material and embedding services
├── static/              # CSS, JavaScript, and uploaded PDFs
├── templates/           # Flask/Jinja pages, including Materials and Tutor
├── tests/               # Offline unit and API tests
├── app.py               # Flask application and HTTP routes
├── config.py            # Environment-based configuration
├── database.py          # SQLite schema initialization
└── requirements.txt     # Python dependencies
```
