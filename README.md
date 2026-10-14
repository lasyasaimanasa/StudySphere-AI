# StudySphere-AI

StudySphere-AI is a Flask-based learning platform for searching course materials and asking source-grounded questions. It ingests PDF files, PowerPoint presentations, and plain-text lecture transcripts with timestamps; stores material and chunk records in SQLite; retrieves semantically relevant passages using Sentence Transformers and FAISS; and generates Tutor answers with source citations using a configured language-model provider.

## Problem and objectives

Course information is often spread across long documents, slide decks, and recorded lectures. StudySphere-AI aims to make that material easier to search and discuss while keeping answers tied to retrievable source passages and their locations.

Project objectives:

- Ingest supported learning materials and preserve their source locations.
- Retrieve relevant passages, optionally limited to one selected material.
- Generate answers from retrieved evidence and show citations rather than presenting unsupported answers as grounded.
- Support local answer generation through Ollama by default, with an optional configured OpenAI provider.

## Implemented features

- Upload and process PDF, PPTX, and timestamped UTF-8 TXT transcript files. The configured maximum upload/request size is 50 MB.
- Extract PDF text by page and PPTX text by slide. Empty PDF pages and slides without extractable text are skipped.
- Parse timestamped transcript lines in the format described below. Malformed, empty, or out-of-order transcripts are rejected.
- Chunk extracted text into overlapping passages and preserve page, slide, or timestamp location metadata.
- Store materials and chunks in SQLite. Generate normalized 384-dimensional embeddings with `sentence-transformers/all-MiniLM-L6-v2`, then incrementally add each material to a FAISS inner-product index with associated JSON metadata.
- Search indexed passages, optionally filtering to a selected material. The Tutor requests up to four passages for a question.
- Generate source-grounded Tutor answers and validate citations against retrieved passage metadata. Insufficient-context handling is retained when there is no usable supporting passage.
- Associate a validated YouTube lecture URL and optional title with an already-indexed transcript. The application does not fetch captions or download video/audio. Transcript citations can show the saved lecture title and timestamp; otherwise they use the transcript filename and timestamp.
- View project-wide progress data based on persisted assessment responses and review indexed passages for topics needing revision.
- Responsive, keyboard-accessible application navigation and layouts.

**Raw MP4 and other video files are not automatically transcribed.** The video-learning workflow requires a timestamped TXT transcript supplied by the user.

The Adaptive Quiz user interface is temporarily unavailable in this prototype. The `/quiz` page displays an unavailable notice and a link to the dashboard; do not expect an interactive quiz from the current main UI.

## Technology stack

- Python and Flask with Jinja templates.
- SQLite for application records and document chunks.
- `pypdf` for PDF text extraction and `python-pptx` for PowerPoint text extraction.
- Sentence Transformers (`all-MiniLM-L6-v2`), NumPy, and `faiss-cpu` for embedding and vector search.
- Ollama through its local HTTP API by default; the optional OpenAI provider uses the `openai` Python package.
- `python-dotenv` for loading local `.env` configuration.
- Python `unittest` for the test suite; `reportlab` is used by tests to generate sample PDFs.

## Architecture and data flow

```text
PDF / PPTX / timestamped TXT upload
                 |
                 v
 MaterialService dispatches to the relevant processor
      |                |                  |
 PDFProcessor     PPTXProcessor    TranscriptProcessor
 page extraction  slide extraction  timestamped lines
      \                |                  /
       +---------------+-----------------+
                       v
              Chunker (1000 chars,
               200-char overlap)
                       |
             +---------+---------+
             |                   |
             v                   v
      SQLite materials      SQLite document_chunks
                                 |
                                 v
                EmbeddingService embeds only the
                 uploaded material's chunks
                                 |
                    +------------+------------+
                    |                         |
                    v                         v
              FAISS index             JSON metadata

Tutor question -> semantic retrieval -> verify retrieved chunks
               -> build evidence-only prompt -> configured LLM
               -> grounded answer + validated source citations
```

SQLite is authoritative for material and chunk content. FAISS and its matching JSON metadata are used as the retrieval index. Upload indexing is incremental per material; a normal upload does not rebuild the entire index. Before using retrieved passages as answer evidence, the RAG service checks their IDs, stored content, material association, and location against SQLite.

## Repository structure

```text
StudySphere-AI/
├── ai/                 # LLM provider implementations
├── ingestion/          # PDF/PPTX/transcript processors and chunker
├── rag/                # Retrieval-augmented answer generation and prompts
├── services/           # Material, embedding, progress, assessment, and lecture services
├── static/
│   ├── css/            # Shared application styles
│   ├── js/             # Navigation, upload, Tutor, and YouTube URL behavior
│   └── uploads/        # Local uploaded materials (runtime data; do not commit)
├── templates/          # Flask/Jinja pages
├── tests/              # Unit and API tests
├── app.py              # Flask app and routes
├── config.py           # Environment-based configuration
├── database.py         # SQLite schema initialization and additive migrations
├── requirements.txt    # Pinned Python dependencies
└── README.md
```

Local SQLite and vector-index data are runtime artifacts and are not part of the source tree to share. Keep actual databases, uploaded materials, credentials, and virtual environments out of version control.

## Prerequisites

- Windows 10/11 and 64-bit Python. Python 3.10 or 3.11 is recommended for the pinned native numerical packages in `requirements.txt`.
- PowerShell and the Python launcher (`py`).
- Ollama installed locally for the default answer provider, plus the configured model. Ollama is not a Python package in `requirements.txt`.
- Internet access to install packages and, on first embedding-service initialization, download the Sentence Transformers model if it is not already cached.

The latest reported test run passed using the project's existing environment. A clean installation of all pinned dependencies on Python 3.14 has not been verified; use Python 3.10/3.11 for a fresh Windows setup if installation issues occur.

## Windows setup

Run these commands from the project root in PowerShell. Create a virtual environment with Python 3.11 (or change `3.11` to an installed compatible version):

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Ollama for Windows, then download the default model:

```powershell
ollama pull qwen2.5:1.5b
ollama list
```

If Ollama is not already running as a local service, start it in a separate PowerShell window and leave that window open:

```powershell
ollama serve
```

## Configuration

`config.py` loads environment variables and a local `.env` file. Start the app from the project root so the default relative SQLite database path resolves consistently. To create a local configuration file without overwriting an existing one:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Configuration variables read by the application:

| Variable | Application fallback | Purpose |
| --- | --- | --- |
| `SECRET_KEY` | Insecure development fallback | Flask session signing. Set a private random value; do not use the fallback when exposing the app beyond local development. |
| `DATABASE_PATH` | `data/studysphere.db` | SQLite database file path. Relative paths are interpreted from the current working directory. |
| `LLM_PROVIDER` | `ollama` | Supported values are `ollama` and `openai`. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for the local Ollama service. |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama model used for answer generation. |
| `OLLAMA_TIMEOUT` | `300` seconds | Positive integer request timeout for Ollama. `.env.example` sets this to `120` seconds. |
| `OPENAI_API_KEY` | Unset | Required if `LLM_PROVIDER=openai`. Never commit this credential. |
| `LLM_MODEL` | `gpt-4o-mini` | Model name used by the optional OpenAI provider. |
| `LLM_TIMEOUT` | `30` seconds | Positive integer request timeout for OpenAI. |

Ollama is the default and does not use an OpenAI API key. To select OpenAI, set `LLM_PROVIDER=openai` and configure a valid `OPENAI_API_KEY`; network access and a working OpenAI account are then required. Provider availability is not guaranteed by configuration alone. The upload limit, upload directory, embedding model, and vector-index locations are currently defined by the application rather than environment variables.

## Start the application

From the project root, with the virtual environment active:

```powershell
python app.py
```

Open <http://127.0.0.1:5000/>. The application starts Flask's built-in development server with debug mode enabled; use it for local development and demos only, not as a production deployment.

Implemented user-facing pages include:

- `/` — landing page
- `/dashboard` — dashboard
- `/materials` — upload and view materials
- `/tutor` — ask questions and associate YouTube links with indexed transcripts
- `/progress` — project-wide progress view
- `/quiz` — temporary-unavailable notice

## Upload and query study materials

1. Open **Materials** and upload a supported PDF, PPTX, or timestamped TXT transcript. PDF and PPTX extraction only indexes text that can be extracted from the file; scanned/image-only pages do not receive OCR.
2. Wait for processing and indexing to complete. The Materials list shows processing status and chunk counts.
3. Open **AI Tutor**, choose all uploaded materials or a specific material, and ask a question.
4. The Tutor retrieves up to four relevant chunks and requests an answer grounded in those passages. Citations identify PDF pages, PowerPoint slides, or transcript timestamps. If the passages do not support an answer, the Tutor uses its insufficient-context response rather than treating unrelated content as evidence.

The Tutor can save a supported YouTube URL and associate it with an already-indexed timestamped transcript. This stores a lecture identity/title and transcript association; it does not download or transcribe the video.

## Timestamped TXT transcript format

Save a UTF-8 plain-text `.txt` file with one timestamp and nonempty transcript segment on each nonblank line. Use `[HH:MM:SS]`, with two or more hour digits and minute/second values from `00` to `59`. Timestamps must be in chronological order.

```text
[00:00:12] Welcome to the lecture on database systems.
[00:01:25] A database index can improve lookup performance.
[01:02:03] We will now review the main concepts.
```

Blank lines are ignored. Each segment is chunked separately as needed, and chunks retain that segment's timestamp for citation. The required legacy page field is stored for compatibility but is not the transcript citation location. Plain MP4, audio, or other video files are not accepted for automatic transcription.

## Tests

Run the suite from the project root:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The latest reported full run passed **168 tests**. This is the result of that run, not a guarantee that future runs or a different environment will pass. Tests use local fixtures and mocks for model/provider behavior; they do not certify the availability of Ollama, OpenAI, external downloads, or browser behavior.

## Limitations and troubleshooting

- **LLM connection/configuration errors:** With the default provider, confirm Ollama is running at `OLLAMA_BASE_URL` and that `OLLAMA_MODEL` has been pulled. With OpenAI, confirm `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, `LLM_MODEL`, and network/account availability.
- **Slow first Tutor use:** The embedding model may need to download the first time the embedding service is initialized. Later runs can use the local model cache.
- **Upload fails or remains unindexed:** Confirm the file type is PDF, PPTX, or a valid timestamped TXT transcript, is within the 50 MB limit, and the local upload directory is writable. Read the Materials page processing status/message. Indexing depends on the embedding model and FAISS being available.
- **No useful Tutor answer:** Check that processing completed and that the selected material is correct. The Tutor is constrained to indexed source passages and may return insufficient context.
- **PDF extraction gaps:** Image-only/scanned PDFs are not OCR-processed. PPTX extraction covers available slide text and table text; it does not transcribe embedded media.
- **Video limitations:** The app can associate a saved YouTube URL/title with a timestamped TXT transcript. It does not fetch captions, download media, or transcribe raw video/audio.
- **Quiz availability:** Adaptive Quiz is not currently accessible through the prototype UI; `/quiz` explains its temporary unavailability.
- **Demo/security scope:** The application uses a placeholder user ID and does not implement authentication or per-user access control. Progress is project-wide. Do not expose the debug development server as a production service.
