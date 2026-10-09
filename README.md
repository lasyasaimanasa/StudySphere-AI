# StudySphere AI

**Multimodal AI Hackathon 2026 - Track D: Personalized Tutoring & Adaptive Learning**

## Current Phase
Phase 1: Flask Foundation

## Project Description
StudySphere AI is a multimodal AI study companion that ingests PDFs, PowerPoint slides, and lecture videos to build a source-grounded knowledge base. It provides RAG-based tutoring, adaptive quizzes, and personalized study recommendations based on student mastery tracking.

## Technology Stack
- Python
- Flask
- SQLite
- HTML
- CSS
- JavaScript
- Jinja2 templates

## Setup Instructions

### Create Virtual Environment
```bash
python -m venv venv
```

### Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
python app.py
```

The application will be available at http://127.0.0.1:5000/

## Project Structure
```
StudySphere-AI/
├── app.py              # Flask application
├── config.py           # Configuration settings
├── database.py         # SQLite database setup
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── .gitignore         # Git ignore rules
├── ai/                # AI module (future)
├── ingestion/         # File ingestion module (future)
├── rag/               # RAG module (future)
├── services/          # Business logic (future)
├── data/              # Database storage
├── static/            # Static assets
│   ├── css/
│   ├── js/
│   └── uploads/
└── templates/         # Jinja2 templates
```

## Features (Phase 1)
- Flask application structure
- SQLite database with users and materials tables
- Home page with project information
- Dashboard with placeholder cards
- Navigation bar
- Responsive frontend design
