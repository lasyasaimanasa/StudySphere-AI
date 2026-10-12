import sqlite3
import os
from config import Config


def get_db_connection():
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    db_dir = os.path.dirname(Config.DATABASE_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            page_count INTEGER,
            processing_status TEXT DEFAULT 'pending',
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Add new columns to existing materials table if they don't exist
    try:
        cursor.execute('ALTER TABLE materials ADD COLUMN file_size INTEGER')
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute('ALTER TABLE materials ADD COLUMN page_count INTEGER')
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute('ALTER TABLE materials ADD COLUMN processing_status TEXT DEFAULT "pending"')
    except sqlite3.OperationalError:
        pass  # Column already exists

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_file TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            location_type TEXT NOT NULL DEFAULT 'page',
            location_value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (material_id) REFERENCES materials (id)
        )
    ''')

    # Add source-location columns to existing databases without rebuilding or
    # rewriting document_chunks. Existing page_number values remain authoritative
    # for legacy rows until a more specific location_value is stored.
    chunk_columns = {
        row[1] for row in cursor.execute('PRAGMA table_info(document_chunks)')
    }
    if 'location_type' not in chunk_columns:
        cursor.execute(
            "ALTER TABLE document_chunks "
            "ADD COLUMN location_type TEXT NOT NULL DEFAULT 'page'"
        )
    if 'location_value' not in chunk_columns:
        cursor.execute(
            'ALTER TABLE document_chunks ADD COLUMN location_value TEXT'
        )

    # Assessment storage is additive so existing materials and chunks remain
    # untouched. One attempt per generated assessment makes submissions safely
    # idempotent at the application boundary (duplicates receive HTTP 409).
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessment_sources (
            assessment_id INTEGER NOT NULL,
            material_id INTEGER NOT NULL,
            PRIMARY KEY (assessment_id, material_id),
            FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE,
            FOREIGN KEY (material_id) REFERENCES materials(id) ON DELETE RESTRICT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessment_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER NOT NULL,
            question_order INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            options_json TEXT NOT NULL,
            correct_option INTEGER NOT NULL,
            explanation TEXT NOT NULL,
            topic TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            source_material_id INTEGER NOT NULL,
            source_file TEXT NOT NULL,
            location_type TEXT NOT NULL,
            location_value TEXT NOT NULL,
            UNIQUE (assessment_id, question_order),
            FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE,
            FOREIGN KEY (source_material_id) REFERENCES materials(id) ON DELETE RESTRICT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessment_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER NOT NULL UNIQUE,
            submitted_answers_json TEXT NOT NULL,
            score INTEGER NOT NULL,
            total_questions INTEGER NOT NULL,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessment_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            selected_option INTEGER NOT NULL,
            is_correct INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
            UNIQUE (attempt_id, question_id),
            FOREIGN KEY (attempt_id) REFERENCES assessment_attempts(id) ON DELETE CASCADE,
            FOREIGN KEY (question_id) REFERENCES assessment_questions(id) ON DELETE RESTRICT
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_assessment_questions_assessment '
                   'ON assessment_questions(assessment_id, question_order)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_assessment_attempts_completed '
                   'ON assessment_attempts(completed_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_assessment_responses_question '
                   'ON assessment_responses(question_id, is_correct)')

    # Persist the single app-scope lecture-to-transcript association. The app
    # does not yet have authenticated user isolation. SET NULL leaves a saved
    # lecture visible if its transcript material is removed later.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS youtube_lectures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL UNIQUE,
            canonical_url TEXT NOT NULL,
            display_title TEXT,
            transcript_material_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (transcript_material_id) REFERENCES materials(id) ON DELETE SET NULL
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_youtube_lectures_transcript '
                   'ON youtube_lectures(transcript_material_id)')

    conn.commit()
    conn.close()
