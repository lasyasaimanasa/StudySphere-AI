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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (material_id) REFERENCES materials (id)
        )
    ''')

    conn.commit()
    conn.close()
