import os
import logging
import sqlite3
import time
from typing import List, Dict, Optional, Tuple, Any
from flask import current_app
from werkzeug.utils import secure_filename
from config import Config
from ingestion import PDFProcessor, Chunker
from ingestion.pptx_processor import PPTXProcessor
from ingestion.transcript_processor import TranscriptProcessor
from database import get_db_connection


def _log_processing_stage(level: str, message: str, *args) -> None:
    try:
        logger = current_app.logger
    except RuntimeError:
        logger = logging.getLogger(__name__)
    getattr(logger, level)(message, *args)


class MaterialService:
    """Service for managing learning materials."""

    ALLOWED_EXTENSIONS = {'pdf', 'pptx', 'txt'}
    MAX_FILE_SIZE = Config.MAX_CONTENT_LENGTH

    def __init__(
        self,
        upload_folder: str,
        embedding_service: Optional[Any] = None,
        database_path: Optional[str] = None
    ):
        self.upload_folder = upload_folder
        self.chunker = Chunker(chunk_size=1000, overlap=200)
        self._embedding_service = embedding_service
        self.database_path = database_path

    @property
    def embedding_service(self):
        """Lazily initialize or return the shared EmbeddingService instance."""
        if self._embedding_service is not None:
            return self._embedding_service
        try:
            import app
            if hasattr(app, 'get_embedding_service'):
                self._embedding_service = app.get_embedding_service()
                return self._embedding_service
        except (ImportError, AttributeError):
            pass
        from services.embedding_service import EmbeddingService
        self._embedding_service = EmbeddingService()
        return self._embedding_service


    def _get_connection(self):
        """Get database connection, respecting optional database_path override."""
        if self.database_path:
            conn = sqlite3.connect(self.database_path)
            conn.row_factory = sqlite3.Row
            return conn
        return get_db_connection()

    def is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS

    def save_uploaded_file(self, file, user_id: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Save uploaded file to disk.

        Args:
            file: File object from Flask request.
            user_id: ID of the user uploading the file.

        Returns:
            Tuple of (file_path, error_message). Error message is None on success.
        """
        if not file or not file.filename:
            return None, "No file provided"

        if not self.is_allowed_file(file.filename):
            return None, "Invalid file type. Only PDF, PowerPoint (.pptx), and timestamped transcript (.txt) files are allowed."

        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size > self.MAX_FILE_SIZE:
            return None, f"File too large. Maximum size is {self.MAX_FILE_SIZE // (1024*1024)}MB."

        # Secure filename
        filename = secure_filename(file.filename)
        if not filename:
            return None, "Invalid filename"

        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_folder, exist_ok=True)

        # Reserve a unique path atomically so concurrent uploads cannot overwrite
        # a file that was just created by another request.
        name, extension = os.path.splitext(filename)
        suffix = 0
        while True:
            candidate = filename if suffix == 0 else f"{name}_{suffix}{extension}"
            file_path = os.path.join(self.upload_folder, candidate)
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
            try:
                file_descriptor = os.open(file_path, flags, 0o666)
                break
            except FileExistsError:
                suffix += 1

        try:
            with os.fdopen(file_descriptor, "wb") as destination:
                file.save(destination)
        except Exception:
            try:
                os.remove(file_path)
            except OSError:
                pass
            raise

        return file_path, None

    def process_pdf(self, file_path: str, filename: str, user_id: int) -> Tuple[Optional[int], Optional[str]]:
        """
        Process uploaded PDF: extract text, chunk, store in database, and index semantically.

        Args:
            file_path: Path to the uploaded PDF file.
            filename: Original filename.
            user_id: ID of the user who uploaded the file.

        Returns:
            Tuple of (material_id, error_message). Error message is None on success.
        """
        material_id = None
        try:
            # Validate, count, and extract from one open PDF reader.
            processor = PDFProcessor(file_path)
            extraction_started = time.perf_counter()
            _log_processing_stage('info', 'PDF text extraction started')
            try:
                is_valid_pdf, page_count, pages = processor.validate_and_extract_text_by_page()
            except Exception:
                _log_processing_stage(
                    'warning',
                    'PDF text extraction failed after %.2f seconds',
                    time.perf_counter() - extraction_started,
                )
                raise
            if not is_valid_pdf:
                return None, "Invalid or corrupted PDF file"
            if page_count == 0:
                return None, "PDF has no pages"
            _log_processing_stage(
                'info',
                'PDF text extraction completed pages=%d duration_seconds=%.2f',
                len(pages),
                time.perf_counter() - extraction_started,
            )
            if not pages:
                return None, "No text could be extracted from PDF"

            # Create material record
            file_size = os.path.getsize(file_path)
            material_id = self._create_material_record(
                user_id=user_id,
                filename=filename,
                file_type='pdf',
                file_path=file_path,
                file_size=file_size,
                page_count=page_count,
                processing_status='processing'
            )

            # Chunk the text
            chunks = self.chunker.chunk_pages(pages, source_file=filename, source_type='textbook')

            # Store chunks in database (transaction commits inside _store_chunks)
            storage_started = time.perf_counter()
            _log_processing_stage('info', 'SQLite chunk storage started chunks=%d', len(chunks))
            try:
                self._store_chunks(material_id, chunks)
            except Exception:
                _log_processing_stage(
                    'warning',
                    'SQLite chunk storage failed after %.2f seconds',
                    time.perf_counter() - storage_started,
                )
                raise
            _log_processing_stage(
                'info',
                'SQLite chunk storage completed chunks=%d duration_seconds=%.2f',
                len(chunks),
                time.perf_counter() - storage_started,
            )

            # Index chunks with EmbeddingService
            indexing_started = None
            try:
                embedding_service = self.embedding_service
                if embedding_service is not None:
                    indexing_started = time.perf_counter()
                    _log_processing_stage('info', 'FAISS indexing started chunks=%d', len(chunks))
                    indexed_count = embedding_service.index_material(
                        material_id,
                        database_path=self.database_path,
                    )
                    _log_processing_stage(
                        'info',
                        'FAISS indexing completed chunks=%d duration_seconds=%.2f',
                        indexed_count,
                        time.perf_counter() - indexing_started,
                    )
                self._update_material_status(material_id, 'processed')
                return material_id, None
            except Exception as index_err:
                if indexing_started is not None:
                    _log_processing_stage(
                        'warning',
                        'FAISS indexing failed after %.2f seconds',
                        time.perf_counter() - indexing_started,
                    )
                else:
                    _log_processing_stage('warning', 'Embedding initialization or indexing failed')
                self._update_material_status(material_id, 'indexing_failed')
                return material_id, f"Material saved, but semantic indexing failed: {str(index_err)}"

        except Exception as e:
            if material_id:
                self._update_material_status(material_id, 'failed')
            return None, f"Processing failed: {str(e)}"

    def process_file(self, file_path: str, filename: str, user_id: int) -> Tuple[Optional[int], Optional[str]]:
        """Dispatch a saved material to its PDF, PPTX, or transcript processor."""
        extension = os.path.splitext(filename)[1].lower()
        if extension == '.pdf':
            return self.process_pdf(file_path, filename, user_id)
        if extension == '.pptx':
            return self.process_pptx(file_path, filename, user_id)
        if extension == '.txt':
            return self.process_transcript(file_path, filename, user_id)
        return None, "Invalid file type. Only PDF, PowerPoint (.pptx), and timestamped transcript (.txt) files are allowed."

    def process_transcript(self, file_path: str, filename: str, user_id: int) -> Tuple[Optional[int], Optional[str]]:
        """Extract, chunk, store, and incrementally index timestamped transcript text."""
        material_id = None
        try:
            processor = TranscriptProcessor(file_path)
            sections = processor.extract_sections()
            file_size = os.path.getsize(file_path)
            material_id = self._create_material_record(
                user_id=user_id,
                filename=filename,
                file_type='transcript',
                file_path=file_path,
                file_size=file_size,
                page_count=len(sections),
                processing_status='processing',
            )

            chunks = self.chunker.chunk_pages(
                sections, source_file=filename, source_type='transcript'
            )
            for chunk in chunks:
                # page_number is required by legacy schemas. Its value is a
                # compatibility sentinel only; timestamp is the citation locator.
                chunk['page'] = TranscriptProcessor.LEGACY_PAGE_NUMBER

            self._store_chunks(material_id, chunks)
            try:
                embedding_service = self.embedding_service
                if embedding_service is not None:
                    embedding_service.index_material(
                        material_id, database_path=self.database_path
                    )
                self._update_material_status(material_id, 'processed')
                return material_id, None
            except Exception as index_err:
                self._update_material_status(material_id, 'indexing_failed')
                return material_id, f"Material saved, but semantic indexing failed: {str(index_err)}"
        except Exception as exc:
            if material_id:
                self._update_material_status(material_id, 'failed')
            return material_id, f"Processing failed: {str(exc)}"

    def process_pptx(self, file_path: str, filename: str, user_id: int) -> Tuple[Optional[int], Optional[str]]:
        """Extract, chunk, store, and incrementally index a PowerPoint presentation."""
        material_id = None
        try:
            processor = PPTXProcessor(file_path)
            extraction_started = time.perf_counter()
            _log_processing_stage('info', 'PPTX text extraction started')
            sections = processor.extract_text_by_slide()
            slide_count = processor.slide_count
            if slide_count == 0:
                return None, "PowerPoint presentation has no slides"
            if not sections:
                return None, "No text could be extracted from PowerPoint presentation"
            _log_processing_stage(
                'info',
                'PPTX text extraction completed slides=%d duration_seconds=%.2f',
                slide_count,
                time.perf_counter() - extraction_started,
            )

            file_size = os.path.getsize(file_path)
            material_id = self._create_material_record(
                user_id=user_id,
                filename=filename,
                file_type='pptx',
                file_path=file_path,
                file_size=file_size,
                page_count=slide_count,
                processing_status='processing',
            )

            chunks = self.chunker.chunk_pages(
                sections, source_file=filename, source_type='slides'
            )
            for chunk in chunks:
                chunk['location_type'] = 'slide'
                chunk['location_value'] = str(chunk['page'])

            storage_started = time.perf_counter()
            _log_processing_stage('info', 'SQLite chunk storage started chunks=%d', len(chunks))
            self._store_chunks(material_id, chunks)
            _log_processing_stage(
                'info',
                'SQLite chunk storage completed chunks=%d duration_seconds=%.2f',
                len(chunks),
                time.perf_counter() - storage_started,
            )

            indexing_started = None
            try:
                embedding_service = self.embedding_service
                if embedding_service is not None:
                    indexing_started = time.perf_counter()
                    _log_processing_stage('info', 'FAISS indexing started chunks=%d', len(chunks))
                    indexed_count = embedding_service.index_material(
                        material_id,
                        database_path=self.database_path,
                    )
                    _log_processing_stage(
                        'info',
                        'FAISS indexing completed chunks=%d duration_seconds=%.2f',
                        indexed_count,
                        time.perf_counter() - indexing_started,
                    )
                self._update_material_status(material_id, 'processed')
                return material_id, None
            except Exception as index_err:
                if indexing_started is not None:
                    _log_processing_stage(
                        'warning',
                        'FAISS indexing failed after %.2f seconds',
                        time.perf_counter() - indexing_started,
                    )
                else:
                    _log_processing_stage('warning', 'Embedding initialization or indexing failed')
                self._update_material_status(material_id, 'indexing_failed')
                return material_id, f"Material saved, but semantic indexing failed: {str(index_err)}"

        except Exception as exc:
            if material_id:
                self._update_material_status(material_id, 'failed')
            return material_id, f"Processing failed: {str(exc)}"

    def retry_indexing(self, material_id: int) -> Tuple[bool, Optional[str]]:
        """
        Retry semantic indexing for an existing material that failed or needs re-indexing.

        Returns:
            Tuple of (success_bool, error_message).
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, processing_status FROM materials WHERE id = ?', (material_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return False, "Material not found"

        chunks_count = self.get_material_chunks_count(material_id)
        if chunks_count == 0:
            return False, "No chunks found for this material"

        try:
            if self.embedding_service is not None:
                self.embedding_service.index_material(material_id, database_path=self.database_path)
            self._update_material_status(material_id, 'processed')
            return True, None
        except Exception as e:
            self._update_material_status(material_id, 'indexing_failed')
            return False, f"Indexing failed: {str(e)}"

    def _create_material_record(self, user_id: int, filename: str, file_type: str,
                                 file_path: str, file_size: int, page_count: int,
                                 processing_status: str) -> int:
        """Create a material record in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO materials (user_id, filename, file_type, file_path, file_size, page_count, processing_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, filename, file_type, file_path, file_size, page_count, processing_status))

        conn.commit()
        material_id = cursor.lastrowid
        conn.close()

        return material_id

    def _store_chunks(self, material_id: int, chunks: List[Dict[str, any]]):
        """Store chunks in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        available_columns = {
            row[1] for row in cursor.execute('PRAGMA table_info(document_chunks)')
        }

        for chunk in chunks:
            columns = [
                'material_id', 'chunk_index', 'content', 'source_type',
                'source_file', 'page_number',
            ]
            values = [
                material_id,
                chunk['chunk_index'],
                chunk['content'],
                chunk['source_type'],
                chunk['source_file'],
                chunk['page'],
            ]
            if 'location_type' in available_columns:
                columns.append('location_type')
                values.append(chunk.get('location_type') or 'page')
            if 'location_value' in available_columns:
                columns.append('location_value')
                location_value = chunk.get('location_value')
                values.append(str(chunk['page']) if location_value is None else str(location_value))

            placeholders = ', '.join('?' for _ in columns)
            cursor.execute(
                f"INSERT INTO document_chunks ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )

        conn.commit()
        conn.close()

    def _update_material_status(self, material_id: int, status: str):
        """Update material processing status."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE materials SET processing_status = ? WHERE id = ?
        ''', (status, material_id))

        conn.commit()
        conn.close()

    def get_user_materials(self, user_id: int) -> List[Dict[str, any]]:
        """Get all materials for a user."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, filename, file_type, file_size, page_count, processing_status, uploaded_at
            FROM materials
            WHERE user_id = ?
            ORDER BY uploaded_at DESC
        ''', (user_id,))

        materials = []
        for row in cursor.fetchall():
            materials.append({
                'id': row['id'],
                'filename': row['filename'],
                'file_type': row['file_type'],
                'file_size': row['file_size'],
                'page_count': row['page_count'],
                'processing_status': row['processing_status'],
                'uploaded_at': row['uploaded_at']
            })

        conn.close()
        return materials

    def get_material_chunks_count(self, material_id: int) -> int:
        """Get the number of chunks for a material."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) as count FROM document_chunks WHERE material_id = ?
        ''', (material_id,))

        count = cursor.fetchone()['count']
        conn.close()
        return count

    def get_material_with_chunks(self, material_id: int) -> Optional[Dict[str, any]]:
        """Get material with its chunks (for debugging)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, filename, file_type, page_count, processing_status FROM materials WHERE id = ?
        ''', (material_id,))

        material = cursor.fetchone()
        if not material:
            conn.close()
            return None

        cursor.execute('''
            SELECT chunk_index, content, source_file, page_number FROM document_chunks
            WHERE material_id = ? ORDER BY chunk_index
        ''', (material_id,))

        chunks = []
        for row in cursor.fetchall():
            chunks.append({
                'chunk_index': row['chunk_index'],
                'content': row['content'],
                'source_file': row['source_file'],
                'page_number': row['page_number']
            })

        conn.close()

        return {
            'id': material['id'],
            'filename': material['filename'],
            'file_type': material['file_type'],
            'page_count': material['page_count'],
            'processing_status': material['processing_status'],
            'chunks': chunks
        }
