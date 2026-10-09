import os
from typing import List, Dict, Optional, Tuple
from werkzeug.utils import secure_filename
from ingestion import PDFProcessor, Chunker
from database import get_db_connection


class MaterialService:
    """Service for managing learning materials."""

    ALLOWED_EXTENSIONS = {'pdf'}
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

    def __init__(self, upload_folder: str):
        self.upload_folder = upload_folder
        self.chunker = Chunker(chunk_size=1000, overlap=200)

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
            return None, "Invalid file type. Only PDF files are allowed."

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

        # Save file
        file_path = os.path.join(self.upload_folder, filename)
        file.save(file_path)

        return file_path, None

    def process_pdf(self, file_path: str, filename: str, user_id: int) -> Tuple[Optional[int], Optional[str]]:
        """
        Process uploaded PDF: extract text, chunk, and store in database.

        Args:
            file_path: Path to the uploaded PDF file.
            filename: Original filename.
            user_id: ID of the user who uploaded the file.

        Returns:
            Tuple of (material_id, error_message). Error message is None on success.
        """
        try:
            # Validate PDF
            processor = PDFProcessor(file_path)
            if not processor.validate_pdf():
                return None, "Invalid or corrupted PDF file"

            # Get page count
            page_count = processor.get_page_count()
            if page_count == 0:
                return None, "PDF has no pages"

            # Extract text by page
            pages = processor.extract_text_by_page()
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

            # Store chunks in database
            self._store_chunks(material_id, chunks)

            # Update material status to processed
            self._update_material_status(material_id, 'processed')

            return material_id, None

        except Exception as e:
            # Update material status to failed if material was created
            return None, f"Processing failed: {str(e)}"

    def _create_material_record(self, user_id: int, filename: str, file_type: str,
                                 file_path: str, file_size: int, page_count: int,
                                 processing_status: str) -> int:
        """Create a material record in the database."""
        conn = get_db_connection()
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
        conn = get_db_connection()
        cursor = conn.cursor()

        for chunk in chunks:
            cursor.execute('''
                INSERT INTO document_chunks (material_id, chunk_index, content, source_type, source_file, page_number)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                material_id,
                chunk['chunk_index'],
                chunk['content'],
                chunk['source_type'],
                chunk['source_file'],
                chunk['page']
            ))

        conn.commit()
        conn.close()

    def _update_material_status(self, material_id: int, status: str):
        """Update material processing status."""
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE materials SET processing_status = ? WHERE id = ?
        ''', (status, material_id))

        conn.commit()
        conn.close()

    def get_user_materials(self, user_id: int) -> List[Dict[str, any]]:
        """Get all materials for a user."""
        conn = get_db_connection()
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
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) as count FROM document_chunks WHERE material_id = ?
        ''', (material_id,))

        count = cursor.fetchone()['count']
        conn.close()
        return count

    def get_material_with_chunks(self, material_id: int) -> Optional[Dict[str, any]]:
        """Get material with its chunks (for debugging)."""
        conn = get_db_connection()
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
