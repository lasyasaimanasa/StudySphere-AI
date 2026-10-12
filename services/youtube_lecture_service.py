"""Persist YouTube lecture identities and their indexed transcript association."""

import re
import sqlite3
from urllib.parse import parse_qs, urlsplit

from database import get_db_connection


VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
}


class YouTubeLectureError(ValueError):
    """A user-correctable YouTube lecture request error."""

    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def canonicalize_youtube_url(value):
    """Validate supported YouTube URLs without making any network requests."""
    if not isinstance(value, str) or not value.strip():
        raise YouTubeLectureError("A YouTube lecture URL is required.")
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError as exc:
        raise YouTubeLectureError("Enter a valid YouTube URL.") from exc

    if (parsed.scheme.lower() != "https" or parsed.username or parsed.password
            or port not in (None, 443)):
        raise YouTubeLectureError("Use a standard HTTPS YouTube link without credentials or a custom port.")

    host = (parsed.hostname or "").lower()
    video_id = None
    if host == "youtu.be":
        match = re.fullmatch(r"/([A-Za-z0-9_-]{11})/?", parsed.path)
        video_id = match.group(1) if match else None
    elif host in YOUTUBE_HOSTS:
        if parsed.path in ("/watch", "/"):
            ids = parse_qs(parsed.query, keep_blank_values=True).get("v", [])
            video_id = ids[0] if len(ids) == 1 else None
        else:
            match = re.fullmatch(r"/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})/?", parsed.path)
            video_id = match.group(1) if match else None

    if not video_id or not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise YouTubeLectureError("This is not a supported YouTube video URL.")
    return video_id, f"https://www.youtube.com/watch?v={video_id}"


class YouTubeLectureService:
    """Store saved video identities and report whether their transcript remains usable."""

    def __init__(self, database_path=None):
        self.database_path = database_path

    def _connection(self):
        if self.database_path:
            connection = sqlite3.connect(self.database_path)
            connection.row_factory = sqlite3.Row
            return connection
        return get_db_connection()

    @staticmethod
    def _lecture_dict(row):
        lecture = dict(row)
        available = bool(lecture.pop("transcript_available"))
        lecture["transcript_available"] = available
        if not available:
            lecture["transcript_material_id"] = None
        return lecture

    def _select_lecture(self, connection, where_clause="", params=()):
        query = '''
            SELECT y.id, y.video_id, y.canonical_url, y.display_title,
                   y.transcript_material_id, y.created_at,
                   m.filename AS transcript_filename,
                   CASE WHEN m.id IS NOT NULL
                         AND m.file_type = 'transcript'
                         AND m.processing_status = 'processed'
                         AND EXISTS (
                             SELECT 1 FROM document_chunks AS dc
                             WHERE dc.material_id = m.id
                               AND dc.location_type = 'timestamp'
                               AND TRIM(COALESCE(dc.location_value, '')) <> ''
                               AND TRIM(dc.content) <> ''
                         ) THEN 1 ELSE 0 END AS transcript_available
            FROM youtube_lectures AS y
            LEFT JOIN materials AS m ON m.id = y.transcript_material_id
        '''
        if where_clause:
            query += " WHERE " + where_clause
        query += " ORDER BY y.created_at DESC, y.id DESC"
        return connection.execute(query, params)

    def list_lectures(self):
        connection = self._connection()
        try:
            return [self._lecture_dict(row) for row in self._select_lecture(connection)]
        finally:
            connection.close()

    def save_lecture(self, url, transcript_material_id, display_title=None):
        video_id, canonical_url = canonicalize_youtube_url(url)
        if isinstance(transcript_material_id, bool) or not isinstance(transcript_material_id, int) or transcript_material_id <= 0:
            raise YouTubeLectureError("Select a valid indexed transcript material.")
        if display_title is not None:
            if not isinstance(display_title, str):
                raise YouTubeLectureError("Lecture title must be text.")
            display_title = display_title.strip() or None
            if display_title and len(display_title) > 200:
                raise YouTubeLectureError("Lecture title must be 200 characters or fewer.")

        connection = self._connection()
        try:
            material = connection.execute(
                "SELECT id, file_type, processing_status FROM materials WHERE id = ?",
                (transcript_material_id,),
            ).fetchone()
            if material is None:
                raise YouTubeLectureError("The selected transcript material was not found.", 404)
            if material["file_type"] != "transcript":
                raise YouTubeLectureError("The selected material is not a transcript.", 422)
            if material["processing_status"] != "processed":
                raise YouTubeLectureError("The transcript has not finished indexing. Process it before saving.", 422)
            timestamp_count = connection.execute(
                '''SELECT COUNT(*) FROM document_chunks
                   WHERE material_id = ? AND location_type = 'timestamp'
                     AND TRIM(COALESCE(location_value, '')) <> ''
                     AND LENGTH(TRIM(content)) > 0''',
                (transcript_material_id,),
            ).fetchone()[0]
            if not timestamp_count:
                raise YouTubeLectureError("The selected transcript has no indexed timestamped chunks.", 422)

            connection.execute(
                '''INSERT INTO youtube_lectures
                       (video_id, canonical_url, display_title, transcript_material_id)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(video_id) DO UPDATE SET
                       canonical_url = excluded.canonical_url,
                       display_title = COALESCE(excluded.display_title, youtube_lectures.display_title),
                       transcript_material_id = excluded.transcript_material_id''',
                (video_id, canonical_url, display_title, transcript_material_id),
            )
            connection.commit()
            row = self._select_lecture(
                connection, "y.video_id = ?", (video_id,)
            ).fetchone()
            return self._lecture_dict(row)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
