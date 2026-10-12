"""Parse timestamped plain-text lecture transcripts."""

import re
from typing import Any, Dict, List


_TIMESTAMPED_LINE = re.compile(
    r"^\[(\d{2,}):([0-5]\d):([0-5]\d)\]\s+(.+?)\s*$"
)


class TranscriptProcessor:
    """Extract timestamped sections from UTF-8 text transcripts.

    Format: one nonempty ``[HH:MM:SS] text`` segment per line. Blank lines are
    ignored. ``page`` is set to 1 solely for compatibility with the required
    legacy database column; timestamp metadata is the authoritative locator.
    """

    LEGACY_PAGE_NUMBER = 1

    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_sections(self) -> List[Dict[str, Any]]:
        """Return timestamped sections using the existing chunker input shape."""
        try:
            with open(self.file_path, "r", encoding="utf-8-sig") as source:
                text = source.read()
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Failed to read transcript: {exc}") from exc

        sections = []
        previous_seconds = -1
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = _TIMESTAMPED_LINE.fullmatch(line)
            if not match:
                raise ValueError(
                    f"Invalid transcript line {line_number}; expected [HH:MM:SS] text"
                )

            hours, minutes, seconds, content = match.groups()
            timestamp = f"{hours}:{minutes}:{seconds}"
            total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            if total_seconds < previous_seconds:
                raise ValueError(
                    f"Transcript timestamps must be in order (line {line_number})"
                )
            content = " ".join(content.split())
            if not content:
                raise ValueError(f"Transcript line {line_number} has no text")

            sections.append({
                "page": self.LEGACY_PAGE_NUMBER,
                "text": content,
                "location_type": "timestamp",
                "location_value": timestamp,
            })
            previous_seconds = total_seconds

        if not sections:
            raise ValueError("Transcript contains no timestamped text")
        return sections
