"""Source-grounded multiple-choice assessment generation and scoring."""

import json
import logging
import re
import sqlite3
import time
import unicodedata
import urllib.error
from typing import Any, Dict, List, Optional

from ai.llm_provider import get_llm_provider
from database import get_db_connection
from rag.prompts import get_source_location

logger = logging.getLogger(__name__)


def _exception_diagnostic_metadata(exception):
    pending = [exception]
    visited = set()
    class_names = []
    http_status = None

    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        class_names.append(type(current).__name__)

        if isinstance(current, urllib.error.HTTPError):
            status = current.code
            if isinstance(status, int) and not isinstance(status, bool):
                http_status = status

        if current.__context__ is not None:
            pending.append(current.__context__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)

    return class_names, http_status


class AssessmentError(ValueError):
    """A safe, expected assessment validation or persistence error."""


class AssessmentService:
    """Generate assessments with deterministic evidence-quote checks.

    Exact quote matching confirms that each evidence span exists in its retrieved
    passage. Whether that span semantically supports the associated claim still
    depends on the generation model's judgment and is not guaranteed by matching.
    """

    MIN_QUESTIONS = 5
    MAX_QUESTIONS = 10
    DIFFICULTIES = {"easy", "medium", "hard"}
    OPTION_COUNT = 4
    _TIMESTAMP_VALUE = re.compile(r"^\d{2,}:[0-5]\d:[0-5]\d$")
    _INLINE_SOURCE_REFERENCE = re.compile(
        r"\[[^\]\r\n]+,\s*(?:page|slide|timestamp)\s+[^\]\r\n]+\]",
        re.IGNORECASE,
    )

    def __init__(self, database_path=None, embedding_service=None, provider_factory=None):
        self.database_path = database_path
        self.embedding_service = embedding_service
        self.provider_factory = provider_factory or get_llm_provider

    def _connection(self):
        if self.database_path:
            connection = sqlite3.connect(self.database_path)
            connection.row_factory = sqlite3.Row
        else:
            connection = get_db_connection()
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def create_assessment(
        self,
        topic: str,
        question_count: int,
        difficulty: str,
        material_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        topic = topic.strip() if isinstance(topic, str) else ""
        difficulty = difficulty.strip().lower() if isinstance(difficulty, str) else ""
        if not topic or len(topic) > 200:
            raise AssessmentError("Topic is required and must be 200 characters or fewer.")
        if isinstance(question_count, bool) or not isinstance(question_count, int):
            raise AssessmentError("Question count must be an integer between 5 and 10.")
        if not self.MIN_QUESTIONS <= question_count <= self.MAX_QUESTIONS:
            raise AssessmentError("Question count must be between 5 and 10.")
        if difficulty not in self.DIFFICULTIES:
            raise AssessmentError("Difficulty must be easy, medium, or hard.")
        if material_id is not None and (
            isinstance(material_id, bool) or not isinstance(material_id, int) or material_id <= 0
        ):
            raise AssessmentError("Material ID must be a positive integer.")
        if self.embedding_service is None:
            raise AssessmentError("Retrieval service is unavailable.")

        try:
            passages = self.embedding_service.search(
                topic, top_k=min(50, max(question_count * 3, 15)), material_id=material_id
            )
        except Exception as exc:
            raise AssessmentError("Study materials could not be searched right now.") from exc
        sources = self._validated_sources(passages, requested_material_id=material_id)
        if not sources:
            raise AssessmentError("No supported study passages were found for this topic.")
        sources = sources[:question_count]

        prompt = self._build_generation_prompt(topic, difficulty, question_count, sources)
        provider_started = time.monotonic()
        try:
            generated = self.provider_factory().generate(
                prompt=prompt,
                system_prompt=(
                    "Use only the supplied passages. If support is insufficient, do not create "
                    "questions. Return JSON only."
                ),
            )
        except Exception as exc:
            elapsed_seconds = time.monotonic() - provider_started
            exception_classes, http_status = _exception_diagnostic_metadata(exc)
            logger.warning(
                "Assessment provider call failed: %s after %.2f seconds "
                "(exception chain: %s; HTTP status: %s)",
                type(exc).__name__, elapsed_seconds,
                ", ".join(exception_classes),
                http_status if http_status is not None else "unavailable",
            )
            raise AssessmentError("Question generation is unavailable right now.") from exc
        else:
            logger.info(
                "Assessment provider call completed in %.2f seconds",
                time.monotonic() - provider_started,
            )

        questions = self._validate_generated_questions(
            generated, question_count, sources, topic, difficulty
        )
        source_material_ids = sorted({source["material_id"] for source in sources})
        connection = self._connection()
        try:
            with connection:
                cursor = connection.execute(
                    "INSERT INTO assessments (topic, difficulty) VALUES (?, ?)",
                    (topic, difficulty),
                )
                assessment_id = cursor.lastrowid
                connection.executemany(
                    "INSERT INTO assessment_sources (assessment_id, material_id) VALUES (?, ?)",
                    [(assessment_id, source_id) for source_id in source_material_ids],
                )
                for order, question in enumerate(questions):
                    source = question["source"]
                    connection.execute(
                        """INSERT INTO assessment_questions (
                               assessment_id, question_order, question_text, options_json,
                               correct_option, explanation, topic, difficulty,
                               source_material_id, source_file, location_type, location_value
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            assessment_id, order, question["question"],
                            json.dumps(question["options"], ensure_ascii=False),
                            question["correct_option"], question["explanation"],
                            topic, difficulty, source["material_id"], source["source_file"],
                            source["location_type"], str(source["location_value"]),
                        ),
                    )
        except sqlite3.IntegrityError as exc:
            raise AssessmentError("A referenced source material is no longer available.") from exc
        finally:
            connection.close()

        return self.get_assessment(assessment_id)

    def _validated_sources(
        self, passages, requested_material_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not isinstance(passages, list):
            return []
        connection = self._connection()
        sources = []
        try:
            for passage in passages:
                if not isinstance(passage, dict):
                    continue
                content = passage.get("content")
                filename = passage.get("source_file")
                material_id = passage.get("material_id")
                if (not isinstance(content, str) or not content.strip()
                        or not isinstance(filename, str) or not filename.strip()
                        or isinstance(material_id, bool) or not isinstance(material_id, int)):
                    continue
                if requested_material_id is not None and material_id != requested_material_id:
                    continue
                raw_location_type = passage.get("location_type")
                location_type, location_value = get_source_location(passage)
                if raw_location_type not in (None, "", "page", "slide", "timestamp"):
                    continue
                if location_value is None or not str(location_value).strip():
                    continue
                material = connection.execute(
                    "SELECT filename, file_type, processing_status FROM materials WHERE id = ?",
                    (material_id,),
                ).fetchone()
                if not material or material["filename"] != filename:
                    continue
                if material["processing_status"] not in (None, "processed"):
                    continue
                expected_location = {
                    "pdf": "page",
                    "pptx": "slide",
                    "transcript": "timestamp",
                }.get(material["file_type"])
                if expected_location != location_type:
                    continue
                locator_value = str(location_value).strip()
                if location_type == "timestamp":
                    if not self._TIMESTAMP_VALUE.fullmatch(locator_value):
                        continue
                elif not locator_value.isdigit() or int(locator_value) < 1:
                    continue
                source = dict(passage)
                source.update({
                    "source_id": f"S{len(sources) + 1}",
                    "source_file": filename,
                    "material_id": material_id,
                    "location_type": location_type,
                    "location_value": locator_value,
                })
                sources.append(source)
        finally:
            connection.close()
        return sources

    @staticmethod
    def _build_generation_prompt(topic, difficulty, question_count, sources):
        passages = []
        for source in sources:
            locator = f"{source['location_type']} {source['location_value']}"
            passages.append(
                f"[{source['source_id']}] {source['source_file']}, {locator}:\n"
                f"{source['content']}"
            )
        schema = {
            "questions": [{
                "question": "Question text",
                "options": ["A", "B", "C", "D"],
                "correct_option": 0,
                "explanation": "Why the correct option is supported",
                "evidence": {"source_id": "S1", "quote": "Exact supporting passage quote"},
            }]
        }
        return (
            f"Create exactly {question_count} {difficulty} difficulty multiple-choice questions "
            f"about {topic!r}, using only the passages below. Each question must have exactly "
            "four distinct options and one correct_option index from 0 to 3. Include one evidence "
            "object per question with exactly two fields: source_id and quote. The evidence object must not contain any other fields. "
            "CRITICAL: evidence must be a JSON object/dictionary with keys source_id and quote. "
            "evidence must NEVER be an array/list. Each question has exactly ONE evidence object. "
            "Example correct format: {\"evidence\": {\"source_id\": \"S1\", \"quote\": \"exact quote\"}}. "
            "The quote must directly support both the question and selected correct option; the "
            "explanation must describe that support. Make distractors plausible but clearly "
            "incorrect according to the same passage. Use no outside knowledge. If one passage "
            "cannot support the question and correct option or distinguish the options, return "
            "{\"questions\": []}. Do not include citations, filenames, or locators in generated "
            "question, option, or explanation text. Return only JSON "
            f"matching this schema: {json.dumps(schema)}\n\nPASSAGES:\n"
            + "\n\n".join(passages)
        )

    def _validate_generated_questions(self, output, count, sources, topic, difficulty):
        if not isinstance(output, str):
            raise AssessmentError("The generated assessment was not valid text.")
        text = output.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            data = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise AssessmentError("The generated assessment was not valid JSON.") from exc
        if not isinstance(data, dict) or set(data) != {"questions"}:
            raise AssessmentError("The generated assessment has an unsupported structure.")
        raw_questions = data["questions"]
        if not isinstance(raw_questions, list) or len(raw_questions) != count:
            raise AssessmentError("The generated assessment did not contain the requested question count.")

        sources_by_id = {source["source_id"]: source for source in sources}
        validated = []
        seen_questions = set()
        allowed_keys = {
            "question", "options", "correct_option", "explanation", "evidence",
        }
        for item in raw_questions:
            if not isinstance(item, dict) or set(item) != allowed_keys:
                raise AssessmentError("A generated question has an invalid structure.")
            question = item["question"]
            explanation = item["explanation"]
            options = item["options"]
            correct = item["correct_option"]
            evidence = item["evidence"]
            if (not isinstance(question, str) or not question.strip()
                    or not isinstance(explanation, str) or not explanation.strip()):
                raise AssessmentError("Generated questions and explanations must be nonempty text.")
            generated_texts = [question, explanation]
            if isinstance(options, list):
                generated_texts.extend(option for option in options if isinstance(option, str))
            if any(self._INLINE_SOURCE_REFERENCE.search(value) for value in generated_texts):
                raise AssessmentError(
                    "Generated questions must use validated passage IDs instead of inline source references."
                )
            if question.strip().casefold() in seen_questions:
                raise AssessmentError("The generated assessment contains duplicate questions.")
            seen_questions.add(question.strip().casefold())
            if (not isinstance(options, list) or len(options) != self.OPTION_COUNT
                    or any(not isinstance(option, str) or not option.strip() for option in options)):
                raise AssessmentError("Each generated question must have four nonempty options.")
            normalized_options = [" ".join(option.split()).casefold() for option in options]
            if len(set(normalized_options)) != self.OPTION_COUNT:
                raise AssessmentError("Each generated question must have distinct options.")
            if isinstance(correct, bool) or not isinstance(correct, int) or not 0 <= correct < self.OPTION_COUNT:
                raise AssessmentError("Each generated question must have one valid correct option.")
            answer_evidence = self._validate_evidence(
                evidence, sources_by_id, "question and correct answer"
            )
            source = sources_by_id[answer_evidence["source_id"]]
            validated.append({
                "question": question.strip(),
                "options": [" ".join(option.split()) for option in options],
                "correct_option": correct,
                "explanation": explanation.strip(),
                "topic": topic,
                "difficulty": difficulty,
                "answer_evidence": answer_evidence,
                "source": source,
            })
        return validated

    @staticmethod
    def _normalize_evidence_text(value):
        normalized = unicodedata.normalize("NFC", value)
        return " ".join(normalized.split()).casefold()

    def _validate_evidence(self, evidence, sources_by_id, field_name):
        """Check evidence schema, retrieved source identity, and exact quote presence."""
        # Normalize single-item list to dict for robustness
        if isinstance(evidence, list):
            if len(evidence) == 1 and isinstance(evidence[0], dict):
                evidence = evidence[0]
            else:
                raise AssessmentError(f"Generated {field_name} evidence is malformed.")
        if not isinstance(evidence, dict) or set(evidence) != {"source_id", "quote"}:
            raise AssessmentError(f"Generated {field_name} evidence is malformed.")
        source_id = evidence["source_id"]
        quote = evidence["quote"]
        if not isinstance(source_id, str) or source_id not in sources_by_id:
            raise AssessmentError(f"Generated {field_name} cited an unsupported source passage.")
        if not isinstance(quote, str) or len(quote.strip()) < 8:
            raise AssessmentError(f"Generated {field_name} evidence quote is missing or too short.")
        passage = sources_by_id[source_id]["content"]
        normalized_passage = self._normalize_evidence_text(passage)
        normalized_quote = self._normalize_evidence_text(quote)
        if normalized_quote not in normalized_passage:
            raise AssessmentError(f"Generated {field_name} evidence quote was not found in its source passage.")
        return evidence

    def get_assessment(self, assessment_id: int) -> Optional[Dict[str, Any]]:
        connection = self._connection()
        try:
            assessment = connection.execute(
                "SELECT id, topic, difficulty, created_at FROM assessments WHERE id = ?",
                (assessment_id,),
            ).fetchone()
            if not assessment:
                return None
            questions = connection.execute(
                """SELECT id, question_order, question_text, options_json, topic, difficulty
                   FROM assessment_questions WHERE assessment_id = ? ORDER BY question_order""",
                (assessment_id,),
            ).fetchall()
            return {
                "id": assessment["id"],
                "topic": assessment["topic"],
                "difficulty": assessment["difficulty"],
                "created_at": assessment["created_at"],
                "questions": [{
                    "id": row["id"],
                    "question_order": row["question_order"],
                    "question_text": row["question_text"],
                    "options": json.loads(row["options_json"]),
                    "topic": row["topic"],
                    "difficulty": row["difficulty"],
                } for row in questions],
            }
        finally:
            connection.close()

    def submit_answers(self, assessment_id: int, answers) -> Dict[str, Any]:
        if not isinstance(answers, list) or not answers:
            raise AssessmentError("Submit one answer for every question.")
        connection = self._connection()
        try:
            with connection:
                assessment = connection.execute(
                    "SELECT id FROM assessments WHERE id = ?", (assessment_id,)
                ).fetchone()
                if not assessment:
                    raise AssessmentError("Assessment not found.")
                question_rows = connection.execute(
                    """SELECT id, question_order, question_text, options_json, correct_option,
                              explanation, topic, difficulty, source_file, location_type,
                              location_value, source_material_id
                       FROM assessment_questions WHERE assessment_id = ? ORDER BY question_order""",
                    (assessment_id,),
                ).fetchall()
                if not question_rows:
                    raise AssessmentError("Assessment has no questions.")
                submitted = {}
                for answer in answers:
                    if not isinstance(answer, dict) or set(answer) != {"question_id", "selected_option"}:
                        raise AssessmentError("Each answer must include question_id and selected_option.")
                    question_id = answer["question_id"]
                    selected = answer["selected_option"]
                    if (isinstance(question_id, bool) or not isinstance(question_id, int)
                            or isinstance(selected, bool) or not isinstance(selected, int)):
                        raise AssessmentError("Question IDs and selected options must be integers.")
                    if question_id in submitted:
                        raise AssessmentError("Duplicate answers for a question are not allowed.")
                    submitted[question_id] = selected
                by_id = {row["id"]: row for row in question_rows}
                expected_ids = set(by_id)
                if set(submitted) != expected_ids:
                    raise AssessmentError("Answers must belong to this assessment and include every question.")

                attempt_exists = connection.execute(
                    "SELECT id FROM assessment_attempts WHERE assessment_id = ?",
                    (assessment_id,),
                ).fetchone()
                if attempt_exists:
                    raise AssessmentError("This assessment has already been submitted.")

                score = 0
                result_rows = []
                for question in question_rows:
                    selected = submitted[question["id"]]
                    options = json.loads(question["options_json"])
                    if selected < 0 or selected >= len(options):
                        raise AssessmentError("A selected option is outside the available choices.")
                    correct = selected == question["correct_option"]
                    score += int(correct)
                    result_rows.append((question, options, selected, correct))
                attempt_cursor = connection.execute(
                    """INSERT INTO assessment_attempts
                       (assessment_id, submitted_answers_json, score, total_questions)
                       VALUES (?, ?, ?, ?)""",
                    (assessment_id, json.dumps(submitted, sort_keys=True), score, len(question_rows)),
                )
                attempt_id = attempt_cursor.lastrowid
                connection.executemany(
                    """INSERT INTO assessment_responses
                       (attempt_id, question_id, selected_option, is_correct)
                       VALUES (?, ?, ?, ?)""",
                    [(attempt_id, question["id"], selected, int(correct))
                     for question, _, selected, correct in result_rows],
                )
                completed_at = connection.execute(
                    "SELECT completed_at FROM assessment_attempts WHERE id = ?", (attempt_id,)
                ).fetchone()[0]
                results = []
                for question, options, selected, correct in result_rows:
                    locator = {
                        "source_file": question["source_file"],
                        "location_type": question["location_type"],
                        "location_value": question["location_value"],
                    }
                    citation = self._citation_text(locator)
                    results.append({
                        "question_id": question["id"],
                        "question_text": question["question_text"],
                        "topic": question["topic"],
                        "difficulty": question["difficulty"],
                        "selected_option": selected,
                        "selected_answer": options[selected],
                        "correct_option": question["correct_option"],
                        "correct_answer": options[question["correct_option"]],
                        "is_correct": correct,
                        "explanation": question["explanation"],
                        "citation": citation,
                        "source_file": question["source_file"],
                        "location_type": question["location_type"],
                        "location_value": question["location_value"],
                        "source_material_id": question["source_material_id"],
                    })
            return {
                "attempt_id": attempt_id,
                "assessment_id": assessment_id,
                "score": score,
                "total_questions": len(question_rows),
                "percentage": round(score * 100 / len(question_rows), 2),
                "completed_at": completed_at,
                "results": results,
            }
        except sqlite3.IntegrityError as exc:
            if "assessment_attempts.assessment_id" in str(exc):
                raise AssessmentError("This assessment has already been submitted.") from exc
            raise AssessmentError("Could not save assessment responses.") from exc
        finally:
            connection.close()

    @staticmethod
    def _citation_text(source):
        source_type = source["location_type"]
        value = source["location_value"]
        label = "page" if source_type == "page" else source_type
        return f"[{source['source_file']}, {label} {value}]"
