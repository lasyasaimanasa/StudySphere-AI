"""Offline integration tests for source-grounded assessment generation and scoring."""

import json
import os
import sqlite3
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

import database

with patch.object(database, "init_db"):
    import app as app_module

from app import app
from services.assessment_service import AssessmentError, AssessmentService


class MockEmbeddingService:
    def __init__(self, passages=None):
        self.passages = passages or [{
            "chunk_id": 91,
            "material_id": 1,
            "chunk_index": 0,
            "source_type": "textbook",
            "source_file": "dbms.pdf",
            "page_number": 4,
            "location_type": "page",
            "location_value": "4",
            "content": (
                "A DBMS stores, manages, and retrieves data. "
                "A DBMS is not an operating system, compiler, or spreadsheet application."
            ),
            "score": 0.93,
        }]
        self.calls = []

    def search(self, query, top_k=5, material_id=None):
        self.calls.append((query, top_k, material_id))
        return self.passages


class MockProvider:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def generate(self, prompt, system_prompt=None):
        self.calls.append((prompt, system_prompt))
        return self.output


def generated_questions(count=5, source_id="S1"):
    return {
        "questions": [{
            "question": f"Which task does a DBMS perform, item {index}?",
            "options": [
                "A DBMS stores data.",
                "A DBMS is an operating system.",
                "A DBMS is a compiler.",
                "A DBMS is a spreadsheet application.",
            ],
            "correct_option": 0,
            "evidence": {
                "source_id": source_id,
                "quote": "A DBMS stores, manages, and retrieves data.",
            },
            "explanation": "The retrieved passage directly supports the correct statement.",
        } for index in range(count)]
    }


class AssessmentTestBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, "assessments.sqlite")
        with patch.object(database.Config, "DATABASE_PATH", self.database_path):
            database.init_db()
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            "INSERT INTO materials (id, user_id, filename, file_type, file_path, processing_status) "
            "VALUES (1, 1, 'dbms.pdf', 'pdf', 'dbms.pdf', 'processed')"
        )
        connection.commit()
        connection.close()
        self.embedding = MockEmbeddingService()
        self.provider = MockProvider(json.dumps(generated_questions()))
        self.service = AssessmentService(
            database_path=self.database_path,
            embedding_service=self.embedding,
            provider_factory=lambda: self.provider,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_assessment(self, service=None):
        return (service or self.service).create_assessment(
            "database management", 5, "medium"
        )

    def passage_for_page(self, page_number, source_file="dbms.pdf"):
        template = self.embedding.passages[0]
        return {
            **template,
            "chunk_id": 100 + page_number,
            "source_file": source_file,
            "page_number": page_number,
            "location_value": str(page_number),
            "content": f"{template['content']} Supporting detail for page {page_number}.",
        }

    def answers_for(self, assessment, choices=None):
        choices = choices or {}
        return [{
            "question_id": question["id"],
            "selected_option": choices.get(index, 0),
        } for index, question in enumerate(assessment["questions"])]


class TestAssessmentGenerationAndScoring(AssessmentTestBase):
    def test_grounded_questions_persist_source_and_hide_answers_before_submission(self):
        assessment = self.create_assessment()

        self.assertEqual(assessment["topic"], "database management")
        self.assertEqual(assessment["difficulty"], "medium")
        self.assertEqual(len(assessment["questions"]), 5)
        self.assertNotIn("correct_option", assessment["questions"][0])
        self.assertNotIn("explanation", assessment["questions"][0])
        self.assertEqual(self.embedding.calls[0][0], "database management")
        self.assertIn("[S1] dbms.pdf, page 4", self.provider.calls[0][0])
        self.assertIn("four distinct options", self.provider.calls[0][0])
        self.assertIn("exactly two fields: source_id and quote", self.provider.calls[0][0])
        self.assertIn("evidence must be a JSON object/dictionary", self.provider.calls[0][0])
        self.assertIn("evidence must NEVER be an array/list", self.provider.calls[0][0])
        self.assertNotIn('"answer_evidence"', self.provider.calls[0][0])
        self.assertIn('"evidence": {"source_id": "S1", "quote": "Exact supporting passage quote"}', self.provider.calls[0][0])
        self.assertNotIn('"question_evidence"', self.provider.calls[0][0])
        self.assertNotIn('"option_evidence"', self.provider.calls[0][0])
        self.assertNotIn('"explanation_evidence"', self.provider.calls[0][0])

        connection = sqlite3.connect(self.database_path)
        row = connection.execute(
            "SELECT topic, difficulty, source_material_id, source_file, location_type, location_value "
            "FROM assessment_questions WHERE assessment_id = ? ORDER BY question_order LIMIT 1",
            (assessment["id"],),
        ).fetchone()
        sources = connection.execute(
            "SELECT material_id FROM assessment_sources WHERE assessment_id = ?",
            (assessment["id"],),
        ).fetchall()
        connection.close()
        self.assertEqual(row, ("database management", "medium", 1, "dbms.pdf", "page", "4"))
        self.assertEqual(sources, [(1,)])

    def test_selected_material_scope_is_passed_to_retrieval(self):
        self.service.create_assessment(
            "database management", 5, "medium", material_id=1
        )
        self.assertEqual(self.embedding.calls[0][2], 1)

    def test_prompt_uses_first_five_valid_ranked_sources_with_sequential_ids(self):
        self.embedding.passages = [
            self.passage_for_page(1),
            self.passage_for_page(99, source_file="missing.pdf"),
            self.passage_for_page(2),
            self.passage_for_page(3),
            self.passage_for_page(4),
            self.passage_for_page(5),
            self.passage_for_page(6),
        ]

        self.create_assessment()

        prompt = self.provider.calls[0][0]
        self.assertEqual(self.embedding.calls[0], ("database management", 15, None))
        for source_id, page_number in enumerate(range(1, 6), start=1):
            self.assertIn(
                f"[S{source_id}] dbms.pdf, page {page_number}:",
                prompt,
            )
        self.assertNotIn("missing.pdf", prompt)
        self.assertNotIn("[S6] dbms.pdf, page 6:", prompt)

    def test_evidence_cannot_cite_a_source_excluded_from_the_prompt(self):
        self.embedding.passages = [self.passage_for_page(page) for page in range(1, 7)]
        excluded_source_output = json.dumps(generated_questions(source_id="S6"))
        service = AssessmentService(
            self.database_path,
            self.embedding,
            provider_factory=lambda: MockProvider(excluded_source_output),
        )

        with self.assertRaisesRegex(AssessmentError, "unsupported source passage"):
            service.create_assessment("database management", 5, "medium")

        connection = sqlite3.connect(self.database_path)
        assessment_count = connection.execute(
            "SELECT COUNT(*) FROM assessments"
        ).fetchone()[0]
        connection.close()
        self.assertEqual(assessment_count, 0)

    def test_answer_evidence_is_derived_from_validated_correct_option(self):
        generated = generated_questions()
        generated_question = generated["questions"][0]
        sources = self.service._validated_sources(self.embedding.passages)

        validated = self.service._validate_generated_questions(
            json.dumps(generated), 5, sources, "database management", "medium"
        )

        self.assertEqual(
            validated[0]["answer_evidence"],
            generated_question["evidence"],
        )
        self.assertEqual(validated[0]["source"]["source_id"], "S1")

    def test_rejects_invalid_generation_and_unsupported_source_reference(self):
        for output in (
            "not json",
            json.dumps({"questions": generated_questions(4)["questions"]}),
        ):
            with self.subTest(output=output), self.assertRaises(AssessmentError):
                AssessmentService(
                    self.database_path, self.embedding,
                    provider_factory=lambda output=output: MockProvider(output),
                ).create_assessment("topic", 5, "easy")

        invalid_source = generated_questions()
        invalid_source["questions"][0]["evidence"]["source_id"] = "invented"
        with self.assertRaisesRegex(AssessmentError, "unsupported source"):
            AssessmentService(
                self.database_path, self.embedding,
                provider_factory=lambda: MockProvider(json.dumps(invalid_source)),
            ).create_assessment("topic", 5, "easy")

        connection = sqlite3.connect(self.database_path)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM assessments").fetchone()[0], 0)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM assessment_questions").fetchone()[0], 0)
        connection.close()

    def test_unrelated_or_unsupported_generation_is_rejected_without_partial_persistence(self):
        unsupported = json.dumps({"questions": []})
        with self.assertRaisesRegex(AssessmentError, "requested question count"):
            AssessmentService(
                self.database_path, self.embedding,
                provider_factory=lambda: MockProvider(unsupported),
            ).create_assessment("black holes", 5, "easy")
        connection = sqlite3.connect(self.database_path)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM assessments").fetchone()[0], 0)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM assessment_questions").fetchone()[0], 0)
        connection.close()

    def test_rejects_fabricated_evidence_for_each_question_component(self):
        unsupported_quote = generated_questions()
        unsupported_quote["questions"][0]["evidence"]["quote"] = (
            "fabricated unsupported evidence"
        )
        with self.assertRaisesRegex(AssessmentError, "evidence quote was not found"):
            AssessmentService(
                self.database_path, self.embedding,
                provider_factory=lambda: MockProvider(json.dumps(unsupported_quote)),
            ).create_assessment("topic", 5, "easy")

        bad_source_id = generated_questions()
        bad_source_id["questions"][0]["evidence"]["source_id"] = "S99"
        with self.assertRaisesRegex(AssessmentError, "unsupported source"):
            AssessmentService(
                self.database_path, self.embedding,
                provider_factory=lambda: MockProvider(json.dumps(bad_source_id)),
            ).create_assessment("topic", 5, "easy")

    def test_accepts_canonically_equivalent_unicode_evidence(self):
        sources_by_id = {
            "S1": {"source_id": "S1", "content": "The cafe\u0301 serves coffee daily."}
        }
        evidence = {
            "source_id": "S1",
            "quote": "The café serves coffee daily.",
        }

        validated = self.service._validate_evidence(
            evidence, sources_by_id, "question and correct answer"
        )

        self.assertIs(validated, evidence)

    def test_rejects_fabricated_quote_for_synthetic_passage(self):
        sources_by_id = {
            "S1": {"source_id": "S1", "content": "The library opens at nine each morning."}
        }
        evidence = {
            "source_id": "S1",
            "quote": "The library closes at midnight every day.",
        }

        with self.assertRaisesRegex(AssessmentError, "evidence quote was not found"):
            self.service._validate_evidence(
                evidence, sources_by_id, "question and correct answer"
            )

    def test_rejects_quote_from_another_passage_when_attributed_to_source(self):
        sources_by_id = {
            "S1": {"source_id": "S1", "content": "The library opens at nine each morning."},
            "S2": {"source_id": "S2", "content": "The museum displays ancient pottery exhibits."},
        }
        evidence = {
            "source_id": "S1",
            "quote": "The museum displays ancient pottery exhibits.",
        }

        with self.assertRaisesRegex(AssessmentError, "evidence quote was not found"):
            self.service._validate_evidence(
                evidence, sources_by_id, "question and correct answer"
            )

    def test_rejects_legacy_per_field_evidence(self):
        legacy_evidence = generated_questions()
        legacy_evidence["questions"][0]["question_evidence"] = {
            "source_id": "S2",
            "quote": "A DBMS stores, manages, and retrieves data.",
        }

        with self.assertRaisesRegex(AssessmentError, "invalid structure"):
            service = AssessmentService(
                self.database_path,
                self.embedding,
                provider_factory=lambda: MockProvider(json.dumps(legacy_evidence)),
            )
            service.create_assessment("database management", 5, "medium")

    def test_rejects_missing_evidence_and_invalid_correct_option(self):
        missing_evidence = generated_questions()
        del missing_evidence["questions"][0]["evidence"]
        with self.assertRaisesRegex(AssessmentError, "invalid structure"):
            AssessmentService(
                self.database_path, self.embedding,
                provider_factory=lambda: MockProvider(json.dumps(missing_evidence)),
            ).create_assessment("topic", 5, "easy")

        invalid_correct_option = generated_questions()
        invalid_correct_option["questions"][0]["correct_option"] = 4
        with self.assertRaisesRegex(AssessmentError, "valid correct option"):
            AssessmentService(
                self.database_path, self.embedding,
                provider_factory=lambda: MockProvider(json.dumps(invalid_correct_option)),
            ).create_assessment("topic", 5, "easy")

    def test_rejects_evidence_as_list_instead_of_dict(self):
        list_evidence = generated_questions()
        list_evidence["questions"][0]["evidence"] = ["S1", "A DBMS stores, manages, and retrieves data."]
        with self.assertRaisesRegex(AssessmentError, "malformed"):
            AssessmentService(
                self.database_path, self.embedding,
                provider_factory=lambda: MockProvider(json.dumps(list_evidence)),
            ).create_assessment("topic", 5, "easy")

    def test_accepts_correctly_shaped_evidence_dict(self):
        correctly_shaped = generated_questions()
        correctly_shaped["questions"][0]["evidence"] = {
            "source_id": "S1",
            "quote": "A DBMS stores, manages, and retrieves data.",
        }
        assessment = AssessmentService(
            self.database_path, self.embedding,
            provider_factory=lambda: MockProvider(json.dumps(correctly_shaped)),
        ).create_assessment("database management", 5, "medium")
        self.assertEqual(len(assessment["questions"]), 5)

    def test_normalizes_single_item_list_containing_valid_evidence_dict(self):
        list_wrapped = generated_questions()
        list_wrapped["questions"][0]["evidence"] = [{
            "source_id": "S1",
            "quote": "A DBMS stores, manages, and retrieves data.",
        }]
        assessment = AssessmentService(
            self.database_path, self.embedding,
            provider_factory=lambda: MockProvider(json.dumps(list_wrapped)),
        ).create_assessment("database management", 5, "medium")
        self.assertEqual(len(assessment["questions"]), 5)

    def test_rejects_multiple_item_evidence_list(self):
        multi_item_list = generated_questions()
        multi_item_list["questions"][0]["evidence"] = [
            {"source_id": "S1", "quote": "First quote"},
            {"source_id": "S2", "quote": "Second quote"},
        ]
        with self.assertRaisesRegex(AssessmentError, "malformed"):
            AssessmentService(
                self.database_path, self.embedding,
                provider_factory=lambda: MockProvider(json.dumps(multi_item_list)),
            ).create_assessment("topic", 5, "easy")

    def test_rejects_empty_evidence_list(self):
        empty_list = generated_questions()
        empty_list["questions"][0]["evidence"] = []
        with self.assertRaisesRegex(AssessmentError, "malformed"):
            AssessmentService(
                self.database_path, self.embedding,
                provider_factory=lambda: MockProvider(json.dumps(empty_list)),
            ).create_assessment("topic", 5, "easy")

    def test_rejects_list_containing_malformed_dict(self):
        malformed_in_list = generated_questions()
        malformed_in_list["questions"][0]["evidence"] = [{"source_id": "S1"}]
        with self.assertRaisesRegex(AssessmentError, "malformed"):
            AssessmentService(
                self.database_path, self.embedding,
                provider_factory=lambda: MockProvider(json.dumps(malformed_in_list)),
            ).create_assessment("topic", 5, "easy")

    def test_rejects_duplicate_options_and_invalid_request_bounds(self):
        duplicate_options = generated_questions()
        duplicate_options["questions"][0]["options"] = ["same", "same", "other", "last"]
        bad_output = json.dumps(duplicate_options)
        bad_service = AssessmentService(
            self.database_path, self.embedding,
            provider_factory=lambda: MockProvider(bad_output),
        )
        with self.assertRaisesRegex(AssessmentError, "distinct options"):
            bad_service.create_assessment("topic", 5, "easy")
        for count, difficulty in ((4, "easy"), (11, "medium"), (5, "expert"), (True, "easy")):
            with self.subTest(count=count, difficulty=difficulty), self.assertRaises(AssessmentError):
                self.service.create_assessment("topic", count, difficulty)

    def test_rejects_sources_that_do_not_match_a_real_material(self):
        fake = MockEmbeddingService([{
            **MockEmbeddingService().passages[0],
            "source_file": "invented.pdf",
        }])
        service = AssessmentService(
            self.database_path, fake,
            provider_factory=lambda: self.provider,
        )
        with self.assertRaisesRegex(AssessmentError, "No supported study passages"):
            service.create_assessment("topic", 5, "easy")

        wrong_locator = MockEmbeddingService([{
            **MockEmbeddingService().passages[0],
            "location_type": "video",
            "location_value": "00:01:25",
        }])
        unsupported_service = AssessmentService(
            self.database_path, wrong_locator,
            provider_factory=lambda: self.provider,
        )
        with self.assertRaisesRegex(AssessmentError, "No supported study passages"):
            unsupported_service.create_assessment("topic", 5, "easy")

    def test_correct_and_incorrect_answers_are_scored_and_persisted(self):
        assessment = self.create_assessment()
        answers = self.answers_for(assessment, choices={1: 1})

        result = self.service.submit_answers(assessment["id"], answers)

        self.assertEqual(result["score"], 4)
        self.assertEqual(result["total_questions"], 5)
        self.assertEqual(result["percentage"], 80.0)
        self.assertEqual(len(result["results"]), 5)
        self.assertFalse(result["results"][1]["is_correct"])
        self.assertIn("[dbms.pdf, page 4]", result["results"][0]["citation"])
        connection = sqlite3.connect(self.database_path)
        attempt = connection.execute(
            "SELECT score, total_questions, submitted_answers_json FROM assessment_attempts"
        ).fetchone()
        response_count = connection.execute(
            "SELECT COUNT(*) FROM assessment_responses"
        ).fetchone()[0]
        connection.close()
        self.assertEqual(attempt[0:2], (4, 5))
        self.assertEqual(json.loads(attempt[2]), {str(a["question_id"]): a["selected_option"] for a in answers})
        self.assertEqual(response_count, 5)

    def test_transcript_source_citation_uses_timestamp_not_legacy_page_value(self):
        passage = {
            **self.embedding.passages[0],
            "source_file": "lecture_transcript.txt",
            "source_type": "transcript",
            "page_number": 1,
            "location_type": "timestamp",
            "location_value": "00:01:25",
        }
        connection = sqlite3.connect(self.database_path)
        connection.execute("UPDATE materials SET filename = ?, file_type = ? WHERE id = 1",
                           ("lecture_transcript.txt", "transcript"))
        connection.commit()
        connection.close()
        self.embedding.passages = [passage]
        assessment = self.create_assessment()
        result = self.service.submit_answers(assessment["id"], self.answers_for(assessment))
        self.assertIn("[lecture_transcript.txt, timestamp 00:01:25]", result["results"][0]["citation"])
        self.assertEqual(result["results"][0]["location_value"], "00:01:25")

    def test_empty_malformed_duplicate_and_cross_assessment_submissions_are_rejected(self):
        first = self.create_assessment()
        second = self.create_assessment()
        with self.assertRaisesRegex(AssessmentError, "every question"):
            self.service.submit_answers(first["id"], [])
        with self.assertRaisesRegex(AssessmentError, "question_id"):
            self.service.submit_answers(first["id"], [{"question_id": 1}])
        valid = self.answers_for(first)
        with self.assertRaisesRegex(AssessmentError, "Duplicate answers"):
            self.service.submit_answers(first["id"], valid + [valid[0]])
        foreign_answers = self.answers_for(first)
        with self.assertRaisesRegex(AssessmentError, "belong to this assessment"):
            self.service.submit_answers(second["id"], foreign_answers)
        self.service.submit_answers(first["id"], valid)
        with self.assertRaisesRegex(AssessmentError, "already been submitted"):
            # A completed assessment cannot be submitted a second time.
            self.service.submit_answers(first["id"], valid)


class TestAssessmentAPI(AssessmentTestBase):
    def setUp(self):
        super().setUp()
        self.original_service = app_module.assessment_service
        app_module.assessment_service = self.service
        app.config["TESTING"] = True
        self.client = app.test_client()

    def tearDown(self):
        app_module.assessment_service = self.original_service
        super().tearDown()

    def test_api_create_fetch_and_submit_preserves_response_contract(self):
        response = self.client.post("/api/assessments", json={
            "topic": "database management",
            "question_count": 5,
            "difficulty": "medium",
        })
        self.assertEqual(response.status_code, 201)
        assessment = response.get_json()["assessment"]
        self.assertEqual(len(assessment["questions"]), 5)
        self.assertNotIn("correct_option", assessment["questions"][0])
        self.assertNotIn("explanation", assessment["questions"][0])

        fetched = self.client.get(f"/api/assessments/{assessment['id']}")
        self.assertEqual(fetched.status_code, 200)
        self.assertNotIn("correct_option", fetched.get_json()["assessment"]["questions"][0])

        answers = [{
            "question_id": question["id"], "selected_option": 0
        } for question in assessment["questions"]]
        submitted = self.client.post(
            f"/api/assessments/{assessment['id']}/submit", json={"answers": answers}
        )
        self.assertEqual(submitted.status_code, 200)
        result = submitted.get_json()
        self.assertEqual(result["total_questions"], 5)
        self.assertEqual(len(result["results"]), 5)
        self.assertIn("correct_answer", result["results"][0])
        self.assertIn("[dbms.pdf, page 4]", result["results"][0]["citation"])

        duplicate = self.client.post(
            f"/api/assessments/{assessment['id']}/submit", json={"answers": answers}
        )
        self.assertEqual(duplicate.status_code, 409)

    def test_api_rejects_malformed_configuration_and_submission(self):
        self.assertEqual(self.client.post("/api/assessments", data="x").status_code, 400)
        bad_count = self.client.post("/api/assessments", json={
            "topic": "topic", "question_count": 2, "difficulty": "easy"
        })
        self.assertEqual(bad_count.status_code, 422)
        bad_submit = self.client.post("/api/assessments/99/submit", json={"answers": []})
        self.assertEqual(bad_submit.status_code, 400)
        self.assertEqual(self.client.get("/api/assessments/99").status_code, 404)

    def test_provider_failure_logs_safely_and_preserves_api_response(self):
        class FailingProvider:
            def generate(self, prompt, system_prompt=None):
                failure = RuntimeError("synthetic sensitive prompt and response")
                timeout = TimeoutError("synthetic sensitive timeout detail")
                http_error = urllib.error.HTTPError(
                    "http://localhost:11434/api/chat",
                    503,
                    "synthetic sensitive response body",
                    {},
                    None,
                )
                failure.__cause__ = timeout
                failure.__context__ = http_error
                timeout.__context__ = failure
                raise failure

        self.service.provider_factory = FailingProvider
        with self.assertLogs("services.assessment_service", level="WARNING") as captured_logs:
            response = self.client.post("/api/assessments", json={
                "topic": "database management",
                "question_count": 5,
                "difficulty": "medium",
            })

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json(), {
            "error": "Question generation is unavailable right now."
        })
        log_output = "\n".join(captured_logs.output)
        self.assertIn("RuntimeError", log_output)
        self.assertIn("TimeoutError", log_output)
        self.assertIn("HTTPError", log_output)
        self.assertIn("HTTP status: 503", log_output)
        self.assertRegex(log_output, r"after \d+\.\d{2} seconds")
        for sensitive_value in (
            "synthetic sensitive prompt and response",
            "synthetic sensitive timeout detail",
            "synthetic sensitive response body",
        ):
            self.assertNotIn(sensitive_value, log_output)

        connection = sqlite3.connect(self.database_path)
        assessment_count = connection.execute(
            "SELECT COUNT(*) FROM assessments"
        ).fetchone()[0]
        connection.close()
        self.assertEqual(assessment_count, 0)

    def test_quiz_page_displays_temporary_unavailable_message(self):
        with patch.object(app_module.material_service, "get_user_materials") as get_materials:
            response = self.client.get("/quiz")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Adaptive Quiz is temporarily unavailable", response.data)
        self.assertIn(b'href="/dashboard"', response.data)
        get_materials.assert_not_called()

    def test_user_facing_pages_hide_adaptive_quiz_entry_points(self):
        for path in ("/", "/dashboard", "/progress"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertNotIn(b"Adaptive Quiz", response.data)
                self.assertNotIn(b'href="/quiz"', response.data)

    def test_sidebar_toggle_exposes_accessible_navigation_state(self):
        response = self.client.get("/dashboard")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="primaryNavigation" inert', response.data)
        self.assertIn(b'aria-label="Open navigation"', response.data)
        self.assertIn(b'aria-expanded="false"', response.data)
        self.assertIn(b'aria-controls="primaryNavigation"', response.data)


if __name__ == "__main__":
    unittest.main()
