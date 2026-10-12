"""Offline integration tests for source-grounded assessment generation and scoring."""

import json
import os
import sqlite3
import tempfile
import unittest
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
            "question_evidence": {
                "source_id": source_id,
                "quote": "A DBMS stores, manages, and retrieves data.",
            },
            "options": [
                "A DBMS stores data.",
                "A DBMS is an operating system.",
                "A DBMS is a compiler.",
                "A DBMS is a spreadsheet application.",
            ],
            "option_evidence": [
                {"source_id": source_id, "quote": "A DBMS stores, manages, and retrieves data."},
                {"source_id": source_id, "quote": "A DBMS is not an operating system"},
                {"source_id": source_id, "quote": "not an operating system, compiler"},
                {"source_id": source_id, "quote": "compiler, or spreadsheet application."},
            ],
            "correct_option": 0,
            "answer_evidence": {
                "source_id": source_id,
                "quote": "A DBMS stores, manages, and retrieves data.",
            },
            "explanation": "The retrieved passage directly supports the correct statement.",
            "explanation_evidence": {
                "source_id": source_id,
                "quote": "A DBMS stores, manages, and retrieves data.",
            },
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
        self.assertIn("each of the four options", self.provider.calls[0][0])

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
        invalid_source["questions"][0]["answer_evidence"]["source_id"] = "invented"
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
        cases = (
            ("question_evidence", "question"),
            ("answer_evidence", "correct answer"),
            ("explanation_evidence", "explanation"),
        )
        for field, expected_error in cases:
            output = generated_questions()
            output["questions"][0][field]["quote"] = "fabricated unsupported evidence"
            with self.subTest(field=field), self.assertRaisesRegex(AssessmentError, expected_error):
                AssessmentService(
                    self.database_path, self.embedding,
                    provider_factory=lambda output=output: MockProvider(json.dumps(output)),
                ).create_assessment("topic", 5, "easy")

    def test_rejects_unsupported_answer_option_and_invalid_option_citation(self):
        unsupported = generated_questions()
        unsupported["questions"][0]["option_evidence"][2]["quote"] = "made up quote"
        for generated, message in (
            (unsupported, "answer option evidence quote"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(AssessmentError, message):
                AssessmentService(
                    self.database_path, self.embedding,
                    provider_factory=lambda generated=generated: MockProvider(json.dumps(generated)),
                ).create_assessment("topic", 5, "easy")

        bad_source_id = generated_questions()
        bad_source_id["questions"][0]["option_evidence"][0]["source_id"] = "S99"
        with self.assertRaisesRegex(AssessmentError, "unsupported source"):
            AssessmentService(
                self.database_path, self.embedding,
                provider_factory=lambda: MockProvider(json.dumps(bad_source_id)),
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

    def test_quiz_page_renders_configuration_for_supported_materials(self):
        materials = [{
            "id": 1, "filename": "dbms.pdf", "file_type": "pdf",
            "page_count": 8, "processing_status": "processed",
        }, {
            "id": 2, "filename": "pending.pdf", "file_type": "pdf",
            "page_count": 2, "processing_status": "processing",
        }]
        with patch.object(app_module.material_service, "get_user_materials", return_value=materials):
            response = self.client.get("/quiz")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="assessmentForm"', response.data)
        self.assertIn(b'dbms.pdf', response.data)
        self.assertNotIn(b'pending.pdf', response.data)
        self.assertIn(b'Number of questions (5', response.data)


if __name__ == "__main__":
    unittest.main()
