"""Offline coverage for computed mastery, recommendations, and progress endpoints."""
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import database
import app as app_module
from services.progress_service import ProgressService


class SearchStub:
    def __init__(self, results=None):
        self.results = results or []
        self.queries = []

    def search(self, query, top_k=5):
        self.queries.append((query, top_k))
        return self.results


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp.name, "progress.sqlite")
        with patch.object(database.Config, "DATABASE_PATH", self.path):
            database.init_db()
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("INSERT INTO users(id,name,email,password_hash) VALUES(1,'x','x@x','x')")
        for mid, filename, filetype in ((1, "book.pdf", "pdf"), (2, "slides.pptx", "pptx"), (3, "lecture.txt", "transcript")):
            self.conn.execute("INSERT INTO materials(id,user_id,filename,file_type,file_path) VALUES(?,1,?,?,?)", (mid, filename, filetype, filename))
        for mid, kind, value in ((1, "page", "12"), (2, "slide", "7"), (3, "timestamp", "00:02:15")):
            self.conn.execute("INSERT INTO document_chunks(id,material_id,chunk_index,content,source_type,source_file,page_number,location_type,location_value) VALUES(?,?,0,?,'text',?,?,?,?)", (mid * 10, mid, f"Useful {kind} material", (("book.pdf","slides.pptx","lecture.txt")[mid-1]), mid, kind, value))
        self.conn.commit()
        self.stub = SearchStub([
            {"material_id": 1, "chunk_id": 10},
            {"material_id": 2, "chunk_id": 20},
            {"material_id": 3, "chunk_id": 30},
            {"material_id": 999, "chunk_id": 999},
        ])
        self.service = ProgressService(self.path, self.stub)
        self.old_progress_service = app_module.progress_service
        self.old_embed = app_module.embedding_service
        self.old_get_embed = app_module.get_embedding_service
        app_module.progress_service = self.service
        app_module.embedding_service = None
        app_module.get_embedding_service = lambda: type("Empty", (), {"index": type("I", (), {"ntotal": 0})()})()
        self.client = app_module.app.test_client()

    def tearDown(self):
        app_module.progress_service = self.old_progress_service
        app_module.embedding_service = self.old_embed
        app_module.get_embedding_service = self.old_get_embed
        self.conn.close()
        self.temp.cleanup()

    def add_attempt(self, topic, outcomes, label=None):
        cur = self.conn.execute("INSERT INTO assessments(topic,difficulty) VALUES(?,'easy')", (label or topic,))
        assessment_id = cur.lastrowid
        cur = self.conn.execute("INSERT INTO assessment_attempts(assessment_id,submitted_answers_json,score,total_questions) VALUES(?,'{}',?,?)", (assessment_id, sum(outcomes), len(outcomes)))
        attempt_id = cur.lastrowid
        for index, correct in enumerate(outcomes):
            cur = self.conn.execute("""INSERT INTO assessment_questions(assessment_id,question_order,question_text,options_json,correct_option,explanation,topic,difficulty,source_material_id,source_file,location_type,location_value) VALUES(?,?,?,'[]',0,'',?,'easy',1,'book.pdf','page','12')""", (assessment_id,index,f"Q{index}",topic))
            self.conn.execute("INSERT INTO assessment_responses(attempt_id,question_id,selected_option,is_correct) VALUES(?,?,0,?)",(attempt_id,cur.lastrowid,int(correct)))
        self.conn.commit()

    def test_empty_progress_api_and_pages(self):
        data = self.client.get("/api/progress").get_json()
        self.assertEqual(data["overall"], {"correct_answers": 0, "answered_questions": 0, "accuracy": None, "assessments_completed": 0})
        self.assertEqual(data["topics"], [])
        self.assertIn(b"No quiz results yet", self.client.get("/dashboard").data)
        self.assertIn(b"Your progress starts with a quiz", self.client.get("/progress").data)

    def test_accuracy_minimum_evidence_repeated_attempts_and_weak_ranking(self):
        self.add_attempt("small sample", [False, False, False, False])
        self.add_attempt("weak", [False, False, True, False, False])
        self.add_attempt("weak", [False, False, False, False, True])
        self.add_attempt("mastered", [True, True, True, True, False])
        data = self.service.get_progress(include_recommendations=False)
        topics = {topic["topic"]: topic for topic in data["topics"]}
        self.assertEqual(topics["small sample"]["mastery_status"], "insufficient_evidence")
        self.assertEqual(topics["small sample"]["accuracy"], 0)
        self.assertEqual(topics["small sample"]["evidence_level"], "insufficient")
        self.assertEqual((topics["weak"]["correct_answers"], topics["weak"]["answered_questions"], topics["weak"]["accuracy"]), (2, 10, 20.0))
        self.assertEqual(topics["weak"]["evidence_level"], "moderate")
        self.assertEqual(topics["weak"]["mastery_status"], "needs_revision")
        self.assertEqual(topics["mastered"]["evidence_level"], "limited")
        self.assertEqual(topics["mastered"]["mastery_status"], "mastered")
        self.assertEqual([item["topic"] for item in data["weak_topics"]], ["weak"])
        self.assertEqual(data["overall"]["assessments_completed"], 4)

    def test_evidence_levels_are_count_based_and_deterministic(self):
        self.add_attempt("low evidence", [True, False, False, False])
        self.add_attempt("moderate evidence", [False] * 10)
        self.add_attempt("strong evidence", [True] * 25)
        data = self.service.get_progress(include_recommendations=False)
        topics = {topic["topic"]: topic for topic in data["topics"]}
        self.assertEqual(topics["low evidence"]["evidence_level"], "insufficient")
        self.assertEqual(topics["moderate evidence"]["evidence_level"], "moderate")
        self.assertEqual(topics["strong evidence"]["evidence_level"], "strong")
        self.assertEqual([item["topic"] for item in data["weak_topics"]], ["moderate evidence"])
        self.assertEqual(data["topics"][0]["topic"], "low evidence")

    def test_progress_pages_display_evidence_level(self):
        self.add_attempt("limited topic", [True] * 5)
        progress_page = self.client.get("/progress").data
        dashboard_page = self.client.get("/dashboard").data
        self.assertIn(b"<th>Evidence</th>", progress_page)
        self.assertIn(b"Limited", progress_page)
        self.assertIn(b"Limited evidence", dashboard_page)
        self.assertIn(b"not statistical confidence", progress_page)
        self.assertIn(b"not statistical confidence", dashboard_page)

    def test_recommendations_use_only_existing_retrieved_source_metadata(self):
        self.add_attempt("weak topic", [False] * 5)
        data = self.service.get_progress()
        recommendations = data["weak_topics"][0]["recommendations"]
        self.assertEqual([(item["source_file"], item["location_type"], item["location_value"]) for item in recommendations], [
            ("book.pdf", "page", "12"), ("slides.pptx", "slide", "7"), ("lecture.txt", "timestamp", "00:02:15")])
        self.assertEqual(self.stub.queries, [("weak topic", 3)])

    def test_recommendations_preserve_search_order_and_drop_invalid_rows(self):
        self.stub.results = [
            {"material_id": 2, "chunk_id": 20},
            {"material_id": 999, "chunk_id": 999},
            {"material_id": 1, "chunk_id": 10},
            {"material_id": 3, "chunk_id": 30},
            {"material_id": 1, "chunk_id": 10},
            {"material_id": 1, "chunk_id": None},
        ]
        self.add_attempt("weak topic", [False] * 5)
        data = self.service.get_progress()
        recommendations = data["weak_topics"][0]["recommendations"]
        self.assertEqual([
            (item["source_file"], item["location_type"], item["location_value"])
            for item in recommendations
        ], [
            ("slides.pptx", "slide", "7"),
            ("book.pdf", "page", "12"),
            ("lecture.txt", "timestamp", "00:02:15"),
        ])
        self.assertEqual(self.stub.queries, [("weak topic", 3)])

    def test_recommendations_preserve_real_source_locations(self):
        self.stub.results = [
            {"material_id": 2, "chunk_id": 20},
            {"material_id": 3, "chunk_id": 30},
            {"material_id": 1, "chunk_id": 10},
        ]
        self.add_attempt("weak topic", [False] * 5)
        data = self.service.get_progress()
        recommendations = data["weak_topics"][0]["recommendations"]
        self.assertEqual(recommendations[0]["location_type"], "slide")
        self.assertEqual(recommendations[0]["location_value"], "7")
        self.assertEqual(recommendations[1]["location_type"], "timestamp")
        self.assertEqual(recommendations[1]["location_value"], "00:02:15")
        self.assertEqual(recommendations[2]["location_type"], "page")
        self.assertEqual(recommendations[2]["location_value"], "12")

    def test_progress_api_contains_topics_and_invalid_methods(self):
        self.add_attempt("weak topic", [False] * 5)
        response = self.client.get("/api/progress")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["weak_topics"][0]["answered_questions"], 5)
        self.assertEqual(self.client.post("/api/progress").status_code, 405)


if __name__ == "__main__":
    unittest.main()
