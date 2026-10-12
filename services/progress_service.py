"""Computed, project-wide learning progress from persisted assessment responses."""

import sqlite3
from typing import Any, Dict, List, Optional

from database import get_db_connection


class ProgressService:
    """Aggregate saved answers and optionally retrieve revision materials."""

    MINIMUM_TOPIC_ANSWERS = 5
    MASTERY_ACCURACY = 80.0

    def __init__(self, database_path: Optional[str] = None, embedding_service=None):
        self.database_path = database_path
        self.embedding_service = embedding_service

    def _connection(self):
        if self.database_path:
            connection = sqlite3.connect(self.database_path)
            connection.row_factory = sqlite3.Row
            return connection
        return get_db_connection()

    def get_progress(self, include_recommendations: bool = True) -> Dict[str, Any]:
        connection = self._connection()
        try:
            attempts = connection.execute(
                """SELECT aa.id, aa.score, aa.total_questions, aa.completed_at,
                          a.id AS assessment_id, a.topic
                   FROM assessment_attempts aa
                   JOIN assessments a ON a.id = aa.assessment_id
                   ORDER BY aa.completed_at DESC, aa.id DESC"""
            ).fetchall()
            aggregates = connection.execute(
                """SELECT q.topic, COUNT(*) AS answered,
                          SUM(r.is_correct) AS correct
                   FROM assessment_responses r
                   JOIN assessment_attempts aa ON aa.id = r.attempt_id
                   JOIN assessment_questions q ON q.id = r.question_id
                   GROUP BY q.topic COLLATE NOCASE
                   ORDER BY q.topic COLLATE NOCASE"""
            ).fetchall()
            topic_rows = []
            weak_topics = []
            for row in aggregates:
                answered, correct = int(row["answered"]), int(row["correct"] or 0)
                accuracy = round(100.0 * correct / answered, 2) if answered else 0.0
                sufficient = answered >= self.MINIMUM_TOPIC_ANSWERS
                status = (
                    "mastered" if sufficient and accuracy >= self.MASTERY_ACCURACY
                    else "needs_revision" if sufficient else "insufficient_evidence"
                )
                topic = {
                    "topic": row["topic"], "correct_answers": correct,
                    "answered_questions": answered, "accuracy": accuracy,
                    "mastery_status": status,
                }
                topic_rows.append(topic)
                if status == "needs_revision":
                    weak_topics.append(topic.copy())
            weak_topics.sort(key=lambda item: (item["accuracy"], -item["answered_questions"], item["topic"].casefold()))
            total_correct = sum(int(row["score"]) for row in attempts)
            total_answered = sum(int(row["total_questions"]) for row in attempts)
            overall = round(total_correct * 100.0 / total_answered, 2) if total_answered else None
        finally:
            connection.close()

        if include_recommendations and self.embedding_service is not None:
            self._add_recommendations(weak_topics)
        elif include_recommendations:
            for topic in weak_topics:
                topic["recommendations"] = []
                topic["recommendation_message"] = "Study material search is currently unavailable."

        return {
            "scope": "project_wide",
            "mastery_rule": {
                "minimum_answered_questions": self.MINIMUM_TOPIC_ANSWERS,
                "mastery_accuracy_percent": self.MASTERY_ACCURACY,
            },
            "overall": {
                "correct_answers": total_correct, "answered_questions": total_answered,
                "accuracy": overall, "assessments_completed": len(attempts),
            },
            "topics": topic_rows,
            "weak_topics": weak_topics,
            "recent_assessments": [dict(row) for row in attempts[:5]],
        }

    def _add_recommendations(self, weak_topics: List[Dict[str, Any]]) -> None:
        for topic in weak_topics:
            try:
                passages = self.embedding_service.search(topic["topic"], top_k=3)
            except Exception:
                passages = []
            recommendations = []
            for item in passages if isinstance(passages, list) else []:
                if not isinstance(item, dict):
                    continue
                material_id = item.get("material_id")
                chunk_id = item.get("chunk_id")
                if isinstance(material_id, bool) or not isinstance(material_id, int):
                    continue
                if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
                    continue
                connection = self._connection()
                try:
                    source = connection.execute(
                        """SELECT dc.content, dc.location_type, dc.location_value,
                                  dc.page_number, m.filename
                           FROM document_chunks dc JOIN materials m ON m.id = dc.material_id
                           WHERE dc.id = ? AND dc.material_id = ?""",
                        (chunk_id, material_id),
                    ).fetchone()
                finally:
                    connection.close()
                if not source:
                    continue
                loc_type = source["location_type"] or "page"
                loc_value = source["location_value"]
                if loc_value is None:
                    loc_value = source["page_number"]
                if loc_value is None:
                    continue
                rec = {
                    "material_id": material_id, "source_file": source["filename"],
                    "location_type": loc_type, "location_value": str(loc_value),
                    "content": source["content"],
                }
                if rec not in recommendations:
                    recommendations.append(rec)
            topic["recommendations"] = recommendations
            topic["recommendation_message"] = (
                "Review these indexed passages." if recommendations
                else "No relevant indexed material was found for this topic."
            )
