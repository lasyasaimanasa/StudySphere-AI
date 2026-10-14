from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from config import Config
from database import init_db
from services import MaterialService, EmbeddingService
from rag import RAGService
from rag.prompts import get_source_location
from services.assessment_service import AssessmentError, AssessmentService
from services.progress_service import ProgressService
from services.youtube_lecture_service import YouTubeLectureError, YouTubeLectureService
import os
import time

app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
with app.app_context():
    init_db()

# Shared embedding service instance (lazily initialized on first use)
embedding_service = None

def get_embedding_service():
    """Return the shared EmbeddingService instance, lazily initializing on first use."""
    global embedding_service
    if embedding_service is None:
        started_at = time.perf_counter()
        app.logger.info("Embedding service initialization started")
        try:
            embedding_service = EmbeddingService()
        except Exception:
            app.logger.warning(
                "Embedding service initialization failed after %.2f seconds",
                time.perf_counter() - started_at,
            )
            raise
        app.logger.info(
            "Embedding service initialization completed in %.2f seconds",
            time.perf_counter() - started_at,
        )
    return embedding_service

material_service = MaterialService(app.config['UPLOAD_FOLDER'])


def _source_location_response_fields(source):
    """Serialize typed location metadata while retaining page-based legacy fields."""
    location_type, location_value = get_source_location(source)
    return {
        "location_type": location_type,
        "location_value": None if location_value is None else str(location_value),
    }

# Shared RAG service instance (lazily initialized on first use)
rag_service = None
assessment_service = None
progress_service = None
youtube_lecture_service = None

def get_rag_service():
    """Return the shared RAGService instance, lazily initializing on first use."""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
        rag_service.set_embedding_service(get_embedding_service())
    return rag_service


def get_assessment_service():
    """Return the assessment service without loading models for reads/submits."""
    global assessment_service
    if assessment_service is None:
        assessment_service = AssessmentService()
    return assessment_service


def get_progress_service():
    """Return the lightweight progress aggregator; retrieval is initialized on demand."""
    global progress_service
    if progress_service is None:
        progress_service = ProgressService()
    return progress_service


def get_youtube_lecture_service():
    """Return the lightweight persisted YouTube lecture service."""
    global youtube_lecture_service
    if youtube_lecture_service is None:
        youtube_lecture_service = YouTubeLectureService()
    return youtube_lecture_service


def load_progress_data():
    """Retrieve recommendations only when weak topics make them useful."""
    service = get_progress_service()
    data = service.get_progress(include_recommendations=False)
    if data["weak_topics"] and service.embedding_service is None:
        try:
            embedding = get_embedding_service()
            if embedding.index is not None and embedding.index.ntotal:
                service.embedding_service = embedding
        except Exception:
            app.logger.info("Indexed search is unavailable for progress recommendations")
    return service.get_progress(include_recommendations=True) if data["weak_topics"] else data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', progress=load_progress_data())


@app.route('/materials', methods=['GET', 'POST'])
def materials():
    # Placeholder user_id - will be replaced with authentication in future
    user_id = 1

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('materials'))

        file = request.files['file']

        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('materials'))

        # Save file
        try:
            file_path, error = material_service.save_uploaded_file(file, user_id)
        except Exception:
            app.logger.error("Uploaded material could not be saved")
            flash('We could not save the material. Check available storage and try again.', 'error')
            return redirect(url_for('materials'))

        if error:
            flash(error, 'error')
            return redirect(url_for('materials'))

        # Process the supported material type and incrementally index its chunks.
        material_id, process_error = material_service.process_file(
            file_path, file.filename, user_id
        )

        if process_error:
            flash(process_error, 'error')
            # Clean up file on error only if database material record was not created
            if not material_id and os.path.exists(file_path):
                os.remove(file_path)
        else:
            chunk_count = material_service.get_material_chunks_count(material_id)
            flash(f'Successfully uploaded and processed {file.filename}. '
                  f'Created {chunk_count} chunks.', 'success')

        return redirect(url_for('materials'))

    # GET request - show materials list
    materials_list = material_service.get_user_materials(user_id)

    # Add chunk counts to each material
    for material in materials_list:
        material['chunk_count'] = material_service.get_material_chunks_count(material['id'])

    return render_template('materials.html', materials=materials_list)


@app.route('/materials/<int:material_id>/retry', methods=['POST'])
def retry_material(material_id):
    """Retry semantic indexing for a material."""
    success, error = material_service.retry_indexing(material_id)
    if success:
        flash('Successfully indexed material.', 'success')
    else:
        flash(error or 'Failed to index material.', 'error')
    return redirect(url_for('materials'))


@app.route('/api/search', methods=['POST'])
def api_search():
    """
    Semantic search API endpoint.

    Expects JSON:
    {
        "query": "string (required)",
        "top_k": int (optional, default 5, between 1 and 50),
        "material_id": int (optional)
    }

    Returns JSON with matched study material chunks and metadata.
    """
    if not request.is_json:
        return jsonify({"error": "Request content type must be application/json"}), 400

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a valid JSON object"}), 400

    query = data.get("query")
    if query is None or not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Query parameter is required and cannot be empty"}), 400
    query = query.strip()

    # Validate top_k
    top_k_raw = data.get("top_k", 5)
    if top_k_raw is None:
        top_k = 5
    else:
        try:
            top_k = int(top_k_raw)
            if top_k <= 0 or top_k > 50:
                return jsonify({"error": "top_k must be an integer between 1 and 50"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "top_k must be a valid integer"}), 400

    # Validate material_id
    material_id = data.get("material_id")
    if material_id is not None:
        try:
            material_id = int(material_id)
            if material_id <= 0:
                return jsonify({"error": "material_id must be a positive integer"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "material_id must be a valid integer"}), 400

        # Check if material exists
        conn = material_service._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename FROM materials WHERE id = ?", (material_id,))
        mat = cursor.fetchone()
        conn.close()

        if not mat:
            return jsonify({"error": f"Material with ID {material_id} not found"}), 404

    # Perform search using the shared embedding_service instance
    try:
        service = get_embedding_service()
        if service.index is None or service.index.ntotal == 0:
            return jsonify({
                "query": query,
                "results": [],
                "total_results": 0,
                "material_id": material_id,
                "message": "No indexed study materials are currently available."
            }), 200

        search_results = service.search(
            query=query,
            top_k=top_k,
            material_id=material_id
        )

        formatted_results = []
        for item in search_results:
            formatted_results.append({
                "chunk_id": item.get("chunk_id"),
                "material_id": item.get("material_id"),
                "chunk_index": item.get("chunk_index"),
                "source_file": item.get("source_file"),
                "page_number": item.get("page_number"),
                **_source_location_response_fields(item),
                "content": item.get("content"),
                "similarity_score": round(float(item.get("score", 0.0)), 4),
            })

        return jsonify({
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "material_id": material_id
        }), 200

    except Exception as e:
        return jsonify({"error": f"Search execution failed: {str(e)}"}), 500


@app.route('/api/answer', methods=['POST'])
def api_answer():
    """
    RAG answer generation API endpoint.

    Expects JSON:
    {
        "query": "string (required)",
        "top_k": int (optional, default 5, between 1 and 50),
        "material_id": int (optional)
    }

    Returns JSON with generated answer and source citations.
    """
    if not request.is_json:
        return jsonify({"error": "Request content type must be application/json"}), 400

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a valid JSON object"}), 400

    query = data.get("query")
    if query is None or not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Query parameter is required and cannot be empty"}), 400
    query = query.strip()

    # Validate top_k
    top_k_raw = data.get("top_k", 5)
    if top_k_raw is None:
        top_k = 5
    else:
        try:
            top_k = int(top_k_raw)
            if top_k <= 0 or top_k > 50:
                return jsonify({"error": "top_k must be an integer between 1 and 50"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "top_k must be a valid integer"}), 400

    # Validate material_id
    material_id = data.get("material_id")
    if material_id is not None:
        try:
            material_id = int(material_id)
            if material_id <= 0:
                return jsonify({"error": "material_id must be a positive integer"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "material_id must be a valid integer"}), 400

        # Check if material exists
        conn = material_service._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename FROM materials WHERE id = ?", (material_id,))
        mat = cursor.fetchone()
        conn.close()

        if not mat:
            return jsonify({"error": f"Material with ID {material_id} not found"}), 404

    # Generate answer using RAG service
    try:
        service = get_rag_service()

        result = service.generate_answer(
            query=query,
            top_k=top_k,
            material_id=material_id
        )

        retrieved_chunks = result.get("sources", []) or []
        formatted_results = [
            {
                "chunk_id": s.get("chunk_id"),
                "material_id": s.get("material_id"),
                "chunk_index": s.get("chunk_index"),
                "source_file": s.get("source_file"),
                "page_number": s.get("page_number"),
                **_source_location_response_fields(s),
                "content": s.get("content"),
                "similarity_score": round(float(s.get("score", 0.0)), 4),
            }
            for s in retrieved_chunks
        ]

        if result.get("error"):
            return jsonify({
                "query": query,
                "answer": None,
                "citations": [],
                "sources": [],
                "insufficient_context": False,
                "error": "Answer generation is unavailable right now. Please try again later.",
                "results": formatted_results,
            }), 500

        return jsonify({
            "query": query,
            "answer": result["answer"],
            "citations": result["citations"],
            "sources": [
                {
                    "source_file": s.get("source_file"),
                    **({"video_title": s["video_title"]} if s.get("video_title") else {}),
                    "page_number": s.get("page_number"),
                    **_source_location_response_fields(s),
                    "similarity_score": round(float(s.get("score", 0.0)), 4)
                }
                for s in result.get("sources", [])
            ],
            "insufficient_context": result.get("insufficient_context", False),
            "material_id": material_id,
            "results": formatted_results,
        }), 200

    except Exception:
        return jsonify({
            "error": "Answer generation failed due to an internal error.",
            "results": [],
        }), 500


@app.route('/tutor')
def tutor():
    user_id = 1
    materials_list = material_service.get_user_materials(user_id)
    lectures = get_youtube_lecture_service().list_lectures()
    return render_template('tutor.html', materials=materials_list, youtube_lectures=lectures)


@app.route('/api/youtube-lectures', methods=['GET', 'POST'])
def api_youtube_lectures():
    service = get_youtube_lecture_service()
    if request.method == 'GET':
        try:
            return jsonify({'lectures': service.list_lectures()}), 200
        except Exception:
            app.logger.exception("Saved YouTube lectures could not be loaded")
            return jsonify({'error': 'Saved lectures could not be loaded. Please try again.'}), 500

    if not request.is_json:
        return jsonify({'error': 'Request content type must be application/json'}), 400
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a valid JSON object'}), 400
    if not set(data).issubset({'url', 'transcript_material_id', 'display_title'}):
        return jsonify({'error': 'Request contains unsupported fields'}), 400

    material_id = data.get('transcript_material_id')
    if isinstance(material_id, str) and material_id.isdigit():
        material_id = int(material_id)
    if isinstance(material_id, bool) or not isinstance(material_id, int):
        return jsonify({'error': 'transcript_material_id must be a positive integer'}), 400
    try:
        lecture = service.save_lecture(
            data.get('url'), material_id, data.get('display_title')
        )
    except YouTubeLectureError as exc:
        return jsonify({'error': str(exc)}), exc.status_code
    except Exception:
        app.logger.exception("YouTube lecture could not be saved")
        return jsonify({'error': 'Lecture could not be saved. Please try again.'}), 500
    return jsonify({'lecture': lecture}), 201


@app.route('/quiz')
def quiz():
    return render_template('quiz_unavailable.html')


@app.route('/api/assessments', methods=['POST'])
def api_create_assessment():
    if not request.is_json:
        return jsonify({"error": "Request content type must be application/json"}), 400
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a valid JSON object"}), 400
    topic = data.get("topic")
    question_count = data.get("question_count")
    difficulty = data.get("difficulty")
    material_id = data.get("material_id")
    if material_id is not None and isinstance(material_id, str) and material_id.isdigit():
        material_id = int(material_id)
    if material_id is not None and (
        isinstance(material_id, bool) or not isinstance(material_id, int) or material_id <= 0
    ):
        return jsonify({"error": "material_id must be a positive integer"}), 400
    try:
        service = get_assessment_service()
        if service.embedding_service is None:
            service.embedding_service = get_embedding_service()
        result = service.create_assessment(
            topic, question_count, difficulty, material_id=material_id
        )
    except AssessmentError as exc:
        status = 409 if "already exists" in str(exc).lower() else 422
        return jsonify({"error": str(exc)}), status
    except Exception:
        app.logger.exception("Assessment creation failed")
        return jsonify({"error": "Assessment generation failed. Please try again."}), 500
    return jsonify({"assessment": result}), 201


@app.route('/api/assessments/<int:assessment_id>', methods=['GET'])
def api_get_assessment(assessment_id):
    assessment = get_assessment_service().get_assessment(assessment_id)
    if assessment is None:
        return jsonify({"error": "Assessment not found"}), 404
    return jsonify({"assessment": assessment}), 200


@app.route('/api/assessments/<int:assessment_id>/submit', methods=['POST'])
def api_submit_assessment(assessment_id):
    if not request.is_json:
        return jsonify({"error": "Request content type must be application/json"}), 400
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a valid JSON object"}), 400
    if set(data) != {"answers"}:
        return jsonify({"error": "Request must contain only an answers list"}), 400
    try:
        result = get_assessment_service().submit_answers(
            assessment_id, data.get("answers")
        )
    except AssessmentError as exc:
        message = str(exc)
        status = 409 if "already been submitted" in message else (
            404 if message == "Assessment not found." else 400
        )
        return jsonify({"error": message}), status
    except Exception:
        app.logger.exception("Assessment submission failed")
        return jsonify({"error": "Assessment submission failed. Please try again."}), 500
    return jsonify(result), 200


@app.route('/progress')
def progress():
    return render_template('progress.html', progress=load_progress_data())


@app.route('/api/progress', methods=['GET'])
def api_progress():
    try:
        return jsonify(load_progress_data()), 200
    except Exception:
        app.logger.exception("Progress aggregation failed")
        return jsonify({"error": "Progress could not be loaded. Please try again."}), 500


# Debug endpoint for testing (development only)
@app.route('/debug/material/<int:material_id>')
def debug_material(material_id):
    """Debug endpoint to inspect a material and its chunks."""
    material = material_service.get_material_with_chunks(material_id)

    if not material:
        return jsonify({'error': 'Material not found'}), 404

    # Return sample chunks (first 3 and last 3)
    all_chunks = material['chunks']
    sample_chunks = []

    if len(all_chunks) > 6:
        sample_chunks = all_chunks[:3] + all_chunks[-3:]
    else:
        sample_chunks = all_chunks

    return jsonify({
        'material': {
            'id': material['id'],
            'filename': material['filename'],
            'file_type': material['file_type'],
            'page_count': material['page_count'],
            'processing_status': material['processing_status'],
            'total_chunks': len(all_chunks)
        },
        'sample_chunks': sample_chunks
    })


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(413)
def request_entity_too_large(e):
    limit_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    message = f'The uploaded file is too large. The maximum upload size is {limit_mb} MB.'
    if request.endpoint == 'materials':
        flash(message, 'error')
        return redirect(url_for('materials'))
    return jsonify({'error': message}), 413


if __name__ == '__main__':
    app.run(debug=True)
