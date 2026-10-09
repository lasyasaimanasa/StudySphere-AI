from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from config import Config
from database import init_db
from services import MaterialService
import os

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = 'dev-secret-key-change-in-production'

# Initialize database
with app.app_context():
    init_db()

# Initialize material service
material_service = MaterialService(app.config['UPLOAD_FOLDER'])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


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
        file_path, error = material_service.save_uploaded_file(file, user_id)

        if error:
            flash(error, 'error')
            return redirect(url_for('materials'))

        # Process PDF
        material_id, process_error = material_service.process_pdf(
            file_path, file.filename, user_id
        )

        if process_error:
            flash(process_error, 'error')
            # Clean up file on error
            if os.path.exists(file_path):
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


@app.route('/tutor')
def tutor():
    return render_template('tutor.html')


@app.route('/quiz')
def quiz():
    return render_template('quiz.html')


@app.route('/progress')
def progress():
    return render_template('progress.html')


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


if __name__ == '__main__':
    app.run(debug=True)
