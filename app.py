import os
import json
import time
import queue
import threading
import random
import fitz 
from captum.attr import IntegratedGradients
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from pdf_processor import process_pdf, get_pdf_metadata
from rag_system import process_texts_with_rag
from checklist_generator import generate_checklist
from config import logger, MAX_QUEUE_SIZE, DOCUMENT_PROCESSING_DELAY, API_USAGE_LOG_PATH, INITIAL_BACKOFF, MAX_BACKOFF, BACKOFF_FACTOR
import openai
from openpyxl import Workbook
import io
import csv
import atexit
from transformers import AutoTokenizer
import requests

app = Flask(__name__)
app.config.from_object('config')
app.config['SECRET_KEY'] = os.urandom(24)  # Set a secret key for session management

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

# In-memory user storage (replace with database in production)
users = {}

@login_manager.user_loader
def load_user(user_id):
    return users.get(int(user_id))

processing_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
documents = {}
api_semaphore = threading.Semaphore(2)
api_queue = queue.Queue()

MAX_RETRIES = 5

def exponential_backoff(attempt):
    backoff = min(INITIAL_BACKOFF * (BACKOFF_FACTOR ** attempt), MAX_BACKOFF)
    jitter = random.uniform(0, backoff * 0.1)
    return backoff + jitter

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify(error="An unexpected error occurred. Please try again later."), 500

@app.route('/', methods=['GET'])
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = next((user for user in users.values() if user.username == username), None)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password:
            if any(user.username == username for user in users.values()):
                flash('Username already exists')
            else:
                user_id = len(users) + 1
                users[user_id] = User(user_id, username, generate_password_hash(password))
                flash('Registration successful. Please log in.')
                return redirect(url_for('login'))
        else:
            flash('Username and password are required')
    return render_template('register.html')

@app.route('/documents', methods=['GET'])
@login_required
def list_documents():
    return render_template('documents.html', documents=documents)

@app.route('/document/<filename>', methods=['GET'])
@login_required
def view_document(filename):
    if filename in documents:
        return render_template('document_details.html', document=documents[filename])
    return redirect(url_for('list_documents'))

@app.route('/delete/<filename>', methods=['POST'])
@login_required
def delete_document(filename):
    if filename in documents:
        del documents[filename]
    return redirect(url_for('list_documents'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No file uploaded. Please select a PDF file.'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files. Please choose at least one PDF file.'}), 400
    
    uploaded_files = []
    
    # Create the upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                processed_chunks = process_pdf(file_path)
                metadata = get_pdf_metadata(file_path)
                documents[filename] = {
                    'filename': filename,
                    'path': file_path,
                    'processed_chunks': processed_chunks,
                    'metadata': metadata,
                    'checklist': None,
                    'feedback': [],
                    'error': None,
                    'processing': False,
                    'retry_count': 0
                }
                uploaded_files.append(filename)
                processing_queue.put(filename)
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                return jsonify({'error': f"Error processing {filename}: {str(e)}"}), 500
        else:
            return jsonify({'error': f"Invalid file type for {file.filename}. Please upload PDF files only."}), 400
    
    if not hasattr(app, 'processing_thread') or not app.processing_thread.is_alive():
        app.processing_thread = threading.Thread(target=process_documents_thread)
        app.processing_thread.start()
    
    return jsonify({'message': 'Files uploaded successfully. Processing will begin shortly.', 'filenames': uploaded_files})

def process_documents_thread():
    while True:
        try:
            filename = processing_queue.get(timeout=1)
            process_document(filename)
            time.sleep(DOCUMENT_PROCESSING_DELAY)
        except queue.Empty:
            time.sleep(1)

def process_document(filename):
    document = documents[filename]
    document['processing'] = True
    document['retry_count'] = 0

    while document['retry_count'] < MAX_RETRIES:
        try:
            logger.info(f"Processing document: {filename} (Attempt {document['retry_count'] + 1})")
            processed_chunks = document['processed_chunks']
            rag_results = process_texts_with_rag(processed_chunks)
            checklist = generate_checklist(rag_results)
            if checklist:
                document['checklist'] = checklist
                document['error'] = None
                logger.info(f"Completed processing document: {filename}")
                break
            else:
                raise Exception("Failed to generate checklist")
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI API rate limit exceeded for {filename}: {str(e)}")
            document['error'] = f"API rate limit exceeded: {str(e)}. Retrying..."
            document['retry_count'] += 1
            backoff = exponential_backoff(document['retry_count'])
            logger.info(f"Retrying in {backoff:.2f} seconds")
            time.sleep(backoff)
        except openai.APIError as e:
            logger.error(f"OpenAI API error for {filename}: {str(e)}")
            document['error'] = f"OpenAI API error: {str(e)}. Please try again later."
            break
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            document['error'] = f"Error processing document: {str(e)}"
            break

    if document['retry_count'] == MAX_RETRIES:
        document['error'] = f"Max retries reached. Please try again later."

    document['processing'] = False

@app.route('/process', methods=['GET'])
@login_required
def process_documents():
    completed = sum(1 for doc in documents.values() if doc['checklist'] or doc['error'])
    total = len(documents)
    
    if completed == total:
        return jsonify({'message': 'Document processing completed'})
    else:
        return jsonify({'message': f'Document processing in progress ({completed}/{total} completed)'})

@app.route('/get_checklists', methods=['GET'])
@login_required
def get_checklists():
    checklists = {}
    errors = {}
    processing = {}
    for filename, doc in documents.items():
        if doc['checklist']:
            checklists[filename] = doc['checklist']
        elif doc['error']:
            errors[filename] = doc['error']
        elif doc['processing']:
            processing[filename] = f"Processing in progress (Attempt {doc['retry_count'] + 1})"
    
    logger.info(f"Returning checklists: {checklists}")
    logger.info(f"Returning errors: {errors}")
    logger.info(f"Documents still processing: {processing}")
    
    return jsonify({
        'checklists': checklists,
        'errors': errors,
        'processing': processing,
        'total_documents': len(documents),
        'completed': len(checklists) + len(errors),
        'in_progress': len(processing)
    })

@app.route('/submit_feedback/<filename>', methods=['POST'])
@login_required
def submit_feedback(filename):
    if filename not in documents:
        return jsonify({'error': 'Document not found'}), 404
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        accuracy = data.get('accuracy')
        comments = data.get('comments')
        
        if accuracy is None:
            return jsonify({'error': 'Accuracy rating is required'}), 400
        
        feedback = {
            'accuracy': int(accuracy),
            'comments': comments
        }
        
        documents[filename]['feedback'].append(feedback)
        
        return jsonify({'message': 'Feedback submitted successfully'})
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in submit_feedback: {str(e)}")
        return jsonify({'error': 'Invalid JSON data provided'}), 400
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({'error': f"Error submitting feedback: {str(e)}"}), 500

@app.route('/retry/<filename>', methods=['POST'])
@login_required
def retry_processing(filename):
    if filename not in documents:
        return jsonify({'error': 'Document not found'}), 404
    
    documents[filename]['error'] = None
    documents[filename]['processing'] = False
    documents[filename]['retry_count'] = 0
    processing_queue.put(filename)
    
    return jsonify({'message': f'Retrying processing for {filename}'})

@app.route('/api_usage', methods=['GET'])
@login_required
def api_usage():
    try:
        with open(API_USAGE_LOG_PATH, 'r') as f:
            usage_data = f.readlines()
        
        usage_summary = {}
        for line in usage_data:
            timestamp, api_name = line.strip().split(',')
            if api_name not in usage_summary:
                usage_summary[api_name] = 0
            usage_summary[api_name] += 1
        
        return jsonify(usage_summary)
    except Exception as e:
        logger.error(f"Error retrieving API usage data: {str(e)}")
        return jsonify({'error': 'Unable to retrieve API usage data'}), 500

@app.route('/explain/<filename>/<int:item_index>', methods=['GET'])
@login_required
def explain_checklist_item(filename, item_index):
    if filename not in documents or not documents[filename]['checklist']:
        return jsonify({'error': 'Document or checklist not found'}), 404
    
    try:
        checklist = documents[filename]['checklist']
        if item_index < 0 or item_index >= len(checklist):
            return jsonify({'error': 'Invalid checklist item index'}), 400
        
        item, score = checklist[item_index]
        original_text = "\n".join([chunk for chunk, _ in documents[filename]['processed_chunks']])
        #model = load_model()  # Define the model before using it
        #integrated_gradients = IntegratedGradients(model)
        #attributions = integrated_gradients.attribute(original_text, item)
        
        nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        tokens = nli_tokenizer.convert_ids_to_tokens(nli_tokenizer.encode(item))
        #explanation = [{'token': token, 'attribution': float(attr)} for token, attr in zip(tokens, attributions)]
        
        return jsonify({
            'item': item,
            'score': score,
            #'explanation': explanation
        })
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return jsonify({'error': 'Unable to generate explanation'}), 500

@app.route('/export/<filename>/<format>', methods=['GET'])
@login_required
def export_checklist(filename, format):
    if filename not in documents or not documents[filename]['checklist']:
        return jsonify({'error': 'Document or checklist not found'}), 404

    checklist = documents[filename]['checklist']

    if format == 'csv':
        return export_csv(filename, checklist)
    elif format == 'excel':
        return export_excel(filename, checklist)
    else:
        return jsonify({'error': 'Invalid export format'}), 400

def export_csv(filename, checklist):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Item', 'Score'])
    for item, score in checklist:
        writer.writerow([item, score])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'{filename}_checklist.csv'
    )

def export_excel(filename, checklist):
    wb = Workbook()
    ws = wb.active
    ws.title = "Checklist"
    ws.append(['Item', 'Score'])
    for item, score in checklist:
        ws.append([item, score])
    
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'{filename}_checklist.xlsx'
    )

def close_running_threads():
    for thread in threading.enumerate():
        if thread.isDaemon() and thread.is_alive():
            thread.join(timeout=1.0)

atexit.register(close_running_threads)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)