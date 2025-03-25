from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os, uuid, logging, json, time
from collections import OrderedDict
import faiss
import numpy as np
import fitz  # PyMuPDF
import re
import threading
import gc
from datetime import datetime, timedelta
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
import base64
from PIL import Image
from io import BytesIO

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS as FAISSVectorStore
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOllama

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
IMAGE_FOLDER = "images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# In-memory session storage
vector_stores = {}  # session_id -> FAISSVectorStore instance
pdf_texts = {}  # session_id -> extracted text
processing_status = {}  # session_id -> status updates

# Add these global variables
MAX_SESSIONS = 10  # Maximum number of sessions to keep in memory
SESSION_TIMEOUT = 60 * 60  # Session timeout in seconds (1 hour)
session_timestamps = {}  # Track when sessions were last accessed

# Check if Gemma 3 4B supports image input
def check_model_capabilities():
    try:
        llm = ChatOllama(model="gemma3:4b")
        model_info = llm.invoke([{"role": "user", "content": "What capabilities do you have? Can you process images?"}])
        logging.info(f"Model capabilities check: {model_info.content[:200]}...")
        return "image" in model_info.content.lower() or "visual" in model_info.content.lower()
    except Exception as e:
        logging.warning(f"Failed to check model capabilities: {e}")
        return False

# Store this result
SUPPORTS_IMAGES = check_model_capabilities()
logging.info(f"Gemma 3 4B image support: {SUPPORTS_IMAGES}")

def extract_text_with_pymupdf(file_path):
    """Extract text from a PDF using PyMuPDF (fitz).
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string and a list of tables
        
    Raises:
        Exception: If there's an error processing the PDF
    """
    text = ""
    tables = []
    doc = None
    
    # First check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")
        
    # Check if it's a valid file
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")
        
    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError(f"PDF file is empty (0 bytes): {file_path}")
    
    logging.info(f"Opening PDF file: {file_path} (size: {file_size} bytes)")
    
    try:
        # Open the document with proper error handling
        doc = fitz.open(file_path)
        if doc.is_closed:
            raise ValueError(f"Failed to open the PDF document: {file_path}")
            
        # Get page count for logging
        page_count = len(doc)
        logging.info(f"Processing PDF with {page_count} pages")
        
        if page_count == 0:
            raise ValueError(f"PDF has no pages: {file_path}")
        
        # Extract text from each page
        for page_num, page in enumerate(doc):
            try:
                # Extract text
                page_text = page.get_text("text")
                
                # Extract tables using the built-in table detection
                try:
                    # Get tables using PyMuPDF's table detection
                    page_tables = page.find_tables()
                    if page_tables and page_tables.tables:
                        for table in page_tables.tables:
                            try:
                                # Convert table to HTML format
                                html_table = "<div class='table-responsive'>\n<table class='styled-table'>\n"
                                
                                # Create a text representation for embedding
                                table_text = f"TABLE FROM PAGE {page_num + 1}:\n"
                                
                                # Add caption with table location
                                html_table += f"<caption>Table from page {page_num + 1}</caption>\n"
                                
                                # Add header row if available
                                if table.header:
                                    html_table += "<tr class='table-header'>\n"
                                    header_text = []
                                    for cell in table.header.cells:
                                        if hasattr(cell, 'text'):
                                            cell_text = cell.text.strip() if cell.text else ""
                                            html_table += f"<th>{cell_text}</th>\n"
                                            header_text.append(cell_text)
                                    html_table += "</tr>\n"
                                    if header_text:
                                        table_text += " | ".join(header_text) + "\n"
                                        table_text += "-" * (sum(len(h) for h in header_text) + 3 * len(header_text)) + "\n"
                                
                                # Add data rows
                                for row_idx, row in enumerate(table.rows):
                                    row_class = "odd-row" if row_idx % 2 == 0 else "even-row"
                                    html_table += f"<tr class='{row_class}'>\n"
                                    row_text = []
                                    for cell in row.cells:
                                        if hasattr(cell, 'text'):
                                            cell_text = cell.text.strip() if cell.text else ""
                                            # Check if cell contains numeric data
                                            is_numeric = re.match(r'^[\d,.%$€£¥]+$', cell_text.replace(' ', ''))
                                            cell_class = "numeric" if is_numeric else ""
                                            html_table += f"<td class='{cell_class}'>{cell_text}</td>\n"
                                            row_text.append(cell_text)
                                    html_table += "</tr>\n"
                                    if row_text:
                                        table_text += " | ".join(row_text) + "\n"
                                
                                html_table += "</table>\n</div>"
                                
                                # Add table to the list
                                tables.append({
                                    "page": page_num + 1,
                                    "html": html_table,
                                    "text": table_text
                                })
                                
                                # Add a placeholder in the main text to indicate table location
                                table_placeholder = f"\n[TABLE {len(tables)} FROM PAGE {page_num + 1}]\n"
                                page_text += table_placeholder
                            except AttributeError as attr_err:
                                logging.warning(f"AttributeError in table processing on page {page_num+1}: {attr_err}")
                                continue
                            except Exception as table_err:
                                logging.warning(f"Error processing table on page {page_num+1}: {table_err}")
                                continue
                except Exception as e:
                    logging.warning(f"Error extracting tables from page {page_num+1}: {e}")
                    # Try alternative table detection method
                    try:
                        # Use text blocks to identify potential tables
                        blocks = page.get_text("dict")["blocks"]
                        table_blocks = [b for b in blocks if b["type"] == 1 and len(b.get("lines", [])) > 2]
                        
                        for i, block in enumerate(table_blocks):
                            if "lines" in block and len(block["lines"]) > 0:
                                # Create a simple text representation of the potential table
                                table_text = f"TABLE FROM PAGE {page_num + 1} (ALTERNATIVE DETECTION):\n"
                                html_table = "<div class='table-responsive'>\n<table class='styled-table'>\n"
                                html_table += f"<caption>Table from page {page_num + 1}</caption>\n"
                                
                                for line_idx, line in enumerate(block["lines"]):
                                    row_text = []
                                    if "spans" in line:
                                        row_class = "table-header" if line_idx == 0 else ("odd-row" if line_idx % 2 == 1 else "even-row")
                                        html_table += f"<tr class='{row_class}'>\n"
                                        
                                        for span in line["spans"]:
                                            if "text" in span:
                                                cell_text = span["text"].strip()
                                                if line_idx == 0:  # Treat first row as header
                                                    html_table += f"<th>{cell_text}</th>\n"
                                                else:
                                                    is_numeric = re.match(r'^[\d,.%$€£¥]+$', cell_text.replace(' ', ''))
                                                    cell_class = "numeric" if is_numeric else ""
                                                    html_table += f"<td class='{cell_class}'>{cell_text}</td>\n"
                                                row_text.append(cell_text)
                                        
                                        html_table += "</tr>\n"
                                        table_text += " | ".join(row_text) + "\n"
                                        if line_idx == 0:  # Add separator after header
                                            table_text += "-" * (sum(len(t) for t in row_text) + 3 * len(row_text)) + "\n"
                                
                                html_table += "</table>\n</div>"
                                
                                # Only add if we have actual content
                                if len(table_text.split('\n')) > 2:
                                    tables.append({
                                        "page": page_num + 1,
                                        "html": html_table,
                                        "text": table_text
                                    })
                                    
                                    # Add a placeholder in the main text
                                    table_placeholder = f"\n[TABLE {len(tables)} FROM PAGE {page_num + 1}]\n"
                                    page_text += table_placeholder
                    except Exception as alt_e:
                        logging.warning(f"Alternative table detection failed on page {page_num+1}: {alt_e}")
                
                text += page_text + "\n"
                logging.debug(f"Extracted {len(page_text)} characters from page {page_num+1}")
            except Exception as e:
                logging.warning(f"Error extracting text from page {page_num+1}: {e}")
                # Continue with other pages even if one fails
                
        # Check if we got any text
        if not text.strip():
            logging.warning(f"No text extracted from PDF: {file_path}")
            
    except Exception as e:
        logging.error(f"Error extracting text with pymupdf: {e}")
        raise Exception(f"Failed to process PDF: {str(e)}")
    finally:
        # Ensure document is closed properly
        if doc and not doc.is_closed:
            try:
                doc.close()
                logging.info(f"Document closed successfully: {file_path}")
            except Exception as e:
                logging.error(f"Error closing document: {e}")
    
    return text, tables


@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Upload a PDF, extract its text using PyMuPDF, split it into chunks,
    and build a FAISS vector store using Ollama embeddings.
    """
    saved_file_paths = []  # Track saved files for cleanup in case of error
    
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No file provided.'}), 400

        session_id = request.form.get('session_id') or str(uuid.uuid4())
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No file provided.'}), 400
            
        # Update session timestamp
        session_timestamps[session_id] = datetime.now()
        
        # Clean up old sessions
        cleanup_old_sessions()
            
        # Validate file types and save them immediately
        file_paths = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                # Clean up any files already saved
                cleanup_files(saved_file_paths)
                return jsonify({'error': f'Invalid file type. Only PDF files are supported. Got: {file.filename}'}), 400
                
            # Create a unique filename to avoid conflicts
            original_filename = file.filename
            unique_filename = f"{uuid.uuid4()}_{original_filename}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            # Save the file to disk immediately
            file.save(file_path)
            saved_file_paths.append(file_path)
            file_paths.append((file_path, original_filename))
            logging.info(f"File saved in upload handler: {original_filename} as {unique_filename}")

        # Initialize processing status for this session
        processing_status[session_id] = {
            'status': 'starting',
            'progress': 0,
            'message': 'Starting document processing...'
        }

        # Start processing in a background thread with file paths instead of file objects
        thread = threading.Thread(target=process_document, args=(session_id, file_paths))
        thread.daemon = True
        thread.start()

        return jsonify({
            'message': 'Document processing started.',
            'session_id': session_id
        })
    except Exception as e:
        # Clean up any saved files in case of error
        cleanup_files(saved_file_paths)
        logging.error(f"Error in upload endpoint: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


def cleanup_files(file_paths):
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Removed temporary file: {file_path}")
        except Exception as e:
            logging.error(f"Error removing file {file_path}: {e}")


def process_document(session_id, file_paths):
    """Process document in background thread with progress updates.
    
    Args:
        session_id: The session ID
        file_paths: List of tuples (file_path, original_filename)
    """
    saved_files = []  # Keep track of saved files
    
    try:
        # Update status to extracting text
        processing_status[session_id] = {
            'status': 'extracting',
            'progress': 10,
            'message': 'Extracting text from PDF...'
        }

        all_text = ""
        all_tables = []
        for file_path, original_filename in file_paths:
            # The file is already saved, just add it to our tracking list
            saved_files.append(file_path)
            
            # Extract text from the saved file
            try:
                logging.info(f"Processing file: {original_filename} at {file_path}")
                extracted_text, extracted_tables = extract_text_with_pymupdf(file_path)
                all_text += extracted_text + "\n\n"
                all_tables.extend(extracted_tables)
                logging.info(f"Successfully extracted text and {len(extracted_tables)} tables from {original_filename}")
            except Exception as e:
                logging.error(f"Error extracting text from {original_filename}: {e}")
                raise Exception(f"Failed to extract text from {original_filename}: {str(e)}")

        if not all_text.strip():
            processing_status[session_id] = {
                'status': 'error',
                'progress': 0,
                'message': 'Failed to extract text from PDF. No text content found.'
            }
            return

        # Store the extracted text and tables
        pdf_texts[session_id] = all_text
        
        # Update status to splitting text
        processing_status[session_id] = {
            'status': 'splitting',
            'progress': 30,
            'message': 'Splitting text into chunks...'
        }

        # Split text into overlapping chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_text(all_text)
        
        # Process tables to create dedicated chunks
        table_chunks = []
        for table in all_tables:
            # Create a dedicated chunk for each table with context
            table_chunk = f"TABLE FROM PAGE {table['page']}:\n{table['text']}"
            
            # Add table chunk with a descriptive prefix
            table_chunks.append(table_chunk)
            
            # For tables with substantial content, create additional chunks with different prefixes
            # to increase the chance of retrieval
            if len(table['text'].split('\n')) > 5:  # Only for larger tables
                table_chunks.append(f"TABULAR DATA FROM PAGE {table['page']}:\n{table['text']}")
                
                # Extract column headers if available
                lines = table['text'].split('\n')
                if len(lines) > 2:
                    headers = lines[0]
                    table_chunks.append(f"TABLE HEADERS FROM PAGE {table['page']}: {headers}")
        
        # Add table chunks to the main chunks list
        chunks.extend(table_chunks)
        
        total_chunks = len(chunks)
        logging.info(f"Created {total_chunks} chunks for session {session_id}, including {len(table_chunks)} table-specific chunks.")

        # Update status to creating embeddings
        processing_status[session_id] = {
            'status': 'embedding',
            'progress': 50,
            'message': f'Creating embeddings for {total_chunks} chunks...'
        }

        # Create FAISS vector store with Ollama embeddings
        embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
        
        # Process chunks in smaller batches (5 at a time) and update progress more frequently
        batch_size = 5
        num_batches = (total_chunks + batch_size - 1) // batch_size  # Ceiling division
        
        all_embeddings = []
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            batch_chunks = chunks[i:batch_end]
            
            # Update progress based on batch processing
            batch_progress = 50 + (i / total_chunks) * 20
            processing_status[session_id] = {
                'status': 'embedding',
                'progress': int(batch_progress),
                'message': f'Processing chunks {i+1}-{batch_end} of {total_chunks}...'
            }
            
            # Process this batch
            batch_embeddings = embedding_model.embed_documents(batch_chunks)
            all_embeddings.extend(batch_embeddings)
            
            # Small delay to make progress updates visible
            time.sleep(0.1)
        
        # Update status to building vector store
        processing_status[session_id] = {
            'status': 'indexing',
            'progress': 80,
            'message': 'Building vector index...'
        }
        
        # Create the vector store
        vector_store = FAISSVectorStore.from_texts(chunks, embedding_model)
        vector_stores[session_id] = vector_store
        logging.info(f"FAISS vector store created for session {session_id}.")

        # Update status to complete
        processing_status[session_id] = {
            'status': 'complete',
            'progress': 100,
            'message': 'Document processing complete.'
        }

    except Exception as e:
        logging.error(f"Error processing document: {e}")
        processing_status[session_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error processing document: {str(e)}'
        }
    finally:
        # Clean up temporary files
        cleanup_files(saved_files)


@app.route('/status/<session_id>', methods=['GET'])
def get_status(session_id):
    """Get the current processing status for a session."""
    # Update session timestamp
    session_timestamps[session_id] = datetime.now()
    
    # Clean up old sessions
    cleanup_old_sessions()
    
    if session_id not in processing_status:
        return jsonify({'error': 'Session not found.'}), 404
    
    return jsonify(processing_status[session_id])


@app.route('/upload-image', methods=['POST'])
def upload_image():
    """Upload an image to be used in the conversation."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided.'}), 400

        session_id = request.form.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session ID provided.'}), 400

        # Update session timestamp
        session_timestamps[session_id] = datetime.now()
        
        # Clean up old sessions
        cleanup_old_sessions()
        
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'Empty image file.'}), 400
            
        # Check file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        file_ext = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Invalid image format. Supported formats: {", ".join(allowed_extensions)}'}), 400
            
        # Create a unique filename
        unique_filename = f"{uuid.uuid4()}_{image_file.filename}"
        image_path = os.path.join(IMAGE_FOLDER, unique_filename)
        
        # Save the image
        image_file.save(image_path)
        logging.info(f"Image saved: {image_path}")
        
        # Return the image path for reference in the conversation
        return jsonify({
            'message': 'Image uploaded successfully.',
            'image_path': image_path
        })
        
    except Exception as e:
        logging.error(f"Error uploading image: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Answer a user question by retrieving context from the FAISS vector store
    and using ChatOllama to generate an HTML-formatted answer.
    """
    data = request.get_json()
    if not data or 'question' not in data or 'session_id' not in data:
        return jsonify({'error': 'Invalid request. Provide both question and session_id.'}), 400

    session_id = data['session_id']
    question = data['question'].strip()
    image_path = data.get('image_path')
    
    # Update session timestamp
    session_timestamps[session_id] = datetime.now()
    
    # Clean up old sessions
    cleanup_old_sessions()
    
    if session_id not in vector_stores:
        return jsonify({'error': 'Session not found. Please upload a PDF first.'}), 400

    # Return a streaming response
    def generate():
        try:
            # Send initial status with the question
            yield json.dumps({
                'status': 'processing',
                'stage': 'retrieving',
                'progress': 10,
                'question': question
            }) + '\n'

            vector_store = vector_stores[session_id]
            retriever = vector_store.as_retriever(search_kwargs={'k': 5})

            # Multi-query retrieval: rephrase the question in multiple ways
            query_prompt = PromptTemplate(
                input_variables=["question"],
                template=(
                    "Rephrase the following question in five different ways to capture various nuances. "
                    "Return each rephrased question on a new line, numbered 1-5.\n\n"
                    "Original question: {question}"
                )
            )
            
            # Update status
            yield json.dumps({
                'status': 'processing',
                'stage': 'rephrasing',
                'progress': 20,
                'question': question
            }) + '\n'
            
            llm = ChatOllama(model="gemma3:4b")
            
            # Get alternative questions
            alt_questions_response = llm.invoke([{"role": "user", "content": query_prompt.format(question=question)}])
            alt_questions_text = alt_questions_response.content
            
            # Parse alternative questions
            alt_questions = []
            for line in alt_questions_text.strip().split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('- ')):
                    # Remove numbering or bullet points
                    cleaned_line = re.sub(r'^[\d\-\.\s]+', '', line).strip()
                    if cleaned_line:
                        alt_questions.append(cleaned_line)
            
            # If we couldn't parse properly, create some basic alternatives
            if len(alt_questions) < 2:
                alt_questions = [
                    question,
                    f"Can you tell me about {question}?",
                    f"What does the document say about {question}?",
                    f"I'd like to know about {question}",
                    f"Please provide information on {question}"
                ]
            
            # Stream each alternative question
            for i, alt_q in enumerate(alt_questions):
                # Calculate progress from 25 to 45 based on question index
                progress = 25 + (i * 5)
                yield json.dumps({
                    'status': 'processing',
                    'stage': 'rephrasing',
                    'progress': progress,
                    'question': question,
                    'alternative_question': alt_q,
                    'alternative_index': i + 1,
                    'total_alternatives': len(alt_questions)
                }) + '\n'
                # Small delay to make the streaming visible
                time.sleep(0.2)
            
            # Update status to searching
            yield json.dumps({
                'status': 'processing',
                'stage': 'searching',
                'progress': 50,
                'question': question
            }) + '\n'
            
            # Check if the question is likely about tabular data
            table_keywords = ['table', 'row', 'column', 'cell', 'data', 'value', 'statistic', 
                             'figure', 'chart', 'percentage', 'total', 'average', 'mean', 
                             'median', 'maximum', 'minimum', 'compare', 'comparison', 'trend']
            
            is_table_query = any(keyword in question.lower() for keyword in table_keywords)
            
            # Adjust retrieval strategy for table queries
            if is_table_query:
                logging.info(f"Detected table-related query: {question}")
                # For table queries, we'll use more documents to ensure we capture the relevant tables
                k_value = 8  # Retrieve more documents for table queries
                
                # Add table-specific queries to alternative questions
                table_specific_queries = [
                    f"table containing information about {question}",
                    f"tabular data related to {question}",
                    f"data in table format about {question}"
                ]
                alt_questions.extend(table_specific_queries)
            else:
                k_value = 5  # Default number of documents to retrieve
            
            # Create multi-query retriever with our alternative questions
            retrieved_docs = []
            for i, alt_q in enumerate(alt_questions):
                # Get documents for each alternative question
                docs = retriever.get_relevant_documents(alt_q, k=k_value)
                retrieved_docs.extend(docs)
                
                # Stream progress for each search
                progress = 50 + ((i + 1) * 5)
                yield json.dumps({
                    'status': 'processing',
                    'stage': 'searching',
                    'progress': progress,
                    'question': question,
                    'search_query': alt_q,
                    'search_index': i + 1,
                    'total_searches': len(alt_questions)
                }) + '\n'
            
            # Remove duplicates from retrieved docs
            unique_docs = []
            seen_content = set()
            
            # Track if we have table content
            has_table_content = False
            
            # First pass: identify if we have table content
            for doc in retrieved_docs:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                if "TABLE FROM PAGE" in content:
                    has_table_content = True
                    break
            
            # Second pass: process documents
            table_docs = []
            text_docs = []
            
            for doc in retrieved_docs:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                if content not in seen_content:
                    seen_content.add(content)
                    
                    # Categorize as table or text document
                    if "TABLE FROM PAGE" in content:
                        table_docs.append(doc)
                    else:
                        text_docs.append(doc)
            
            # Ensure we have a mix of both table and text content if tables are present
            if has_table_content:
                # Aim for a balanced mix of table and text content
                max_tables = min(3, len(table_docs))  # At most 3 tables
                max_texts = 5 - max_tables  # Remaining slots for text
                
                # Select documents
                selected_table_docs = table_docs[:max_tables]
                selected_text_docs = text_docs[:max_texts]
                
                # Combine and sort by relevance (assuming earlier docs are more relevant)
                top_docs = selected_table_docs + selected_text_docs
            else:
                # If no tables, just use the top text documents
                top_docs = text_docs[:5]
            
            context = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else doc for doc in top_docs])
            logging.info(f"Retrieved {len(top_docs)} unique documents for session {session_id}, including {len([d for d in top_docs if 'TABLE FROM PAGE' in (d.page_content if hasattr(d, 'page_content') else str(d))])} table documents.")

            # Update status
            yield json.dumps({
                'status': 'processing',
                'stage': 'generating',
                'progress': 70,
                'question': question
            }) + '\n'

            # Prepare messages for the LLM
            messages = []
            
            # Add image if provided and supported
            if image_path and SUPPORTS_IMAGES and os.path.exists(image_path):
                try:
                    # Read the image and convert to base64
                    with open(image_path, "rb") as img_file:
                        img_data = img_file.read()
                        base64_img = base64.b64encode(img_data).decode('utf-8')
                    
                    # Add system message about image capability
                    messages.append({
                        "role": "system", 
                        "content": "You can analyze both text and images. When an image is provided, consider both the image content and the text context to provide a comprehensive answer. Format your response with proper HTML structure using appropriate tags like <h3>, <p>, <strong>, <ul>, <li>, and <em>. Only use tables when presenting structured data that benefits from tabular format. If you need to provide information not found in the document context, clearly indicate this with a visually distinct notice in a div with class 'note-box'. For tables, use the 'styled-table' class and wrap them in a div with class 'table-responsive'."
                    })
                    
                    # Add the image as user content
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": base64_img},
                            {"type": "text", "text": f"I have a question about this image and the document: {question}\n\nContext from the document:\n{context}"}
                        ]
                    })
                    
                    logging.info(f"Added image to the conversation: {image_path}")
                except Exception as e:
                    logging.error(f"Error processing image: {e}")
                    # Fall back to text-only if image processing fails
                    messages = []
            
            # If no image or image processing failed, use text-only prompt
            if not messages:
                # Construct prompt for answer generation with HTML formatting instructions
                rag_prompt = (
                    "You are a helpful assistant that answers questions about documents. "
                    "Your answers must be formatted in valid HTML using appropriate tags.\n\n"
                    "Context from the document:\n{context}\n\n"
                    "Question: {question}\n\n"
                    "Instructions:\n"
                    "1. Answer the question based ONLY on the provided context.\n"
                    "2. If the context doesn't contain the answer, you may provide information from your knowledge, but you MUST clearly indicate this with a visually distinct notice like: "
                    "   <div class='note-box' style='background-color: #fff8e1; border-left: 4px solid #ffc107; padding: 12px; margin: 15px 0; border-radius: 4px;'>"
                    "   <p><strong>Note:</strong> This information is not from the document but from my general knowledge.</p>"
                    "   </div>\n"
                    "3. Format your text response with proper HTML structure:\n"
                    "   - Use <h3> tags for section headings\n"
                    "   - Use <p> tags for paragraphs\n"
                    "   - Use <strong> for emphasis or important points\n"
                    "   - Use <ul> and <li> for lists\n"
                    "   - Use <em> for italics\n"
                    "   - Use <blockquote> for quoted text from the document\n"
                    "4. ONLY use tables when presenting structured data that benefits from tabular format. Don't force information into tables when paragraph text is more appropriate.\n"
                    "5. When the question is specifically about data that appears in a table:\n"
                    "   - First explain the key insights in well-formatted text paragraphs\n"
                    "   - Then present the relevant data in a well-formatted HTML table\n"
                    "   - Use the 'styled-table' class for tables\n"
                    "   - Wrap tables in a div with class 'table-responsive'\n"
                    "   - Add the 'table-header' class to header rows\n"
                    "   - Add 'odd-row' and 'even-row' classes to alternate rows\n"
                    "   - Use the 'numeric' class for cells containing numeric data\n"
                    "6. When presenting tabular data, use this structure:\n"
                    "   <div class='table-responsive'>\n"
                    "     <table class='styled-table'>\n"
                    "       <caption>Table Caption (if applicable)</caption>\n"
                    "       <tr class='table-header'><th>Header 1</th><th>Header 2</th></tr>\n"
                    "       <tr class='odd-row'><td>Data 1</td><td class='numeric'>123</td></tr>\n"
                    "       <tr class='even-row'><td>Data 2</td><td class='numeric'>456</td></tr>\n"
                    "     </table>\n"
                    "   </div>\n"
                    "7. If the context contains table placeholders like [TABLE X FROM PAGE Y], check if the actual table content is provided elsewhere in the context.\n"
                    "8. Do NOT include any markdown formatting.\n"
                    "9. Do NOT include any HTML document structure tags like <html>, <body>, etc.\n"
                    "10. Do NOT include code blocks or backticks in your response.\n"
                    "11. Balance your response - provide well-formatted text with appropriate HTML tags.\n\n"
                    "Your HTML-formatted answer:"
                )
                
                messages.append({
                    "role": "user", 
                    "content": rag_prompt.format(context=context, question=question)
                })
            
            # Update status to show we're generating the answer
            yield json.dumps({
                'status': 'processing',
                'stage': 'generating',
                'progress': 80,
                'question': question
            }) + '\n'
            
            logging.info(f"Sending {len(messages)} messages to LLM")
            answer_response = llm.invoke(messages)
            raw_answer = answer_response.content
            
            # Ensure the answer is properly formatted as HTML
            formatted_answer = ensure_html_formatting(raw_answer)
            
            # Send final response
            yield json.dumps({
                'status': 'complete',
                'progress': 100,
                'answer': formatted_answer,
                'source': context
            }) + '\n'
            
        except Exception as e:
            logging.error(f"Error processing question: {e}")
            yield json.dumps({
                'status': 'error',
                'message': f'Error processing question: {str(e)}'
            }) + '\n'

    return Response(generate(), mimetype='text/event-stream')


def ensure_html_formatting(text):
    """Ensure the text is properly formatted as HTML."""
    # Remove any HTML, body, or head tags
    text = re.sub(r'</?html>|</?body>|</?head>', '', text, flags=re.IGNORECASE)
    
    # Remove any code block markers
    text = re.sub(r'```\w*|```', '', text)
    
    # Remove any inline backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Improve table formatting if present
    if '<table>' in text.lower():
        # Make sure tables have proper structure
        text = re.sub(r'<table>(?!\s*<tr>)', '<table><tr>', text, flags=re.IGNORECASE)
        
        # Add missing closing tags
        if '<table>' in text.lower() and '</table>' not in text.lower():
            text += '</table>'
            
        # Add responsive wrapper and styling classes if not present
        if '<table class=' not in text.lower() and '<table style=' not in text.lower():
            text = re.sub(r'<table>', r'<table class="styled-table">', text, flags=re.IGNORECASE)
            
        if '<div class="table-responsive">' not in text.lower():
            text = re.sub(r'<table', r'<div class="table-responsive"><table', text, flags=re.IGNORECASE)
            text = re.sub(r'</table>', r'</table></div>', text, flags=re.IGNORECASE)
            
        # Add row styling classes if not present
        if 'table-header' not in text.lower():
            # Add table-header class to first row if it has th elements
            text = re.sub(r'(<tr>(?:\s*<th[^>]*>.*?</th>\s*)+\s*</tr>)', 
                         r'<tr class="table-header">\1', text, flags=re.IGNORECASE)
            
        # Add alternating row classes
        rows = re.findall(r'<tr(?:\s+[^>]*)?>.*?</tr>', text, re.DOTALL | re.IGNORECASE)
        for i, row in enumerate(rows):
            if 'class=' not in row.lower():
                row_class = 'odd-row' if i % 2 == 1 else 'even-row'
                replacement = re.sub(r'<tr', f'<tr class="{row_class}"', row, flags=re.IGNORECASE)
                text = text.replace(row, replacement)
    
    # Check if the text already has paragraph tags
    if not re.search(r'<p>.*?</p>', text, re.DOTALL):
        # Split by newlines and wrap each paragraph in <p> tags
        paragraphs = text.split('\n\n')
        text = ''.join([f'<p>{p.strip()}</p>' for p in paragraphs if p.strip()])
    
    return text


# Add this function to clean up old sessions
def cleanup_old_sessions():
    """Clean up old sessions to prevent memory leaks."""
    current_time = datetime.now()
    sessions_to_remove = []
    
    # Find sessions that have timed out
    for session_id, timestamp in session_timestamps.items():
        if current_time - timestamp > timedelta(seconds=SESSION_TIMEOUT):
            sessions_to_remove.append(session_id)
    
    # If we have too many sessions, remove the oldest ones
    if len(session_timestamps) > MAX_SESSIONS:
        # Sort sessions by timestamp (oldest first)
        sorted_sessions = sorted(session_timestamps.items(), key=lambda x: x[1])
        # Add oldest sessions to the removal list until we're under the limit
        for session_id, _ in sorted_sessions[:len(sorted_sessions) - MAX_SESSIONS]:
            if session_id not in sessions_to_remove:
                sessions_to_remove.append(session_id)
    
    # Remove the sessions
    for session_id in sessions_to_remove:
        if session_id in vector_stores:
            del vector_stores[session_id]
        if session_id in pdf_texts:
            del pdf_texts[session_id]
        if session_id in processing_status:
            del processing_status[session_id]
        if session_id in session_timestamps:
            del session_timestamps[session_id]
        
        logging.info(f"Cleaned up session: {session_id}")
    
    # Force garbage collection
    if sessions_to_remove:
        gc.collect()
        logging.info(f"Cleaned up {len(sessions_to_remove)} old sessions")


# Add this function to periodically clean up the uploads folder
def cleanup_upload_folder():
    """Clean up old files in the uploads folder."""
    try:
        current_time = time.time()
        count = 0
        
        # Get all files in the upload folder
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Skip if not a file
            if not os.path.isfile(file_path):
                continue
                
            # Check file age (24 hours)
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > 24 * 60 * 60:  # 24 hours in seconds
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    logging.error(f"Error removing old file {file_path}: {e}")
        
        if count > 0:
            logging.info(f"Cleaned up {count} old files from uploads folder")
            
        # Also clean up sessions
        cleanup_old_sessions()
        
    except Exception as e:
        logging.error(f"Error in cleanup_upload_folder: {e}")


# Set up scheduler for periodic cleanup
scheduler = BackgroundScheduler()
scheduler.add_job(func=cleanup_upload_folder, trigger="interval", hours=1)
scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    # Clean up on startup
    cleanup_upload_folder()
    app.run(host='0.0.0.0', port=5000, debug=True)
