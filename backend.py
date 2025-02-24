from flask import Flask, request, jsonify
from flask_cors import CORS
import os, uuid, logging
from collections import OrderedDict
import faiss
import numpy as np
import fitz  # PyMuPDF

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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory session storage
vector_stores = {}  # session_id -> FAISSVectorStore instance
pdf_texts = {}  # session_id -> extracted text


def extract_text_with_pymupdf(file_path):
    """Extract text from a PDF using PyMuPDF (fitz)."""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            page_text = page.get_text("text")
            text += page_text + "\n"
        doc.close()
    except Exception as e:
        logging.error(f"Error extracting text with pymupdf: {e}")
    return text


@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Upload a PDF, extract its text using PyMuPDF, split it into chunks,
    and build a FAISS vector store using Ollama embeddings.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    session_id = request.form.get('session_id') or str(uuid.uuid4())
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No file provided.'}), 400

    all_text = ""
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        logging.info(f"File saved: {file.filename}")
        extracted = extract_text_with_pymupdf(file_path)
        all_text += extracted + "\n\n"

    if not all_text.strip():
        return jsonify({'error': 'Failed to extract text from PDF.'}), 400

    pdf_texts[session_id] = all_text

    # Split text into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(all_text)
    logging.info(f"Created {len(chunks)} chunks for session {session_id}.")

    # Create FAISS vector store with Ollama embeddings
    embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = FAISSVectorStore.from_texts(chunks, embedding_model)
    vector_stores[session_id] = vector_store
    logging.info(f"FAISS vector store created for session {session_id}.")

    return jsonify({'message': 'PDF processed successfully.', 'session_id': session_id})


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
    if session_id not in vector_stores:
        return jsonify({'error': 'Session not found. Please upload a PDF first.'}), 400

    vector_store = vector_stores[session_id]
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})

    # Multi-query retrieval: rephrase the question in multiple ways
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "Rephrase the following question in five different ways to capture various nuances:\n"
            "Original question: {question}"
        )
    )
    llm = ChatOllama(model="llama3.1")
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever, llm, prompt=query_prompt)
    retrieved_docs = multi_query_retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else doc for doc in retrieved_docs])
    logging.info(f"Retrieved {len(retrieved_docs)} documents for session {session_id}.")

    # Construct prompt for answer generation with HTML formatting instructions
    rag_prompt = ChatPromptTemplate.from_template(
        (
            "<html><body>"
            "<h2>Context</h2><p>{context}</p>"
            "<h2>Question</h2><p>{question}</p>"
            "<h2>Answer</h2>"
            "<p>Please provide a clear answer that must be formatted in valid HTML. Use <strong>, <em>, <ul>, <li>, and <p> as needed.</p>"
            "</body></html>"
        )
    )
    full_prompt = rag_prompt.format(context=context, question=question)
    logging.info("LLM prompt (first 500 chars): " + full_prompt[:500] + "...")
    answer_response = llm.invoke([{"role": "user", "content": full_prompt}])
    answer = answer_response.content

    return jsonify({'answer': answer, 'source': context})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
