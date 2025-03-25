# PDF Chat Tool

An interactive tool that allows users to chat with their PDF documents using advanced language models and semantic search. Built with Angular frontend and Flask backend.

![pdf-chat](https://github.com/user-attachments/assets/ed689c81-85a6-48d6-8321-0328c7a38cda)

## Features

- **PDF Document Processing**
  - Upload and process PDF files
  - Automatic text extraction
  - Smart table detection and formatting
  - Real-time processing status updates

- **Intelligent Chat Interface**
  - Natural language question answering
  - Context-aware responses
  - HTML-formatted answers with proper styling
  - Support for tables and structured data
  - Image upload and analysis capabilities

- **Advanced Search & Retrieval**
  - Semantic search using FAISS vector store
  - Multi-query retrieval for better results
  - Automatic question rephrasing
  - Table-aware search optimization

## Technology Stack

### Frontend
- Angular 15
- Angular Material UI
- RxJS for reactive programming
- PDF.js for PDF rendering

### Backend
- Flask
- FAISS for vector search
- PyMuPDF for PDF processing
- Ollama for embeddings and chat
- LangChain for document processing

## Getting Started

### Prerequisites
- Python 3.12+
- Node.js and npm
- Ollama installed and running locally

### Backend Setup
1. Navigate to the backend directory
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the backend server:
   ```bash
   python backend.py
   ```
   The server will start at http://localhost:5000

### Frontend Setup
1. Navigate to the frontend directory
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   ng serve
   ```
   The application will be available at http://localhost:4200

## Usage

1. **Upload PDF**
   - Click "Select PDF" to upload your document
   - Wait for processing to complete
   - Progress indicators will show current status

2. **Ask Questions**
   - Type your question in the chat input
   - Get HTML-formatted responses
   - View tables and structured data
   - Upload images for visual context

3. **View Results**
   - See PDF preview in the right panel
   - Get context-aware answers
   - View formatted tables and lists
   - Track processing stages
