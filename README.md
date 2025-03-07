# PDF Chat Tool

A tool for chatting with a PDF using local LLMs. This project leverages Flask, PyMuPDF, and local models such as Ollama for embeddings and ChatOllama for generating responses.

![{F4DB45FD-9A07-4FE0-97EF-8861AE527DBB}](https://github.com/user-attachments/assets/2a541109-eff3-47d7-855d-1054e023b703)

## Requirements
- Python 3.x
- [Ollama](https://www.ollama.com) (for embeddings and chat model)
- Local LLMs (e.g., ChatOllama)
- Additional Python libraries: Flask, PyMuPDF, faiss, numpy, langchain, etc.

*Performance may vary depending on the used hardware.*

## Features
- **PDF Upload:** Extracts text using PyMuPDF.
- **Text Splitting:** Divides content into overlapping chunks.
- **Vector Store:** Builds a FAISS index from text chunks.
- **Question Answering:** Retrieves context and generates HTML-formatted answers.

## Installation
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
1. Start the ```backend.py```
2. ```cd frontend``` and compile frontend with ```ng serve```
3. Download Models from Ollama (can be easily changed in ```backend.py```)
