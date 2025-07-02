# RAG-Powered PDF Question Answering App

This project implements a Retrieval-Augmented Generation (RAG) pipeline that enables users to query a PDF document using a locally hosted Large Language Model (LLM). It extracts and indexes the contents of a PDF, retrieves relevant information using vector search, and generates answers through a chat-based LLM such as LLaMA 3 via Ollama.

## Features

* Upload and process PDF documents
* Chunk and embed content using `nomic-embed-text`
* Perform vector similarity search with FAISS
* Generate answers using a local LLM (e.g., `llama3`)
* Streamlit-based user interface for interactive question answering
* Fully local processing without reliance on external APIs

## Project Structure

```
.
├── data/
│   └── awsbook.pdf              # Input PDF file
├── index/
│   ├── faiss_index.index        # Vector index
│   └── chunks.json              # Stored text chunks
├── rag/
│   ├── chunker.py               # Text chunking logic
│   ├── embedder.py              # Embedding functions using Ollama
│   ├── faiss_store.py           # FAISS index handling
│   ├── llm.py                   # Local LLM interaction
│   ├── pdf_loader.py            # PDF text extraction
├── ingest.py                    # Script for processing and indexing the PDF
├── query.py                     # Streamlit app for querying
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/tarun0matta/rag-implementation
cd rag-implementation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure that Ollama is installed and running locally with the following models available:

* `nomic-embed-text`
* `llama3` (or a compatible chat model)

## Usage

### Step 1: Ingest the PDF

This step parses the PDF, creates embeddings, and builds the FAISS index:

```bash
python ingest.py
```

### Step 2: Launch the Streamlit App

Start the user interface to begin querying:

```bash
streamlit run query.py
```

## How It Works

1. PDF to Text: Extracts text from the PDF using PyPDF2
2. Chunking: Splits the text into overlapping segments
3. Embedding: Converts chunks into vectors using `nomic-embed-text`
4. Indexing: Stores vectors in a FAISS index for fast retrieval
5. Retrieval and Generation: Finds relevant chunks and passes them to `llama3` for answer generation

## Dependencies

* `faiss-cpu`
* `PyPDF2`
* `langchain`
* `requests`
* `streamlit`
* `numpy`

## Customization

* Replace the PDF in `data/awsbook.pdf` to use a different document
* Modify `embedder.py` or `llm.py` to change embedding or LLM models
* Adjust chunking strategy in `chunker.py`

## Acknowledgements

* Ollama for local model serving
* FAISS for similarity search
* LangChain for modular RAG components
* Hugging Face for model and tokenizer support

