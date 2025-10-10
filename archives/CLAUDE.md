# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a Retrieval-Augmented Generation (RAG) system built with LangChain, using Groq's LLaMA models and Chroma vector database. The system processes documents (text and CSV files) to enable conversational Q&A about their contents.

### Core Components

- **Document Processing**: Loads and processes `.txt` and `.csv` files from the `./documents/` directory
- **Vector Store**: Uses Chroma with HuggingFace embeddings (`all-MiniLM-L6-v2` model) for document storage and retrieval
- **LLM**: Groq's `llama-3.1-8b-instant` model for question answering
- **Text Chunking**: RecursiveCharacterTextSplitter with 1000 char chunks and 200 char overlap

### System Architecture

The repository contains multiple RAG implementations:

1. **Simple RAG** (`simple_rag.py`): Basic document processing with temporal grouping
2. **Structured RAG** (`structured_rag.py`): Advanced system with hybrid search capabilities
3. **Web Interface** (`app.py`): Streamlit web application for user interaction

### Data Processing Strategy

CSV files are processed using two different approaches:

**Simple RAG Approach:**
- Rows are grouped by `year` and `month` columns
- Each group becomes a single document in the vector store
- Individual rows are converted to natural language descriptions via `create_readable_text_from_row()`

**Structured RAG Approach:**
- Groups by unique cruise (product, departure, ship, sail date)
- Creates rich metadata with cabin categories, pricing, and temporal information
- Supports advanced filtering and hybrid search capabilities

## Running the Applications

### Simple RAG System
```bash
python simple_rag.py
```
Interactive Q&A session using basic RAG with temporal grouping.

### Structured RAG System
```bash
python structured_rag.py
```
Advanced system with metadata filtering, pandas agent integration, and hybrid search.

### Streamlit Web Interface
```bash
streamlit run app.py
```
Web interface providing:
- File upload capabilities
- Interactive chat interface
- System logs and monitoring
- Visual status indicators

### Data Preprocessing
```bash
jupyter notebook notebook.ipynb
```
Contains examples for CSV date formatting, temporal column extraction, and data preparation.

## Environment Setup

Required environment variable in `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Key Dependencies
- `langchain` and `langchain-community` for RAG framework
- `langchain-groq` for Groq LLM integration
- `chromadb` for vector storage
- `sentence-transformers` for embeddings
- `pandas` for CSV processing
- `streamlit` for web interface
- `python-dotenv` for environment management

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Directory Structure

```
documents/
├── celebrity-cruises.csv     # Main cruise data with temporal grouping
├── celebrity-cruises-info.txt # Explains CSV structure and query patterns
└── [additional .txt/.csv files]
```

The system automatically discovers and processes all `.txt` and `.csv` files in the documents directory.

## Vector Store Persistence

- Simple RAG: `./chroma_groq_db/`
- Structured RAG: `./chroma_structured_db/`

Both systems persist vector stores to avoid re-processing documents on subsequent runs.

## Key Implementation Details

**Simple RAG (`simple_rag.py`):**
- Basic document chunking and retrieval
- Temporal grouping by year/month
- Interactive CLI interface
- Debug mode enabled (`langchain.debug = True`)

**Structured RAG (`structured_rag.py`):**
- Advanced metadata extraction and filtering
- Pandas agent integration for analytical queries
- Hybrid search combining vector similarity and structured filtering
- Deduplication of cruise results
- Rich natural language query processing

**Web Interface (`app.py`):**
- Streamlit-based user interface
- File upload and management
- Real-time chat interface
- System monitoring and logging
- Configurable result limits

## Development Workflow

1. **Data Preparation**: Use `notebook.ipynb` to preprocess CSV files
2. **Testing**: Use `structured_rag_test.py` for system validation
3. **Development**: Modify core RAG logic in respective Python files
4. **Deployment**: Use `app.py` for production web interface

## Query Capabilities

The structured RAG system supports complex queries such as:
- Date-based filtering ("cruises in October 2025")
- Location-based searches ("cruises from Amsterdam")
- Duration-specific queries ("7-night cruises")
- Price and cabin category analysis
- Analytical summaries and comparisons