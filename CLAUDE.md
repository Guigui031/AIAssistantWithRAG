# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a Retrieval-Augmented Generation (RAG) system built with LangChain, using Groq's LLaMA models and Chroma vector database. The system processes documents (text and CSV files) to enable conversational Q&A about their contents.

### Core Components

- **Document Processing**: Loads and processes `.txt` and `.csv` files from the `./documents/` directory
- **Vector Store**: Uses Chroma with HuggingFace embeddings (`all-MiniLM-L6-v2` model) for document storage and retrieval
- **LLM**: Groq's `llama-3.1-8b-instant` model for question answering
- **Text Chunking**: RecursiveCharacterTextSplitter with 1000 char chunks and 200 char overlap

### Data Processing Strategy

CSV files are processed using a specialized approach:
- Rows are grouped by `year` and `month` columns
- Each group becomes a single document in the vector store
- Individual rows are converted to natural language descriptions via `create_readable_text_from_row()`
- Maintains temporal context while enabling efficient retrieval

## Running the Application

### Main Application
```bash
python simple_rag.py
```

This starts an interactive Q&A session. The system will:
1. Load documents from `./documents/`
2. Create embeddings and vector store in `./chroma_groq_db/`
3. Start interactive prompt for questions

### Jupyter Notebook
The `notebook.ipynb` contains data preprocessing examples, particularly for CSV date formatting and adding temporal columns.

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

## Data Directory Structure

```
documents/
├── celebrity-cruises.csv     # Main cruise data with temporal grouping
├── celebrity-cruises-info.txt # Explains CSV structure and query patterns
└── [additional .txt/.csv files]
```

The system automatically discovers and processes all `.txt` and `.csv` files in the documents directory.

## Key Implementation Details

- Vector store persists to `./chroma_groq_db/` to avoid re-processing documents
- CSV processing includes date parsing and temporal column extraction
- Retrieval uses top-k=5 similar documents for context
- QA chain returns both answers and source documents for transparency
- Debug mode enabled (`langchain.debug = True`) for development visibility