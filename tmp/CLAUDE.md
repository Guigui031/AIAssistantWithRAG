# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) system that combines SQL database queries with vector store retrieval using LangGraph agents. The system enables conversational Q&A about cruise data stored in a SQLite database.

### Core Components

- **LLM**: Ollama with llama3.1 (local inference) or Groq's llama-3.3-70b-versatile (cloud)
- **Vector Store**: Chroma with HuggingFace embeddings (`all-MiniLM-L6-v2` model)
- **Database**: SQLite database (`cruises.db`) containing cruise information
- **Agent Framework**: LangGraph with ReAct agent pattern
- **Web Interface**: Streamlit application for user interaction

### System Architecture

The repository contains:

1. **CruiseRAG Class** (`rag.py`): Main RAG system combining SQL and vector retrieval
2. **Streamlit App** (`app.py`): Web interface for interacting with the RAG system
3. **Jupyter Notebook** (`cruises-rag.ipynb`): Development and testing environment

## Core Files

### rag.py

Main RAG implementation with the `CruiseRAG` class:

```python
class CruiseRAG:
    def __init__(self, db_path=None, chroma_persist_dir="./chroma_langchain_db")
    def query(self, question: str, thread_id: str = "default") -> str
    def query_stream(self, question: str, thread_id: str = "default")
    def get_table_info(self) -> str
```

**Key Features:**
- Automatic database path resolution (uses `cruises.db` in same directory)
- SQL toolkit with database query capabilities
- Vector store retrieval for semantic search
- LangGraph ReAct agent with conversation memory
- Thread-based conversation history using `MemorySaver`

**Agent Tools:**
- SQL database schema inspection
- SQL query execution and checking
- Vector store similarity search (`retrieve` tool)

### app.py

Streamlit web interface providing:

**User Interface:**
- Chat interface with message history
- Real-time streaming with step-by-step logging
- System status indicators (connection, database, message count)

**Sidebar Controls:**
- System initialization/reload
- Database connection status
- Thread ID configuration for conversation management
- Chat history clearing
- Scrollable logs viewer

**Logging System:**
- Captures all agent steps (AI messages, tool calls, tool results)
- Pretty-formatted logs with separators
- Scrollable log container (400px height)
- Download logs functionality

## Running the Applications

### Streamlit Web Interface
```bash
streamlit run app.py
```

Web interface features:
- Interactive chat with RAG system
- Streaming responses with step logging
- Thread-based conversation memory
- Visual system monitoring

### Jupyter Notebook
```bash
jupyter notebook cruises-rag.ipynb
```

Used for development, testing, and experimentation.

## Environment Setup

Required environment variable in `.env`:
```
GROQ_API_KEY=your_groq_api_key_here  # Optional, if using Groq instead of Ollama
```

### Key Dependencies
- `langchain` and `langchain-community` for RAG framework
- `langchain-groq` for Groq LLM integration (optional)
- `langchain-ollama` for local Ollama inference
- `langgraph` for agent workflow
- `chromadb` for vector storage
- `sentence-transformers` for embeddings
- `streamlit` for web interface
- `python-dotenv` for environment management

### Ollama Setup

For local inference, install Ollama and pull a compatible model:

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.1  # Recommended (supports tools)
# or
ollama pull llama3.2  # Alternative
ollama pull mistral   # Alternative
ollama pull qwen2.5   # Alternative
```

**Important:** Use models that support function calling/tools. Models like `llama3` (without .1/.2) do NOT support tools and will fail.

## Database Structure

The SQLite database (`cruises.db`) contains cruise information with schema:

```sql
CREATE TABLE celebrity_cruises (
    product TEXT,
    city_of_departure TEXT,
    ship TEXT,
    sail_date TEXT,
    group_id INTEGER,
    cabin_category TEXT,
    cabins_requested INTEGER,
    cabins_named INTEGER,
    cabins_unnamed INTEGER,
    price TEXT,
    "group_fare*" TEXT,
    "laf*" TEXT,
    group_fare_advantage TEXT,
    taxes_per_guest INTEGER,
    currency TEXT,
    group_name TEXT,
    sail_nights INTEGER,
    num_date TIMESTAMP,
    year INTEGER,
    month INTEGER,
    day INTEGER
)
```

## Vector Store Persistence

- Vector store directory: `./chroma_langchain_db/`
- Collection name: `example_collection`
- Persists to avoid re-processing documents on subsequent runs

## Key Implementation Details

### CruiseRAG System (`rag.py`)

**Initialization:**
- Sets up LLM (Ollama llama3.1 by default, configurable to Groq)
- Initializes Chroma vector store with HuggingFace embeddings
- Connects to SQLite database with automatic path resolution
- Creates LangGraph ReAct agent with SQL toolkit and retrieval tools

**Query Methods:**
- `query()`: Returns final answer as string
- `query_stream()`: Yields intermediate agent messages for logging/streaming
- Thread-based conversation memory for context retention

**Agent Prompt:**
- Instructs agent to query database schema first
- Limits results to top 5 by default
- Prevents DML operations (INSERT, UPDATE, DELETE, DROP)
- Emphasizes query validation and error recovery

### Streamlit App (`app.py`)

**Session State:**
- `rag_system`: CruiseRAG instance
- `messages`: Chat history
- `logs`: Agent step logs

**Streaming Implementation:**
- Uses `query_stream()` to capture agent steps
- Logs all intermediate messages (AI, tool, human)
- Displays only final answer in chat
- Pretty-formatted logs with separators and full content

**UI Components:**
- System status indicators (Ready/Not Initialized, Database, Messages)
- Thread ID input for conversation management
- Scrollable logs with monospace formatting
- Download logs functionality

## Development Workflow

1. **Database Preparation**: Ensure `cruises.db` exists in the cruises-rag directory
2. **Model Setup**: Pull Ollama model (`ollama pull llama3.1`) or configure Groq API key
3. **Testing**: Use `cruises-rag.ipynb` for experimentation
4. **Development**: Modify `rag.py` for core logic changes
5. **Deployment**: Run `streamlit run app.py` for web interface

## Query Capabilities

The system supports queries such as:
- Date-based filtering: "Show me cruises in October 2025"
- Location-based searches: "Cruises departing from Amsterdam"
- Duration queries: "What 7-night cruises are available?"
- Price and cabin analysis: "Compare cabin categories for European cruises"
- Analytical queries using SQL: "How many cruises in 2026?"
- Semantic search via vector store: "Relaxing Caribbean vacations"

## Switching Between LLM Providers

### Using Ollama (Local - Default)
```python
from langchain.chat_models import init_chat_model
self.llm = init_chat_model("llama3.1", model_provider="ollama")
```

### Using Groq (Cloud)
```python
from langchain_groq import ChatGroq
self.llm = ChatGroq(model_name="llama-3.3-70b-versatile")
```

Edit line 59 in `rag.py` to switch between providers.

## Important Notes

- **Tool Support**: Ensure Ollama model supports function calling (llama3.1+, not llama3)
- **Database Path**: Automatically resolved to `cruises.db` in the same directory as `rag.py`
- **Vector Store**: Persisted locally to avoid re-embedding on each run
- **Conversation Memory**: Thread-based with configurable thread IDs in the UI
- **Logging**: Enable "Show Logs" in sidebar to see agent reasoning steps
