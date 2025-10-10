# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) system for Celebrity Cruises data. It combines SQL database queries with vector store retrieval to answer questions about cruise availability, pricing, schedules, and general cruise information. The system uses LangGraph to orchestrate a React agent that can access both structured (SQL) and unstructured (vector store) data sources.

## Architecture

### Core Components

**`rag.py`**: The main RAG system implementation (`CruiseRAG` class)
- Initializes LLM (ChatGroq with llama-3.3-70b-versatile)
- Sets up vector store (Chroma with HuggingFace embeddings using all-MiniLM-L6-v2)
- Configures SQL database access (SQLite with celebrity_cruises table)
- Creates a unified React agent with both SQL toolkit and vector retrieval tools
- Handles conversation memory via LangGraph's MemorySaver checkpointer

**`app.py`**: Gradio web interface
- Provides chat interface for user queries
- Streams responses from the RAG system
- Displays agent steps and tool calls (optional)
- Shows system status (database, documents, vector store)

**Database**: `cruises.db` (SQLite)
- Table: `celebrity_cruises`
- Columns include: group_id, product, city_of_departure, ship, sail_date, sail_nights, year, month, day, cabin_category, min_group_fare, max_group_fare, etc.
- Processed from raw cruise data via `preprocess_data.ipynb`

**Vector Store**: Chroma (persisted to `./chroma_langchain_db`)
- Stores embeddings of documents from `documents/` directory
- Currently contains: `celebrity-cruises-info.txt` (general cruise information)

**Documents**: `documents/` directory
- Text files containing unstructured information about Celebrity Cruises
- Automatically indexed into vector store on first run

### Agent Architecture

The system uses a single React agent with multiple tools:
- SQL database tools (from SQLDatabaseToolkit): list_tables, schema, query, query_checker
- Vector retrieval tool: `retrieve()` for similarity search
- The agent decides which tools to use based on the user query
- System prompt provides instructions for both SQL queries and general conversation

### Data Flow

1. User question → Gradio UI (`app.py`)
2. Question sent to `rag_system.query_stream()`
3. React agent analyzes question and decides which tools to call
4. Tools execute (SQL queries and/or vector retrieval)
5. Agent synthesizes response from tool results
6. Response streamed back to UI

## Development Commands

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Run Gradio web interface (default)
python app.py

# Run CLI interface
python rag.py
```

The Gradio app launches on port 8501 and binds to 0.0.0.0 by default.

### Data Preprocessing

The Jupyter notebook `preprocess_data.ipynb` processes raw cruise CSV data:
- Parses dates and creates year/month/day columns
- Aggregates by group_id to get one row per cruise
- Exports to both CSV (`documents/celebrity-cruises.csv`) and SQLite (`cruises.db`)

To update the database with new cruise data:
1. Place raw CSV in `source-documents/`
2. Run cells in `preprocess_data.ipynb`
3. Database and documents will be updated

### Environment Variables

Required in `.env` file:
- `GROQ_API_KEY`: API key for Groq LLM service

## Key Implementation Details

### Agent Prompt Strategy

The system uses a single unified prompt (`system_message` in `rag.py:100-141`) that:
- Handles general chat and greetings
- Instructs on vector store usage for document retrieval
- Provides SQL query guidelines (limit results, check schema first, no DML)
- Specifies SQLite dialect and available tables

### Vector Store Management

`_setup_vector_store()` in `rag.py:292-336`:
- Checks if vector store already has documents (avoids re-indexing)
- Loads all .txt files from `documents/` directory
- Splits text into chunks (1000 chars with 200 overlap)
- Persists to `chroma_langchain_db/` for reuse

### Streaming Responses

`query_stream()` in `rag.py:372-389`:
- Uses LangGraph's stream mode with both "values" and "messages"
- Events contain message objects with type, content, and metadata
- UI filters events by langgraph_node to distinguish agent vs supervisor
- Final response captured from last AI message (excluding supervisor)

### Deployment Configuration

**systemd service** (`systemd/rag-app.service`):
- Runs as www-data user
- Sets model cache directories for HuggingFace
- References `app_gradio.py` (note: actual file is `app.py`)

**nginx** (`nginx/rag.conf`):
- Proxies to localhost:8501
- WebSocket support for Gradio streaming
- SSL configured for production domain

## Common Patterns

### Adding New Documents

1. Place .txt file in `documents/` directory
2. Delete `chroma_langchain_db/` to force re-indexing
3. Restart the application

### Modifying Agent Behavior

Edit the `system_message` in `rag.py:100-141` to change:
- How the agent approaches queries
- SQL query guidelines
- Response formatting
- Tool usage priorities

### Debugging Agent Steps

In Gradio UI:
- Enable "Show Agent Steps" checkbox
- View tool calls and intermediate results in response

In CLI (`python rag.py`):
- Agent automatically prints all messages with `pretty_print()`
- Shows full conversation flow and tool usage

## Architecture Notes

**Commented Code**: The codebase contains commented-out code for a multi-agent supervisor architecture (`rag.py:85-96, 236-290`). This was an earlier design with separate sql_agent, chat_agent, and retrieval_agent coordinated by a supervisor. The current implementation uses a single unified React agent, which is simpler and performs well for this use case.

**Thread-based Memory**: The system uses thread IDs to maintain separate conversation contexts. Each thread has its own memory via LangGraph's checkpointer.

**Model Flexibility**: Code includes commented Ollama integration (`rag.py:62`) for local LLM usage as an alternative to Groq.
