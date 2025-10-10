# Celebrity Cruises RAG Assistant

An intelligent conversational assistant for Celebrity Cruises information, powered by Retrieval-Augmented Generation (RAG). This system combines SQL database queries with vector store retrieval to answer questions about cruise availability, pricing, schedules, and general cruise information.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)

## Features

- **Intelligent Query Routing**: Automatically determines whether to use SQL queries or document retrieval based on the question
- **Natural Conversation**: Handles greetings, casual chat, and complex cruise queries
- **Hybrid Search**: Combines structured database queries with semantic document search
- **Real-time Streaming**: See agent reasoning and tool usage in real-time
- **Modern UI**: Clean Gradio interface with conversation history
- **Memory**: Maintains conversation context across questions using thread-based memory

## Architecture

The system uses **LangGraph** to orchestrate a React agent with access to:

- **SQL Database** (SQLite): Structured cruise data including dates, prices, ships, and itineraries
- **Vector Store** (Chroma): Semantic search over cruise information documents
- **LLM** (Groq): llama-3.3-70b-versatile for natural language understanding and generation

## Quick Start

### Prerequisites

- Python 3.8+
- Groq API key ([get one here](https://console.groq.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AIAssistantWithRAG.git
   cd AIAssistantWithRAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**

   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**

   Navigate to `http://localhost:8501`

## Usage

### Web Interface

The Gradio interface provides:

- **Chat Interface**: Ask questions in natural language
- **Thread Management**: Change thread ID to start new conversations
- **Agent Steps**: Toggle to see detailed reasoning and tool usage
- **System Status**: View database and document availability

Example queries:
- "Show me cruises in October 2025"
- "What are the available 7-night cruises?"
- "Tell me about European cruises from Amsterdam"
- "What's the price range for Caribbean cruises?"

### CLI Interface

For command-line interaction:

```bash
python rag.py
```

This provides a text-based interface with full agent step visibility.

## Data Management

### Database Schema

The `celebrity_cruises` table includes:
- `group_id`: Unique cruise identifier
- `product`: Cruise region (Europe, Caribbean, etc.)
- `city_of_departure`: Departure city
- `ship`: Ship code
- `sail_date`: Formatted sail date
- `year`, `month`, `day`: Parsed date components
- `sail_nights`: Cruise duration
- `cabin_category`: Available cabin types
- `min_group_fare`, `max_group_fare`: Price range
- And more...

### Updating Cruise Data

To add or update cruise information:

1. Place raw CSV data in `source-documents/`
2. Open and run `preprocess_data.ipynb`
3. The notebook will:
   - Parse and clean the data
   - Create date fields
   - Aggregate by cruise group
   - Export to `cruises.db` and `documents/`

### Adding Documents

To add new information documents:

1. Place `.txt` files in the `documents/` directory
2. Delete the `chroma_langchain_db/` folder to force re-indexing
3. Restart the application

Documents are automatically:
- Loaded and split into chunks
- Embedded using HuggingFace all-MiniLM-L6-v2
- Stored in Chroma vector database
- Made available for semantic search

## Project Structure

```
AIAssistantWithRAG/
├── app.py                    # Gradio web interface
├── rag.py                    # Core RAG system (CruiseRAG class)
├── requirements.txt          # Python dependencies
├── cruises.db               # SQLite database
├── preprocess_data.ipynb    # Data preprocessing notebook
├── documents/               # Text files for vector store
│   └── celebrity-cruises-info.txt
├── source-documents/        # Raw data files
├── chroma_langchain_db/     # Vector store persistence (auto-generated)
├── nginx/                   # Nginx configuration for deployment
├── systemd/                 # Systemd service for deployment
└── archives/                # Archived/experimental code
```

## Deployment

### Systemd Service

The `systemd/rag-app.service` file provides a template for running as a system service:

```bash
sudo cp systemd/rag-app.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rag-app
sudo systemctl start rag-app
```

### Nginx Reverse Proxy

The `nginx/rag.conf` provides SSL-enabled reverse proxy configuration:

```bash
sudo cp nginx/rag.conf /etc/nginx/sites-available/rag
sudo ln -s /etc/nginx/sites-available/rag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

**Note**: Update domain name and SSL certificate paths as needed.

## Technology Stack

- **LangChain**: Framework for LLM applications
- **LangGraph**: Agent orchestration and workflow management
- **Groq**: Fast LLM inference
- **Chroma**: Vector database for semantic search
- **HuggingFace**: Embedding models
- **Gradio**: Web UI framework
- **SQLite**: Relational database
- **Pandas**: Data processing

## Configuration

### LLM Model

Default: `llama-3.3-70b-versatile` via Groq

To use a different model, edit `rag.py`:
```python
self.llm = ChatGroq(model_name="your-model-name")
```

For local models via Ollama, uncomment the relevant lines in `rag.py:62`.

### Vector Store

Default: Chroma with `all-MiniLM-L6-v2` embeddings

To change the embedding model, edit `rag.py:65`:
```python
self.embeddings = HuggingFaceEmbeddings(model_name="your-model-name")
```

### Agent Behavior

The system prompt in `rag.py:100-141` controls agent behavior. Modify this to change:
- Response style and tone
- SQL query approach
- Tool usage priorities
- Output formatting

## Troubleshooting

**Vector store not loading documents:**
- Ensure `.txt` files are in `documents/` directory
- Check file encoding is UTF-8
- Delete `chroma_langchain_db/` to force re-indexing

**Database queries failing:**
- Verify `cruises.db` exists in project root
- Run `preprocess_data.ipynb` to regenerate database
- Check SQL query syntax in agent output

**API errors:**
- Verify `GROQ_API_KEY` is set in `.env`
- Check Groq API quota and status
- Review error messages in console output

## Contributing

Contributions are welcome! Areas for enhancement:

- Add more cruise lines/data sources
- Implement advanced filtering (price ranges, date ranges)
- Add booking integration
- Improve agent prompts for better responses
- Add unit tests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Powered by [Groq](https://groq.com/) for fast inference
- UI by [Gradio](https://gradio.app/)

## Contact

Guillaume Genois - guillaume.genois.ca

---

**Celebrity Cruises RAG Assistant** | Built with AI
