"""
Streamlit Web Interface for Structured RAG System
Deploy at rag.guillaume.genois.ca
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
from rag import CruiseRAG
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Celebrity Cruises RAG Assistant",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'logs' not in st.session_state:
    st.session_state.logs = []

def log_message(message, level="INFO"):
    """Add message to logs"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}"
    st.session_state.logs.append(log_entry)
    logger.info(message)

@st.cache_resource  # Cache the RAG system across all users
def initialize_rag_system():
    """Initialize the RAG system"""
    try:
        # CruiseRAG automatically uses cruises.db in same directory
        rag_system = CruiseRAG(chroma_persist_dir="./chroma_langchain_db")
        return rag_system
    except Exception as e:
        st.error(f"❌ Error initializing RAG system: {str(e)}")
        return None
    
def parse_message_content(content):
    """Parse message content, handling JSON if present"""
    # Extract text from JSON if content is a dict/JSON string
    if isinstance(content, dict):
        content = content.get("text", str(content))
    elif isinstance(content, str) and content.strip().startswith("{"):
        try:
            parsed = json.loads(content)
            content = parsed.get("text", content)
        except:
            pass  # Keep original if parsing fails

    return content


# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Controls")

    # Initialize button
    if st.button("🔄 Initialize/Reload RAG System"):
        st.session_state.rag_system = initialize_rag_system()
        if st.session_state.rag_system:
            st.success("✅ System initialized!")
            log_message(st.session_state.rag_system.get_table_info())

    st.markdown("---")

    # Database info
    st.markdown("### 📊 Database Info")
    if st.session_state.rag_system:
        db_path = Path("./cruises.db")
        if db_path.exists():
            st.success(f"✅ Database connected")
            st.text(f"📄 cruises.db")
        else:
            st.warning("⚠️ Database file not found")
    else:
        st.info("Initialize system to check database")

    st.markdown("---")

    # Documents info
    st.markdown("### 📁 Documents")
    docs_dir = Path("./documents")
    if docs_dir.exists():
        doc_files = list(docs_dir.glob("*"))
        if doc_files:
            for doc_file in sorted(doc_files):
                if doc_file.is_file():
                    st.text(f"📄 {doc_file.name}")
        else:
            st.info("No documents found")
    else:
        st.info("Documents folder not found")

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Settings")

    # Thread ID for conversation memory
    thread_id = st.text_input("Thread ID", value="default", help="Change to start a new conversation thread")

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Logs toggle
    show_logs = st.checkbox("📋 Show Logs", value=False)

# Main content
st.markdown('<h1 class="main-header">🚢 Celebrity Cruises RAG Assistant</h1>', unsafe_allow_html=True)

# System status
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.session_state.rag_system:
        st.success("✅ System Ready")
    else:
        st.warning("⚠️ System Not Initialized")

with col2:
    db_exists = Path("./cruises.db").exists()
    if db_exists:
        st.info(f"📊 Database connected")
    else:
        st.warning("⚠️ No database")

with col3:
    st.info(f"💬 {len(st.session_state.messages)} messages")

st.markdown("---")

# Chat interface
st.markdown("### 💬 Chat with your data")

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    if role == "user":
        st.markdown(f'<div class="chat-message user-message">{content}</div>',
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message assistant-message">{content}</div>',
                   unsafe_allow_html=True)

# Chat input
user_query = st.chat_input("Ask about cruises... (e.g., 'Show me cruises in October 2026')")

if user_query:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.markdown(f'<div class="chat-message user-message">{user_query}</div>',
                   unsafe_allow_html=True)
    
    # Check if system is initialized
    if not st.session_state.rag_system:
        st.warning("⚠️ Please initialize the RAG system first (click button in sidebar)")
        log_message("Query attempted before system initialization", "WARNING")
    else:
        # Get response with streaming (logs steps, displays final answer)
        try:
            with st.spinner("🔍 Processing your query..."):
                final_response = ""

                # Stream through agent steps
                for event in st.session_state.rag_system.query_stream(user_query, thread_id=thread_id):
                    
                    # Extract message from event
                    if event[0] == "values":
                        message = event[1]['messages'][-1]  # Latest message
                    elif event[0] == "messages":
                        message = event[1][0]

                    # Log each step
                    if hasattr(message, 'type'):
                        if message.type == "ai":
                            if hasattr(message, 'tool_calls') and message.tool_calls:
                                # Log tool calls
                                for tool_call in message.tool_calls:
                                    log_message(f"Tool call: {tool_call.get('name', 'unknown')}")
                            if hasattr(message, 'content') and message.content:
                                final_response = parse_message_content(message.content)
                                log_message(f"AI response: {final_response}")
                        elif message.type == "tool":
                            log_message(f"Tool result: {str(message.content)}")
                        elif message.type == "human":
                            log_message(f"User query: {message.content}")

                # Add final response to chat
                if final_response:
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    log_message(f"Response generated successfully")
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "No response generated"})
                    log_message(f"No response generated", "WARNING")

                # Rerun to update chat display
                st.rerun()
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.error(f"❌ {error_msg}")
            log_message(error_msg, "ERROR")

# Logs section (collapsible)
if show_logs:
    st.markdown("---")
    st.markdown("### 📋 System Logs")

    if st.session_state.logs:
        # Create scrollable container with all logs
        st.markdown("""
        <style>
        .log-container {
            height: 400px;
            overflow-y: scroll;
            border: 1px solid #ddd;
            padding: 10px;
            background-color: #f8f9fa;
            font-family: monospace;
            font-size: 0.85rem;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

        logs_html = "<div class='log-container'>"
        for log in st.session_state.logs:
            logs_html += f"<div>{log}</div>"
        logs_html += "</div>"

        st.markdown(logs_html, unsafe_allow_html=True)

        # Download logs button
        full_logs = "\n".join(st.session_state.logs)
        st.download_button(
            label="💾 Download Full Logs",
            data=full_logs,
            file_name=f"rag_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    else:
        st.info("No logs yet")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>Celebrity Cruises RAG Assistant | Powered by LangChain & Groq</p>
    <p>📧 guillaume.genois.ca</p>
</div>
""", unsafe_allow_html=True)

# Auto-initialize on first load
if st.session_state.rag_system is None:
    st.session_state.rag_system = initialize_rag_system()