"""
Streamlit Web Interface for Structured RAG System
Deploy at rag.guillaume.genois.ca
"""

import streamlit as st
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from structured_rag import StructuredRAG
import logging
from io import StringIO

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

def initialize_rag_system():
    """Initialize the RAG system"""
    try:
        csv_path = "./documents/celebrity-cruises.csv"
        if not os.path.exists(csv_path):
            st.error(f"❌ CSV file not found: {csv_path}")
            log_message(f"CSV file not found: {csv_path}", "ERROR")
            return None

        with st.spinner("🔄 Initializing RAG system..."):
            rag_system = StructuredRAG(csv_path)
            log_message("RAG system initialized successfully")
            return rag_system
    except Exception as e:
        st.error(f"❌ Error initializing RAG system: {str(e)}")
        log_message(f"Error initializing RAG system: {str(e)}", "ERROR")
        return None

def save_uploaded_file(uploaded_file, directory="./documents"):
    """Save uploaded file to documents directory"""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(directory, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        log_message(f"File uploaded: {uploaded_file.name}")
        return file_path
    except Exception as e:
        log_message(f"Error uploading file: {str(e)}", "ERROR")
        return None

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Controls")

    # Initialize button
    if st.button("🔄 Initialize/Reload RAG System"):
        st.session_state.rag_system = initialize_rag_system()
        if st.session_state.rag_system:
            st.success("✅ System initialized!")

    st.markdown("---")

    # Document upload section
    st.markdown("### 📁 Document Management")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a new CSV file to add to the documents directory"
    )

    if uploaded_file is not None:
        if st.button("💾 Save File"):
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                st.success(f"✅ File saved: {uploaded_file.name}")
                st.info("ℹ️ Reinitialize the system to use the new file")

    # Show current documents
    st.markdown("#### Current Documents")
    docs_dir = Path("./documents")
    if docs_dir.exists():
        csv_files = list(docs_dir.glob("*.csv"))
        txt_files = list(docs_dir.glob("*.txt"))

        if csv_files:
            st.markdown("**CSV files:**")
            for f in csv_files:
                st.text(f"📄 {f.name}")

        if txt_files:
            st.markdown("**Text files:**")
            for f in txt_files:
                st.text(f"📝 {f.name}")

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Settings")
    max_results = st.slider("Max results to show", 5, 30, 15)

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
    docs_count = len(list(Path("./documents").glob("*.csv"))) if Path("./documents").exists() else 0
    st.info(f"📁 {docs_count} CSV files")

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
        st.markdown(f'<div class="chat-message user-message">👤 **You:** {content}</div>',
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message assistant-message">🤖 **Assistant:** {content}</div>',
                   unsafe_allow_html=True)

# Chat input
user_query = st.chat_input("Ask about cruises... (e.g., 'Show me cruises in October 2026')")

if user_query:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    log_message(f"User query: {user_query}")

    # Check if system is initialized
    if not st.session_state.rag_system:
        st.warning("⚠️ Please initialize the RAG system first (click button in sidebar)")
        log_message("Query attempted before system initialization", "WARNING")
    else:
        # Get response
        try:
            with st.spinner("🔍 Processing your query..."):
                response = st.session_state.rag_system.hybrid_search(
                    user_query,
                    max_results=max_results
                )

                # Add assistant response to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                log_message(f"Response generated successfully")

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
        logs_text = "\n".join(st.session_state.logs[-50:])  # Show last 50 logs
        st.text_area("Logs", logs_text, height=300, disabled=True)

        # Download logs button
        if st.button("💾 Download Full Logs"):
            full_logs = "\n".join(st.session_state.logs)
            st.download_button(
                label="📥 Download",
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