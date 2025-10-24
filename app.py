"""
Gradio Web Interface for Cruise RAG System
Lightweight alternative to Streamlit
"""

import gradio as gr
from rag import CruiseRAG
from pathlib import Path
import traceback
import uuid

# Initialize RAG system once at startup
print("🚀 Initializing RAG system...")
try:
    rag_system = CruiseRAG(chroma_persist_dir="./chroma_langchain_db")
    print("✅ RAG system initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing RAG system: {e}")
    traceback.print_exc()
    rag_system = None

def chat_function(message, history, thread_id, show_steps=False):
    """Process user message and return response"""
    if not rag_system:
        return "❌ RAG system not initialized. Please check the logs."

    try:
        # Stream response and collect steps
        steps_log = []
        final_response = ""

        for event in rag_system.query_stream(message, thread_id=thread_id):
            # Extract message from event tuple
            if event[0] == "values":
                latest_message = event[1]['messages'][-1]
            elif event[0] == "messages":
                latest_message = event[1][0]
                metadata = event[1][1] if len(event[1]) > 1 else {}
            else:
                continue

            # Process message
            if hasattr(latest_message, 'type'):
                agent_name = metadata.get('langgraph_node', 'unknown') if event[0] == "messages" else 'agent'

                if latest_message.type == "ai":
                    # Capture AI responses
                    if hasattr(latest_message, 'content') and latest_message.content:
                        # Always update final_response with latest AI content
                        if agent_name != 'supervisor':
                            final_response = latest_message.content

                        if show_steps:
                            steps_log.append(f"**[{agent_name}]** {latest_message.content[:]}...")

                    # Log tool calls
                    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls and show_steps:
                        for tool_call in latest_message.tool_calls:
                            tool_name = tool_call.get('name', 'unknown')
                            steps_log.append(f"🔧 **Tool:** {tool_name}")

                elif latest_message.type == "tool" and show_steps:
                    tool_name = getattr(latest_message, 'name', 'unknown')
                    content_preview = str(latest_message.content)[:]
                    steps_log.append(f"📊 **Tool Result ({tool_name}):** {content_preview}...")

        # Format final response
        if not final_response:
            final_response = "No response generated. Please check the logs or try again."
            print(f"⚠️ Warning: No final response captured from stream")

        # Add steps if requested
        if show_steps and steps_log:
            steps_section = "\n\n---\n### 🔍 Agent Steps:\n" + "\n".join(steps_log)
            return final_response + steps_section

        return final_response

    except Exception as e:
        error_msg = f"❌ Error processing query: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

def get_system_info():
    """Get system information for display"""
    info = "## 📊 System Status\n\n"

    # RAG System Status
    if rag_system:
        info += "✅ **RAG System:** Connected\n\n"
    else:
        info += "❌ **RAG System:** Not initialized\n\n"

    # Database Status
    db_path = Path("./cruises.db")
    if db_path.exists():
        info += "✅ **Database:** Connected (`cruises.db`)\n\n"
    else:
        info += "⚠️ **Database:** Not found\n\n"

    # Documents
    docs_dir = Path("./documents")
    if docs_dir.exists():
        doc_files = [f for f in docs_dir.glob("*") if f.is_file()]
        info += f"📁 **Documents:** {len(doc_files)} files\n\n"
        for doc in sorted(doc_files):
            info += f"  - 📄 {doc.name}\n"
    else:
        info += "📁 **Documents:** No documents folder\n"

    return info

# Create Gradio interface
with gr.Blocks(
    title="Celebrity Cruises RAG Assistant",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("""
    # 🚢 Celebrity Cruises RAG Assistant

    Ask questions about cruise availability, pricing, schedules, and more!
    """)

    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                show_copy_button=True,
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="Ask about cruises... (e.g., 'Show me cruises in October 2026')",
                    scale=4,
                )
                submit = gr.Button("Send", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "Show me cruises in October 2025",
                    "What are the available 7-night cruises?",
                    "Tell me about European cruises from Amsterdam",
                    "What's the price range for Caribbean cruises?",
                    "Hi, how are you?",
                ],
                inputs=msg,
            )

            clear = gr.Button("🗑️ Clear Chat")

        with gr.Column(scale=1):
            # Sidebar with settings and info
            gr.Markdown("### ⚙️ Settings")

            thread_id = gr.Textbox(
                label="Thread ID",
                value=str(uuid.uuid4()),
                info="Change to start a new conversation"
            )

            show_steps = gr.Checkbox(
                label="Show Agent Steps",
                value=False,
                info="Display tool calls and agent reasoning"
            )

            gr.Markdown("---")

            system_info = gr.Markdown(get_system_info())

            refresh_btn = gr.Button("🔄 Refresh Info")

    # Event handlers
    def respond(message, chat_history, thread_id, show_steps):
        if not message.strip():
            return chat_history, ""

        # Add user message to history
        chat_history.append((message, None))

        # Get bot response
        bot_response = chat_function(message, chat_history, thread_id, show_steps)

        # Update history with bot response
        chat_history[-1] = (message, bot_response)

        return chat_history, ""

    # Submit on button click
    submit.click(
        respond,
        inputs=[msg, chatbot, thread_id, show_steps],
        outputs=[chatbot, msg],
    )

    # Submit on enter
    msg.submit(
        respond,
        inputs=[msg, chatbot, thread_id, show_steps],
        outputs=[chatbot, msg],
    )

    # Clear chat
    clear.click(lambda: None, None, chatbot, queue=False)

    # Refresh system info
    refresh_btn.click(
        get_system_info,
        outputs=system_info,
    )

    gr.Markdown("""
    ---
    **Celebrity Cruises RAG Assistant** | Powered by LangChain & Groq
    📧 guillaume.genois.ca
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8501,
        share=False,
        show_error=True,
    )
