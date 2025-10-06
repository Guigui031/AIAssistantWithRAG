import os
import traceback
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import Annotated
from langchain_groq import ChatGroq
from langchain_core.tools import tool
# from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool


load_dotenv()


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class AgentState(TypedDict):
    question: str
    selected_agent: str
    messages: list
    final_answer: str

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


class CruiseRAG:
    """RAG system combining SQL database queries with vector store retrieval"""

    def __init__(self, db_path: str = None, chroma_persist_dir: str = "./chroma_langchain_db"):
        """Initialize the RAG system

        Args:
            db_path: Path to SQLite database. If None, uses cruises.db in same directory
            chroma_persist_dir: Directory for Chroma vector store persistence
        """
        # Set up database path
        if db_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(script_dir, "cruises.db")

        # Initialize LLM
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile")
        # self.llm = init_chat_model("llama3.1", model_provider="ollama", temperature=0.1)

        # Initialize embeddings and vector store
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=self.embeddings,
            persist_directory=chroma_persist_dir,
        )

        # Initialize SQL database
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

        # Set up or load vector store
        self._setup_vector_store()

        # Set up system message and prompts
        self._setup_prompts()

        # Set up agent with tools
        self._setup_agent()

        # Create specialized agents
        # self.sql_agent = self._create_sql_agent()
        # self.chat_agent = self._create_chat_agent()
        # self.retrieval_agent = self._create_retrieval_agent()
        
        # # Create supervisor/router
        # self.supervisor = self._create_supervisor()
        
        # # Compile app
        # self.agent_executor = self.supervisor.compile(
        #     checkpointer=MemorySaver(), 
        #     store=InMemoryStore(),
        #     )

    def _setup_prompts(self):
        """Set up system prompts and templates"""
        self.system_message = """
        You are a friendly cruise booking assistant.
        Answer general questions, greetings, and casual conversation.
        Use the vector store to find relevant information from documents.
        Provide concise, accurate answers based on retrieved context.
        We want to keep our customers, so don't redirect them to another source, instead suggest to get in contact with an agent.

        If asked about specific cruise data, you are also an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most {top_k} results.

        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.

        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.

        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.

        Then you should query the schema of the most relevant tables.

        Never query for all the columns from a specific table, only ask for a the
        few relevant columns given the question.

        Pay attention to use only the column names that you can see in the schema
        description. Be careful to not query for columns that do not exist. Also,
        pay attention to which column is in which table.

        Only use the following tables:
        {table_info}
        """.format(
            dialect="SQLite",
            top_k=15,
            table_info=self.db.get_table_info(),
        )

        user_prompt = "Question: {input}"
        self.query_prompt_template = ChatPromptTemplate(
            [("system", self.system_message), ("user", user_prompt)]
        )

        # Supervisor/Router Prompt
        self.supervisor_prompt = """
        You are a routing supervisor for a cruise booking assistant.
        Analyze the user's query and decide which agent should handle it.
        
        Available agents:
        - sql_agent: For queries about cruise data, availability, prices, schedules
        - chat_agent: For greetings, general conversation, non-cruise questions
        - retrieval_agent: For general information about cruises from documents
        
        Respond with ONLY the agent name: sql_agent, chat_agent, or retrieval_agent
        """
        
        # SQL Agent Prompt
        self.sql_agent_prompt = """You are a SQL database expert agent with access to database tools.

        Your job is to:
        1. Use sql_db_list_tables to see available tables
        2. Use sql_db_schema to get table schemas for relevant tables
        3. Write a SQL query based on the schema
        4. Use sql_db_query_checker to validate your SQL query
        5. Use sql_db_query to EXECUTE the query and get results
        6. Answer the user's question based on the query results

        IMPORTANT: You must EXECUTE queries using sql_db_query tool, not just generate them!

        Guidelines:
        - ALWAYS check the schema first using sql_db_schema before writing queries
        - Limit queries to {top_k} results unless specified
        - Only query relevant columns, not all columns
        - Use WHERE clauses for filtering (e.g., WHERE month = 10 for October)
        - Validate queries with sql_db_query_checker before executing
        - NO DML statements (INSERT, UPDATE, DELETE, DROP)
        - If errors occur, fix the query and retry

        Database dialect: {dialect}

        Available tables:
        {table_info}
        """.format(
            dialect="SQLite",
            top_k=15,
            table_info=self.db.get_table_info(),
        )
        
        # Chat Agent Prompt
        self.chat_agent_prompt = """
        You are a friendly cruise booking assistant.
        Answer general questions, greetings, and casual conversation.
        Keep responses concise and helpful.
        If asked about specific cruise data, politely inform that you can help with that.
        """
        
        # Retrieval Agent Prompt
        self.retrieval_agent_prompt = """
        You are an information retrieval specialist for cruise information.
        Use the vector store to find relevant information from documents.
        Provide concise, accurate answers based on retrieved context.
        """

    def _setup_agent(self):
        """Set up the agent with tools"""
        # Create retrieve tool bound to this instance
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = self.vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        # Set up SQL toolkit and tools
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        tools = toolkit.get_tools()
        tools.append(retrieve)
        tools = ToolNode(tools)

        # Create memory and agent
        memory = MemorySaver()
        self.agent_executor = create_react_agent(
            self.llm,
            tools,
            prompt=self.system_message,
            checkpointer=memory
        )
    
    def _create_sql_agent(self):
        """SQL agent with database tools"""
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        tools = toolkit.get_tools()
        sql_agent = create_react_agent(
            self.llm,
            tools,
            name="SQLAgent",
            prompt=self.sql_agent_prompt,
        )
        return sql_agent

    def _create_chat_agent(self):
        """Simple chat agent without tools"""
        react_agent = create_react_agent(
            self.llm, 
            [], 
            name="ChatAgent",
            prompt=self.chat_agent_prompt, 
        )
        return react_agent

    def _create_retrieval_agent(self):
        """Agent with vector retrieval tool"""
        # Create retrieve tool bound to this instance
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = self.vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        react_agent = create_react_agent(
            self.llm, 
            [retrieve],
            name="RetrievalAgent",
            prompt=self.retrieval_agent_prompt
        )
        return react_agent

    def _create_supervisor(self):
        # Create supervisor workflow
        forwarding_tool = create_forward_message_tool("supervisor") # The argument is the name to assign to the resulting forwarded message
        supervisor = create_supervisor(
            [self.sql_agent, self.chat_agent, self.retrieval_agent],
            model=self.llm,
            prompt=self.supervisor_prompt,
            output_mode="full_history",  # Changed from "full_history" to "messages" to include all sub-agent tool calls
            tools=[forwarding_tool],
            name="SupervisorAgent",
        )
        return supervisor

    def _setup_vector_store(self):
        """Set up or load vector store"""

        script_dir = os.path.dirname(os.path.abspath(__file__))
        docs_path = os.path.join(script_dir, "documents")

        # Check if vector store already has documents (avoid re-indexing)
        try:
            collection = self.vector_store._collection
            doc_count = collection.count()

            if doc_count > 0:
                print(f"✅ Vector store already loaded with {doc_count} documents, skipping indexing")
                return
        except Exception as e:
            print(f"ℹ️ Could not check vector store status: {e}")

        # Only index if vector store is empty
        print("📄 Indexing documents into vector store...")

        if not os.path.exists(docs_path):
            print(f"⚠️ Documents folder not found at {docs_path}, skipping")
            return

        docs = []
        for filename in os.listdir(docs_path):
            file_path = os.path.join(docs_path, filename)
            if filename.endswith('.txt'):
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs.extend(loader.load())
                    print(f"  Loaded: {filename}")
                except Exception as e:
                    print(f"  ⚠️ Error loading {filename}: {e}")

        if docs:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_splits = text_splitter.split_documents(docs)

            # Index chunks
            _ = self.vector_store.add_documents(documents=all_splits)
            print(f"✅ Indexed {len(all_splits)} document chunks from {len(docs)} files")
        else:
            print("⚠️ No documents found to index")

    def query(self, question: str, thread_id: str = "default") -> str:
        """Query the RAG system

        Args:
            question: User's question
            thread_id: Thread ID for conversation memory

        Returns:
            Answer as a string
        """
        config = {"configurable": {"thread_id": thread_id}}

        final_response = ""
        for event in self.agent_executor.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="messages",
            config=config,
        ):
            # Extract message from event
            if isinstance(event, tuple):
                message = event[0]
                metadata = event[1] if len(event) > 1 else {}
            else:
                message = event
                metadata = {}

            # Capture final response from sub-agents only
            if hasattr(message, 'type') and message.type == "ai":
                agent_name = metadata.get('langgraph_node', 'supervisor')
                if agent_name != 'supervisor' and hasattr(message, 'content') and message.content:
                    final_response = message.content

        return final_response if final_response else "No response generated"

    def query_stream(self, question: str, thread_id: str = "default"):
        """Stream query responses

        Args:
            question: User's question
            thread_id: Thread ID for conversation memory

        Yields:
            Message events from the agent
        """
        config = {"configurable": {"thread_id": thread_id}}

        for event in self.agent_executor.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode=["values", "messages"],
            config=config,
        ):
            yield event

    def get_table_info(self) -> str:
        """Get database table information"""
        return self.db.get_table_info()


def main():
    """Interactive CLI for CruiseRAG system"""
    print("🚀 Starting CruiseRAG System...")

    try:
        # Initialize RAG system
        rag_system = CruiseRAG()
        print("✅ System initialized successfully!")
        print("\n💡 Ask questions about cruises or type 'quit' to exit.\n")

        # Interactive loop
        while True:
            question = input("❓ Your question: ")

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not question.strip():
                continue

            try:
                # Stream responses and show agent steps
                print("\n🤖 Processing...")
                for event in rag_system.query_stream(question):
                    # Extract message from event
                    if event[0] == "values":
                        message = event[1]['messages'][-1]  # Latest message
                    elif event[0] == "messages":
                        message = event[1][0]

                    # Show agent name and message
                    if hasattr(message, 'pretty_print'):
                        message.pretty_print()
                    else:
                        print(message)

                print("\n" + "-" * 80 + "\n")

            except Exception as e:
                print(f"❌ Error: {e}")
                print(f"\nStack trace:")
                traceback.print_exc()
                print()

    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print(f"\nStack trace:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
