import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase
from langgraph.graph import MessagesState, StateGraph
from typing_extensions import TypedDict
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from typing import Dict, Any
# from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


load_dotenv()


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


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
        # self.llm = ChatGroq(model_name="llama-3.3-70b-versatile")
        self.llm = init_chat_model("llama3.1", model_provider="ollama")

        # Initialize embeddings and vector store
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=self.embeddings,
            persist_directory=chroma_persist_dir,
        )

        # Initialize SQL database
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

        # Set up system message and prompts
        self._setup_prompts()

        # Create agent executor
        self._setup_agent()

    def _setup_prompts(self):
        """Set up system prompts and templates"""
        self.system_message = """
        You are an agent designed to interact with a SQL database.
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
            top_k=5,
            table_info=self.db.get_table_info(),
        )

        user_prompt = "Question: {input}"
        self.query_prompt_template = ChatPromptTemplate(
            [("system", self.system_message), ("user", user_prompt)]
        )

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

        self.config = {"configurable": {"thread_id": "default"}}

    def _setup_vector_store(self):
        """Set up or load vector store"""
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        docs_path = os.path.join(script_dir, "documents")
        
        # Vector store is initialized in __init__, so this can be used for additional setup if needed
        docs = []  # Load or define documents here
        for filename in os.listdir(docs_path):
            file_path = os.path.join(docs_path, filename)
            if filename.endswith('.txt'):
                loader = TextLoader(file_path)
                docs.extend(loader.load())

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        
        # Index chunks
        _ = vector_store.add_documents(documents=all_splits)

    def query(self, question: str, thread_id: str = "default") -> str:
        """Query the RAG system

        Args:
            question: User's question
            thread_id: Thread ID for conversation memory

        Returns:
            Answer as a string
        """
        config = {"configurable": {"thread_id": thread_id}}

        messages = []
        for event in self.agent_executor.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
            config=config,
        ):
            messages.append(event["messages"][-1])

        # Return the last AI message content
        if messages:
            last_message = messages[-1]
            return last_message.content if hasattr(last_message, 'content') else str(last_message)
        return "No response generated"

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
            stream_mode="values",
            config=config,
        ):
            yield event["messages"][-1]

    def get_table_info(self) -> str:
        """Get database table information"""
        return self.db.get_table_info()


# Legacy functions for backward compatibility
def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}


def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


# History
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


# message
def main():
    global db, llm, vector_store, query_prompt_template, system_message

    # Global variables
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    # llm = init_chat_model("claude-sonnet-4-20250514")
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

    # SQL
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "cruises.db")
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")


    system_message = """
    You are an agent designed to interact with a SQL database.
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
        top_k=10,
        table_info=db.get_table_info(),
    )

    user_prompt = "Question: {input}"

    query_prompt_template = ChatPromptTemplate(
        [("system", system_message), ("user", user_prompt)]
    )

    # Agents
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    tools = toolkit.get_tools()
    tools.append(retrieve)
    tools = ToolNode(tools)
    
    memory = MemorySaver()
    
    agent_executor = create_react_agent(llm, tools, prompt=system_message, checkpointer=memory)

    # Specify an ID for the thread    

    config = {"configurable": {"thread_id": "abc123"}}

    input_message = (
        "What are the cruises available in october?\n\n"
        "Once you get the answer, what are the ones available in 2025?"
    )
    
    for event in agent_executor.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        event["messages"][-1].pretty_print() 
    

if __name__ == "__main__":
    main()
