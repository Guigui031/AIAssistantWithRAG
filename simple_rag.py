import os
import langchain
from unittest import result
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document
import pandas as pd

langchain.debug = True 
def create_readable_text_from_row(row):
    """
    Convert a single CSV row into a natural language description
    """
    # Customize this based on your CSV structure
    # This example assumes columns: Name, HEX, RGB
    description_parts = []
    for column_name, value in row.items():
        if pd.notna(value):  # Only include non-empty values
            description_parts.append(f"{column_name}: {value}")
    # Join everything into one readable sentence
    return ". ".join(description_parts) + "."


# Load environment variables
load_dotenv()
def load_documents(directory_path):
    """Load all text and CSV files from a directory"""
    documents = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if filename.endswith('.txt'):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            
        elif filename.endswith('.csv'):
            # loader = CSVLoader(file_path)
            # data = loader.load()
            # documents.extend(data)
            
            df = pd.read_csv(file_path)
            # Group rows by year and month
            for (year, month), group in df.groupby(["year", "month"]):
                # Convert all rows in the group into text
                rows_text = [create_readable_text_from_row(row) for _, row in group.iterrows()]
                
                # Join them into a single "document"
                combined_text = "\n".join(rows_text)
                
                # Create Document with metadata
                doc = Document(
                    page_content=combined_text,
                    metadata={"year": int(year), "month": int(month)}
                )
                documents.append(doc)
            
    print(f"Loaded {len(documents)} documents")
    return documents


def chunk_documents(documents):
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Each chunk ~1000 characters
        chunk_overlap=200,      # 200 characters overlap for context
        length_function=len,    # Measure by character count
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store_groq(chunks):
    """Create embeddings using HuggingFace and store in Chroma"""
    # Use a small, fast sentence transformer model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_groq_db"
    )
    
    print("Vector store created successfully")
    return vectorstore


def setup_qa_chain_groq(vectorstore):
    """Set up QA chain using Groq's LLaMA model"""
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",  # Fast and capable
        temperature=0.1               # Low temperature for consistent answers
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )
    
    return qa_chain


def main():
    print("🚀 Starting RAG System Setup...")
    
    # Step 1: Load documents
    print("📄 Loading documents...")
    documents = load_documents("./documents")
    
    if not documents:
        print("❌ No documents found! Add some .txt or .csv files to the ./documents folder")
        return
    
    # Step 2: Chunk documents  
    print("✂️  Chunking documents...")
    chunks = chunk_documents(documents)
    
    # Step 3: Create vector store (choose your version)
    print("🧠 Creating embeddings and vector store...")
    # For Groq version:
    vectorstore = create_vector_store_groq(chunks)
    # For OpenAI version:
    # vectorstore = create_vector_store_openai(chunks)
    
    # Step 4: Set up QA chain
    print("🔗 Setting up QA chain...")
    # For Groq version:
    qa_chain = setup_qa_chain_groq(vectorstore)
    # For OpenAI version:
    # qa_chain = setup_qa_chain_openai(vectorstore)
    
    # Step 5: Interactive Q&A
    print("\n✅ RAG System Ready! Ask me anything about your documents.")
    print("💡 Type 'quit' to exit.\n")
    
    while True:
        question = input("❓ Your question: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not question.strip():
            continue
            
        try:
            result = qa_chain.invoke({"query": question})
            print(f"\n💬 Answer: {result['result']}\n")
            
            # Show source documents for transparency
            print("📚 Sources:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"{i}. {doc.page_content[:150]}...")
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try rephrasing your question.\n")

if __name__ == "__main__":
    main()