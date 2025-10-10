"""
Structured RAG System for CSV Data
Based on the Medium article approach: storing metadata, schema, and relationships
to enable accurate querying over structured data.
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import langchain

langchain.debug = True

load_dotenv()

class StructuredRAG:
    def __init__(self, csv_path, persist_directory="./structured_chroma_db"):
        self.csv_path = csv_path
        self.persist_directory = persist_directory
        self.df = pd.read_csv(csv_path)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)

    def extract_metadata(self):
        """
        Extract schema metadata: column names, types, sample values
        """
        metadata = {
            "table_name": "celebrity_cruises",
            "description": "Celebrity Cruises booking data with pricing, dates, and locations",
            "columns": [],
            "sample_rows": []
        }

        # Extract column information
        for col in self.df.columns:
            col_info = {
                "name": col,
                "type": str(self.df[col].dtype),
                "nullable": self.df[col].isnull().any(),
                "unique_count": self.df[col].nunique(),
                "sample_values": self.df[col].dropna().unique()[:5].tolist()
            }
            metadata["columns"].append(col_info)

        # Extract 5 sample rows
        sample_df = self.df.head(5)
        metadata["sample_rows"] = sample_df.to_dict(orient='records')

        return metadata

    def create_knowledge_documents(self, metadata):
        """
        Create documents for the knowledge base:
        1. Overall table metadata
        2. Individual column descriptions
        3. Sample data patterns
        """
        documents = []

        # Document 1: Overall table description
        table_desc = f"""
Table: {metadata['table_name']}
Description: {metadata['description']}
Total Columns: {len(metadata['columns'])}
Column Names: {', '.join([c['name'] for c in metadata['columns']])}

This table contains cruise booking information including:
- Destinations (Product, City of departure)
- Timing (Sail Date, year, month, day, Sail Nights)
- Ship information (Ship name)
- Pricing (Price, Group Fare, Taxes per guest, Currency)
- Cabin details (Cabin Category, Cabins Requested)
- Group information (Group ID, Group Name)
"""
        documents.append(Document(
            page_content=table_desc,
            metadata={"type": "table_overview", "table_name": metadata['table_name']}
        ))

        # Document 2: Column-specific details
        for col in metadata['columns']:
            col_desc = f"""
Column: {col['name']}
Type: {col['type']}
Nullable: {col['nullable']}
Unique Values: {col['unique_count']}
Sample Values: {', '.join(map(str, col['sample_values']))}

Context: In the celebrity_cruises table, {col['name']} represents {"" if col['name'] in ['Product', 'City of departure', 'Ship', 'Sail Date', 'month', 'year', 'day'] else "booking "}information.
"""
            if col['name'] == 'month':
                col_desc += "\nNote: Month values are numeric (1-12). Use this to filter by month."
            elif col['name'] == 'Sail Date':
                col_desc += "\nNote: Contains full date information in string format."
            elif col['name'] == 'Product':
                col_desc += "\nNote: Represents cruise destinations/regions."

            documents.append(Document(
                page_content=col_desc,
                metadata={"type": "column_info", "column_name": col['name'], "table_name": metadata['table_name']}
            ))

        # Document 3: Sample data representation
        sample_desc = f"""
Sample Cruise Records from celebrity_cruises table:

"""
        for i, row in enumerate(metadata['sample_rows'], 1):
            sample_desc += f"\nRecord {i}:\n"
            for key, val in row.items():
                if key in ['Product', 'City of departure', 'Ship', 'Sail Date', 'month', 'year', 'Price', 'Cabin Category']:
                    sample_desc += f"  - {key}: {val}\n"

        documents.append(Document(
            page_content=sample_desc,
            metadata={"type": "sample_data", "table_name": metadata['table_name']}
        ))

        # Document 4: Query patterns and relationships
        query_patterns = f"""
Query Patterns for celebrity_cruises table:

To find cruises by month:
- Use the 'month' column (numeric: 1-12)
- Example: month == 10 for October

To find cruises by destination:
- Use the 'Product' column for region (e.g., 'Europe', 'Asia', 'Long Caribbean')
- Use the 'City of departure' column for specific cities

To find cruises by price range:
- Use the 'Price' column (note: some values may be 'GROUPX' or 'STANDARD')

To find cruises by date:
- Use 'year', 'month', 'day' columns for filtering
- Or use 'Sail Date' column for full date

Available regions: {', '.join(self.df['Product'].dropna().unique().tolist())}
Available departure cities: {', '.join(self.df['City of departure'].dropna().unique()[:10].tolist())}
"""
        documents.append(Document(
            page_content=query_patterns,
            metadata={"type": "query_patterns", "table_name": metadata['table_name']}
        ))

        return documents

    def build_knowledge_base(self):
        """
        Build the vector store with structured knowledge
        """
        print("📊 Extracting metadata from CSV...")
        metadata = self.extract_metadata()

        print("📝 Creating knowledge documents...")
        documents = self.create_knowledge_documents(metadata)

        print(f"💾 Building vector store with {len(documents)} documents...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        print("✅ Knowledge base created successfully!")
        return vectorstore

    def query_with_structured_context(self, user_query, vectorstore):
        """
        Query the system using structured context retrieval
        """
        # Step 1: Retrieve relevant schema/metadata
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(user_query)

        # Step 2: Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Step 3: Create a prompt that uses the schema to generate a filtering strategy
        prompt_template = """You are a data assistant helping users query cruise booking data.

Retrieved Schema Context:
{context}

User Question: {question}

Based on the schema context above, provide a helpful answer. If the question requires filtering data:
1. Explain what columns to use
2. Provide the specific filter conditions
3. Give a sample result if possible

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(context=context, question=user_query)

        return response, relevant_docs

    def execute_pandas_query(self, filter_condition):
        """
        Execute a pandas query on the dataframe
        Example: "month == 10" or "Product == 'Europe'"
        """
        try:
            result = self.df.query(filter_condition)
            return result
        except Exception as e:
            return f"Error executing query: {e}"


def main():
    print("🚀 Starting Structured RAG System...")

    # Initialize the system
    csv_path = "./documents/celebrity-cruises.csv"
    rag = StructuredRAG(csv_path)

    # Build knowledge base
    vectorstore = rag.build_knowledge_base()

    # Interactive Q&A
    print("\n✅ Structured RAG System Ready!")
    print("💡 Ask questions about the cruise data (e.g., 'suggest cruises in October')")
    print("💡 Type 'quit' to exit.\n")

    while True:
        question = input("❓ Your question: ")

        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not question.strip():
            continue

        try:
            # Get AI response with structured context
            response, relevant_docs = rag.query_with_structured_context(question, vectorstore)

            print(f"\n💬 Answer:\n{response}\n")

            # Show retrieved schema context
            print("📚 Retrieved Schema Context:")
            for i, doc in enumerate(relevant_docs, 1):
                print(f"{i}. Type: {doc.metadata.get('type', 'N/A')}")
                print(f"   Preview: {doc.page_content[:150]}...\n")

            # If user wants actual data, they can ask for filtering
            if "month" in question.lower() and any(str(i) in question for i in range(1, 13)):
                # Try to extract month number
                for i in range(1, 13):
                    if str(i) in question or [
                        "january", "february", "march", "april", "may", "june",
                        "july", "august", "september", "october", "november", "december"
                    ][i-1] in question.lower():
                        print(f"\n🔍 Fetching actual cruises for month {i}...")
                        results = rag.df.query(f"month == {i}")
                        print(f"Found {len(results)} cruises:")
                        if len(results) > 0:
                            print(results[['Product', 'City of departure', 'Sail Date', 'Price', 'Ship']].head(10).to_string())
                        break

            print("-" * 80)

        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try rephrasing your question.\n")


if __name__ == "__main__":
    main()