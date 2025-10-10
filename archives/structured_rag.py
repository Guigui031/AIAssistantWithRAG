import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import json
from typing import List, Dict, Any
import re
from datetime import datetime
import langchain

load_dotenv()
langchain.debug = True
langchain.allow_dangerous_code=True


class StructuredRAG:
    def __init__(self, csv_path: str):
        """Initialize the Structured RAG system with cruise data"""
        self.df = pd.read_csv(csv_path)
        self.df['num_date'] = pd.to_datetime(self.df['num_date'])
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self._setup_metadata_store()
        self._setup_pandas_agent()

    def _setup_metadata_store(self):
        """Create a vector store with rich metadata - one doc per unique cruise (not cabin)"""
        documents = []

        # Fill NaN values before grouping to avoid issues with missing data
        df_clean = self.df.copy()
        df_clean['City of departure'] = df_clean['City of departure'].fillna('Unknown')
        df_clean['Product'] = df_clean['Product'].fillna('Unknown')

        # Group by unique cruise (same product, departure, ship, sail date)
        grouped = df_clean.groupby(['Product', 'City of departure', 'Ship', 'Sail Date', 'num_date'])

        for (product, city, ship, sail_date, num_date), group in grouped:
            # Create a description that mentions available cabin categories
            content = self._create_cruise_description_grouped(group)

            # Use metadata from first row, but add cabin variety info
            first_row = group.iloc[0]

            # Extract rich metadata with proper handling of missing values
            metadata = {
                'product': str(product) if product else 'Unknown',
                'departure_city': str(city) if city else 'Unknown',
                'ship': str(ship) if ship else 'Unknown',
                'sail_date': str(sail_date),
                'num_date': num_date.strftime('%Y-%m-%d'),
                'year': int(first_row['year']),
                'month': int(first_row['month']),
                'day': int(first_row['day']),
                'month_name': num_date.strftime('%B'),
                'sail_nights': int(first_row['Sail Nights']),
                'currency': str(first_row['Currency']) if pd.notna(first_row['Currency']) else 'CAD',
                'cabin_categories': ', '.join(group['Cabin Category'].dropna().unique().astype(str)),
                'num_cabin_types': len(group['Cabin Category'].dropna().unique()),
                'group_name': str(first_row['Group Name']) if pd.notna(first_row['Group Name']) else 'Unknown'
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_structured_db"
        )
        print(f"Created vector store with {len(documents)} unique cruise documents (grouped from {len(self.df)} rows)")

    def _setup_pandas_agent(self):
        """Set up pandas agent for complex queries"""
        self.pandas_agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            allow_dangerous_code=True
        )

    def _create_cruise_description(self, row):
        """Create a rich natural language description of a cruise"""
        try:
            fare = str(row['Group Fare*']).replace(',', '')
            advantage = str(row['Group Fare Advantage']).replace(',', '')

            description = f"""
            {row['Product']} cruise departing from {row['City of departure']} on ship {row['Ship']}.
            Sail date: {row['Sail Date']} ({row['num_date'].strftime('%B %Y')}).
            Duration: {row['Sail Nights']} nights.
            Cabin category: {row['Cabin Category']}.
            Group fare: {fare} {row['Currency']} with advantage of {advantage} {row['Currency']}.
            {row['Cabins Requested']} cabins requested for group {row['Group Name']}.
            """
            return description.strip()
        except Exception as e:
            return f"Cruise: {row['Product']} from {row['City of departure']} on {row['Sail Date']}"

    def _create_cruise_description_grouped(self, group_df):
        """Create a description for a cruise with multiple cabin options"""
        try:
            first_row = group_df.iloc[0]

            # Get price range across all cabin types
            fares = []
            for _, row in group_df.iterrows():
                fare_str = str(row['Group Fare*']).replace(',', '')
                if fare_str.replace('.', '').replace('-', '').isdigit():
                    fares.append(float(fare_str))

            price_info = ""
            if fares:
                min_fare = min(fares)
                max_fare = max(fares)
                if min_fare == max_fare:
                    price_info = f"Pricing: {min_fare:.0f} {first_row['Currency']}"
                else:
                    price_info = f"Pricing: {min_fare:.0f} - {max_fare:.0f} {first_row['Currency']}"

            # List cabin categories
            cabin_categories = group_df['Cabin Category'].unique()
            cabin_info = f"Available cabin categories: {', '.join(cabin_categories)}"

            description = f"""
            {first_row['Product']} cruise departing from {first_row['City of departure']} on ship {first_row['Ship']}.
            Sail date: {first_row['Sail Date']} ({first_row['num_date'].strftime('%B %Y')}).
            Duration: {first_row['Sail Nights']} nights.
            {cabin_info}.
            {price_info}.
            """
            return description.strip()
        except Exception as e:
            first_row = group_df.iloc[0]
            return f"Cruise: {first_row['Product']} from {first_row['City of departure']} on {first_row['Sail Date']}"

    def _extract_date_filters(self, query: str) -> Dict[str, Any]:
        """Extract date-related filters from natural language query"""
        filters = {}
        query_lower = query.lower()

        # Month extraction
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }

        for month_name, month_num in months.items():
            if month_name in query_lower:
                filters['month'] = month_num
                break

        # Year extraction
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            filters['year'] = int(year_match.group(1))

        # Duration extraction
        night_match = re.search(r'(\d+)\s*night', query_lower)
        if night_match:
            filters['sail_nights'] = int(night_match.group(1))

        return filters

    def _extract_other_filters(self, query: str) -> Dict[str, Any]:
        """Extract non-date filters from query"""
        filters = {}
        query_lower = query.lower()

        # Product/region extraction
        if 'europe' in query_lower:
            filters['product'] = 'Europe'
        elif 'caribbean' in query_lower:
            filters['product'] = '7n Caribbean'
        elif 'asia' in query_lower:
            filters['product'] = 'Asia'

        # Departure city extraction
        cities = ['amsterdam', 'reykjavik', 'southampton', 'tokyo', 'sydney', 'fort lauderdale']
        for city in cities:
            if city in query_lower:
                filters['departure_city_contains'] = city.title()
                break

        return filters

    def search_with_filters(self, query: str, k: int = 10) -> List[Document]:
        """Search with metadata filtering based on query analysis"""
        # Extract filters from query
        date_filters = self._extract_date_filters(query)
        other_filters = self._extract_other_filters(query)

        # Combine filters
        all_filters = {**date_filters, **other_filters}

        print(f"Extracted filters: {all_filters}")

        # Create metadata filter for Chroma using $and operator
        where_filter = None
        filter_conditions = []

        for key, value in all_filters.items():
            if key == 'departure_city_contains':
                # Handle partial matches in post-processing
                continue
            else:
                filter_conditions.append({key: {"$eq": value}})

        # Build the where clause properly
        if len(filter_conditions) == 1:
            where_filter = filter_conditions[0]
        elif len(filter_conditions) > 1:
            where_filter = {"$and": filter_conditions}

        # Search with filters - get more when filtering to ensure we capture all matches
        fetch_k = k * 3 if where_filter else k * 2

        if where_filter:
            results = self.vectorstore.similarity_search(
                query,
                k=fetch_k,
                filter=where_filter
            )
        else:
            results = self.vectorstore.similarity_search(query, k=fetch_k)

        # Apply partial string matching filters
        if 'departure_city_contains' in all_filters:
            city_filter = all_filters['departure_city_contains'].lower()
            results = [doc for doc in results
                      if city_filter in doc.metadata.get('departure_city', '').lower()]

        print(f"Found {len(results)} matching cruises after filtering")
        return results[:k]

    def query_with_pandas(self, query: str) -> str:
        """Use pandas agent for complex analytical queries"""
        try:
            # Extract filters and do direct pandas operations
            date_filters = self._extract_date_filters(query)
            other_filters = self._extract_other_filters(query)

            # Start with full dataframe
            filtered_df = self.df.copy()

            # Apply filters
            if 'month' in date_filters:
                filtered_df = filtered_df[filtered_df['month'] == date_filters['month']]
            if 'year' in date_filters:
                filtered_df = filtered_df[filtered_df['year'] == date_filters['year']]
            if 'product' in other_filters:
                filtered_df = filtered_df[filtered_df['Product'] == other_filters['product']]
            if 'departure_city_contains' in other_filters:
                city = other_filters['departure_city_contains']
                filtered_df = filtered_df[filtered_df['City of departure'].str.contains(city, case=False, na=False)]
            if 'sail_nights' in date_filters:
                filtered_df = filtered_df[filtered_df['Sail Nights'] == date_filters['sail_nights']]

            # Generate summary statistics
            if len(filtered_df) > 0:
                result = f"Found {len(filtered_df)} matching cruises.\n"
                result += f"Destinations: {', '.join(filtered_df['Product'].dropna().unique().astype(str).tolist())}\n"
                result += f"Departure cities: {', '.join(filtered_df['City of departure'].dropna().unique().astype(str).tolist())}\n"
                result += f"Ships: {', '.join(filtered_df['Ship'].dropna().unique().astype(str).tolist())}\n"
                result += f"Date range: {filtered_df['Sail Date'].min()} to {filtered_df['Sail Date'].max()}\n"

                # Price analysis
                result += f"\nPrice analysis:\n"
                result += f"- Average cabins requested: {filtered_df['Cabins Requested'].mean():.1f}\n"
                if len(filtered_df['Cabin Category'].mode()) > 0:
                    result += f"- Most common cabin category: {filtered_df['Cabin Category'].mode().iloc[0]}\n"

                return result
            else:
                return "No cruises found matching the criteria."

        except Exception as e:
            return f"Error with pandas analysis: {str(e)}"

    def hybrid_search(self, query: str, max_results: int = 10) -> str:
        """Combine vector search with structured data analysis"""
        print(f"\n🔍 Processing query: '{query}'")

        # Step 1: Try vector search with filters
        print("\n📊 Step 1: Semantic search with metadata filtering...")
        vector_results = self.search_with_filters(query, k=max_results)

        # Step 2: Use pandas agent for analytical queries
        print("\n🐼 Step 2: Pandas agent analysis...")
        pandas_result = self.query_with_pandas(query)

        # Step 3: Combine results using LLM
        print("\n🤖 Step 3: Synthesizing results...")

        # Deduplicate vector results by complete cruise identifier
        seen_cruises = set()
        unique_results = []
        for doc in vector_results:
            # Create unique identifier: product-departure_city-ship-sail_date-sail_nights
            cruise_id = (
                doc.metadata.get('product', ''),
                doc.metadata.get('departure_city', ''),
                doc.metadata.get('ship', ''),
                doc.metadata.get('sail_date', ''),
                doc.metadata.get('sail_nights', '')
            )
            if cruise_id not in seen_cruises:
                seen_cruises.add(cruise_id)
                unique_results.append(doc)

        if len(unique_results) < len(vector_results):
            print(f"   Deduplicated from {len(vector_results)} to {len(unique_results)} unique cruises")
        else:
            print(f"   Found {len(unique_results)} unique cruises to present")

        # Prepare context from vector search - show full descriptions
        vector_context = "\n\n".join([
            f"Option {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(unique_results)
        ])

        # Create synthesis prompt
        synthesis_prompt = PromptTemplate(
            input_variables=["query", "vector_context", "pandas_result"],
            template="""
            Based on the following information about cruises, provide a comprehensive answer to the user's question.

            User Question: {query}

            Available Cruise Options:
            {vector_context}

            Summary Statistics:
            {pandas_result}

            IMPORTANT: Each "Option" above represents ONE complete cruise itinerary.
            When you see "Available cabin categories: A1, C2, SV", these are different cabin types available on that SAME cruise.

            Please provide a helpful answer that:
            - Presents each cruise option clearly with its key details (destination, ship, date, departure city)
            - Mentions the cabin categories available for each cruise
            - Includes pricing information when available
            - Uses the summary statistics to provide additional context about the overall results
            - Lists all the unique cruise options shown above
            """
        )

        chain = LLMChain(llm=self.llm, prompt=synthesis_prompt)

        final_answer = chain.run(
            query=query,
            vector_context=vector_context,
            pandas_result=pandas_result
        )

        return final_answer

    def interactive_query(self):
        """Start interactive query session"""
        print("\n✅ Structured RAG System Ready!")
        print("💡 Try queries like:")
        print("   - 'Show me cruises in October 2025'")
        print("   - 'What are the cheapest European cruises?'")
        print("   - 'Suggest 7-night cruises from Amsterdam'")
        print("   - 'Compare prices for different cabin categories'")
        print("💡 Type 'quit' to exit.\n")

        while True:
            query = input("❓ Your question: ")

            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not query.strip():
                continue

            try:
                # Default to showing up to 10 results, can be increased
                answer = self.hybrid_search(query, max_results=20)
                print(f"\n💬 Answer:\n{answer}\n")
                print("-" * 80)

            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try rephrasing your question.\n")

def main():
    print("🚀 Starting Structured RAG System...")

    # Initialize with cruise data
    csv_path = "./documents/celebrity-cruises.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    try:
        rag_system = StructuredRAG(csv_path)
        rag_system.interactive_query()
    except Exception as e:
        print(f"❌ Error initializing system: {e}")

if __name__ == "__main__":
    main()