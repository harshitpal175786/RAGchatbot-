import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import re
from tqdm import tqdm

# Constants
DATA_PATH = "data"
DB_PATH = "vectorstores/db/"

# Move clean_text function to global scope
def clean_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
    # Normalize spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    return text.strip()

def build_or_update_vector_db():
    # Check if database already exists
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print("Loading existing database...")
        db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        )
        print("Existing database loaded successfully!")
        
        # Get current document count
        current_count = db._collection.count()
        print(f"Current documents in DB: {current_count}")
        
        # Load new documents
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        new_documents = loader.load()
        
        # Process new documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        )
        new_chunks = text_splitter.split_documents(new_documents)
        
        # Clean new chunks
        for doc in new_chunks:
            doc.page_content = clean_text(doc.page_content)
        
        # Add new chunks to existing database
        print(f"Adding {len(new_chunks)} new chunks...")
        db.add_documents(new_chunks)
        # Remove the persist() call since it's deprecated
        
        print(f"Database updated! Total documents: {db._collection.count()}")
        
        # Test the updated database
        test_database(db)
        
    else:
        print("Creating new database...")
        # Your existing code for first-time creation
        create_new_database()

def test_database(db):
    """Test the database with queries"""
    print("\nTesting the retriever...")
    # Configure retriever with advanced options
    retriever = db.as_retriever(
        search_type="mmr",           # Maximum Marginal Relevance
        search_kwargs={
            "k": 8,                  # Retrieve more candidates
            "fetch_k": 20,           # Fetch more for diversity
            "lambda_mult": 0.7       # Balance relevance vs diversity
        }
    )
    sample_query = "What is the main topic of the documents?"
    retrieved_docs = retriever.invoke(sample_query)

    print(f"Query: '{sample_query}'")
    print(f"Retrieved {len(retrieved_docs)} documents.")
    print("\n--- All Retrieved Content ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Document {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(doc.page_content)
        print("-" * 50)

    # Test with more sophisticated queries
    advanced_queries = [
        "Explain the difference between paging and segmentation with examples",
        "What are the trade-offs in CPU scheduling algorithms?",
        "How does virtual memory solve memory management problems?",
        "Compare and contrast different deadlock prevention strategies"
    ]

    for query in tqdm(advanced_queries, desc="Testing queries"):
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        # Add timing and performance metrics
        import time
        start_time = time.time()
        try:
            retrieved_docs = retriever.invoke(query)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            continue
        end_time = time.time()

        print(f"Retrieved {len(retrieved_docs)} documents in {end_time - start_time:.2f}s")
        print(f"Average document length: {sum(len(doc.page_content) for doc in retrieved_docs) / len(retrieved_docs):.0f} chars")
        
        for i, doc in enumerate(retrieved_docs[:3]):  # Show top 3
            print(f"\n--- Result {i+1} ---")
            print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)

    # Save results for analysis
    with open("retrieval_results.txt", "w") as f:
        f.write(f"Vector DB Test Results\n{'='*50}\n")
        # ... save all results

def create_new_database():
    # Your existing database creation code here
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    # Better text splitting with semantic boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,           # Optimal for most LLMs
        chunk_overlap=80,         # Better context preservation
        separators=[
            "\n\n",               # Paragraph breaks
            "\n",                 # Line breaks  
            ". ",                 # Sentence endings
            "? ",                 # Question endings
            "! ",                 # Exclamation endings
            "; ",                 # Semicolon breaks
            ", ",                 # Comma breaks
            " ",                  # Word boundaries
            ""                    # Character level (fallback)
        ]
    )
    chunked_documents = text_splitter.split_documents(documents)

    # Apply cleaning to all chunks
    for doc in chunked_documents:
        doc.page_content = clean_text(doc.page_content)

    # --- 2. Create Embeddings ---
    # We use a powerful, open-source embedding model
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- 3. Build and Persist the Vector Store ---
    print("Building vector store... This may take a moment.")
    db = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    print(f"Vector store created successfully at {DB_PATH}")

    # Test the new database
    test_database(db)

if __name__ == "__main__":
    build_or_update_vector_db()
