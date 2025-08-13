import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup the path to your data directory
DATA_PATH = "data"
loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # The number of characters in each chunk
    chunk_overlap=100 # The number of characters to overlap between chunks
)

# Split the documents into chunks
chunked_documents = text_splitter.split_documents(documents)

print(f"Loaded {len(documents)} document(s).")
print(f"Split into {len(chunked_documents)} chunks.")

# You can print a sample chunk to see the result
# print("\n--- Sample Chunk ---")
# print(chunked_documents[0].page_content)
# print("--------------------")