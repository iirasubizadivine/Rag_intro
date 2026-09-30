import os
import shutil
import time
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"Directory {docs_path} does not exist, "
            "please create it and add your documents."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in {docs_path}, please add your documents."
        )

    for i, doc in enumerate(documents[:2]):
        print(f"Document {i+1}: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=200):
    #split documents into chunks
    print(f"Splitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    
    if chunks:
        for i, chunk in enumerate(chunks[:2]):
            print(f"Chunk {i+1}: {chunk.metadata['source']}")
            print(f"Content length: {len(chunk.page_content)} characters")
            print(f"Content: {chunk.page_content}")
            print(f"_" * 50)
            
            if len(chunks) > 5:
                print(f"\n... and {len(chunks) - 5} more chunks")
                
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and storing in ChromaDB...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        max_retries=10,
    )

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Cleared existing vector store at {persist_directory}")

    os.makedirs(persist_directory, exist_ok=True)

    batch_size = 50
    total = len(chunks)

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  Processed {min(i + batch_size, total)}/{total} chunks")
        if i + batch_size < total:
            time.sleep(65)

    print("Finished creating vector store")
    print(f"Persisted vector store to {persist_directory}...")
    return vectorstore
    

load_dotenv()


def main():
    print("Main function initialized.")
    documents = load_documents(docs_path="docs")
    
    chunks = split_documents(documents)
    
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()
