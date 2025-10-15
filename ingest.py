import os
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
DATA_PATH = "data/"
VECTOR_STORE_PATH = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"  # Smaller, faster model

def create_vector_db():
    """
    Processes PDF documents in the data directory, splits them into chunks,
    generates embeddings, and saves them to a Chroma vector store.
    """
    print("Starting data ingestion process...")

    # 1. Load documents from the specified directory
    try:
        print(f"Loading documents from '{DATA_PATH}'...")
        loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            print(f"No PDF files found in '{DATA_PATH}'. Please add your NCERT textbooks to this folder.")
            return
        print(f"Loaded {len(documents)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # 2. Split the documents into smaller chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    # 3. Initialize the embedding model
    print(f"Initializing embedding model: {MODEL_NAME}...")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"} # Use CPU for broader compatibility
        )
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return

    # 4. Create the Chroma vector store from the text chunks and embeddings
    print("Creating Chroma vector store... (This may take a while depending on the number of documents)")
    try:
        # Remove existing database if it exists
        if os.path.exists(VECTOR_STORE_PATH):
            import shutil
            shutil.rmtree(VECTOR_STORE_PATH)
        
        # Create Chroma vector store
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_PATH
        )
        
        print("Chroma vector store created successfully.")
    except Exception as e:
        print(f"Error creating Chroma vector store: {e}")
        return

    # 5. Persist the vector store
    try:
        vectorstore.persist()
        print(f"Chroma vector store saved to '{VECTOR_STORE_PATH}'.")
        print("\nIngestion complete! You can now run 'streamlit run streamlit_app.py' to start the application.")
    except Exception as e:
        print(f"Error saving Chroma vector store: {e}")


if __name__ == "__main__":
    # Check if the data directory exists, create it if it doesn't
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created directory '{DATA_PATH}'. Please add your NCERT textbook PDFs to this folder and run the script again.")
    else:
        create_vector_db()

