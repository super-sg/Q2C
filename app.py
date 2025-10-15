import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# --- INITIALIZE FLASK APP ---
app = Flask(__name__)
CORS(app) # Allow cross-origin requests

# --- CONFIGURATION ---
VECTOR_STORE_PATH = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"  # Smaller, faster model
GEMINI_MODEL_NAME = "gemini-1.0-pro"

# --- LOAD MODELS AND VECTOR STORE (GLOBAL) ---
# This section runs only once when the server starts.
print("Loading models and Chroma vector store for the backend...")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # Load Chroma vector store
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(f"Chroma database not found at {VECTOR_STORE_PATH}")
    
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )
    
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME, 
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    prompt_template = """
    You are an expert educational assistant. Your task is to answer the user's question based *only* on the provided context from their textbook.
    If the information to answer the question is not in the context, you must state "I cannot find the answer in the provided text."
    Do not add any information that is not present in the context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=llm, prompt=PROMPT)
    print("Backend is ready.")
except Exception as e:
    print(f"Error initializing backend models: {e}")
    vectorstore, chain = None, None

@app.route('/ask', methods=['POST'])
def ask_question():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400
        
    if not vectorstore or not chain:
        return jsonify({"error": "Backend models not initialized."}), 500

    print(f"Received query: {query}")
    try:
        # 1. Retrieve relevant documents using Chroma similarity search
        docs = vectorstore.similarity_search(query, k=3)
        
        # 2. Prepare context for the LLM
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. Generate an answer
        result = chain.run(context=context, question=query)

        # 4. Extract sources
        sources = []
        for doc in docs:
            source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
            source_info = {
                "source": source_file,
                "page": doc.metadata.get('page', 'N/A')
            }
            if source_info not in sources:
                sources.append(source_info)
        
        print(f"Generated Answer: {result}")
        print(f"Sources: {sources}")

        return jsonify({"answer": result, "sources": sources})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(debug=True, port=5000)

