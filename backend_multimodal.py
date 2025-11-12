# backend_multimodal.py
"""
Backend logic for EduRAG Multilingual: model loading, search, and preprocessing functions.
"""
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
import io
import base64

# Import common utilities to avoid code duplication
from backend_common import hybrid_search, preprocess_query, process_image_input

# --- CONFIGURATION ---
VECTOR_STORE_PATH = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

load_dotenv()

def load_models():
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(f"Chroma database not found at '{VECTOR_STORE_PATH}'. Please run 'python ingest.py' first.")
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
You are an expert legal and policy analysis assistant specializing in insurance policies, contracts, and compliance documents. The user has provided both text and an image.
Please:
1. Parse the query to extract key metadata (age, condition, policy details, dates, etc.)
2. Identify and retrieve relevant clauses from the provided context
3. Evaluate the conditions and apply policy logic
4. Provide a clear decision (approved/rejected/conditional) with reasoning
5. Reference specific clauses that support your conclusion
6. Calculate any amounts or benefits if applicable
7. Ensure your response is traceable and explainable for audit purposes

Previous conversation:
{conversation_history}

Context from metadata(s):
{context}

Client's Question: {question}

Your Response:
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["conversation_history", "context", "question"])
    chain = LLMChain(llm=llm, prompt=PROMPT)
    return vectorstore, chain
