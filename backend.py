# backend.py
"""
Backend logic for EduRAG: model loading, search, and preprocessing functions.
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
from backend_common import hybrid_search, preprocess_query

# --- CONFIGURATION ---
VECTOR_STORE_PATH = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

load_dotenv()

def load_models():
    """
    Loads the embedding model, LLM, and the Chroma vector store.
    """
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
You are an expert educational assistant helping students with NCERT textbooks. You are having a conversation with a student and should respond in a helpful, conversational manner.

Instructions:
1. Use the provided context from textbooks to answer questions
2. Reference previous parts of the conversation when relevant
3. Be conversational and encouraging
4. If no direct answer exists in the context, look for related information
5. Mention chapter titles, sections, or topics when they help explain concepts
6. Only if absolutely no relevant information exists, suggest how the student might rephrase their question
7. Keep your responses engaging and educational

Previous conversation:
{conversation_history}

Context from textbook(s):
{context}

Student's Question: {question}

Your Response:
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["conversation_history", "context", "question"])
    chain = LLMChain(llm=llm, prompt=PROMPT)
    return vectorstore, chain
