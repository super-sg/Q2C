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

def hybrid_search(vectorstore, query, k=5):
    """
    Perform hybrid search combining semantic and keyword-based retrieval.
    Optimized to avoid loading all documents into memory.
    """
    from langchain.schema import Document
    
    # Get semantic search results
    semantic_docs = vectorstore.similarity_search_with_score(query, k=k)
    
    # For keyword search, only retrieve a limited set (k*10) instead of all documents
    # This prevents memory issues with large databases
    all_docs = vectorstore.get(limit=k * 10)
    keyword_matches = []
    query_words = set(query.lower().split())
    
    if 'documents' in all_docs and 'metadatas' in all_docs:
        for doc_text, metadata in zip(all_docs['documents'], all_docs['metadatas']):
            doc_words = set(doc_text.lower().split())
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                doc = Document(page_content=doc_text, metadata=metadata)
                score = 1.0 / (overlap + 1)
                keyword_matches.append((doc, score))
    
    # Combine and deduplicate results using full content hash
    all_results = semantic_docs + keyword_matches
    seen_content = set()
    unique_results = []
    for doc, score in all_results:
        # Use hash of full content for better deduplication
        content_hash = hash(doc.page_content)
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_results.append((doc, score))
    
    return sorted(unique_results, key=lambda x: x[1])[:k]

def preprocess_query(query):
    """
    Preprocess query with optimized synonym expansion.
    Only expands terms that exactly match known keywords.
    """
    processed_query = query.lower().strip()
    physics_synonyms = {
        "chapters": ["topics", "sections", "units"],
        "physics": ["physical science", "mechanics", "motion"],
        "energy": ["power", "force", "work"],
        "conservation": ["preservation", "constant"],
        "law": ["principle", "rule", "theorem"],
        "motion": ["movement", "kinematics"],
        "electricity": ["electric", "electrical", "current"],
        "magnetism": ["magnetic", "magnet"],
        "light": ["optics", "optical", "rays"],
        "waves": ["wave", "vibration", "oscillation"]
    }
    query_terms = processed_query.split()
    expanded_terms = list(query_terms)  # Start with original terms
    seen_expansions = set(query_terms)  # Track to avoid duplicates
    
    for term in query_terms:
        # Only expand if term exactly matches a key
        if term in physics_synonyms:
            for synonym in physics_synonyms[term]:
                if synonym not in seen_expansions:
                    expanded_terms.append(synonym)
                    seen_expansions.add(synonym)
    
    expanded_query = " ".join(expanded_terms)
    return query, expanded_query
