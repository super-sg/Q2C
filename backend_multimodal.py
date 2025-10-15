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

# --- CONFIGURATION ---
VECTOR_STORE_PATH = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

load_dotenv()

def load_models():
    from langchain_community.embeddings import HuggingFaceEmbeddings
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

def hybrid_search(vectorstore, query, k=5):
    semantic_docs = vectorstore.similarity_search_with_score(query, k=k)
    all_docs = vectorstore.get()
    keyword_matches = []
    query_words = set(query.lower().split())
    if 'documents' in all_docs and 'metadatas' in all_docs:
        for i, (doc_text, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
            doc_words = set(doc_text.lower().split())
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                from langchain.schema import Document
                doc = Document(page_content=doc_text, metadata=metadata)
                score = 1.0 / (overlap + 1)
                keyword_matches.append((doc, score))
    all_results = semantic_docs + keyword_matches
    seen_content = set()
    unique_results = []
    for doc, score in all_results:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_results.append((doc, score))
    return sorted(unique_results, key=lambda x: x[1])[:k]

def preprocess_query(query):
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
    expanded_terms = []
    for term in query_terms:
        expanded_terms.append(term)
        for key, synonyms in physics_synonyms.items():
            if term in key or key in term:
                expanded_terms.extend(synonyms)
    expanded_query = " ".join(expanded_terms)
    return query, expanded_query

def process_image_input(uploaded_image):
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode()
            return {
                "image": image,
                "base64": img_base64,
                "description": "User uploaded an image related to their question"
            }
        except Exception as e:
            return None
    return None
