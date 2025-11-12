# backend_common.py
"""
Common backend utilities shared across backend modules.
Extracted to avoid code duplication and improve maintainability.
"""
from langchain.schema import Document


def hybrid_search(vectorstore, query, k=5):
    """
    Perform hybrid search combining semantic and keyword-based retrieval.
    Optimized to avoid loading all documents into memory.
    
    Args:
        vectorstore: The Chroma vectorstore instance
        query: The search query string
        k: Number of results to return (default: 5)
    
    Returns:
        List of (document, score) tuples sorted by relevance
    """
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
    
    Args:
        query: The original query string
    
    Returns:
        Tuple of (original_query, expanded_query)
    """
    processed_query = query.lower().strip()
    
    # Domain-specific synonyms for better semantic matching
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


def process_image_input(uploaded_image):
    """
    Process uploaded image input.
    Returns image data dictionary or None if processing fails.
    
    Args:
        uploaded_image: Uploaded file object from Streamlit
    
    Returns:
        Dictionary with image data or None
    """
    if uploaded_image is not None:
        try:
            from PIL import Image
            import io
            import base64
            
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
