import os
import streamlit as st
import json
from backend_multimodal import load_models, hybrid_search, preprocess_query, process_image_input

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Q2C",
    page_icon="📃",
    layout="wide",
)





# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = ""
if "user_language" not in st.session_state:
    st.session_state.user_language = "en"


vectorstore, chain = load_models()





def process_voice_input():
    """
    Process voice input using speech recognition.
    Returns the transcribed text or None if failed.
    """
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        
        # Create audio recording interface
        st.info("🎤 Click the button below and speak your question...")
        
        if st.button("🎙️ Start Recording", key="voice_button"):
            with st.spinner("Listening... Speak now!"):
                try:
                    # Use microphone as source
                    with sr.Microphone() as source:
                        r.adjust_for_ambient_noise(source, duration=1)
                        audio = r.listen(source, timeout=5, phrase_time_limit=10)
                    
                    with st.spinner("Processing speech..."):
                        # Use Google's speech recognition
                        text = r.recognize_google(audio)
                        
                        st.success(f"🎯 I heard: '{text}'")
                        
                        return text
                        
                except sr.WaitTimeoutError:
                    st.warning("⏰ No speech detected. Please try again.")
                    return None
                except sr.UnknownValueError:
                    st.warning("🤔 Sorry, I couldn't understand what you said. Please try again.")
                    return None
                except sr.RequestError as e:
                    st.error(f"❌ Speech recognition error: {e}")
                    return None
                except Exception as e:
                    st.warning("🎤 Voice input not available. Please ensure you have a microphone connected.")
                    return None
    except ImportError:
        st.warning("🎤 Voice input requires additional packages. Please install: pip install speechrecognition pyaudio")
        return None
    
    return None



def create_multimodal_prompt(text_query, image_data=None, conversation_history=""):
    """
    Create a prompt that can handle both text and image inputs for policy and legal document analysis.
    """
    if image_data:
        return f"""
You are an expert legal and policy analysis assistant specializing in insurance policies, contracts, and compliance documents. The user has provided both text and an image.

Previous conversation:
{conversation_history}

User's query: {text_query}
User has also uploaded an image that may contain: policy clauses, contract terms, claim forms, legal documents, or other relevant content.

Please:
1. Analyze the image if it contains relevant policy clauses, terms, or legal content
2. Extract key entities from the query (e.g., age, condition, policy duration, location, procedure type)
3. Map the query to applicable clauses in the provided context
4. Evaluate conditions and eligibility based on the policy logic
5. Provide a clear decision with supporting explanation grounded in specific clauses
6. If applicable, calculate any payout amounts or benefits

Context from policy document(s):
{{context}}

Your Response (provide structured analysis with decision, reasoning, and clause references):
"""
    else:
        return f"""
You are an expert legal and policy analysis assistant specializing in insurance policies, contracts, HR policies, and compliance documents. You help users interpret complex policy language and make informed decisions based on natural language queries.

Previous conversation:
{conversation_history}

Context from policy document(s):
{{context}}

User's Query: {text_query}

Please:
1. Parse the query to extract key metadata (age, condition, policy details, dates, etc.)
2. Identify and retrieve relevant clauses from the provided context
3. Evaluate the conditions and apply policy logic
4. Provide a clear decision (approved/rejected/conditional) with reasoning
5. Reference specific clauses that support your conclusion
6. Calculate any amounts or benefits if applicable
7. Ensure your response is traceable and explainable for audit purposes

Your Response:
"""



# --- STREAMLIT APP ---
st.title("Query to Clause 📃")
st.markdown("Chat with your legal documents! Ask questions and have a conversation about your queries.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📖 Sources"):
                for source in message["sources"]:
                    st.write(f"- **{source['source']}** (Page: {source['page']})")
        # Display images in user messages
        if message["role"] == "user" and "image" in message:
            st.image(message["image"], caption="Uploaded Image", width=300)

# --- MULTIMODAL INPUT SECTION ---
st.markdown("### 💬 Ask your question using:")

# Create columns for different input types
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Text input (primary)
    text_input = st.chat_input("Type your question here...")

with col2:
    # Voice input
    st.markdown("**🎤 Voice**")
    voice_text = process_voice_input()

with col3:
    # Image input
    st.markdown("**🖼️ Image**")
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=['png', 'jpg', 'jpeg'],
        key="image_uploader",
        label_visibility="collapsed"
    )

# Process the input (text, voice, or both)
prompt = text_input or voice_text
image_data = process_image_input(uploaded_image) if uploaded_image else None

if prompt or image_data:
    if vectorstore is None or chain is None:
        st.error("Backend models are not loaded. Please check the error message above.")
    else:
        # Prepare the user message
        user_message = {"role": "user", "content": prompt or "Please analyze this image"}
        if image_data:
            user_message["image"] = image_data["image"]
            user_message["has_image"] = True
        
        # Add user message to chat history
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            if prompt:
                st.markdown(prompt)
            if image_data:
                st.image(image_data["image"], caption="Uploaded Image", width=300)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Use the prompt or default for image-only queries
                    query_text = prompt or "Please analyze this image and explain what you see"
                    
                    # Preprocess the query
                    original_query, expanded_query = preprocess_query(query_text)
                    
                    # 1. Try hybrid search first
                    try:
                        docs_with_scores = hybrid_search(vectorstore, original_query, k=5)
                    except Exception as e:
                        # Fallback to regular semantic search if hybrid search fails
                        docs_with_scores = vectorstore.similarity_search_with_score(original_query, k=5)
                    
                    # 2. If scores are too high (not similar enough), try with expanded query
                    if docs_with_scores and docs_with_scores[0][1] > 1.0:
                        try:
                            expanded_docs = hybrid_search(vectorstore, expanded_query, k=3)
                        except:
                            expanded_docs = vectorstore.similarity_search_with_score(expanded_query, k=3)
                        
                        # Combine and deduplicate results
                        all_docs = docs_with_scores + expanded_docs
                        seen_content = set()
                        unique_docs = []
                        for doc, score in all_docs:
                            content_hash = hash(doc.page_content[:100])
                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                unique_docs.append((doc, score))
                        docs_with_scores = sorted(unique_docs, key=lambda x: x[1])[:5]
                    
                    # 3. Filter documents by similarity threshold
                    similarity_threshold = 1.2
                    relevant_docs = [doc for doc, score in docs_with_scores if score < similarity_threshold]
                    
                    # If no documents meet the threshold, use the top 3 anyway
                    if not relevant_docs:
                        relevant_docs = [doc for doc, score in docs_with_scores[:3]]
                    
                    # 4. Prepare context with more information
                    context_parts = []
                    for i, doc in enumerate(relevant_docs):
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'N/A')
                        context_parts.append(f"Source {i+1} ({source}, Page {page}):\n{doc.page_content}")
                    
                    context = "\n\n".join(context_parts)
                    
                    # 5. Prepare conversation history
                    conversation_history = ""
                    if len(st.session_state.messages) > 1:  # More than just the current message
                        recent_messages = st.session_state.messages[-6:]  # Last 3 exchanges (6 messages)
                        history_parts = []
                        for msg in recent_messages[:-1]:  # Exclude current message
                            role = "Student" if msg["role"] == "user" else "Assistant"
                            content = msg['content']
                            if msg.get('has_image'):
                                content += " [Student also shared an image]"
                            history_parts.append(f"{role}: {content}")
                        conversation_history = "\n".join(history_parts)
                    
                    # 6. Generate response
                    result = chain.run(
                        conversation_history=conversation_history,
                        context=context, 
                        question=original_query
                    )
                    
                    # Display the response
                    st.markdown(result)
                    
                    # Prepare sources information
                    sources = []
                    for doc in relevant_docs:
                        source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        source_info = {
                            "source": source_file,
                            "page": doc.metadata.get('page', 'N/A')
                        }
                        if source_info not in sources:
                            sources.append(source_info)
                    
                    # Display sources
                    with st.expander("📖 Sources"):
                        for source in sources:
                            st.write(f"- **{source['source']}** (Page: {source['page']})")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result,
                        "sources": sources
                    })
                    
                    # Check if the answer indicates no relevant information was found
                    if "cannot find" in result.lower() or "no relevant information" in result.lower():
                        st.info("💡 **Tip**: Try rephrasing your question or use more specific terms!")

                except Exception as e:
                    error_msg = f"I'm sorry, I encountered an error while processing your question: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# --- SIDEBAR ---
with st.sidebar:
    st.header("💬 Chat Controls")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_context = ""
        st.rerun()
    

    
    # Language selector
    languages = {
        "en": "🇺🇸 English",
        "hi": "🇮🇳 हिंदी (Hindi)",
        "bn": "🇧🇩 বাংলা (Bengali)", 
        "te": "🇮🇳 తెలుగు (Telugu)",
        "mr": "🇮🇳 मराठी (Marathi)",
        "ta": "🇮🇳 தமிழ் (Tamil)",
        "ur": "🇵🇰 اردو (Urdu)",
        "gu": "🇮🇳 ગુજરાતી (Gujarati)",
        "kn": "🇮🇳 ಕನ್ನಡ (Kannada)",
        "ml": "🇮🇳 മലയാളം (Malayalam)",
        "es": "🇪🇸 Español",
        "fr": "🇫🇷 Français",
        "de": "🇩🇪 Deutsch",
        "zh": "🇨🇳 中文",
        "ar": "🇸🇦 العربية"
    }

    
    # Chat statistics
    if st.session_state.messages:
        st.metric("Messages in chat", len(st.session_state.messages))
        st.metric("Your questions", len([m for m in st.session_state.messages if m["role"] == "user"]))
    
    st.header("About EduRAG")
    st.info("This is a multilingual, multimodal conversational AI assistant for your Legal Documents and Contracts . Chat in your native language using text, voice, and images!")
    
    st.header("🎯 Input Methods")
    st.markdown(f"""
    **💬 Text**: Type in any supported language
    - Current: {languages.get(st.session_state.user_language, '🇺🇸 English')}
    
    **🎤 Voice**: Speak in your preferred language
    - Auto-detects language when enabled
    - Supports 15+ languages including all major Indian languages
    
    **🖼️ Images**: Upload diagrams, equations, or textbook pages
    - Supports PNG, JPG, JPEG formats
    - Perfect for homework problems or diagrams
    
    **🌍 Multilingual**: 
    - Auto-detects your language or manually select
    - Responses in your preferred language
    - Seamless translation between languages
    """)
    
    st.header("💡 Chat Tips")
    st.markdown(f"""
    **For better multilingual conversations:**
    - Ask in any language: हिंदी, English, বাংলা, தமிழ், etc.
    - Mix languages freely - the AI adapts automatically
    - Use voice in your native language for natural interaction
    - Combine voice + image: Speak while uploading a diagram
    
    **Sample conversation starters:**
    - 🗣️ English: "Hi! What topics can you help me with?"
    - �️ Hindi: "मुझे भौतिकी में मदद चाहिए" 
    - 🗣️ Bengali: "পদার্থবিজ্ঞান সম্পর্কে বলুন"
    - 🖼️ Upload diagram + ask: "এটি ব্যাখ্যা করুন" (Explain this)
    - 🎤 Voice: "Energy conservation के बारे में बताओ"
    
    **Current Language**: {languages.get(st.session_state.user_language, '🇺🇸 English')}
    """)
    
    st.header("🔧 Advanced")
    if st.checkbox("Show debug info"):
        st.write("**Session State:**")
        st.write(f"Messages: {len(st.session_state.messages)}")
        st.write(f"Conversation context length: {len(st.session_state.conversation_context)}")
    
    st.warning("**Note:** The AI's knowledge is limited to the indexed NCERT textbooks.")
