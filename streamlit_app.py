import os
import streamlit as st
from backend import load_models, hybrid_search, preprocess_query, process_image_input

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EduRAG",
    page_icon="📚",
    layout="wide",
)





# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = ""


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
    Create a prompt that can handle both text and image inputs.
    """
    if image_data:
        return f"""
You are an expert educational assistant helping students with NCERT textbooks. The student has provided both text and an image.

Previous conversation:
{conversation_history}

Student's text question: {text_query}
Student has also uploaded an image that may contain: diagrams, equations, text from textbooks, or other educational content.

Please:
1. Analyze the image if it contains relevant educational content
2. Answer the student's question using both the image context and the textbook context provided
3. Be conversational and helpful
4. If the image shows a problem or diagram, help explain it step by step

Context from textbook(s):
{{context}}

Your Response:
"""
    else:
        return f"""
You are an expert educational assistant helping students with NCERT textbooks. You are having a conversation with a student and should respond in a helpful, conversational manner.

Previous conversation:
{conversation_history}

Context from textbook(s):
{{context}}

Student's Question: {text_query}

Your Response:
"""

# --- STREAMLIT APP ---
st.title("📚 EduRAG: Your Personal NCERT Assistant")
st.markdown("Chat with your NCERT textbooks! Ask questions and have a conversation about your studies.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📖 Sources"):
                for source in message["sources"]:
                    st.write(f"**{source['source']}** (Page {source['page']})")
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
                        sources.append(source_info)
                    
                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant", 
                        "content": result,
                        "sources": sources
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    # Update conversation context
                    new_context = f"Student: {query_text}\nAssistant: {result}"
                    if len(st.session_state.conversation_context) > 2000:  # Keep context manageable
                        # Keep only the last portion of the conversation
                        st.session_state.conversation_context = st.session_state.conversation_context[-1000:] + "\n" + new_context
                    else:
                        st.session_state.conversation_context += "\n" + new_context

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error("Please try rephrasing your question or check if the database is properly loaded.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("💬 Chat Controls")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_context = ""
        st.rerun()
    
    # Chat statistics
    if st.session_state.messages:
        st.metric("Messages in chat", len(st.session_state.messages))
        st.metric("Your questions", len([m for m in st.session_state.messages if m["role"] == "user"]))
    
    st.header("About EduRAG")
    st.info("This is a conversational AI assistant for your NCERT textbooks. Chat using text, voice, and images!")
    
    st.header("🎯 Input Methods")
    st.markdown("""
    **💬 Text**: Type your questions directly
    
    **🎤 Voice**: Speak your questions using the voice input feature
    
    **🖼️ Images**: Upload diagrams, equations, or textbook pages
    - Supports PNG, JPG, JPEG formats
    - Perfect for homework problems or diagrams
    """)
    
    st.header("💡 Chat Tips")
    st.markdown("""
    **Sample conversation starters:**
    - 🗣️ "Hi! What topics can you help me with?"
    - 🗣️ "Explain the concept of energy conservation"
    - 🗣️ "What are the laws of motion?"
    - 🖼️ Upload a diagram and ask: "Explain this diagram"
    - 🎤 Use voice input for natural interaction
    """)
    
    st.header("🔧 Advanced")
    if st.checkbox("Show debug info"):
        st.write("**Session State:**")
        st.write(f"Messages: {len(st.session_state.messages)}")
        st.write(f"Conversation context length: {len(st.session_state.conversation_context)}")
    
    st.warning("**Note:** The AI's knowledge is limited to the indexed NCERT textbooks.")