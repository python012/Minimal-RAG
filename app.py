"""Streamlit RAG Knowledge Assistant Frontend.

This file implements the chat interface and calls the backend `RAGEngine` to complete the following steps:
1. Load environment variables to get model name and vector database path.
2. Page initialization (title / icon / layout).
3. Initialize and cache engine instance and session chat history.
4. Sidebar: Engine reload, database statistics, retrieval parameters, clear history, usage tips.
5. Main area: Render historical messages in order, source content can be collapsed for viewing.
6. After user input, execute RAG: Generate embedding -> Similarity search -> Construct context -> Call model to generate answer.
7. Bottom displays currently used Ollama model and ChromaDB information.

"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGEngine

# ====================== Environment Variable Loading ======================
# Load .env early to ensure CHROMA_DB_PATH / OLLAMA_MODEL etc. are available.

# Load environment variables
load_dotenv()

# ====================== Page Configuration ======================
# Set title, icon, and wide layout.
st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="🤖", layout="wide")

# ====================== Session State Initialization ======================
# chat_history: [{role, content, sources?}]
# rag_engine: Cached RAGEngine singleton
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None


@st.cache_resource
def initialize_rag_engine():
    """Construct and cache RAGEngine singleton.

    Uses `@st.cache_resource` to avoid repeated initialization (time-consuming operations like
    connecting to vector database), only rebuilds on first run or after manual cache clear.
    Displays error message in UI when exceptions occur.
    """
    db_path = os.getenv("CHROMA_DB_PATH", "./vector_db")
    try:
        engine = RAGEngine(db_path=db_path)
        return engine
    except Exception as e:
        st.error(f"Error initializing RAG engine: {e}")
        return None


# ====================== Page Header ======================
# Display Logo + title + brief description.
# st.image("static/robot.png", width=48)
st.title("📚 RAG Knowledge Assistant")
st.markdown("Ask questions and get answers based on your knowledge base")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # Engine reload: Clear cache and rerun, suitable for refreshing after adding new data.
    if st.button("🔄 Reload Engine"):
        st.cache_resource.clear()
        st.session_state.rag_engine = None
        st.rerun()

    # Lazy initialization: Execute on first run or after reload, with progress indicator.
    if st.session_state.rag_engine is None:
        with st.spinner("Initializing RAG engine..."):
            st.session_state.rag_engine = initialize_rag_engine()

    # Engine health status and vector database statistics.
    if st.session_state.rag_engine:
        st.success("✅ Engine loaded")
        stats = st.session_state.rag_engine.get_stats()
        st.markdown(f"**Documents in DB:** :red[{stats['total_documents']}]")

        # Display model information
        st.divider()
        st.subheader("🤖 Model Configuration")
        # Check if using Aliyun API
        api_key = os.getenv("EMBEDDING_MODEL_API_KEY", "") or os.getenv("GENERATION_MODEL_API_KEY", "")
        if api_key:
            # Using Aliyun cloud models
            gen_model = os.getenv("ALIYUN_CHAT_MODEL", "qwen-flash")
            embed_model = os.getenv("ALIYUN_EMBED_MODEL", "text-embedding-v4")
            model_provider = "Aliyun DashScope"
        else:
            # Fallback to local Ollama models
            gen_model = os.getenv("OLLAMA_MODEL", "llama2:7b")
            embed_model = os.getenv("OLLAMA_EMBED_MODEL", gen_model)
            model_provider = "Local Ollama"
        st.info(f"**Provider:** {model_provider}")
        st.info(f"**Generation Model:**\n`{gen_model}`")
        st.info(f"**Embedding Model:**\n`{embed_model}`")

        # Retrieval parameters: Control number of results and minimum relevance threshold.
        st.divider()
        st.subheader("🔍 Search Settings")
        n_results = st.slider("Number of results", 1, 10, 5)
        min_relevance = st.slider("Min relevance to use KB", 0.0, 1.0, 0.35, 0.05)

        # Source filter (placeholder, can be extended with actual data source tags later).
        filter_source = st.selectbox("Filter by source", ["All", "file", "git", "jira", "confluence"])
        if filter_source == "All":
            filter_source = None
    else:
        st.error("❌ Engine not loaded")
        st.stop()

    st.divider()

    # Clear chat: Reset history without rebuilding engine.
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    # Usage instructions and operation tips.
    st.markdown(
        """
    ### 📚 How to use:
    1. Load documents via `data_loader.py`
    2. Ask a focused question
    3. Inspect sources for verification
    
    ### 💡 Tips:
    - Be specific for better retrieval
    - Use the source expander to audit relevance
    - Reload engine after adding new data
    """
    )

# Main chat interface
if st.session_state.rag_engine is None:
    st.warning("⚠️ RAG Engine not loaded, please click Reload Engine in sidebar to retry.")
    st.stop()

# ====================== Chat History Rendering ======================
# Loop through and display messages; collapse sources to avoid cluttering main interface.
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}** (Relevance: {source['relevance_score']:.2%})")
                    # Text content
                    st.text(source["content"])
                    # Display thumbnail if image path exists
                    img_path = source.get("metadata", {}).get("image_path")
                    if img_path:
                        st.image(img_path, width=160)
                    if source.get("metadata"):
                        st.caption(f"File: {source['metadata'].get('file_name', 'Unknown')}")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your knowledge base..."):
    # ====================== User Input Processing ======================
    # 1. Write user message
    # 2. Render user bubble
    # 3. Call RAG engine to retrieve and generate answer, attach sources
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Use RAG engine query method (handles relevance check internally)
            print(f"\n{'='*60}")
            print(f"🔍 Processing query: {prompt}")
            print(f"   Min relevance threshold: {min_relevance}")
            print(f"{'='*60}")
            
            # Get search results and check relevance
            search_results = st.session_state.rag_engine.search(
                query=prompt,
                n_results=n_results,
                filter_source=filter_source
            )
            
            documents = search_results["documents"][0] if search_results["documents"] else []
            metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []
            distances = search_results["distances"][0] if search_results["distances"] else []
            
            # Check relevance
            max_relevance = 0.0
            if distances:
                max_relevance = max(1 - d for d in distances)
            
            is_irrelevant = (not documents) or (max_relevance < max(0.0, min_relevance))
            
            print(f"   Retrieved distances: {distances[:3] if distances else 'N/A'}")
            print(f"   Max relevance score: {max_relevance:.6f}")
            print(f"   Relevance above threshold: {max_relevance >= max(0.0, min_relevance)}")
            
            if is_irrelevant:
                print("   ℹ️  No relevant documents found - using model's native knowledge")
            else:
                print(f"   ✅ Found {len(documents)} relevant documents - using knowledge base")
            
            # Stream answer using generator
            full_response = ""
            
            # Use RAG engine's generate_answer method (supports both Aliyun API and Ollama fallback)
            # When is_irrelevant=True, it passes empty context_docs, and the engine will use model's native knowledge
            answer_gen = st.session_state.rag_engine.generate_answer(
                query=prompt,
                context_docs=documents if not is_irrelevant else [],
                stream=True
            )
            
            # Stream and display answer
            if answer_gen:
                for chunk in answer_gen:
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + " ⏳")
            else:
                full_response = "Error: Failed to generate answer"
            
            # Final display without loading indicator
            message_placeholder.markdown(full_response)
            
            print(f"✅ Answer generated ({len(full_response)} characters)")
            print(f"{'='*60}\n")
            
            # Prepare sources for display
            sources = []
            if not is_irrelevant:
                sources = [
                    {
                        "content": doc[:200] + "..." if len(doc) > 200 else doc,
                        "metadata": meta,
                        "relevance_score": 1 - dist,
                    }
                    for doc, meta, dist in zip(documents, metadatas, distances)
                ]
            
            # Display sources if found
            if sources:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}** (Relevance: {source['relevance_score']:.2%})")
                        st.text(source["content"])
                        img_path = source.get("metadata", {}).get("image_path")
                        if img_path:
                            st.image(img_path, width=160)
                        if source.get("metadata"):
                            st.caption(f"File: {source['metadata'].get('file_name', 'Unknown')}")
                        st.divider()
                
                # Display top image
                top_img = sources[0].get("metadata", {}).get("image_path")
                if top_img:
                    st.image(top_img, width=240)
            
            # Save to chat history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": full_response, "sources": sources}
            )
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            err_msg = str(e)
            print(f"❌ Error: {err_msg}")
            # Highlight memory/model errors
            if "out of memory" in err_msg.lower():
                st.error(f"❌ Ollama out of memory or model too large:\n{err_msg}")
            else:
                st.error(f"❌ Ollama API error:\n{err_msg}")
            st.session_state.chat_history.append({"role": "assistant", "content": err_msg, "sources": []})

# Footer
st.divider()
# ====================== Footer Model Information ======================
# Display current model provider + ChromaDB description.
api_key = os.getenv("EMBEDDING_MODEL_API_KEY", "") or os.getenv("GENERATION_MODEL_API_KEY", "")
if api_key:
    model_info = f"Aliyun DashScope ({os.getenv('ALIYUN_CHAT_MODEL', 'qwen-flash')})"
else:
    model_info = f"Ollama ({os.getenv('OLLAMA_MODEL', 'llama2:7b')})"
st.caption(f"🤖 Powered by {model_info} and ChromaDB")
