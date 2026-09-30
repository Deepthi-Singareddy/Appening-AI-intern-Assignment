import streamlit as st
import requests
import json
from pathlib import Path

from src.config import settings
from src.graph import build_rag_graph
from src.utils import ensure_knowledge_base_pdf

# Page Configuration
st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for enterprise aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #1F2937;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #374151;
        margin-bottom: 1rem;
    }
    .chunk-card {
        background-color: #111827;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">Agentic AI Knowledge Base RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">LangGraph Stateful Orchestration • Pinecone Vector DB • Strict Grounding</div>', unsafe_allow_html=True)

# Ensure PDF document
ensure_knowledge_base_pdf()

# Cache LangGraph instance
@st.cache_resource
def get_graph():
    return build_rag_graph(index_name=settings.PINECONE_INDEX_NAME)

graph = get_graph()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_response" not in st.session_state:
    st.session_state.latest_response = None

# Sidebar Diagnostics & Inspector Panel
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/brain.png", width=64)
    st.title("System Diagnostics")
    
    st.markdown("---")
    st.subheader("Configuration Status")
    
    openai_status = "🟢 Configured" if settings.is_openai_configured else "🟡 Local Fallback Mode"
    pinecone_status = "🟢 Configured" if settings.is_pinecone_configured else "🟡 Local Fallback Mode"
    
    st.write(f"**OpenAI Embeddings:** {openai_status}")
    st.write(f"**Pinecone Vector DB:** {pinecone_status}")
    st.write(f"**Target Document:** `{settings.DEFAULT_PDF_PATH.name}`")
    
    st.markdown("---")
    st.subheader("Sample Verification Queries")
    
    sample_queries = [
        "What is the core definition of Agentic AI as outlined in the eBook?",
        "What are the main architectural components required to build agentic systems?",
        "What real-world industry use cases for Agentic AI are discussed in the eBook?",
        "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
        "What key challenges or limitations of Agentic AI are mentioned in the document?",
        "What is the capital of France?"
    ]
    
    selected_sample = st.selectbox("Click to auto-populate test query:", ["-- Select Sample Query --"] + sample_queries)
    if selected_sample != "-- Select Sample Query --":
        st.session_state.sample_prompt = selected_sample

    st.markdown("---")
    st.subheader("Retrieved Context Inspector")
    
    if st.session_state.latest_response:
        resp = st.session_state.latest_response
        score = resp.get("confidence_score", 0.0)
        
        # Display Score Meter
        score_color = "#10B981" if score > 0.7 else "#F59E0B" if score > 0.3 else "#EF4444"
        st.markdown(f"### Confidence Score")
        st.markdown(f"<h2 style='color: {score_color};'>{score * 100:.1f}%</h2>", unsafe_allow_html=True)
        st.progress(score)
        
        st.markdown("#### Retrieved Chunks:")
        chunks = resp.get("retrieved_context_chunks", [])
        if chunks:
            for idx, chunk in enumerate(chunks, 1):
                with st.expander(f"Chunk #{idx}", expanded=(idx == 1)):
                    st.markdown(f'<div class="chunk-card">{chunk}</div>', unsafe_allow_html=True)
        else:
            st.info("No context chunks retrieved.")
    else:
        st.info("Ask a query to inspect vector retrieval and confidence scoring.")

# Main Chat Interface Layout
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "payload" in msg:
            st.caption(f"Score: {msg['payload']['confidence_score']} | Chunks: {len(msg['payload']['retrieved_context_chunks'])}")

# User Input Box
user_prompt = st.chat_input("Ask any question from the Agentic AI eBook...")

# Check if sample was selected
if "sample_prompt" in st.session_state and st.session_state.sample_prompt:
    user_prompt = st.session_state.sample_prompt
    st.session_state.sample_prompt = None

if user_prompt:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    # Process query via LangGraph state machine
    with st.chat_message("assistant"):
        with st.spinner("Retrieving vector context & executing LangGraph workflow..."):
            initial_state = {
                "question": user_prompt,
                "context": [],
                "answer": "",
                "score": 0.0,
                "is_grounded": False
            }
            
            try:
                result = graph.invoke(initial_state)
                
                payload = {
                    "query": user_prompt,
                    "final_answer": result.get("answer", "No answer generated."),
                    "retrieved_context_chunks": result.get("context", []),
                    "confidence_score": result.get("score", 0.0)
                }
                
                st.session_state.latest_response = payload
                st.write(payload["final_answer"])
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": payload["final_answer"],
                    "payload": payload
                })
                
                st.rerun()

            except Exception as e:
                st.error(f"Error processing query: {e}")
