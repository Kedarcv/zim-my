"""
Zim-my Streamlit Application
============================
Interactive web UI for the Zim-my agriculture AI assistant.

Model: Zim-my (fine-tuned Qwen2.5-3B-Instruct, Q4_K_M quantized)
Developer: Michael Mlungisi Nkomo — AI Engineer from Zimbabwe

Usage:
    streamlit run src/app.py -- --model models/gguf/zim-my-q4_k_m.gguf
"""

import argparse
import os
import sys
import time
import psutil

import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zim-my 🇿🇼",
    page_icon="🇿🇼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Zim-my Identity ──────────────────────────────────────────────────────────
ZIM_MY_SYSTEM_PROMPT = (
    "You are Zim-my, an AI assistant developed by Michael Mlungisi Nkomo, "
    "an artificial intelligence engineer from Zimbabwe. "
    "You specialize in Zimbabwean agriculture and can communicate in Shona and English. "
    "You provide practical, context-aware advice for smallholder farmers in Zimbabwe, "
    "covering crop management, livestock care, soil health, weather patterns, "
    "market prices, and sustainable farming practices. "
    "When asked in Shona, respond in Shona. When asked in English, respond in English."
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .zim-header {
        background: linear-gradient(135deg, #006633 0%, #FFD700 50%, #EF3340 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .zim-subtitle {
        color: #666;
        font-size: 1rem;
    }
    .chat-message-user {
        background-color: #e8f5e9;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .chat-message-assistant {
        background-color: #fff8e1;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .metric-card {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .shona-badge {
        background: #006633;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    .english-badge {
        background: #1565c0;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str, n_ctx: int = 2048, n_threads: int = 4):
    """Load Zim-my model (cached across Streamlit reruns)."""
    from llama_cpp import Llama

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=0,  # CPU-only
        verbose=False,
    )
    return llm


@st.cache_resource
def load_rag():
    """Load RAG pipeline (cached across Streamlit reruns)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rag import ZimMyRAG
    return ZimMyRAG()


def generate_response(llm, message: str, use_rag: bool = False, rag_engine=None) -> dict:
    """Generate a response from Zim-my."""
    messages = [{"role": "system", "content": ZIM_MY_SYSTEM_PROMPT}]

    # Add RAG context if enabled
    rag_context = None
    if use_rag and rag_engine:
        rag_context = rag_engine.query(message, n_results=3)
        if rag_context and "No documents indexed" not in rag_context:
            augmented = (
                f"Reference information:\n{rag_context}\n\n"
                f"User question: {message}"
            )
            messages.append({"role": "user", "content": augmented})
        else:
            messages.append({"role": "user", "content": message})
    else:
        messages.append({"role": "user", "content": message})

    start_time = time.time()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
    )
    elapsed = time.time() - start_time

    answer = response["choices"][0]["message"]["content"]
    tokens = len(answer.split())
    tokens_per_sec = tokens / elapsed if elapsed > 0 else 0

    return {
        "answer": answer,
        "time_sec": round(elapsed, 2),
        "tokens": tokens,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "rag_context": rag_context,
    }


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇿🇼 Zim-my Settings")

    model_path = st.text_input(
        "Model Path",
        value="models/gguf/zim-my-q4_k_m.gguf",
        help="Path to the Zim-my GGUF model file",
    )

    use_rag = st.checkbox(
        "Enable RAG",
        value=False,
        help="Use ChromaDB retrieval to augment responses",
    )

    n_ctx = st.slider("Context Window", 512, 4096, 2048, step=512)
    n_threads = st.slider("CPU Threads", 1, 8, 4)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.1)
    max_tokens = st.slider("Max Tokens", 64, 1024, 512, step=64)

    st.markdown("---")
    st.markdown("### 📊 System Info")

    ram = psutil.virtual_memory()
    st.metric("RAM Used", f"{ram.used / 1e9:.1f} / {ram.total / 1e9:.1f} GB")
    st.metric("RAM Available", f"{ram.available / 1e9:.1f} GB")

    st.markdown("---")
    st.markdown("""
    **About Zim-my** 🇿🇼

    Developed by **Michael Mlungisi Nkomo**, an AI engineer from Zimbabwe.

    Built for the **ADTC 2026 Challenge** — running entirely on CPU within 7GB RAM.

    *Specializes in Zimbabwean agriculture with Shona language support.*
    """)

# ── Main Content ─────────────────────────────────────────────────────────────
st.markdown('<p class="zim-header">🇿🇼 Zim-my</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="zim-subtitle">Zimbabwean Agriculture AI Assistant — '
    'Developed by Michael Mlungisi Nkomo</p>',
    unsafe_allow_html=True,
)

# Language toggle
col1, col2 = st.columns([1, 5])
with col1:
    language = st.selectbox("Language", ["English", "ChiShona"], index=0)

# ── Load Model ───────────────────────────────────────────────────────────────
if not os.path.exists(model_path):
    st.error(f"Model not found: `{model_path}`")
    st.info("Run `notebooks/03_quantize.ipynb` to generate the GGUF model first.")
    st.stop()

with st.spinner("Loading Zim-my model..."):
    llm = load_model(model_path, n_ctx=n_ctx, n_threads=n_threads)

rag_engine = None
if use_rag:
    with st.spinner("Loading RAG pipeline..."):
        rag_engine = load_rag()

# ── Chat Interface ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask Zim-my about Zimbabwean agriculture..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Zim-my is thinking..."):
            result = generate_response(llm, prompt, use_rag=use_rag, rag_engine=rag_engine)

        st.markdown(result["answer"])

        # Show metrics
        cols = st.columns(3)
        with cols[0]:
            st.metric("⏱ Time", f"{result['time_sec']}s")
        with cols[1]:
            st.metric("📝 Tokens", f"{result['tokens']}")
        with cols[2]:
            st.metric("⚡ Speed", f"{result['tokens_per_sec']} t/s")

        # Show RAG sources if enabled
        if use_rag and result.get("rag_context"):
            with st.expander("📚 RAG Sources"):
                st.markdown(result["rag_context"])

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

# ── Quick Prompts ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💬 Quick Prompts")

quick_prompts = {
    "English": [
        "What are the best crops to grow in Mashonaland during the rainy season?",
        "How can I improve soil fertility on my small farm in Zimbabwe?",
        "What livestock is most suitable for smallholder farmers in Matabeleland?",
        "How do I control fall armyworm in my maize field?",
        "What are the current best practices for conservation agriculture in Zimbabwe?",
    ],
    "ChiShona": [
        "Nzvimbo dzinorimwa chibhorani mudunhu reMashonaland ndedzipi?",
        "Nzira dzekuwedzera hunyoro mivhu yangu ndedzipi?",
        "Zvipfuyo zvinoenderana nemurimi mudiki ndezvipi?",
        "Nzira yekurwisa hove yemombe munderere rangu ndeyipi?",
        "Zvinhu zvakanakira kurima kusingashandise muti ndezvipi?",
    ],
}

for prompt_text in quick_prompts.get(language, quick_prompts["English"]):
    if st.button(prompt_text, key=prompt_text[:30]):
        st.chat_input(prompt_text)
