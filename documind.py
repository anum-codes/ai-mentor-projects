import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

# Works both locally and on Streamlit Cloud
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except:
        return os.getenv("GEMINI_API_KEY")

# Page config
st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-top: 0;
    }
    .stat-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=get_api_key(),
        temperature=0.3
    )

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=get_api_key()
    )

def process_pdf(uploaded_file):
    """Process uploaded PDF and create RAG chain"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyMuPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory="./documind_db"
    )

    os.unlink(tmp_path)
    return vectorstore, len(documents), len(chunks)

def create_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""
You are DocuMind, an expert AI document analyst.
Answer questions based ONLY on the provided context.
If the answer isn't in the context, say so clearly.
Be concise, accurate, and helpful.
Use bullet points for lists.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_history(history):
        if not history:
            return "No previous conversation."
        return "\n".join([f"Human: {h}\nAI: {a}" for h, a in history])

    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history") | RunnablePassthrough()
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return chain, retriever

# ─── HEADER ───────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-header">🧠 DocuMind</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered document intelligence — ask anything about your PDFs</p>', unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Reading and indexing document..."):
                vectorstore, pages, chunks = process_pdf(uploaded_file)
                chain, retriever = create_chain(vectorstore)

                st.session_state.vectorstore = vectorstore
                st.session_state.chain = chain
                st.session_state.retriever = retriever
                st.session_state.doc_stats = {
                    "name": uploaded_file.name,
                    "pages": pages,
                    "chunks": chunks
                }
                st.session_state.chat_history = []
                st.session_state.messages = []

            st.success("✅ Document ready!")

    if st.session_state.doc_stats:
        st.divider()
        st.subheader("📊 Document Stats")
        stats = st.session_state.doc_stats

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{stats['pages']}</div>
                <div class="stat-label">Pages</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{stats['chunks']}</div>
                <div class="stat-label">Chunks</div>
            </div>""", unsafe_allow_html=True)

        st.caption(f"📄 {stats['name']}")
        st.divider()

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown("**💡 Try asking:**")
    st.caption("• Summarize this document")
    st.caption("• What are the key findings?")
    st.caption("• What does it say about [topic]?")
    st.caption("• List the main recommendations")

# ─── MAIN CHAT AREA ───────────────────────────────────────
if not st.session_state.vectorstore:
    st.markdown("### 👈 Upload a PDF to get started")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📋 **Summarize**\nGet instant summaries of any document")
    with col2:
        st.info("🔍 **Search**\nFind specific information instantly")
    with col3:
        st.info("💬 **Converse**\nAsk follow-up questions naturally")
else:
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                st.caption(f"📌 Sources: Pages {message['sources']}")

    # Chat input
    if question := st.chat_input("Ask anything about your document..."):
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history_str = "\n".join([
                    f"Human: {h}\nAI: {a}"
                    for h, a in st.session_state.chat_history
                ])

                source_docs = st.session_state.retriever.invoke(question)
                answer = st.session_state.chain.invoke({
                    "question": question,
                    "chat_history": history_str
                })

                pages = sorted(set(
                    doc.metadata['page'] + 1
                    for doc in source_docs
                    if 'page' in doc.metadata
                ))

            st.markdown(answer)
            if pages:
                st.caption(f"📌 Sources: Pages {pages}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": pages
        })
        st.session_state.chat_history.append((question, answer))