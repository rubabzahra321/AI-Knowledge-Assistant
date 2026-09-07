# ============================================================
# AI KNOWLEDGE ASSISTANT
# COMPLETE RAG PROJECT
# ============================================================
#
# PDF
#   ↓
# Text Extraction
#   ↓
# Text Splitting
#   ↓
# Gemini Embeddings
#   ↓
# ChromaDB
#   ↓
# Retrieval
#   ↓
# LangChain Prompt
#   ↓
# Gemini
#   ↓
# Answer + Sources
#
# Features:
# - Multiple PDF uploads
# - Duplicate prevention
# - Persistent ChromaDB
# - Conversation memory
# - Source display
# - Clear chat
# - Clear knowledge base
# - Error handling
# - FIX: safely extracts plain text from Gemini responses,
#        even when LangChain returns content as a list of
#        blocks instead of a plain string
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import hashlib

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "ai_knowledge_documents"

EMBEDDING_MODEL = "gemini-embedding-001"

LLM_MODEL = "gemini-3.1-flash-lite"

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

RETRIEVAL_K = 4


# ============================================================
# GLOBAL STYLES
#
# Purely cosmetic — no application logic lives here.
# Injects a refined, corporate-grade visual system:
# typography, palette, spacing, and component skins.
# ============================================================

def inject_custom_css():

    st.markdown(
        """
        <style>

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

        /* ============================================================
           ONE UNIFIED DARK THEME
           Same base color, same borders, same text tones — sidebar and
           main canvas are visually one surface, not two clashing panels.
        ============================================================ */

        :root {
            --aka-bg: #0B1220;
            --aka-bg-alt: #0E1626;
            --aka-panel: #131C2E;
            --aka-panel-alt: #172033;
            --aka-border: #22304A;
            --aka-text: #E6EAF2;
            --aka-text-dim: #94A3B8;
            --aka-muted: #64748B;
            --aka-blue: #3B82F6;
            --aka-blue-dim: #2563EB;
            --aka-success: #34D399;
            --aka-success-bg: #10241C;
            --aka-success-border: #1E4A38;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ---------- Whole app: one continuous dark surface ---------- */
        .stApp {
            background: linear-gradient(180deg, var(--aka-bg) 0%, var(--aka-bg-alt) 100%);
        }

        .block-container {
            padding-top: 2rem;
        }

        /* Default text color across the main canvas */
        .stApp, .stApp p, .stApp span, .stApp label,
        .stApp .stMarkdown, .stApp .stCaption, .stApp li {
            color: var(--aka-text);
        }

        /* ---------- Hide default Streamlit chrome ---------- */
        #MainMenu, footer, header {visibility: hidden;}

        /* ---------- Hero banner (same palette family as the rest) ---------- */
        .aka-hero {
            background: linear-gradient(120deg, #101A2E 0%, #16244A 55%, #1D3A73 100%);
            border: 1px solid var(--aka-border);
            border-radius: 16px;
            padding: 2.4rem 2.6rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 10px 30px -14px rgba(0, 0, 0, 0.55);
            position: relative;
            overflow: hidden;
        }

        .aka-hero::after {
            content: "";
            position: absolute;
            top: -60px;
            right: -60px;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, rgba(59,130,246,0) 70%);
            border-radius: 50%;
        }

        .aka-hero-title {
            font-family: 'Sora', sans-serif;
            font-size: 2.1rem;
            font-weight: 800;
            color: #FFFFFF;
            margin: 0;
            letter-spacing: -0.02em;
        }

        .aka-hero-subtitle {
            color: var(--aka-text-dim);
            font-size: 1.02rem;
            margin-top: 0.55rem;
            font-weight: 400;
            max-width: 640px;
            line-height: 1.5;
        }

        .aka-badge-row {
            margin-top: 1.1rem;
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .aka-pill {
            background: rgba(59,130,246,0.10);
            border: 1px solid rgba(59,130,246,0.28);
            color: #DBEAFE;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 500;
            backdrop-filter: blur(4px);
        }

        /* ---------- Sidebar: same surface family, one shade darker ---------- */
        section[data-testid="stSidebar"] {
            background: var(--aka-bg);
            border-right: 1px solid var(--aka-border);
        }

        section[data-testid="stSidebar"] * {
            color: var(--aka-text) !important;
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-family: 'Sora', sans-serif;
            font-weight: 700;
        }

        .aka-sidebar-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--aka-text-dim) !important;
            font-weight: 600;
            margin-top: 0.3rem;
        }

        section[data-testid="stSidebar"] .stButton button {
            background: var(--aka-panel);
            border: 1px solid var(--aka-border);
            border-radius: 10px;
            font-weight: 500;
            transition: all 0.15s ease;
        }

        section[data-testid="stSidebar"] .stButton button:hover {
            background: var(--aka-blue-dim);
            border-color: var(--aka-blue-dim);
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: var(--aka-panel);
            border: 1.5px dashed var(--aka-border);
            border-radius: 12px;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            background: var(--aka-panel-alt);
            border: 1px solid var(--aka-border);
        }

        /* ---------- Metric card ---------- */
        div[data-testid="stMetric"] {
            background: var(--aka-panel);
            border: 1px solid var(--aka-border);
            border-radius: 12px;
            padding: 0.9rem 1rem;
        }

        div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-family: 'Sora', sans-serif;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--aka-text-dim) !important;
        }

        /* ---------- Status blocks everywhere (success/info/error/warning) ---------- */
        .stAlert {
            border-radius: 10px !important;
            background: var(--aka-panel) !important;
            border: 1px solid var(--aka-border) !important;
        }

        .stAlert p {
            color: var(--aka-text) !important;
        }

        /* ---------- Section headers on main page ---------- */
        .aka-section-title {
            font-family: 'Sora', sans-serif;
            font-weight: 700;
            font-size: 1.0rem;
            color: var(--aka-text);
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin: 1.4rem 0 0.6rem 0;
        }

        /* ---------- Ready banner ---------- */
        .aka-ready-banner {
            background: var(--aka-success-bg);
            border: 1px solid var(--aka-success-border);
            color: #A7F3D0;
            padding: 0.6rem 1rem;
            border-radius: 10px;
            font-size: 0.88rem;
            margin-bottom: 0.6rem;
        }

        .aka-ready-banner b {
            color: #FFFFFF;
        }

        /* ---------- Chat bubbles ---------- */
        div[data-testid="stChatMessage"] {
            background: var(--aka-panel);
            border: 1px solid var(--aka-border);
            border-radius: 14px;
            padding: 0.35rem 0.4rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
            margin-bottom: 0.6rem;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] span {
            color: var(--aka-text) !important;
        }

        /* ---------- Chat input ---------- */
        .stChatInput textarea, .stChatInput {
            border-radius: 12px !important;
            background: var(--aka-panel) !important;
            border: 1px solid var(--aka-border) !important;
            color: var(--aka-text) !important;
        }

        /* ---------- Source expander cards ---------- */
        div[data-testid="stExpander"] {
            border: 1px solid var(--aka-border);
            border-radius: 10px;
            background: var(--aka-panel);
            margin-bottom: 0.45rem;
        }

        div[data-testid="stExpander"] summary {
            font-weight: 600;
            color: var(--aka-text) !important;
        }

        div[data-testid="stExpander"] p {
            color: var(--aka-text-dim) !important;
        }

        /* ---------- Footer ---------- */
        .aka-footer {
            text-align: center;
            color: var(--aka-muted);
            font-size: 0.8rem;
            margin-top: 2.4rem;
            padding-top: 1.2rem;
            border-top: 1px solid var(--aka-border);
        }

        .aka-footer b {
            color: var(--aka-text-dim);
        }

        </style>
        """,
        unsafe_allow_html=True
    )


inject_custom_css()


# ============================================================
# LOAD AI MODELS
# ============================================================

@st.cache_resource
def load_models():

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.2
    )

    return embeddings, llm


# ============================================================
# LOAD CHROMA VECTOR DATABASE
#
# IMPORTANT:
# `_embeddings` has an underscore.
#
# Streamlit will NOT try to hash this argument.
# This fixes:
#
# Cannot hash argument 'embeddings'
# ============================================================

@st.cache_resource
def load_vectorstore(_embeddings):

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings,
        persist_directory=CHROMA_PATH
    )

    return vectorstore


# ============================================================
# INITIALIZE MODELS AND VECTOR DATABASE
# ============================================================

try:

    embeddings, llm = load_models()

    vectorstore = load_vectorstore(
        embeddings
    )

except Exception as e:

    st.error(
        "Could not initialize the AI models or ChromaDB."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# CREATE UNIQUE FILE ID
# ============================================================

def create_file_id(file_bytes):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()[:16]


# ============================================================
# EXTRACT PLAIN TEXT FROM GEMINI RESPONSE
#
# WHY THIS EXISTS:
#
# Depending on the Gemini model / LangChain version, the
# `.content` field of an AIMessage is NOT always a plain
# string. Sometimes it comes back as a LIST of content
# blocks, e.g.:
#
#   [
#     {"type": "text", "text": "The actual answer..."},
#     {"extras": {"signature": "EnEKb..."}}
#   ]
#
# If you pass that raw list straight into st.markdown(),
# Streamlit just prints the Python object as text, which is
# exactly the "unstructured" / garbled output you were
# seeing (with the base64-looking "signature" blob).
#
# This helper normalizes ANY shape of `.content` down to a
# clean, plain string containing only the actual answer
# text — nothing else.
# ============================================================

def extract_text(content):

    # Already a plain string — nothing to do

    if isinstance(content, str):

        return content.strip()


    # List of content blocks — pull out only text parts

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, str):

                text_parts.append(block)

            elif isinstance(block, dict):

                # Standard shape: {"type": "text", "text": "..."}

                if block.get("type") == "text" and "text" in block:

                    text_parts.append(block["text"])

                # Fallback: some providers just use {"text": "..."}
                # without a "type" key

                elif "text" in block and "type" not in block:

                    text_parts.append(block["text"])

                # Anything else (e.g. {"extras": {...}}) is metadata,
                # not answer text — intentionally skipped

        return "\n".join(text_parts).strip()


    # Fallback for any other unexpected type

    return str(content).strip()


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    # --------------------------------------------------------
    # Read file bytes
    # --------------------------------------------------------

    file_bytes = uploaded_file.getvalue()

    file_id = create_file_id(
        file_bytes
    )

    file_name = uploaded_file.name


    # --------------------------------------------------------
    # Open PDF
    # --------------------------------------------------------

    reader = PdfReader(
        uploaded_file
    )


    # --------------------------------------------------------
    # Extract text page by page
    # --------------------------------------------------------

    pages_text = []


    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if page_text:

            pages_text.append(
                (
                    page_number,
                    page_text
                )
            )


    # --------------------------------------------------------
    # Check extracted text
    # --------------------------------------------------------

    if not pages_text:

        return {
            "success": False,
            "message": (
                f"Could not extract text from "
                f"'{file_name}'. "
                f"It may be a scanned/image-only PDF."
            )
        }


    # --------------------------------------------------------
    # TEXT SPLITTER
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP
    )


    documents = []

    ids = []


    # --------------------------------------------------------
    # Split every page
    # --------------------------------------------------------

    for page_number, page_text in pages_text:

        page_chunks = splitter.split_text(
            page_text
        )


        for chunk_number, chunk in enumerate(
            page_chunks
        ):

            # Unique ID for every chunk

            chunk_id = (
                f"{file_id}_"
                f"page_{page_number}_"
                f"chunk_{chunk_number}"
            )


            documents.append(
                {
                    "text": chunk,

                    "metadata": {

                        "source": file_name,

                        "file_id": file_id,

                        "page": page_number,

                        "chunk": chunk_number
                    }
                }
            )


            ids.append(
                chunk_id
            )


    # --------------------------------------------------------
    # CHECK FOR DUPLICATES
    # --------------------------------------------------------

    try:

        existing = vectorstore.get(
            ids=ids
        )

        existing_ids = set(
            existing.get(
                "ids",
                []
            )
        )

    except Exception:

        existing_ids = set()


    # --------------------------------------------------------
    # Keep only NEW chunks
    # --------------------------------------------------------

    new_documents = []

    new_ids = []


    for document, chunk_id in zip(
        documents,
        ids
    ):

        if chunk_id not in existing_ids:

            new_documents.append(
                document
            )

            new_ids.append(
                chunk_id
            )


    # --------------------------------------------------------
    # PDF ALREADY EXISTS
    # --------------------------------------------------------

    if not new_documents:

        return {

            "success": True,

            "duplicate": True,

            "message": (
                f"'{file_name}' is already "
                f"stored in the knowledge base."
            ),

            "chunks": 0
        }


    # --------------------------------------------------------
    # ADD TO CHROMADB
    # --------------------------------------------------------

    vectorstore.add_texts(

        texts=[
            document["text"]
            for document in new_documents
        ],

        metadatas=[
            document["metadata"]
            for document in new_documents
        ],

        ids=new_ids
    )


    return {

        "success": True,

        "duplicate": False,

        "message": (
            f"'{file_name}' added successfully."
        ),

        "chunks": len(new_documents)
    }


# ============================================================
# CLEAR KNOWLEDGE BASE
# ============================================================

def clear_knowledge_base():

    try:

        data = vectorstore.get()

        ids = data.get(
            "ids",
            []
        )


        if ids:

            vectorstore.delete(
                ids=ids
            )


        return True


    except Exception as e:

        st.error(
            f"Could not clear knowledge base: {e}"
        )

        return False


# ============================================================
# RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(

    search_kwargs={
        "k": RETRIEVAL_K
    }
)


# ============================================================
# RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are an AI Knowledge Assistant.

Your job is to answer the user's question using
ONLY the information provided in the document context.

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION:
{question}


RULES:

1. Use only the document context for factual answers.

2. Do not invent information.

3. Do not use outside knowledge.

4. If the answer cannot be found in the document
   context, say:

   "I couldn't find that information in the
   uploaded documents."

5. Keep the answer clear and easy to understand.

6. Use conversation history when the user asks
   a follow-up question.

7. If multiple document sections are relevant,
   combine them into one useful answer.


ANSWER:
"""
)


# ============================================================
# LANGCHAIN CHAIN
# ============================================================

chain = prompt | llm


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.2rem;">
            <div style="font-size:1.6rem;">🧠</div>
            <div style="font-family:'Sora',sans-serif; font-weight:700; font-size:1.15rem; color:#FFFFFF;">
                Knowledge Base
            </div>
        </div>
        <div style="color:#94A3B8; font-size:0.85rem; margin-bottom:1.2rem;">
            Retrieval-Augmented Generation Console
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="aka-sidebar-label">Upload Documents</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Upload one or more PDF documents "
        "to create your knowledge base."
    )


    # ========================================================
    # PDF UPLOADER
    # ========================================================

    uploaded_files = st.file_uploader(

        "Upload PDF files",

        type=["pdf"],

        accept_multiple_files=True,

        label_visibility="collapsed"
    )


    # ========================================================
    # PROCESS PDF FILES
    # ========================================================

    if uploaded_files:

        st.markdown(
            '<div class="aka-sidebar-label" style="margin-top:1rem;">Processing Documents</div>',
            unsafe_allow_html=True
        )


        for uploaded_file in uploaded_files:

            file_bytes = uploaded_file.getvalue()


            file_id = create_file_id(
                file_bytes
            )


            # Unique session key

            processed_key = (
                f"processed_{file_id}"
            )


            # ------------------------------------------------
            # Only process once per session
            # ------------------------------------------------

            if processed_key not in st.session_state:

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):

                    result = process_pdf(
                        uploaded_file
                    )


                # --------------------------------------------
                # Processing successful
                # --------------------------------------------

                if result["success"]:

                    st.session_state[
                        processed_key
                    ] = True


                    # Duplicate

                    if result.get(
                        "duplicate",
                        False
                    ):

                        st.info(
                            "ℹ️  " + result["message"]
                        )


                    # New document

                    else:

                        st.success(
                            "✅  " + result["message"]
                        )

                        st.caption(
                            f"📄  Chunks added: "
                            f"**{result['chunks']}**"
                        )


                # --------------------------------------------
                # Processing failed
                # --------------------------------------------

                else:

                    st.error(
                        "⚠️  " + result["message"]
                    )


    # ========================================================
    # KNOWLEDGE BASE STATISTICS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="aka-sidebar-label">Vector Store Status</div>',
        unsafe_allow_html=True
    )

    try:

        total_chunks = (
            vectorstore._collection.count()
        )

    except Exception:

        total_chunks = 0


    st.metric(
        "Stored Chunks",
        total_chunks
    )


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    st.markdown(
        '<div class="aka-sidebar-label" style="margin-top:1rem;">Session Controls</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "🧹  Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


    # ========================================================
    # CLEAR KNOWLEDGE BASE
    # ========================================================

    if st.button(
        "🗑️  Clear Knowledge Base",
        use_container_width=True
    ):

        if clear_knowledge_base():

            # Remove processed flags

            keys_to_remove = [

                key

                for key in st.session_state.keys()

                if key.startswith(
                    "processed_"
                )
            ]


            for key in keys_to_remove:

                del st.session_state[key]


            st.success(
                "Knowledge base cleared."
            )

            st.rerun()

    st.markdown(
        """
        <div style="margin-top:2rem; padding-top:1rem; border-top:1px solid #22304A;
                    color:#64748B; font-size:0.72rem; text-align:center;">
            Powered by LangChain · Gemini · ChromaDB
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN PAGE — HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="aka-hero">
        <div class="aka-hero-title">🧠 AI Knowledge Assistant</div>
        <div class="aka-hero-subtitle">
            An enterprise-style Retrieval-Augmented Generation (RAG) system —
            upload your PDFs and get grounded, source-cited answers powered
            by Gemini, LangChain, and ChromaDB.
        </div>
        <div class="aka-badge-row">
            <span class="aka-pill">🔗 LangChain</span>
            <span class="aka-pill">✨ Gemini</span>
            <span class="aka-pill">🗄️ ChromaDB</span>
            <span class="aka-pill">📎 Source-Grounded Answers</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

try:

    total_chunks = (
        vectorstore._collection.count()
    )

except Exception:

    total_chunks = 0


if total_chunks == 0:

    st.info(
        "👈 Upload one or more PDF files "
        "from the sidebar to start."
    )

else:

    st.markdown(
        f"""
        <div class="aka-ready-banner">
            ✅ Knowledge base ready — <b>{total_chunks}</b> chunks indexed and searchable.
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    '<div class="aka-section-title">💬 Conversation</div>',
    unsafe_allow_html=True
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# ============================================================
# HANDLE QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Make sure documents exist
    # --------------------------------------------------------

    if total_chunks == 0:

        st.warning(
            "Please upload a PDF before asking a question."
        )

        st.stop()


    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append(

        {
            "role": "user",

            "content": question
        }
    )


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # Generate AI response
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching documents..."
        ):

            try:

                # =================================================
                # RETRIEVAL
                # =================================================

                retrieved_docs = retriever.invoke(
                    question
                )


                # =================================================
                # CREATE CONTEXT
                # =================================================

                context_parts = []


                for doc in retrieved_docs:

                    source = doc.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = doc.metadata.get(
                        "page",
                        "Unknown"
                    )


                    context_parts.append(

                        f"""
Source: {source}
Page: {page}

Content:
{doc.page_content}
"""
                    )


                context = "\n\n".join(
                    context_parts
                )


                # =================================================
                # CREATE CHAT HISTORY
                # =================================================

                previous_messages = (
                    st.session_state.messages[:-1]
                )


                # Keep only recent messages

                recent_messages = (
                    previous_messages[-6:]
                )


                history_parts = []


                for message in recent_messages:

                    role = message["role"]

                    content = message["content"]


                    history_parts.append(

                        f"{role.upper()}: "
                        f"{content}"
                    )


                history = "\n".join(
                    history_parts
                )


                if not history:

                    history = (
                        "No previous conversation."
                    )


                # =================================================
                # SEND TO GEMINI THROUGH LANGCHAIN
                # =================================================

                response = chain.invoke(

                    {
                        "context": context,

                        "history": history,

                        "question": question
                    }
                )


                # =================================================
                # GET ANSWER
                #
                # FIX: response.content is not always a plain
                # string — normalize it with extract_text() so
                # we only ever display clean answer text, never
                # raw block dicts / signature metadata.
                # =================================================

                answer = extract_text(
                    response.content
                )


                # Safety net: if extraction somehow produced an
                # empty string, fall back to a clear message
                # instead of showing a blank bubble.

                if not answer:

                    answer = (
                        "I couldn't generate a readable "
                        "response for that question. "
                        "Please try rephrasing it."
                    )


                # =================================================
                # DISPLAY ANSWER
                # =================================================

                st.markdown(
                    answer
                )


                # =================================================
                # DISPLAY SOURCES
                # =================================================

                if retrieved_docs:

                    st.markdown(
                        """
                        <div style="margin-top:0.9rem; margin-bottom:0.4rem;
                                    font-weight:600; font-size:0.85rem; color:#94A3B8;
                                    display:flex; align-items:center; gap:0.4rem;">
                            📚 Sources referenced
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    for index, doc in enumerate(

                        retrieved_docs,

                        start=1
                    ):

                        source = doc.metadata.get(
                            "source",
                            "Unknown"
                        )

                        page = doc.metadata.get(
                            "page",
                            "Unknown"
                        )


                        with st.expander(

                            f"📄  Source {index} · "
                            f"{source} — "
                            f"Page {page}"
                        ):

                            st.write(
                                doc.page_content
                            )


                # =================================================
                # SAVE AI MESSAGE
                #
                # Store the CLEANED text, not the raw response
                # object, so chat history replay also stays plain.
                # =================================================

                st.session_state.messages.append(

                    {
                        "role": "assistant",

                        "content": answer
                    }
                )


            # =====================================================
            # ERROR HANDLING
            # =====================================================

            except Exception as e:

                st.error(
                    "Something went wrong while "
                    "processing your question."
                )


                with st.expander(
                    "Technical error"
                ):

                    st.code(
                        str(e)
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="aka-footer">
        Built with <b>Streamlit</b>, <b>LangChain</b>, <b>Gemini</b> &amp; <b>ChromaDB</b>
        — a Retrieval-Augmented Generation reference implementation.
    </div>
    """,
    unsafe_allow_html=True
)