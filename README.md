# 🧠 AI Knowledge Assistant

An AI-powered Knowledge Assistant built using Retrieval-Augmented Generation (RAG). The application allows users to upload PDF documents and ask questions based on their content.

Instead of relying only on the language model's general knowledge, the system retrieves relevant information from the uploaded documents and uses that context to generate accurate and context-aware responses.

Built with **Python, Streamlit, LangChain, Google Gemini, and ChromaDB**.

---

## 🚀 Features

- 📄 Upload and process multiple PDF documents
- 🔍 Extract text from uploaded PDFs
- ✂️ Split documents into manageable text chunks
- 🧮 Generate vector embeddings using Gemini
- 🗄️ Store embeddings in a persistent ChromaDB vector database
- 🔎 Retrieve relevant document chunks using semantic search
- 🤖 Generate context-aware answers using Google Gemini
- 💬 Maintain conversation history during the session
- 📚 Display source information for retrieved answers
- 🚫 Prevent duplicate document processing
- 🧹 Clear chat history
- 🗑️ Clear the knowledge base
- ⚠️ Error handling for invalid or unsupported documents

---

# 🏗️ How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

```text
                PDF Documents
                      │
                      ▼
               Text Extraction
                      │
                      ▼
                Text Splitting
                      │
                      ▼
              Gemini Embeddings
                      │
                      ▼
             ChromaDB Vector Store
                      │
                      ▼
                User Question
                      │
                      ▼
              Semantic Retrieval
                      │
                      ▼
              Relevant Documents
                      │
                      ▼
               LangChain Prompt
                      │
                      ▼
                Google Gemini
                      │
                      ▼
          Context-Aware AI Response
                      │
                      ▼
                Answer + Sources
