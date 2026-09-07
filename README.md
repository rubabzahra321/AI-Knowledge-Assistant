# 🧠 AI Knowledge Assistant

> An AI-powered document question-answering application built using Retrieval-Augmented Generation (RAG).

The AI Knowledge Assistant allows users to upload PDF documents and interact with their content through natural language questions.

Instead of relying solely on the general knowledge of a Large Language Model (LLM), the application retrieves relevant information from uploaded documents and provides it as context to generate accurate and context-aware responses.

This project was built as a hands-on implementation of modern Generative AI concepts, including **RAG, embeddings, vector databases, semantic search, and LLM-powered question answering**.

---

# 🚀 Features

- 📄 Upload and process PDF documents
- 📚 Support for multiple document uploads
- 🔍 Extract text from PDF files
- ✂️ Split large documents into manageable text chunks
- 🧮 Generate embeddings for document chunks
- 🗄️ Store embeddings in ChromaDB
- 🔎 Perform semantic similarity search
- 🤖 Generate context-aware responses using Google Gemini
- 💬 Maintain conversation history during the session
- 📌 Display sources used to generate answers
- 🚫 Prevent duplicate document processing
- 🧹 Clear conversation history
- 🗑️ Clear the knowledge base
- ⚠️ Handle document processing errors

---

# 🏗️ System Architecture

The application follows a Retrieval-Augmented Generation (RAG) architecture.

```text
                        ┌──────────────────┐
                        │   PDF Documents  │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Text Extraction  │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  Text Chunking   │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Gemini Embeddings│
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ ChromaDB Vector  │
                        │    Database      │
                        └────────┬─────────┘
                                 │
                                 ▼
                            USER QUESTION
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Query Embedding  │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Semantic Search  │
                        │    Retrieval     │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Relevant Context │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  Google Gemini   │
                        │       LLM        │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ AI Response with │
                        │ Source References│
                        └──────────────────┘
```

---

# 🧠 How It Works

The application follows a complete RAG pipeline:

### 1. Document Upload

Users upload one or multiple PDF documents through the Streamlit interface.

### 2. Text Extraction

The application extracts text content from the uploaded PDF documents.

### 3. Text Chunking

Large documents are divided into smaller chunks to make them easier to process and retrieve.

Chunking improves retrieval performance because the system searches for relevant sections instead of entire documents.

### 4. Embedding Generation

Each text chunk is converted into a numerical vector representation called an embedding.

Embeddings capture the semantic meaning of text.

### 5. Vector Storage

The generated embeddings are stored in ChromaDB, which acts as the application's vector database.

### 6. User Question

When a user asks a question, the question is also converted into an embedding.

### 7. Semantic Retrieval

The application compares the question embedding with stored document embeddings.

The most semantically relevant document chunks are retrieved.

### 8. Context Generation

The retrieved chunks are combined and provided to the Large Language Model as context.

### 9. AI Response

Google Gemini uses the retrieved context and the user's question to generate a relevant and context-aware answer.

---

# 🔄 RAG Workflow

## Document Processing Pipeline

```text
PDF Documents
      │
      ▼
Text Extraction
      │
      ▼
Text Chunking
      │
      ▼
Embedding Generation
      │
      ▼
ChromaDB Vector Database
```

## Question Answering Pipeline

```text
User Question
      │
      ▼
Query Embedding
      │
      ▼
Semantic Similarity Search
      │
      ▼
Retrieve Relevant Document Chunks
      │
      ▼
Provide Context to LLM
      │
      ▼
Google Gemini
      │
      ▼
Context-Aware Response
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| LangChain | RAG pipeline and LLM orchestration |
| Google Gemini | Large Language Model |
| Gemini Embeddings | Text embedding generation |
| ChromaDB | Vector database |
| PyPDF | PDF text extraction |
| Python Dotenv | Environment variable management |

---

# 📚 Concepts Implemented

This project includes hands-on implementation of important Generative AI concepts:

- Artificial Intelligence
- Generative AI
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Text Chunking
- Chunk Overlap
- Vector Embeddings
- Vector Databases
- ChromaDB
- Semantic Search
- Similarity Search
- Document Retrieval
- Context Injection
- Prompt Engineering
- Conversation History
- Source Attribution
- Persistent Vector Storage

---

# 📂 Project Structure

```text
AI-Knowledge-Assistant/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── .env
│
├── documents/
│
├── chroma_db/
│
└── screenshots/
    ├── home.png
    ├── upload.png
    ├── chat.png
    └── sources.png
```

> **Note:** `.env`, `.venv`, `chroma_db`, and uploaded documents should not be pushed to GitHub.

---

# ⚙️ Installation and Setup

## 1. Clone the Repository

```bash
git clone https://github.com/rubabzahra321/AI-Knowledge-Assistant.git
```

Navigate to the project directory:

```bash
cd AI-Knowledge-Assistant
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment.

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the root directory of the project.

Add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

⚠️ Never upload your `.env` file or API key to GitHub.

---

## 5. Run the Application

```bash
streamlit run app.py
```

The application will start locally and open in your browser.

---

# ⚙️ Core Configuration

The document processing pipeline uses configurable parameters.

| Parameter | Value |
|---|---|
| Chunk Size | 1000 |
| Chunk Overlap | 200 |
| Retrieval Results | 4 |

These parameters help balance document context, retrieval quality, and response relevance.

---

# 🤖 AI Models

## Language Model

The application uses Google Gemini as the Large Language Model for generating responses.

The LLM receives:

- The user's question
- Relevant retrieved document chunks
- Conversation context

It then generates a context-aware response.

## Embedding Model

Google's embedding model is used to convert:

- Document text chunks
- User questions

into numerical vector representations.

These vectors allow the application to perform semantic similarity searches.

---

# 💡 Why RAG?

Large Language Models have limitations.

They may:

- Not know information from private documents
- Have outdated knowledge
- Generate incorrect information
- Hallucinate responses

Retrieval-Augmented Generation helps address these limitations by retrieving relevant information before generating an answer.

Without RAG:

```text
User Question
      │
      ▼
LLM
      │
      ▼
Answer
```

With RAG:

```text
User Question
      │
      ▼
Retrieve Relevant Information
      │
      ▼
Provide Context to LLM
      │
      ▼
Generate Context-Aware Answer
```

This allows the AI assistant to answer questions based on the user's uploaded documents.

---

# ✨ Application Features in Detail

## 📄 Multiple PDF Upload

Users can upload multiple PDF documents and create a combined knowledge base.

The application processes the documents and makes their information available for question answering.

---

## 🗄️ Persistent Vector Database

Document embeddings are stored in ChromaDB.

This allows the knowledge base to persist instead of being recreated every time the application processes a question.

---

## 🔍 Semantic Search

The application does not rely only on exact keyword matching.

Instead, it uses embeddings to understand the semantic meaning of the user's question.

For example:

```text
Question:
What is Retrieval-Augmented Generation?

Document:
RAG improves LLM responses by retrieving relevant external information.
```

Although the wording is different, semantic search can identify that both statements are related.

---

## 📌 Source References

The application displays information about the source documents used to generate responses.

This improves transparency and allows users to understand where the information originated.

---

## 💬 Conversation History

The application maintains conversation history during the active session.

This allows users to interact naturally with the AI assistant and maintain context during a conversation.

---

## 🚫 Duplicate Prevention

The application prevents already processed documents from being added repeatedly to the knowledge base.

This avoids unnecessary duplicate data and repeated embedding generation.

---

## 🧹 Knowledge Base Management

Users can clear the existing knowledge base and remove stored document embeddings.

This allows users to create a new knowledge base when needed.

---

# 🎯 Project Purpose

This project was developed as part of my learning journey in Artificial Intelligence and Generative AI.

The goal was to move beyond theoretical learning and gain practical experience by building a complete Retrieval-Augmented Generation application.

Through this project, I gained hands-on experience with:

- Building AI-powered applications
- Integrating Large Language Models
- Working with embeddings
- Implementing vector databases
- Designing a RAG pipeline
- Semantic document retrieval
- Prompt engineering
- Context-aware AI responses
- Building an interactive AI interface

This project represents an important step in my journey toward building more advanced AI and Agentic AI applications.

---

# 🔮 Future Improvements

Potential improvements for the project include:

- Support for additional document formats
- DOCX document support
- CSV and structured data support
- Persistent chat history
- User authentication
- Multi-user support
- Streaming AI responses
- Advanced document management
- Metadata filtering
- Hybrid search
- Improved retrieval strategies
- Conversation memory
- Cloud deployment
- Agent-based document analysis
- Multi-modal document support

---

# 📈 Learning Journey

This project is part of my ongoing learning journey:

```text
Python
   │
   ▼
Generative AI
   │
   ▼
Large Language Models
   │
   ▼
LangChain
   │
   ▼
Embeddings & Vector Databases
   │
   ▼
Retrieval-Augmented Generation
   │
   ▼
AI Agents
   │
   ▼
Agentic AI
```

I believe in learning by building practical projects, experimenting with new technologies, and continuously improving my understanding of Artificial Intelligence.

---

# 👩‍💻 Author

## Rubab Zahra

**Computer Science Undergraduate | AI & Software Development Enthusiast**

Currently exploring:

- Artificial Intelligence
- Generative AI
- Large Language Models
- Retrieval-Augmented Generation
- Vector Databases
- LangChain
- LangGraph
- AI Agents
- Agentic AI

---

# 🤝 Connect With Me

- LinkedIn: [Rubab Zahra](https://www.linkedin.com/in/rubab-zahra-765827332/)
- GitHub: [rubabzahra321](https://github.com/rubabzahra321)

---

# ⭐ Support

If you find this project interesting, consider giving the repository a star.

Your support motivates me to continue learning, building, and sharing my projects.

---

> This project was built as a hands-on learning project to explore and understand Retrieval-Augmented Generation (RAG) and modern Generative AI application development.
