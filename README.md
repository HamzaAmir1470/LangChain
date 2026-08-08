# 🚀 Generative AI & LangChain Masterclass — Complete Course Repository

[![YouTube Playlist](https://img.shields.io/badge/YouTube-Playlist-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/playlist?list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0)
[![Course Creator](https://img.shields.io/badge/Channel-CampusX-blue?style=for-the-badge&logo=youtube)](https://www.youtube.com/@CampusX-official)
[![Tech Stack](https://img.shields.io/badge/Tech-Python_%7C_LangChain_%7C_OpenAI_%7C_VectorDB-green?style=for-the-badge)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

Welcome to the ultimate repository for the **Generative AI using LangChain** course series by **CampusX**! This comprehensive guide and codebase hub covers everything from fundamental LLM concepts to advanced Retrieval-Augmented Generation (RAG) pipelines, Autonomous AI Agents, and full-stack GenAI application deployment.

---

## 📌 Table of Contents

- [📖 About the Series](#-about-the-series)
- [🎯 Learning Objectives](#-learning-objectives)
- [🛠️ Tech Stack & Prerequisites](#️-tech-stack--prerequisites)
- [🗺️ Curriculum & Detailed Module Breakdown](#️-curriculum--detailed-module-breakdown)
  - [Module 1: Foundations of Generative AI & LLMs](#module-1-foundations-of-generative-ai--llms)
  - [Module 2: LangChain Essentials (Models, Prompts, Parsers)](#module-2-langchain-essentials-models-prompts-parsers)
  - [Module 3: LangChain Chains & LCEL](#module-3-langchain-chains--lcel)
  - [Module 4: Memory Systems in Conversational AI](#module-4-memory-systems-in-conversational-ai)
  - [Module 5: Indexes, Vector Stores & Embeddings](#module-5-indexes-vector-stores--embeddings)
  - [Module 6: Retrieval-Augmented Generation (RAG) Architecture](#module-6-retrieval-augmented-generation-rag-architecture)
  - [Module 7: LangChain Agents & Custom Tools](#module-7-langchain-agents--custom-tools)
  - [Module 8: End-to-End GenAI Projects](#module-8-end-to-end-genai-projects)
- [💻 Environment Setup & Quickstart](#-environment-setup--quickstart)
- [🚀 Hands-on Project Architecture](#-hands-on-project-architecture)
- [⭐ Resource Links & Acknowledgments](#-resource-links--acknowledgments)

---

## 📖 About the Series

This course series, taught on **CampusX**, provides a hands-on, production-grade dive into **LangChain** and **Generative AI**. It bridges the gap between understanding Large Language Models (LLMs) theoretically and building scalable, real-world GenAI products.

- 📺 **Playlist URL:** [CampusX - Generative AI using LangChain](https://www.youtube.com/playlist?list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0)
- 👨‍🏫 **Instructor:** Nitish Singh (CampusX)
- 🎯 **Target Audience:** Data Scientists, Software Developers, AI/ML Engineers, and Tech Enthusiasts wanting to build GenAI applications.

---

## 🎯 Learning Objectives

By going through this repository and playlist, you will master:

1. Core LLM architecture, Tokenization, Prompting, and Temperature/Top-P parameters.
2. Building modular pipelines using **LangChain Core** and **LCEL (LangChain Expression Language)**.
3. Managing state and contextual conversation memory in AI chatbots.
4. Implementing semantic search with **Embeddings**, **Vector Databases** (FAISS, ChromaDB, Pinecone).
5. Designing enterprise **RAG (Retrieval-Augmented Generation)** systems over PDF, HTML, and custom datasets.
6. Creating reasoning-driven **Autonomous AI Agents** equipped with Custom Python Tools and Search APIs.
7. Deploying full-stack GenAI web apps using **Streamlit** / **FastAPI**.

---

## 🛠️ Tech Stack & Prerequisites

### Tech Stack

- **Language:** Python 3.10+
- **Framework:** LangChain (`langchain`, `langchain-community`, `langchain-openai`, `langchain-core`)
- **LLM Providers:** OpenAI (GPT-4o, GPT-3.5-turbo), Google Gemini, Hugging Face Hub
- **Vector Databases:** FAISS, ChromaDB, Pinecone
- **Frontend / Deployment:** Streamlit, FastAPI, Docker

### Prerequisites

- Proficiency in **Python** (OOP, Async, Functions)
- Basic understanding of API requests and JSON handling
- Basic knowledge of Machine Learning & Data Processing (Pandas, NumPy)

---

## 🗺️ Curriculum & Detailed Module Breakdown

### Module 1: Foundations of Generative AI & LLMs

- Introduction to Generative AI vs Traditional Machine Learning
- How Large Language Models (LLMs) work: Transformer Architecture, Tokens, and Context Windows
- API Integrations: OpenAI API, Google Gemini API, Hugging Face Inference API
- Key LLM Parameters: `temperature`, `top_p`, `max_tokens`, `frequency_penalty`

### Module 2: LangChain Essentials (Models, Prompts, Parsers)

- **Model I/O:** `ChatModels` vs `LLMs`
- **Prompt Engineering:** `PromptTemplate`, `ChatPromptTemplate`, `FewShotPromptTemplate`
- **Output Parsers:** `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser`, `StructuredOutputParser`
- Handling structured output generation from unstructured text responses

### Module 3: LangChain Chains & LCEL

- Classic Chains: `LLMChain`, `SimpleSequentialChain`, `SequentialChain`, `RouterChain`
- **LangChain Expression Language (LCEL):**
  - Runnable primitives: `RunnableSequence`, `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`
  - Pipe operator (`|`) syntax and composition
  - Streaming, batch processing, and async execution in LCEL

### Module 4: Memory Systems in Conversational AI

- Why LLMs are stateless and how Memory solves state management
- `ConversationBufferMemory` & `ConversationBufferWindowMemory`
- `ConversationSummaryMemory` & `ConversationSummaryBufferMemory`
- `VectorStoreRetrieverMemory` for long-term historical context
- Persisting chat histories into databases (SQLite, Redis, MongoDB)

### Module 5: Indexes, Vector Stores & Embeddings

- Text Embeddings: What are semantic vectors? (OpenAI Embeddings, HuggingFace BGE)
- **Document Loaders:** PyPDF, DirectoryLoader, WebBaseLoader, CSVLoader
- **Text Splitters:** `CharacterTextSplitter`, `RecursiveCharacterTextSplitter`
- **Vector Databases:** Local (FAISS, ChromaDB) vs Cloud (Pinecone, Qdrant)
- Similarity search techniques: Cosine Similarity, Euclidean Distance, Dot Product

### Module 6: Retrieval-Augmented Generation (RAG) Architecture

- Understanding the RAG Pipeline: Ingestion, Indexing, Retrieval, and Generation
- Naive RAG vs Advanced RAG:
  - Query Transformation (Multi-Query Expansion, Sub-Query Generation)
  - Context Compression & Re-ranking (Cohere Rerank)
  - Hybrid Search (Keyword + Dense Vector Search)
- Building an End-to-End "Chat with your PDF" System

### Module 7: LangChain Agents & Custom Tools

- Agent Architecture: ReAct (Reasoning + Acting) Framework
- Pre-built LangChain Tools: SerpAPI, DuckDuckGo Search, Wikipedia Tool, Python REPL
- Defining Custom Tools using `@tool` decorator and `BaseTool`
- Agent Executors and handling structured agent decisions

### Module 8: End-to-End GenAI Projects

- **Project 1:** Interactive Customer Support Chatbot with Memory
- **Project 2:** Enterprise Document QA (Chat with PDF / Multi-Doc RAG System)
- **Project 3:** AI Research Agent capable of Web Search & Automated Report Generation
- **Project 4:** Code Assistant and Explanation Bot

---

## 💻 Environment Setup & Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/generative-ai-langchain-campusx.git
cd generative-ai-langchain-campusx
```

### 2. Create and Activate Virtual Environment

```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
PINECONE_API_KEY=your_pinecone_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here
```

---

## 🚀 Hands-on Code Example: Minimal RAG Pipeline

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 1. Load Document
loader = PyPDFLoader("sample.pdf")
docs = loader.load()

# 2. Split Document into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Create Vector Store & Retriever
vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Define Prompt & Model
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 5. Build LCEL RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Execute Query
response = rag_chain.invoke("What are the key highlights of the document?")
print("AI Response:", response)
```

---

## ⭐ Resource Links & Acknowledgments

- 📺 **Official Playlist:** [Generative AI using LangChain on YouTube](https://www.youtube.com/playlist?list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0)
- 🎓 **CampusX Channel:** [CampusX Official YouTube Channel](https://www.youtube.com/@CampusX-official)
- 🦜🔗 **LangChain Documentation:** [python.langchain.com](https://python.langchain.com/)
- 🤝 **Community & Support:** Join the CampusX Discord / Telegram community for discussions, Q&A, and project reviews.

---

<p center>
<i>Crafted with ❤️ for AI Engineers and Developers learning Generative AI. Happy Coding! 🚀</i>
</p>
