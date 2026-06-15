# Redi  Resturant Assistant – Redi Restaurant AI Waiter

A restaurant menu chatbot built with Retrieval-Augmented Generation (RAG).

The system helps users ask natural language questions about menu items, prices, allergens, vegetarian options, vegan options, and recommendations.

This project uses structured menu data, PDF menu information, ChromaDB retrieval, Hugging Face embeddings, Groq LLM, and Streamlit.

## Team Contribution

### Vaishnavi

- Data cleaning
- Data loading
- Chunking
- Basic pipeline

### Emily
- Data source
- Embeddings
- Vector store
- RAG chain

### Watcharapon Anaketanaset
- RAG architecture Preparation
- PDF to CSV conversion
- Retriever
- Streamlit app


## Project Objective

The objective of this project is to build an AI-powered restaurant assistant that can:

- Answer menu-related questions in natural language
- Provide dish recommendations
- Explain allergens and ingredients
- Support customer preferences such as vegetarian, vegan, allergy, budget, and spice level
- Generate menu-grounded answers using RAG

## What is RAG?

RAG means Retrieval-Augmented Generation.

Instead of asking the LLM to answer from memory, the system first retrieves relevant menu information from the vector database and then sends only that context to the LLM.

```text
User Question
      ↓
Retriever
      ↓
Relevant Menu Context
      ↓
Groq LLM
      ↓
Final Answer
```

This reduces hallucination and keeps answers grounded in the restaurant menu data.

## Project Architecture

```text
Raw PDF / CSV Menu Data
        ↓
PDF to CSV Conversion
        ↓
Data Cleaning
        ↓
Data Loader
        ↓
Chunking
        ↓
Embeddings
        ↓
ChromaDB Vector Store
        ↓
Retriever + Structured Filters
        ↓
RAG Chain
        ↓
Groq LLM
        ↓
Streamlit Chatbot
```

## Project Structure

```text
restaurant-rag/
│
├── data/
│   ├── swagat_menu_raw.csv
│   ├── swagat_menu.csv
│   └── Menu of Swagat Indian Restaurant.pdf
│
├── clean_menu_csv.py
├── data_loader.py
├── chunking.py
├── embeddings.py
├── vector_store.py
├── retriever.py
├── rag_chain.py
├── streamlit_app.py
│
├── .env
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Main Components

### 1. Data Cleaning

File:

```text
clean_menu_csv.py
```

This file cleans the raw menu CSV and creates the final cleaned CSV used by the RAG system.

Main tasks:

- Normalize boolean values
- Normalize allergen codes
- Convert allergen codes to readable names
- Detect drinks
- Fix special cases such as Kingfisher beer and Lassi
- Create a search_text field for better retrieval

Packages used:

```text
pandas
pathlib
```

### 2. Data Loading

File:

```text
data_loader.py
```

This file loads CSV and PDF files and converts them into LangChain documents.

Loaders used:

```text
CSVLoader
PyPDFLoader
```

The metadata added includes:

```text
source_type
file_name
```

### 3. Chunking

File:

```text
chunking.py
```

This file splits documents for retrieval.

Strategy:

- CSV rows are kept complete because each row already represents one menu item
- PDF documents are split into smaller chunks

Text splitter used:

```text
RecursiveCharacterTextSplitter
```

Chunk settings:

```text
chunk_size = 500
chunk_overlap = 100
```

### 4. Embeddings

File:

```text
embeddings.py
```

Embedding model used:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Package used:

```text
langchain_huggingface
```

Embeddings convert text into numerical vectors so similar meanings can be searched.

### 5. Vector Store

File:

```text
vector_store.py
```

Vector database used:

```text
ChromaDB
```

Package used:

```text
langchain_chroma
```

The vector store is saved in:

```text
chroma_db/
```

### 6. Retriever

File:

```text
retriever.py
```

The retriever searches ChromaDB and applies structured filters.

It supports:

- Query expansion
- Price filtering
- Vegetarian filtering
- Vegan filtering
- Allergy filtering
- Meal type filtering
- Drink detection
- Spice safety notes
- PDF fallback for non-strict questions

### 7. RAG Chain

File:

```text
rag_chain.py
```

LLM provider:

```text
Groq
```

Model used:

```text
llama-3.1-8b-instant
```

Package used:

```text
langchain_groq
```

The RAG chain:

1. Receives the user question
2. Calls the retriever
3. Builds a safe prompt
4. Sends filtered menu context to Groq LLM
5. Returns the final customer-friendly answer

### 8. Streamlit App

File:

```text
streamlit_app.py
```

Features:

- Chat interface
- Chat history
- Example questions
- Sidebar
- Clear chat button
- Friendly AI waiter responses

Package used:

```text
streamlit
```

## Security

The API key is stored in a `.env` file and should not be pushed to GitHub.

Example `.env` file:

```text
GROQ_API_KEY=your_api_key_here
```

Make sure `.gitignore` contains:

```text
.env
.venv/
chroma_db/
__pycache__/
*.pkl
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd restaurant-rag
```

### 2. Create and activate virtual environment

Using uv:

```bash
uv sync
```

### 3. Add Groq API key

Create a `.env` file:

```text
GROQ_API_KEY=your_api_key_here
```

### 4. Clean the CSV data

```bash
python clean_menu_csv.py
```

### 5. Create the vector store

```bash
python vector_store.py
```

### 6. Test the RAG chain

```bash
python rag_chain.py
```

### 7. Run the Streamlit app

```bash
python -m streamlit run streamlit_app.py
```

## Example Questions

```text
I want mild chicken under 12 euros and no nuts.
```

```text
Show me vegetarian lunch menu items.
```

```text
Do you have vegan drinks?
```

```text
I am allergic to milk. What can I eat?
```

```text
Which dishes are under 5 euros?
```

## Key Features

- Menu-grounded RAG chatbot
- Structured CSV filtering
- PDF and CSV support
- Allergen-aware responses
- Budget-aware recommendations
- Vegetarian and vegan support
- ChromaDB vector search
- Groq Llama model integration
- Streamlit chat interface



## Summary

This project demonstrates how RAG can be used to build a practical restaurant AI assistant.

By combining cleaned menu data, embeddings, ChromaDB retrieval, structured filtering, Groq LLM, and Streamlit, the chatbot can answer customer questions in a reliable and menu-grounded way.
