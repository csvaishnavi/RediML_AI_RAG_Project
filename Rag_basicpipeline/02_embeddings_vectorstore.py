# %% [markdown]
# # 02 Embeddings and Vector Store
#
# In this file, we create embeddings for our restaurant documents.
#
# Embeddings convert text into numbers/vectors.
# ChromaDB stores those vectors and helps us search similar menu information.

# %%
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from data_loading import load_all_documents
from langchain_huggingface import HuggingFaceEmbeddings
print("Step 2: Embeddings and Vector Store Started")

# %% [markdown]
# ## Load Documents from data_loading.py
#
# We reuse the function from `data_loading.py`.
# This avoids repeating the CSV and PDF loading code.

# %%

# We load all documents (CSV + PDF) into one list.
all_documents = load_all_documents()

print(f"Documents loaded: {len(all_documents)}")

# %% [markdown]
# ## Chunk Documents
#
# We split long documents into smaller chunks.
#
# This improves:
# - Retrieval quality
# - Search accuracy
# - LLM responses
#%%
# We use RecursiveCharacterTextSplitter to split documents into chunks of 500 characters with 100 characters overlap.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(all_documents)

print(f"Original documents: {len(all_documents)}")
print(f"Chunks created: {len(chunks)}")
# %%
# %% [markdown]
# ## Inspect One Chunk
#
# Let's look at the first chunk after splitting.
# %%
print(chunks[0].page_content)
print("\nMetadata:")
print(chunks[0].metadata)
# %%
# %% [markdown]
# ## Load Embedding Model
#
# We use sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 to convert text chunks into vectors.
#
# Similar meaning -> similar vectors.

# %%
# We load the HuggingFace embedding model.

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Embedding model loaded successfully")
# %% [markdown]
# ## Create Chroma Vector Database
#
# ChromaDB stores the embeddings of our chunks.
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("ChromaDB vector store created successfully")
# %% [markdown]
# ## Create Retriever
#
# The retriever searches ChromaDB and returns the most relevant chunks.
#
# `k=5` means it returns the top 5 most relevant chunks.
# %%
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# %% [markdown]
# ## Test Retriever
#
# We test whether the retriever can find allergy information.
# %%
query = "allergen code f meaning allergy legend?"

retrieved_docs = retriever.invoke(query)

print(f"Retrieved documents: {len(retrieved_docs)}")

for i, doc in enumerate(retrieved_docs, start=1):
    print("=" * 50)
    print(f"Document {i}")
    print(doc.page_content[:700])
    print(doc.metadata)
# %%
# %%
pdf_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source_type": "pdf_menu"}
    }
)

query = "allergen code f meaning"

pdf_results = pdf_retriever.invoke(query)

for i, doc in enumerate(pdf_results, start=1):
    print("=" * 50)
    print(f"PDF Document {i}")
    print(doc.page_content[:1200])
    print(doc.metadata)
# %%
# %%
dish_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source_type": "csv_menu"}
    }
)

allergy_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source_type": "pdf_menu"}
    }
)

query = "What does allergen code f mean?"

dish_results = dish_retriever.invoke(query)
allergy_results = allergy_retriever.invoke(query)

combined_results = dish_results + allergy_results

print(f"Dish documents: {len(dish_results)}")
print(f"Allergy PDF documents: {len(allergy_results)}")
print(f"Combined documents: {len(combined_results)}")

for i, doc in enumerate(combined_results, start=1):
    print("=" * 50)
    print(f"Document {i}")
    print(doc.page_content[:800])
    print(doc.metadata)
# %%
