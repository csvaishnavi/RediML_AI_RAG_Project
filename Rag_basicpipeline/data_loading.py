# %% [markdown]
# # Redi AI Waiter
#
# ## Step 1: Data Loading
#
# In this file we will:
#
# 1. Load CSV menu data
# 2. Load PDF menu data
# 3. Convert them into LangChain Documents

# %%
print("Step 1: Data Loading Started")

# %%
from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader,PyPDFLoader

print("Libraries imported successfully")

# %%
print("Step 1: Data Loading Started")
# %% [markdown]
# ## Find All CSV and PDF Files
#
# Instead of hardcoding one file name, we search the whole `data/` folder.
#
# This makes the project reusable for more restaurants later.

# %%
DATA_DIR = Path("data")

csv_files = list(DATA_DIR.glob("*.csv"))
pdf_files = list(DATA_DIR.glob("*.pdf"))

print("CSV files found:")
for file in csv_files:
    print("-", file)

print("\nPDF files found:")
for file in pdf_files:
    print("-", file)

# %% [markdown]
# ## Function 1: Load One CSV File
# CSVLoader converts each CSV row into one LangChain Document.

# %%
def load_one_csv(csv_path: Path):
    loader = CSVLoader(
        file_path=str(csv_path),
        encoding="utf-8"
    )
    documents = loader.load()
    for doc in documents:
        doc.metadata["source_type"] = "csv_menu"
        doc.metadata["file_name"] = csv_path.name
    return documents

# %% [markdown]
# ## Function 2: Load One PDF File
# PyPDFLoader converts each PDF page into one LangChain Document.

# %%
# We load the PDF and add metadata to each page document.
def load_one_pdf(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    for doc in documents:
        doc.metadata["source_type"] = "pdf_menu"
        doc.metadata["file_name"] = pdf_path.name
    return documents

# %% [markdown]
# ## Load All CSV Files

# %%
# We loop through all CSV files and load them into documents.
csv_documents = []
for csv_file in csv_files:
    docs = load_one_csv(csv_file)
    csv_documents.extend(docs)
print(f"Total CSV Documents: {len(csv_documents)}")

# %% [markdown]
# ## Load All PDF Files
# The PDF may contain extra information such as allergen code explanations.

# %%
# We loop through all PDF files and load them into documents.
pdf_documents = []
for pdf_file in pdf_files:
    docs = load_one_pdf(pdf_file)
    pdf_documents.extend(docs)
print(f"Total PDF Documents: {len(pdf_documents)}")

# %% [markdown]
# ## Combine CSV and PDF Documents
# This becomes the complete knowledge base for the AI waiter.

# %%
# We combine all CSV and PDF documents into one list.
all_documents = csv_documents + pdf_documents
print(f"Total Documents: {len(all_documents)}")

# %% [markdown]
# ## Check One CSV Document
# %%
# We check the first CSV document to see how it looks.
if csv_documents:
    print(csv_documents[0].page_content)
    print(csv_documents[0].metadata)
else:
    print("No CSV documents found.")

# %% [markdown]
# ## Check One PDF Document

# %%
if pdf_documents:
    print(pdf_documents[0].page_content[:1000])
    print(pdf_documents[0].metadata)
else:
    print("No PDF documents found.")

# %% [markdown]
# ## Final Check

# %%
print("=" * 50)
print("DATA LOADING COMPLETED")
print(f"CSV Documents: {len(csv_documents)}")
print(f"PDF Documents: {len(pdf_documents)}")
print(f"Total Documents: {len(all_documents)}")
print("=" * 50)

# %%
# We can also create a function to load all documents at once.
def load_all_documents():
    all_documents = []
    for csv_file in csv_files:
        all_documents.extend(load_one_csv(csv_file))
    for pdf_file in pdf_files:
        all_documents.extend(load_one_pdf(pdf_file))
    return all_documents
# %%
if __name__ == "__main__":
    all_documents = load_all_documents()
    print(f"Total documents loaded: {len(all_documents)}")