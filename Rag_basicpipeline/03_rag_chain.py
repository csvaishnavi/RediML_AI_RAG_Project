# %% [markdown]
# # 03 RAG Chain
#
# In this file, we connect:
#
# - ChromaDB vector store
# - Retriever
# - Prompt
# - Groq LLM
#
# Final flow:
#
# User question
# ↓
# Retriever gets relevant chunks
# ↓
# Groq LLM generates final answer

# %%
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

print("Step 3: RAG Chain Started")
# %%
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    print("Groq API key loaded successfully")
else:
    print("Groq API key not found. Please check your .env file.")

# %%
# %% [markdown]
# ## Load Embedding Model
#
# We must use the same embedding model that we used when creating ChromaDB.

# %%
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Embedding model loaded")

# %% [markdown]
# ## Load Existing ChromaDB
#
# We already created the vector database in `02_embeddings_vectorstore.py`.
# Now we load it from the `chroma_db` folder.

# %%
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

print("ChromaDB loaded")

# %% [markdown]
# ## Create Retriever
#
# The retriever searches ChromaDB and returns relevant chunks.
# %%
dish_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source_type": "csv_menu"}
    }
)

allergy_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"source_type": "pdf_menu"}
    }
)


def allergy_aware_retriever(question):
    dish_docs = dish_retriever.invoke(question)

    allergy_docs = allergy_retriever.invoke(
        "Allergene Zusatzstoffe F Milcherzeugnisse laktosehaltig D Fisch"
    )

    return dish_docs + allergy_docs
# %%
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 8}
)

print("Retriever created")

# %% [markdown]
# ## Helper Function to Format Retrieved Documents

# %%
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# %% [markdown]
# ## Create AI Waiter Prompt
#
# The prompt tells the LLM how to behave.
#
# We instruct it to answer only from the restaurant menu context.

# %%
prompt = ChatPromptTemplate.from_template(
    """
Your responsibilities:

You are a helpful multilingual AI waiter for an Indian restaurant.
Use only the provided menu context.

Language and Translation Rules:
- Detect the language of the customer's question.
- If the question is English, answer only in English.
- If the question is German, answer only in German.
- Translate German menu item descriptions, categories, drinks, and side dishes into the customer's language.
- Keep original Indian dish names, such as Aloo Gobi, Channa Masala, Butter Chicken, Malai Kofta.
- Do not leave German words in an English answer unless they are original Indian dish names.
- If a German menu item has a clear English meaning, translate it.
Menu Rules:
1. Do not invent dishes.
2. If a dish has `is_vegan_possible: true`, say it can be prepared vegan.
3. If a dish has `is_vegetarian: true`, say it is vegetarian.
4. Do not say a dish is not vegan only because the normal description contains egg, milk, cheese, or yogurt, if `is_vegan_possible: true` is shown.
5. Mention prices when available.

6. For allergens, mention both the code and the meaning if available.say further details about the allergen if available, such as the allergen group or the specific ingredient that contains the allergen. 
7. If the information is missing, say you could not find it in the menu.
8. If the question is not related to the menu, say you can only answer questions related to the menu.
9.do not hallucinate information that is not in the menu.
10.if the question is vague, ask for clarification.
11. Use bullet points when listing dishes.
12.If information is not found in the menu context,
say:
"I could not find that information in the menu."
{context}

Customer Question:
{question}

Answer:
"""
)

print("Prompt created")

# %% [markdown]
# ## Create Groq LLM

# %%
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

print("Groq LLM created")

# %% [markdown]
# ## Build RAG Chain
#
# Chain flow:
#
# Question
# ↓
# Retriever gets context
# ↓
# Prompt combines context + question
# ↓
# Groq generates answer
# ↓
# Output parser returns text

# %%
# %%
rag_chain = (
    {
        "context": RunnableLambda(allergy_aware_retriever) | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("Allergy-aware RAG chain created")

# %% [markdown]
#  Test Question 1: Allergy

# %%
question = "What does allergen code f mean?"
answer = rag_chain.invoke(question)
print(answer)
# %% [markdown]
# ## Test Question 2: Vegetarian Dishes

# %%
question = "What vegetarian dishes do you recommend?"

answer = rag_chain.invoke(question)

print(answer)

# %% [markdown]
# ## Test Question 3: German Question

# %%
question = "Welche veganen Gerichte gibt es?"

answer = rag_chain.invoke(question)

print(answer)
# %%
question = "What vegetarian dishes do you recommend?"

answer = rag_chain.invoke(question)

print(answer)
# %%
