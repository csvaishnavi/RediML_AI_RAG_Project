# %% [markdown]
# # Redi AI Waiter
#
# Streamlit application for restaurant RAG chatbot
#
# Features:
# - Chatbot
# - Chat History
# - Language Selection
# - Example Questions
# - Menu Preview
# - Allergy-Aware Retrieval


#%%
import os
import pandas as pd
from dotenv import load_dotenv
import base64
import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
#%%
load_dotenv()
# %% [markdown]
# ## Load Menu Data
#
# Loads CSV menu for menu preview tab
# %% [markdown]
# ## Load RAG Chain
#
# Creates:
# - Embeddings
# - ChromaDB
# - Retrievers
# - Prompt
# - Groq LLM
# - RAG Chain
# %%[markdown]
# %% [markdown]
# ### Dish Retriever
#
# Searches menu items
#%%
@st.cache_resource
def load_rag_chain():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

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
        "Allergene Zusatzstoffe F Milcherzeugnisse laktosehaltig D Fisch G Nüsse I Senf"
    )

        return dish_docs + allergy_docs
# %% [markdown]
# ### Format Retrieved Documents
# ### AI Waiter Prompt
#%%
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful multilingual AI waiter for an Indian restaurant.

You MUST answer in the SAME LANGUAGE as the customer's question.
If the question is English, answer only in English.
If the question is German, answer only in German.

Use only the provided menu context.

Rules:
- Do not invent dishes.
- If `is_vegan_possible: true`, say the dish can be prepared vegan.
- If `is_vegetarian: true`, say the dish is vegetarian.
- Mention prices when available.
- For allergens, mention both the code and the meaning if available.
- D = fish.
- F = milk products / lactose.
- G = nuts / nut products.
- I = mustard.
- Do not tell the customer to contact restaurant staff.
- Use bullet points for recommendations.

Context:
{context}

Customer question:
{question}

Answer:
"""
    )
# %% [markdown]
#### Groq LLM
#%%
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    rag_chain = (
        {
            "context": RunnableLambda(allergy_aware_retriever) | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# %% [markdown]
# ## Streamlit Page Configuration

# %%
st.set_page_config(
    page_title="Redi AI Waiter",
    page_icon="🍛",
    layout="wide"
)

st.title("🍛 Redi AI Waiter")
st.write("Ask about ingredients, allergens, vegetarian/vegan dishes, prices, or recommendations.")
st.caption("A multilingual RAG chatbot for Indian restaurant menu questions")
with st.sidebar:
    st.header("👋 Welcome")
    st.write(
        "Ask me about dishes, ingredients, allergens, vegan options, vegetarian dishes, prices, and recommendations."
    )

    language_choice = st.selectbox(
        "🌐 Preferred language",
        ["Auto-detect", "English", "German"]
    )

    st.divider()

    st.subheader("💬 Example Questions")
    example_question = st.selectbox(
        "Try one:",
        [
            "",
            "What vegan dishes do you have?",
            "What does allergen code F mean?",
            "Which dishes are under 5 euros?",
            "What vegetarian dishes do you recommend?",
            "Welche veganen Gerichte gibt es?",
            "Was bedeutet Allergen-Code F?"
        ]
    )

    st.divider()

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

tab1, tab2 = st.tabs(["💬 Chatbot", "📋 Menu Preview"])

with tab1:
    st.info("You can ask in English or German. The answer will follow your question language.")

    rag_chain = load_rag_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! 👋 I am your Redi AI Waiter. Ask me anything about the menu."
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask your question about the menu...")
    if user_question:
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Checking the menu..."):
                answer = rag_chain.invoke(user_question)
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.rerun()
# %% [markdown]
# ### Process User Question

# %%
    if example_question:
        user_question = example_question

    if user_question:
        if language_choice == "English":
            user_question = user_question + "\nPlease answer in English."
        elif language_choice == "German":
            user_question = user_question + "\nBitte antworte auf Deutsch."

        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Checking the menu..."):
                answer = rag_chain.invoke(user_question)
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
# %% [markdown]
# ## Menu Preview Tab

# %%
with tab2:
   with tab2:
    st.subheader("📋 Original Menu PDF")

    pdf_path = "data/Menu of Swagat Indian Restaurant.pdf"

    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="800px"
        type="application/pdf">
    </iframe>
    """

    st.markdown(pdf_display, unsafe_allow_html=True) 
# %%
