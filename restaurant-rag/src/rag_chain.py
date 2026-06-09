# This script implements a Retrieval-Augmented Generation (RAG) chain for a restaurant menu chatbot.
# It retrieves relevant menu items based on a customer's query and then generates a response using a language model.
# Import necessary libraries and modules for file handling, environment variable loading, and the language model.
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from retriever import describe_spice_safety, retrieve_filtered_items
# Go up from src/rag_chain.py to the main restaurant-rag folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Path to the .env file that stores the Groq API key.
ENV_PATH = PROJECT_ROOT / ".env"
# Load environment variables from .env.
# This makes GROQ_API_KEY available to LangChain/Groq.
load_dotenv(ENV_PATH)
# Convert each filtered menu item into text for the LLM prompt. This includes important fields and a spice safety note based on the customer's taste preference.
def format_items_for_prompt(filtered_items, intent):
    # Store formatted menu item text blocks here.
    item_blocks = []
    # Convert each filtered menu item into text for the LLM prompt.
    for item in filtered_items:
        # Create a spice warning/note based on the user taste request.
        spice_note = describe_spice_safety(item, intent["taste"])
        # Format one menu item with the important fields.
        block = f"""
Name: {item.get("name")}
Price: {item.get("price")} euros
German description: {item.get("description_de")}
Category: {item.get("category")}
Meal type: {item.get("meal_type")}
Vegetarian: {item.get("is_vegetarian")}
Vegan possible: {item.get("is_vegan_possible")}
Allergens: {item.get("allergen_names")}
Spice note: {spice_note}
"""
        item_blocks.append(block)
    # Join all item blocks into one large text context.
    return "\n".join(item_blocks)

# Function to build the prompt for the LLM, combining the customer query, extracted intent, and formatted menu items.
def build_prompt(query, intent, filtered_items):
    # Convert filtered menu items into prompt context.
    menu_context = format_items_for_prompt(filtered_items, intent)

    # Create the instruction prompt for the LLM.
    prompt = f"""
You are a helpful AI waiter for Swagat Indian Restaurant.

Answer the customer in clear English.

Use only the menu items provided below.
Do not invent dishes, prices, ingredients, or allergens.

Important reasoning rules:
- If max_price is 12, then prices like 11.9 or 11.90 are within budget.
- Do not reject an item unless it violates the extracted intent.
- The customer asked to avoid nuts only. Do not reject milk_lactose unless the customer asks to avoid dairy or lactose.
- All provided menu items have already passed structured filters, so treat them as valid matches unless a spice note warns otherwise.
- Do not make claims about kitchen cross-contamination unless it is explicitly provided in the menu items.
- For drinks, do not mention spice notes unless the menu explicitly says something about spice.
- If the user asks for beer or bier, answer only with beer items from the provided menu items.

If a German description is provided, translate or summarize it in English.
If allergens are "not_listed", say that allergen information is not clearly listed and the customer should confirm with the restaurant.
If the user asks for mild food, use the spice note carefully.

Customer question:
{query}

Extracted customer intent:
{intent}

Menu items:
{menu_context}

Write a friendly answer with:
- best matching options
- price
- short English description
- allergen information
- spice warning if needed
- final safety note for allergies
"""
    return prompt
# Function to generate the final answer for the customer by retrieving relevant menu items and then using the LLM to create a response.
def generate_llm_answer(query):
    # Use retriever.py to get structured intent and filtered menu items.
    intent, filtered_items = retrieve_filtered_items(query, top_k=200)

    # If no matching items were found, return a safe fallback message.
    if not filtered_items:
        return (
            "I could not find a matching menu item. "
            "Please contact the restaurant directly, especially for allergy-sensitive requests."
        )

    # Build the prompt using the customer query and filtered menu items.
    prompt = build_prompt(query, intent, filtered_items)

    # Create the Groq LLM client.
    # Groq runs the Llama model through its API.
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
    )

    # Send the prompt to the LLM.
    response = llm.invoke(prompt)

    # Return the generated English answer.
    return response.content
# Main function to test the RAG chain with an example customer query.
def main():
    # Example customer question for testing.
    query = "I want mild chicken under 12 euros and no nuts"

    # Generate final answer using retrieval + filtering + LLM.
    answer = generate_llm_answer(query)

    # Print the final customer-friendly answer.
    print(answer)

# Run main only when this file is executed directly.
if __name__ == "__main__":
    main()