# This script is responsible for retrieving relevant document chunks from the Chroma vector store based on a user query. It loads the existing Chroma vector database, performs similarity search, and prints the retrieved chunks along with their metadata.
# Import necessary libraries for vector storage and embedding creation.
from pathlib import Path
from langchain_chroma import Chroma
from sqlalchemy import text
from embeddings import get_embedding_model

# Go up from src/retriever.py to the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder where Chroma vector database is saved.
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


# Load the existing Chroma vector database from disk.
def load_vector_store():
    # Load the same Hugging Face embedding model used to create Chroma.
    embedding_model = get_embedding_model()

    # Load the existing Chroma vector database.
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model,
    )

    return vector_store


# Retrieve the most relevant document chunks from Chroma using similarity search.
def retrieve_relevant_chunks(query, top_k=200):
    # Load Chroma vector database.
    vector_store = load_vector_store()

    # Search for the most relevant chunks.
    results = vector_store.similarity_search(
        query,
        k=top_k,
    )

    return results


# Convert true/false text into Python boolean.
def parse_bool(value):
    text = str(value).strip().lower()

    return text == "true"


# Convert price text into a number.
def parse_price(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Convert LangChain Document text into a Python dictionary.
def document_to_item(doc):
    item = {}

    for line in doc.page_content.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            item[key.strip()] = value.strip()

    return item


# Extract structured intent from the user's natural language query.
def extract_user_intent(query):
    text = query.lower()

    # Default intent values.
    intent = {
        "max_price": None,
        "require_chicken": False,
        "require_vegetarian": False,
        "require_vegan": False,
        "require_lunch_menu": False,
        "require_beer": False,
        "meal_type": None,
        "exclude_drinks": False,
        "avoid_allergens": [],
        "taste": None,
    }

    # Detect budget like "under 12 euros", "below 15", or "less than 10".
    words = text.replace("€", " euros").split()

    for index, word in enumerate(words):
        if word in ["under", "below", "less"]:
            for next_word in words[index + 1:index + 4]:
                try:
                    intent["max_price"] = float(next_word)
                    break
                except ValueError:
                    continue

    # Detect chicken preference.
    if "chicken" in text:
        intent["require_chicken"] = True

    # Detect vegetarian preference.
    if "vegetarian" in text or "veg" in text:
        intent["require_vegetarian"] = True

    # Detect vegan preference.
    if "vegan" in text:
        intent["require_vegan"] = True

    # Detect lunch menu request.
    if "lunch" in text or "mittag" in text or "mittagsmenü" in text:
        intent["require_lunch_menu"] = True

    # If the user asks what they can eat, avoid returning drinks.
    if "eat" in text or "food" in text or "dish" in text or "meal" in text:
        intent["exclude_drinks"] = True

    # Detect drink-related requests.
    if (
        "drink" in text
        or "drinks" in text
        or "lassi" in text
        or "beer" in text
        or "bier" in text
        or "kingfisher" in text
        or "hefeweizen" in text
        or "alcohol" in text
    ):
        intent["meal_type"] = "drink"

    # Detect beer-specific requests.
    # This prevents the retriever from returning general drinks or food items
    # when the user specifically asks for beer.
    if "beer" in text or "bier" in text or "kingfisher" in text or "hefeweizen" in text:
        intent["require_beer"] = True

    # Detect starter/appetizer requests.
    if "starter" in text or "appetizer" in text or "pakora" in text:
        intent["meal_type"] = "starter"

    # Detect main dish requests.
    if "main" in text or "main dish" in text or "meal" in text:
        intent["meal_type"] = "main"

    # Detect allergens to avoid.
    allergen_keywords = {
        "nuts": "nuts",
        "nut": "nuts",
        "milk": "milk_lactose",
        "dairy": "milk_lactose",
        "lactose": "milk_lactose",
        "gluten": "gluten",
        "fish": "fish",
        "egg": "egg",
        "soy": "soy",
        "mustard": "mustard",
    }

    allergy_signals = ["no", "avoid", "allergic", "allergy", "without"]

    for keyword, allergen_name in allergen_keywords.items():
        if keyword in text and any(signal in text for signal in allergy_signals):
            if allergen_name not in intent["avoid_allergens"]:
                intent["avoid_allergens"].append(allergen_name)

    # Detect taste/spice preference.
    if "mild" in text or "not spicy" in text:
        intent["taste"] = "mild"

    if "spicy" in text or "hot" in text:
        intent["taste"] = "spicy"

    return intent


# Check allergens using structured allergen names and safety keywords from item text.
def item_contains_avoided_allergen(item, avoid_allergens):
    allergen_names = str(item.get("allergen_names", "")).lower()

    item_text = " ".join(
        [
            str(item.get("name", "")),
            str(item.get("description_de", "")),
            str(item.get("category", "")),
        ]
    ).lower()

    # Extra allergen keywords catch cases where allergen codes are missing
    # but the name or description clearly suggests an allergen.
    allergen_keyword_map = {
        "milk_lactose": [
            "milk",
            "lactose",
            "butter",
            "cream",
            "paneer",
            "cheese",
            "yogurt",
            "sahne",
            "käse",
            "kaese",
            "milch",
            "lassi",
        ],
        "nuts": [
            "nuts",
            "nut",
            "cashew",
            "cashewn",
            "almond",
            "mandel",
            "nüsse",
            "nuesse",
        ],
        "gluten": [
            "gluten",
            "wheat",
            "weizen",
            "naan",
            "roti",
            "paratha",
            "noodles",
            "nudeln",
        ],
        "egg": ["egg", "ei"],
        "soy": ["soy", "soja"],
        "fish": ["fish", "fisch"],
        "mustard": ["mustard", "senf"],
    }

    for allergen in avoid_allergens:
        allergen = allergen.lower()

        # First check the structured allergen_names field.
        if allergen in allergen_names:
            return True

        # Then check dish name/description/category for safety keywords.
        for keyword in allergen_keyword_map.get(allergen, []):
            if keyword in item_text:
                return True

    return False


# Check whether a menu item is a beer item.
# This is used after retrieval to keep only beer results for beer-related queries.
def item_is_beer(item):
    item_text = " ".join(
        [
            str(item.get("name", "")),
            str(item.get("description_de", "")),
            str(item.get("category", "")),
        ]
    ).lower()

    beer_words = ["beer", "bier", "kingfisher", "hefeweizen"]

    return any(word in item_text for word in beer_words)


# Apply exact filters to retrieved menu items.
def filter_menu_items(
    documents,
    max_price=None,
    require_chicken=False,
    require_vegetarian=False,
    require_vegan=False,
    require_lunch_menu=False,
    require_beer=False,
    meal_type=None,
    exclude_drinks=False,
    avoid_allergens=None,
):
    filtered = []

    if avoid_allergens is None:
        avoid_allergens = []

    for doc in documents:
        # Only filter CSV menu rows, not PDF page chunks.
        if doc.metadata.get("source_type") != "csv_menu":
            continue

        # Convert document content into a dictionary.
        item = document_to_item(doc)

        # Read price.
        price = parse_price(item.get("price"))

        # Filter by maximum price.
        if max_price is not None:
            if price is None or price > max_price:
                continue

        # Filter by chicken requirement.
        if require_chicken:
            if not parse_bool(item.get("contains_chicken")):
                continue

        # Filter by vegetarian requirement.
        if require_vegetarian:
            if not parse_bool(item.get("is_vegetarian")):
                continue

        # Filter by vegan requirement.
        if require_vegan:
            if not parse_bool(item.get("is_vegan_possible")):
                continue

        # Filter by lunch menu requirement.
        if require_lunch_menu:
            if not parse_bool(item.get("is_lunch_menu")):
                continue

        # Filter by meal type, such as drink, starter, or main.
        if meal_type is not None:
            if item.get("meal_type", "").lower() != meal_type:
                continue

        # Filter out drinks if the user seems to be asking for food.
        if exclude_drinks:
            if item.get("meal_type", "").lower() == "drink":
                continue

        # If the user specifically asked for beer, remove non-beer drinks and food items.
        if require_beer:
            if not item_is_beer(item):
                continue

        # Filter by allergens to avoid.
        if item_contains_avoided_allergen(item, avoid_allergens):
            continue

        filtered.append(item)

    return filtered


# Create spice safety note based on description and user taste preference.
def describe_spice_safety(item, taste):
    description = item.get("description_de", "").lower()
    meal_type = item.get("meal_type", "").lower()

    # Drinks do not need spice notes.
    if meal_type == "drink":
        return ""

    spicy_words = ["scharf", "würzig", "pikant", "chili", "spicy", "würziger"]

    if taste == "mild" and any(word in description for word in spicy_words):
        return "This may not be mild because the description suggests it is spicy or strongly seasoned."

    if taste == "mild":
        return "The menu does not clearly list spice level, so please confirm if you need it very mild."

    return ""


# Expand short beer queries with related German and menu-specific terms.
# This improves retrieval when the user types "beer" but the menu uses "Bier".
def expand_query_for_retrieval(query):
    text = query.lower()

    if ("beer" in text or "bier" in text) and "kingfisher" not in text:
        return query + " bier beer indisches bier kingfisher hefeweizen"

    return query


# Retrieve relevant menu chunks and apply structured filters based on the user query.
def retrieve_filtered_items(query, top_k=200):
    intent = extract_user_intent(query)

    expanded_query = expand_query_for_retrieval(query)

    retrieved_docs = retrieve_relevant_chunks(expanded_query, top_k=top_k)

    filtered_items = filter_menu_items(
        retrieved_docs,
        max_price=intent["max_price"],
        require_chicken=intent["require_chicken"],
        require_vegetarian=intent["require_vegetarian"],
        require_vegan=intent["require_vegan"],
        require_lunch_menu=intent["require_lunch_menu"],
        require_beer=intent["require_beer"],
        meal_type=intent["meal_type"],
        exclude_drinks=intent["exclude_drinks"],
        avoid_allergens=intent["avoid_allergens"],
    )

    return intent, filtered_items


# Main function to test the retrieval and filtering process.
def main():
    query = "beer"

    intent, filtered_items = retrieve_filtered_items(query, top_k=200)

    print(f"User query: {query}")
    print("Extracted intent:")
    print(intent)
    print("=" * 50)

    print("Filtered menu items:")
    print("=" * 50)

    if not filtered_items:
        print("No matching menu items found.")

    for item in filtered_items:
        spice_note = describe_spice_safety(item, intent["taste"])

        print(f"Name: {item.get('name')}")
        print(f"Price: {item.get('price')} euros")
        print(f"Description: {item.get('description_de')}")
        print(f"Allergens: {item.get('allergen_names')}")

        if spice_note:
            print(f"Spice note: {spice_note}")

        print("-" * 50)


# Run the main function when this script is executed directly.
if __name__ == "__main__":
    main()
