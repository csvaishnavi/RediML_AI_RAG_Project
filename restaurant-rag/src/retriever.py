# This script is responsible for retrieving relevant document chunks from the Chroma vector store based on a user query. It loads the existing Chroma vector database, performs similarity search, and prints the retrieved chunks along with their metadata.
# Import necessary libraries for vector storage and embedding creation.
from pathlib import Path
import re
from langchain_chroma import Chroma
from embeddings import get_embedding_model

# Go up from src/retriever.py to the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder where ChromaDB is saved.
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# These words help the retriever understand ingredient requests.
INGREDIENT_FILTERS = {
    "chicken": "contains_chicken",
    "hähnchen": "contains_chicken",
    "haehnchen": "contains_chicken",
    "fish": "contains_fish",
    "fisch": "contains_fish",
    "lamb": "contains_lamb",
    "lamp": "contains_lamb",
    "lamm": "contains_lamb",
    "duck": "contains_duck",
    "ente": "contains_duck",
    "shrimp": "contains_shrimp",
    "prawn": "contains_shrimp",
    "garnelen": "contains_shrimp",
    "paneer": "contains_paneer",
}

# These words improve search for menu groups.
MENU_KEYWORD_EXPANSIONS = {
    "salad": "salat salad",
    "salat": "salat salad",
    "soup": "suppe shorba soup",
    "suppe": "suppe shorba soup",
    "shorba": "suppe shorba soup",
    "bread": "naan roti paratha bread",
    "naan": "naan roti paratha bread",
    "roti": "naan roti paratha bread",
    "rice": "reis biryani rice",
    "reis": "reis biryani rice",
    "biryani": "reis biryani rice",
    "dessert": "dessert gulab jamun mango cream",
    "sweet": "dessert gulab jamun mango cream",
    "drink": "getränke drink lassi cola wasser bier",
    "drinks": "getränke drink lassi cola wasser bier",
    "getränke": "getränke drink lassi cola wasser bier",
}

# This function loads the Chroma vector store from disk and returns it for use in retrieval.
def load_vector_store():
    # Load the same embedding model used when creating the vector database.
    embedding_model = get_embedding_model()

    # Load the saved Chroma vector store.
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model,
    )

    return vector_store

# This function expands the user query with helpful keywords to improve retrieval results. For example, if the user asks for
def expand_query_for_retrieval(query):
    # Convert query to lowercase for easy keyword matching.
    text = query.lower()

    # Store extra search words here.
    extra_words = []

    # Add menu group expansions like salad -> salat salad.
    for keyword, expansion in MENU_KEYWORD_EXPANSIONS.items():
        if keyword in text:
            extra_words.append(expansion)

    # Add special beer support.
    if ("beer" in text or "bier" in text) and "kingfisher" not in text:
        extra_words.append("bier beer indisches bier kingfisher hefeweizen")

    # Add ingredient keywords like fish, lamb, duck, shrimp, paneer.
    for keyword in INGREDIENT_FILTERS:
        if keyword in text:
            extra_words.append(keyword)

    # If no extra words are needed, return the original query.
    if not extra_words:
        return query

    # Return original query plus helpful search words.
    return query + " " + " ".join(extra_words)

# This function retrieves relevant document chunks from the Chroma vector store based on the user query. It first expands the query with helpful keywords, then performs a similarity search in Chroma to get relevant documents.
def retrieve_relevant_chunks(query, top_k=200):
    # Expand query before semantic search.
    expanded_query = expand_query_for_retrieval(query)

    # Load Chroma vector store.
    vector_store = load_vector_store()

    # Retrieve relevant documents from Chroma.
    documents = vector_store.similarity_search(expanded_query, k=top_k)

    return documents

# This function converts a retrieved LangChain Document into a normal Python dictionary representing a menu item. It extracts metadata and parses the document text to fill in fields like name, price, description, category, meal type, allergens, and ingredient flags.
def document_to_item(doc):
    # Convert one retrieved LangChain Document into a normal Python dictionary.
    item = {}

    # First copy metadata if useful fields are stored there.
    for key, value in doc.metadata.items():
        item[str(key).strip().lower()] = str(value).strip()

    # Get document text.
    content = doc.page_content

    # Parse normal CSVLoader lines like:
    # name: Butter Chicken
    # price: 11.9
    for line in content.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            clean_key = key.strip().lower().replace(" ", "_")
            clean_value = value.strip()
            item[clean_key] = clean_value

    # Parse pipe-separated search text like:
    # Name: Butter Chicken | Price: 11.9 euros | Meal type: main
    for part in content.split("|"):
        if ":" in part:
            key, value = part.split(":", 1)
            clean_key = key.strip().lower().replace(" ", "_")
            clean_value = value.strip().replace(" euros", "")
            item[clean_key] = clean_value

    return item

# This function extracts price information from the user query. It looks for patterns like "under 12 euros", "below 15", "less than 10", "maximum 20", or "max 8" and returns the price as a float. If no price is found, it returns None.
def extract_price_from_query(query):
    # Find price expressions like:
    # under 12 euros
    # below 15
    # less than 10
    text = query.lower()

    price_patterns = [
        r"under\s+(\d+(?:\.\d+)?)",
        r"below\s+(\d+(?:\.\d+)?)",
        r"less than\s+(\d+(?:\.\d+)?)",
        r"maximum\s+(\d+(?:\.\d+)?)",
        r"max\s+(\d+(?:\.\d+)?)",
    ]

    for pattern in price_patterns:
        match = re.search(pattern, text)

        if match:
            return float(match.group(1))

    return None

# This function extracts structured user intent from the natural language query. It looks for keywords to determine if the user wants vegetarian or vegan food, a lunch menu, beer, specific ingredients, or wants to avoid certain allergens. It also extracts any price constraints and taste preferences.
def extract_user_intent(query):
    # Convert query to lowercase for easier matching.
    text = query.lower()

    # Store what the customer wants.
    intent = {
        "max_price": extract_price_from_query(query),
        "require_vegetarian": False,
        "require_vegan": False,
        "require_lunch_menu": False,
        "require_beer": False,
        "meal_type": None,
        "exclude_drinks": False,
        "avoid_allergens": [],
        "taste": None,
        "required_ingredient_fields": [],
    }

    # Vegetarian intent.
    if "vegetarian" in text or "veggie" in text:
        intent["require_vegetarian"] = True

    # Vegan intent.
    if "vegan" in text:
        intent["require_vegan"] = True

    # Lunch menu intent.
    if "lunch" in text or "mittag" in text or "mittagsmenü" in text:
        intent["require_lunch_menu"] = True

    # Drink intent.
    drink_words = ["drink", "drinks", "getränke", "lassi", "beer", "bier", "kingfisher", "hefeweizen"]
    if any(word in text for word in drink_words):
        intent["meal_type"] = "drink"

    # Beer intent.
    beer_words = ["beer", "bier", "kingfisher", "hefeweizen"]
    if any(word in text for word in beer_words):
        intent["require_beer"] = True
        intent["meal_type"] = "drink"

    # If customer asks for food, avoid showing drinks.
    food_words = ["eat", "food", "dish", "meal", "curry", "starter", "main"]
    if any(word in text for word in food_words):
        intent["exclude_drinks"] = True

    # Taste intent.
    if "mild" in text:
        intent["taste"] = "mild"
    elif "spicy" in text or "hot" in text or "scharf" in text:
        intent["taste"] = "spicy"

    # Ingredient intent: fish, lamb, duck, shrimp, paneer, chicken.
    for keyword, field_name in INGREDIENT_FILTERS.items():
        if keyword in text and field_name not in intent["required_ingredient_fields"]:
            intent["required_ingredient_fields"].append(field_name)

    # Allergy intent.
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

    return intent

# This function checks if a menu item contains any of the allergens the customer wants to avoid. It looks at the allergen_names field and also checks the name, description, and category for hidden keywords that might indicate the presence of allergens.
def item_contains_avoided_allergen(item, avoid_allergens):
    # Read allergen names from the menu item.
    allergen_names = str(item.get("allergen_names", "")).lower()

    # Also check name/category/description for hidden allergy keywords.
    item_text = " ".join(
        [
            str(item.get("name", "")),
            str(item.get("description_de", "")),
            str(item.get("category", "")),
        ]
    ).lower()

    # Extra safety keyword map.
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
        # Check allergen_names column.
        if allergen in allergen_names:
            return True

        # Check text keywords.
        for keyword in allergen_keyword_map.get(allergen, []):
            if keyword in item_text:
                return True

    return False

# This function checks if a menu item is beer. It looks for beer-related keywords in the name, description, category, and search_text fields to identify beers that might not be explicitly labeled as drinks.
def item_is_beer(item):
    # Check if a menu item is beer.
    text = " ".join(
        [
            str(item.get("name", "")),
            str(item.get("description_de", "")),
            str(item.get("category", "")),
            str(item.get("search_text", "")),
        ]
    ).lower()

    beer_words = ["beer", "bier", "kingfisher", "hefeweizen"]

    return any(word in text for word in beer_words)

# This function converts price text into a float safely. It handles cases where the price might be missing, malformed, or contain commas instead of dots. If conversion fails, it returns None.
def safe_float(value):
    # Convert price text into a float safely.
    try:
        return float(str(value).replace(",", ".").strip())
    except ValueError:
        return None

# This function applies structured filters to the retrieved documents based on the extracted user intent. It checks each menu item against criteria like price, vegetarian/vegan status, meal type, allergens, and required ingredients. It also avoids duplicates and returns a list of matching items.
def filter_menu_items(
    documents,
    max_price=None,
    require_vegetarian=False,
    require_vegan=False,
    require_lunch_menu=False,
    require_beer=False,
    meal_type=None,
    exclude_drinks=False,
    avoid_allergens=None,
    required_ingredient_fields=None,
):
    # Avoid mutable default values.
    if avoid_allergens is None:
        avoid_allergens = []

    if required_ingredient_fields is None:
        required_ingredient_fields = []

    # Store final matching menu items here.
    filtered_items = []

    # Track item names so duplicates do not appear.
    seen_names = set()

    for doc in documents:
        # Convert document into dictionary.
        item = document_to_item(doc)

        # Skip rows without a name.
        name = item.get("name")
        if not name:
            continue

        # Avoid duplicate menu items.
        if name in seen_names:
            continue

        # Price filter.
        price = safe_float(item.get("price"))
        if max_price is not None:
            if price is None or price > max_price:
                continue

        # Ingredient filter: fish, lamb, chicken, duck, shrimp, paneer.
        ingredient_match = True

        for field_name in required_ingredient_fields:
            if str(item.get(field_name, "")).lower() != "true":
                ingredient_match = False

        if not ingredient_match:
            continue

        # Vegetarian filter.
        if require_vegetarian:
            if str(item.get("is_vegetarian", "")).lower() != "true":
                continue

        # Vegan filter.
        if require_vegan:
            if str(item.get("is_vegan_possible", "")).lower() != "true":
                continue

        # Lunch menu filter.
        if require_lunch_menu:
            if str(item.get("is_lunch_menu", "")).lower() != "true":
                continue

        # Meal type filter.
        if meal_type:
            if str(item.get("meal_type", "")).lower() != meal_type:
                continue

        # Exclude drinks when customer asks for food.
        if exclude_drinks:
            if str(item.get("meal_type", "")).lower() == "drink":
                continue

        # Beer filter.
        if require_beer:
            if not item_is_beer(item):
                continue

        # Allergy filter.
        if item_contains_avoided_allergen(item, avoid_allergens):
            continue

        # Add valid item.
        filtered_items.append(item)
        seen_names.add(name)

    return filtered_items[:10]

# This function checks the spice level of a menu item based on its description and the customer's taste preference. If the customer wants mild food but the description contains spicy keywords, it returns a warning note. If the taste is mild but no spice information is found, it advises the customer to confirm if they need it very mild.
def describe_spice_safety(item, taste):
    # Do not give spice notes for drinks.
    if str(item.get("meal_type", "")).lower() == "drink":
        return ""

    description = str(item.get("description_de", "")).lower()

    spicy_words = ["scharf", "würzig", "pikant", "chili", "spicy"]

    if taste == "mild" and any(word in description for word in spicy_words):
        return "This may not be mild because the description suggests it is spicy or strongly seasoned."

    if taste == "mild":
        return "The menu does not clearly list spice level, so please confirm if you need it very mild."

    return ""

# This is the main function that retrieves and filters menu items based on the user query. It first extracts structured intent from the query, then retrieves relevant chunks from Chroma, and finally applies structured filters to return a list of matching menu items.
def retrieve_filtered_items(query, top_k=200):
    # Extract structured user intent from the question.
    intent = extract_user_intent(query)

    # Retrieve relevant chunks from Chroma.
    retrieved_docs = retrieve_relevant_chunks(query, top_k=top_k)

    # Apply structured filters.
    filtered_items = filter_menu_items(
        retrieved_docs,
        max_price=intent["max_price"],
        require_vegetarian=intent["require_vegetarian"],
        require_vegan=intent["require_vegan"],
        require_lunch_menu=intent["require_lunch_menu"],
        require_beer=intent["require_beer"],
        meal_type=intent["meal_type"],
        exclude_drinks=intent["exclude_drinks"],
        avoid_allergens=intent["avoid_allergens"],
        required_ingredient_fields=intent["required_ingredient_fields"],
    )

    return intent, filtered_items

# Main function for testing the retriever. You can change the query to test different scenarios like asking for fish, lamb, drinks, salad, etc. It prints the extracted intent and the filtered menu items in a readable format.
def main():
    # Test query. Change this query to test fish, lamb, drinks, salad, etc.
    query = "I want fish meal"

    intent, filtered_items = retrieve_filtered_items(query)

    print(f"User query: {query}")
    print("Extracted intent:")
    print(intent)
    print("=" * 50)
    print("Filtered menu items:")
    print("=" * 50)

    for item in filtered_items:
        spice_note = describe_spice_safety(item, intent["taste"])

        print(f"Name: {item.get('name')}")
        print(f"Price: {item.get('price')} euros")
        print(f"Description: {item.get('description_de')}")
        print(f"Category: {item.get('category')}")
        print(f"Meal type: {item.get('meal_type')}")
        print(f"Allergens: {item.get('allergen_names')}")

        if spice_note:
            print(f"Spice note: {spice_note}")

        print("-" * 50)

# Run the main function when this script is executed.
if __name__ == "__main__":
    main()
