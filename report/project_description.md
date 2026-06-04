# Redi AI Waiter - Project Description

## 1. What is the one core feature your project must have to work?

The core feature of my project is a Retrieval-Augmented Generation (RAG) chatbot that can answer customer questions about the restaurant menu accurately using the restaurant's menu data. The chatbot should retrieve relevant information from the menu and provide helpful responses about dishes, ingredients, allergens, prices, spice levels, and dietary preferences.

Without the retrieval system, the chatbot would not be able to provide reliable restaurant-specific answers.

---

## 2. What dataset will you be using?

I will use two datasets:

### 1. Restaurant Menu CSV Dataset

* File: `swagat_menu.csv`
* Contains structured menu information such as:

  * Dish name
  * Category
  * Price
  * Ingredients
  * Allergens
  * Vegetarian/Vegan status
  * Spice level
  * Description

### 2. Restaurant Menu PDF

* File: `swagat_menu.pdf`
* Contains additional menu information and descriptions available to restaurant customers.

These datasets will be converted into LangChain Documents, embedded using the BAAI/bge-m3 embedding model, stored in ChromaDB, and used by the RAG chatbot to answer customer questions.
