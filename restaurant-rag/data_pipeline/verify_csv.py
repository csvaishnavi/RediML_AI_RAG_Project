from pathlib import Path
import pandas as pd
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

df = pd.read_csv(PROJECT_ROOT / "data" / "swagat_menu.csv")
output_path = CURRENT_DIR / "verification_results.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("--- CSV Sanity Checks ---\n")
    f.write(f"Total rows: {len(df)}\n")
    f.write(f"Columns: {list(df.columns)}\n\n")

    f.write("Categories distribution:\n")
    f.write(df["category"].value_counts().to_string())
    f.write("\n\n")

    f.write("Meal type distribution:\n")
    f.write(df["meal_type"].value_counts().to_string())
    f.write("\n\n")

    f.write("Vegetarian count:\n")
    f.write(df["is_vegetarian"].value_counts().to_string())
    f.write("\n\n")

    f.write("Vegan possible count:\n")
    f.write(df["is_vegan_possible"].value_counts().to_string())
    f.write("\n\n")

    f.write("Null counts:\n")
    f.write(df.isnull().sum().to_string())
    f.write("\n\n")

    f.write("Ingredient Flags (True counts):\n")
    for flag in ["contains_chicken", "contains_lamb", "contains_fish", "contains_shrimp", "contains_duck", "contains_paneer"]:
        true_count = df[flag].astype(str).str.lower().value_counts().get('true', 0)
        f.write(f"  {flag}: {true_count}\n")
    f.write("\n")

    lunch_with_allergens = df[df["is_lunch_menu"] == True]["allergen_codes"].dropna().count()
    f.write(f"Lunch menu items with inherited allergen codes: {lunch_with_allergens} out of {len(df[df['is_lunch_menu'] == True])}\n\n")

    f.write("Sample records:\n")
    sample = df.sample(10, random_state=42)
    for idx, row in sample.iterrows():
        f.write(f"  ID: {row['item_id']} | Name: {row['name']} | Price: {row['price']} | Category: {row['category']} | Veg: {row['is_vegetarian']} | Vegan: {row['is_vegan_possible']} | Allergens: {row['allergen_codes']}\n")

print(f"Verification results written to {output_path}")
