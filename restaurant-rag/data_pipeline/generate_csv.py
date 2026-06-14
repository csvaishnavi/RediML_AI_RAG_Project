import re
import csv
import json

# Read the extracted menu text
with open(r"c:\Users\Bright\.gemini\antigravity-ide\scratch\extracted_menu.txt", "r", encoding="utf-8") as f:
    content = f.read()

pages_raw = content.split("--- PAGE ")
pages = {}

for p_raw in pages_raw:
    if not p_raw.strip():
        continue
    parts = p_raw.split(" ---")
    if len(parts) < 2:
        continue
    page_num = int(parts[0].strip())
    page_text = parts[1]
    pages[page_num] = page_text.strip().split("\n")

HEADERS_TO_IGNORE = [
    "MITTAGSMENÜ (GÜLTIG NUR MITTAGS VON 11:00 BIS 14:30 UHR)",
    "Salat der Saison Zu jedem Hauptgericht erhalten Sie im Haus einen Salat der Saison als Beilage.",
    "SUPPEN",
    "SALATE",
    "VORSPEISEN",
    "Alle Vorspeisen werden mit Joghurt-Pfefferminz und Tamarindensoße serviert.",
    "NAAN AUS DEM TANDOOR (LEHMOFEN)",
    "Alle Naan basieren auf Weizenmehl",
    "BIRYANI-SPEZIALITÄTEN (TEILWEISE VEGAN MÖGLICH)",
    "Unsere Biryani-Spezialitäten servieren wir zusammen mit Raita (gewürzter Joghurt).",
    "VEGETARISCHE GERICHTE (AUCH VEGAN MÖGLICH)",
    "Alle Gerichte werden mit frischem Koriander und mit Basmatireis serviert",
    "TANDOORI-SPEZIALITÄTEN",
    "Werden auf einem Weißkohlbett serviert. Dazu wird Butter Naan und eine hausgemachte Gewürzsoße",
    "gereicht.",
    "HÄHNCHEN-SPEZIALITÄTEN",
    "Spezialitäten vom Huhn werden mit frischem Koriander und mit Basmatireis serviert",
    "LAMB-SPEZIALITÄTEN",
    "Spezialitäten vom Lamm werden mit frischem Koriander und mit Basmatireis serviert",
    "FISCH-SPEZIALITÄTEN",
    "Fisch-Spezialitäten werden mit frischem Koriander und mit Basmatireis serviert",
    "ENTEN-SPEZIALITÄTEN",
    "Enten-Spezialitäten werden mit frischem Koriander und mit Basmatireis serviert",
    "REDI RESTAURANT-SPEZIALITÄTEN",
    "BEILAGEN",
    "DESSERTS",
    "ALKOHOLFREIE GETRÄNKE",
    "inkl. Pfand",
    "INDISCHE GETRÄNKE",
    "ALKOHOLISCHE GETRÄNKE",
    "Flasche 0,3l"
]

def is_header_or_metadata(line):
    line_s = line.strip()
    if not line_s:
        return True
    if any(x in line_s for x in ["Swagat Indian Restaurant", "ReDi Restaurant", "https://www.swagat", "Online Menu Of", "See MENU & Order", "INDIAN FOOD DELIVERY", "We offer Pick-up"]):
        return True
    if line_s in HEADERS_TO_IGNORE:
        return True
    return False

# 1. Process all pages to extract lists of raw items and raw descriptions
page_data = {}

for p_num in range(1, 13):
    lines = pages.get(p_num, [])
    items = []
    text_lines = []
    
    for line in lines:
        line_s = line.strip()
        if is_header_or_metadata(line_s):
            continue
        
        if re.match(r"^\d+", line_s):
            m = re.match(r"^(\d+)[.,\s]+(.*?)\s+(\d+,\d{2})$", line_s)
            if m:
                items.append({
                    "id": int(m.group(1)),
                    "name": m.group(2).strip(),
                    "price": m.group(3).strip(),
                    "page": p_num
                })
            else:
                text_lines.append(line_s)
        else:
            text_lines.append(line_s)
            
    # Merge descriptions with continuation support
    merged_descriptions = []
    temp_desc = ""
    for tl in text_lines:
        if not temp_desc:
            temp_desc = tl
        else:
            # Check for specific continuation words or pattern
            is_continuation = False
            if temp_desc.endswith("und") or temp_desc.endswith("aus") or temp_desc.endswith("süß-") or temp_desc.endswith("Cashewnüssen") or temp_desc.endswith("Joghurtmarinade aus"):
                is_continuation = True
            elif tl.startswith("Sahne verfeinert") or tl.startswith("und Mandeln") or tl.startswith("dem Tandoor") or tl.startswith("sauren Schärfe"):
                is_continuation = True
            elif tl.startswith("(") and tl.endswith(")") and len(tl) <= 10 and not temp_desc.endswith(")"):
                # Allergen codes split to next line like (f)
                is_continuation = True
            
            if is_continuation:
                if temp_desc.endswith("-"):
                    temp_desc += tl
                else:
                    temp_desc += " " + tl
            else:
                merged_descriptions.append(temp_desc)
                temp_desc = tl
    if temp_desc:
        merged_descriptions.append(temp_desc)
        
    page_data[p_num] = {
        "items": items,
        "descriptions": merged_descriptions
    }

# 2. Pair items with descriptions according to alignment rules
final_pairs = []

# Group A: Pages 1, 2, 3 (Mittagsmenü, Suppen, Salate)
# These pages share descriptions continuously.
group_1_items = []
group_1_descs = []
for p_num in [1, 2, 3]:
    group_1_items.extend(page_data[p_num]["items"])
    group_1_descs.extend(page_data[p_num]["descriptions"])

# Pair Group A
for i, item in enumerate(group_1_items):
    desc = group_1_descs[i] if i < len(group_1_descs) else ""
    final_pairs.append({
        "item": item,
        "desc": desc
    })

# Group B: Page 4 to 10
# Each page matches items to descriptions one-to-one
for p_num in range(4, 11):
    items = page_data[p_num]["items"]
    descs = page_data[p_num]["descriptions"]
    for i, item in enumerate(items):
        desc = descs[i] if i < len(descs) else ""
        final_pairs.append({
            "item": item,
            "desc": desc
        })

# Group C: Page 11 (Beilagen & Desserts)
# Skip items 160 and 161 (rice) when matching descriptions
p11_items = page_data[11]["items"]
p11_descs = page_data[11]["descriptions"]
desc_idx = 0
for item in p11_items:
    if item["id"] in [160, 161]:
        final_pairs.append({
            "item": item,
            "desc": ""
        })
    else:
        desc = p11_descs[desc_idx] if desc_idx < len(p11_descs) else ""
        final_pairs.append({
            "item": item,
            "desc": desc
        })
        desc_idx += 1

# Group D: Page 12 (Drinks)
# Only items 190, 191, 192, 193, 194 get descriptions in order. Others are empty.
p12_items = page_data[12]["items"]
p12_descs = page_data[12]["descriptions"]
desc_idx = 0
for item in p12_items:
    if 190 <= item["id"] <= 194:
        desc = p12_descs[desc_idx] if desc_idx < len(p12_descs) else ""
        final_pairs.append({
            "item": item,
            "desc": desc
        })
        desc_idx += 1
    else:
        final_pairs.append({
            "item": item,
            "desc": ""
        })

# Helper functions for processing fields
def extract_allergens(desc_text):
    if not desc_text:
        return ""
    # Find allergen letters in parenthetical suffixes, e.g. (f), (f,g), (a,d.o.g), ((f)
    m = re.search(r'\(([^)]+)\)?$', desc_text)
    if m:
        code_str = m.group(1)
        # Normalize and filter only letters within a-m
        clean_code = re.sub(r'[^a-zA-Z,]', ',', code_str)
        codes = [c.strip().lower() for c in clean_code.split(',') if c.strip() and len(c.strip()) == 1]
        valid_codes = [c for c in codes if c in 'abcdefghijklm']
        if valid_codes:
            return ",".join(sorted(list(set(valid_codes))))
    return ""

def clean_description(desc_text):
    if not desc_text:
        return ""
    # Remove allergen code suffix from text, e.g. (f), (f.g), ((f)
    cleaned = re.sub(r'\s*\({1,2}[a-zA-Z,.\s\(\)]+\)?\s*$', '', desc_text)
    return cleaned.strip()

def get_category_and_type(item_id):
    if 500 <= item_id <= 564:
        return "Mittagsmenü / lunch menu", "main"
    elif item_id in [1, 2, 3]:
        return "Suppen", "starter"
    elif item_id in [5, 6, 7, 8]:
        return "Salate", "starter"
    elif 10 <= item_id <= 19:
        return "Vorspeisen", "starter"
    elif 25 <= item_id <= 33:
        return "Naan / Brot", "side"
    elif 35 <= item_id <= 41:
        return "Biryani-Spezialitäten", "main"
    elif 45 <= item_id <= 61:
        return "Vegetarische Gerichte", "main"
    elif 65 <= item_id <= 77:
        return "Tandoori-Spezialitäten", "main"
    elif 80 <= item_id <= 94:
        return "Hähnchen-Spezialitäten", "main"
    elif 110 <= item_id <= 119:
        return "Lamm-Spezialitäten", "main"
    elif 125 <= item_id <= 128:
        return "Fisch-Spezialitäten", "main"
    elif 135 <= item_id <= 138:
        return "Enten-Spezialitäten", "main"
    elif item_id in [145, 146]:
        return "ReDi Restaurant-Spezialitäten", "main"
    elif 155 <= item_id <= 162:
        return "Beilagen", "side"
    elif 165 <= item_id <= 169:
        return "Desserts", "dessert"
    elif (180 <= item_id <= 185) or (190 <= item_id <= 194) or (201 <= item_id <= 204):
        return "Getränke", "drink"
    else:
        return "Unknown", "main"

def check_vegetarian(item_name, item_desc, category, item_id):
    item_name_lower = item_name.lower()
    item_desc_lower = item_desc.lower()
    
    if category == "Vegetarische Gerichte":
        return True
    if category == "Desserts":
        return True
    if category == "Getränke":
        return True
    if category == "Beilagen":
        return True
    if category == "Naan / Brot":
        if "keema" in item_name_lower:
            return False
        return True
    
    if category == "Suppen":
        if "chicken" in item_name_lower or "hühn" in item_desc_lower:
            return False
        return True
    
    if category == "Salate":
        if "chicken" in item_name_lower or "hähnchen" in item_desc_lower:
            return False
        return True
        
    if category == "Vorspeisen":
        if any(w in item_name_lower for w in ["chicken", "fish"]) or any(w in item_desc_lower for w in ["hähnchen", "fisch", "gemischte platte"]):
            return False
        return True
        
    if category == "Biryani-Spezialitäten":
        if any(w in item_name_lower for w in ["sabzi", "tofu", "paneer"]):
            return True
        return False
        
    if category == "Tandoori-Spezialitäten":
        if any(w in item_name_lower for w in ["paneer", "veggie"]):
            return True
        return False
        
    if category == "Mittagsmenü / lunch menu":
        if 500 <= item_id <= 509:
            return True
        return False
        
    return False

def check_vegan_possible(item_name, category, is_veg):
    item_name_lower = item_name.lower()
    if not is_veg:
        return False
        
    if any(w in item_name_lower for w in ["paneer", "käse", "cheese"]):
        return False
    if category == "Desserts":
        return False
    if category == "Getränke":
        if "lassi" in item_name_lower and "vegan" not in item_name_lower:
            return False
        return True
    if category == "Naan / Brot":
        if "roti" in item_name_lower:
            return True
        return False
    if category == "Beilagen":
        if "raita" in item_name_lower:
            return False
        return True
    
    if category in ["Vegetarische Gerichte", "Mittagsmenü / lunch menu"]:
        if "tofu" in item_name_lower:
            return True
        non_vegan_veg_dishes = ["makhni", "shahi", "kofta", "malai", "butter"]
        if any(w in item_name_lower for w in non_vegan_veg_dishes):
            return False
        return True
        
    if category == "Biryani-Spezialitäten":
        if any(w in item_name_lower for w in ["tofu", "sabzi"]):
            return True
        return False
        
    if category == "Vorspeisen":
        if "paneer" in item_name_lower:
            return False
        if any(w in item_name_lower for w in ["pakoras", "onion bhaji", "samosa", "papadamm"]):
            return True
        return False
        
    if category == "Salate":
        if any(w in item_name_lower for w in ["gemischter salat", "channa chat"]):
            return True
        return False
        
    if category == "Suppen":
        if any(w in item_name_lower for w in ["daal", "tomatar"]):
            return True
        return False
        
    if category == "Tandoori-Spezialitäten":
        if "veggie" in item_name_lower:
            return True
        return False

    return False

# Build mapping of regular item names to their properties for Mittagmenü lookup
regular_items_map = {}
for p in final_pairs:
    item = p["item"]
    desc = p["desc"]
    item_id = item["id"]
    if item_id < 500:
        cleaned_desc = clean_description(desc)
        allergens = extract_allergens(desc)
        name_normalized = item["name"].lower().strip()
        regular_items_map[name_normalized] = {
            "desc": cleaned_desc,
            "allergens": allergens
        }

# Generate the final clean rows
csv_rows = []
for p in final_pairs:
    item = p["item"]
    desc = p["desc"]
    item_id = item["id"]
    name = item["name"]
    price = item["price"].replace(",", ".")
    source_page = item["page"]
    category, meal_type = get_category_and_type(item_id)
    is_lunch_menu = (500 <= item_id <= 564)
    
    # Process description and allergens
    raw_allergens = extract_allergens(desc)
    description_de = clean_description(desc)
    
    # If Mittagsmenü, inherit description and allergens if available from regular menu
    if is_lunch_menu:
        name_norm = name.lower().strip()
        # Clean some specific Mittagmenü text patterns
        if name_norm.startswith("gebratene veg.-nudeln"):
            name_norm = "gebratene veg.-nudeln"
        elif name_norm.startswith("gebratene nudeln mit hähnchen"):
            name_norm = "gebratene nudeln mit hähnchen"
            
        # Try direct or partial match
        matched_regular = None
        for reg_name in regular_items_map:
            if reg_name == name_norm or reg_name in name_norm or name_norm in reg_name:
                matched_regular = regular_items_map[reg_name]
                break
        
        if matched_regular:
            if not description_de:
                description_de = matched_regular["desc"]
            if not raw_allergens:
                raw_allergens = matched_regular["allergens"]
                
    # Veg / Vegan flags
    is_vegetarian = check_vegetarian(name, description_de, category, item_id)
    is_vegan_possible = check_vegan_possible(name, category, is_vegetarian)
    
    # Content flags
    name_desc_lower = (name + " " + description_de).lower()
    contains_chicken = any(w in name_desc_lower for w in ["chicken", "hähnchen", "huhn"])
    contains_lamb = any(w in name_desc_lower for w in ["lamb", "lamm"]) or "seekh kebab" in name_desc_lower
    contains_fish = any(w in name_desc_lower for w in ["fisch", "fish", "maschli", "seelachs"])
    contains_shrimp = any(w in name_desc_lower for w in ["jheenga", "garnelen"])
    contains_duck = any(w in name_desc_lower for w in ["ente", "entenbrust"])
    
    # Grillplatte has chicken, lamb, fish
    if "grillplatte" in name_desc_lower:
        contains_chicken = True
        contains_lamb = True
        contains_fish = True
        
    contains_paneer = "paneer" in name_desc_lower or "indischer käse" in name_desc_lower or "indischem käse" in name_desc_lower or "indischen käse" in name_desc_lower
    
    # Generate search text index
    search_text = f"Name: {name} | Category: {category} | Description: {description_de}"
    if raw_allergens:
        search_text += f" | Allergens: {raw_allergens}"
    
    csv_rows.append({
        "item_id": item_id,
        "name": name,
        "description_de": description_de,
        "price": price,
        "category": category,
        "meal_type": meal_type,
        "is_lunch_menu": str(is_lunch_menu).lower(),
        "is_vegetarian": str(is_vegetarian).lower(),
        "is_vegan_possible": str(is_vegan_possible).lower(),
        "contains_chicken": str(contains_chicken).lower(),
        "contains_lamb": str(contains_lamb).lower(),
        "contains_fish": str(contains_fish).lower(),
        "contains_shrimp": str(contains_shrimp).lower(),
        "contains_duck": str(contains_duck).lower(),
        "contains_paneer": str(contains_paneer).lower(),
        "allergen_codes": raw_allergens,
        "source_page": source_page,
        "search_text": search_text
    })

# Write to CSV file in target workspace
csv_path = r"d:\ReDi\RAG_project\EJ\data\swagat_menu_raw.csv"
fieldnames = [
    "item_id", "name", "description_de", "price", "category", "meal_type",
    "is_lunch_menu", "is_vegetarian", "is_vegan_possible",
    "contains_chicken", "contains_lamb", "contains_fish", "contains_shrimp", "contains_duck", "contains_paneer",
    "allergen_codes", "source_page", "search_text"
]

with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"Successfully generated CSV at {csv_path} with {len(csv_rows)} items.")
# Let's also print a small sample for summary
print("\nSample Items:")
for r in csv_rows[:5]:
    name_safe = r['name'].encode('ascii', errors='replace').decode('ascii')
    cat_safe = r['category'].encode('ascii', errors='replace').decode('ascii')
    print(f"  ID: {r['item_id']}, Name: {name_safe}, Category: {cat_safe}, Price: {r['price']} EUR, Veg: {r['is_vegetarian']}, Allergen: {r['allergen_codes']}")
