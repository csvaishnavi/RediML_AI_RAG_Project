import pypdf

pdf_path = r"d:\ReDi\Final Project\raw_data\Menu of ReDi Restaurant.pdf"
output_path = r"c:\Users\Bright\.gemini\antigravity-ide\scratch\extracted_menu.txt"

reader = pypdf.PdfReader(pdf_path)
print(f"Total pages: {len(reader.pages)}")

with open(output_path, "w", encoding="utf-8") as f:
    for i, page in enumerate(reader.pages):
        page_num = i + 1
        text = page.extract_text()
        f.write(f"--- PAGE {page_num} ---\n")
        f.write(text)
        f.write("\n\n")

print(f"Extracted text written to {output_path}")
