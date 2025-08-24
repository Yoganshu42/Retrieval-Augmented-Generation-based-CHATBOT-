import fitz
import re

path = r"C:\Users\yogan\OneDrive\Desktop\my rag project\AI Training Document.pdf"
# Extract raw text from the document
def extract_text_from_document(path):
    with fitz.open(path) as file:
        text = ""
        for page in file:
            text += page.get_text()
        return text

extracted_text = extract_text_from_document(path)
# print(f"Contents: {extracted_text}")

def clean_text(extracted_text):
    # Remove multiple spaces and tabs
    text = re.sub(r'\s+', ' ',extracted_text)

    # Remove headers and footers
    text = re.sub(r'Page \d+ of \d+','', extracted_text, flags = re.IGNORECASE)
    text = re.sub(r'\n\d+\n',' ', extracted_text)  #Standalone Numbers

    # Remove unwanted special characters
    text = re.sub(r'[^\x00-\x7F]+',' ', extracted_text)

    # Remove excessive punctuation
    text = re.sub(r'[_.<>|]',' ', extracted_text)

    # Fix white spaces again
    text = re.sub(r'\s+', ' ', extracted_text).strip()

    return text

cleaned_text = clean_text(extracted_text)
# print(f"{cleaned_text}")

