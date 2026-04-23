import pdfplumber
import re

def extract_from_pdf(pdf_path):
    extracted_data = {"glucose": None, "age": 23, "gender": "Female", "name": None}
    
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text()
        
        # 1. Age/Gender Extraction 
        age_gender_match = re.search(r'(\d+)/(Male|Female)', text)
        if age_gender_match:
            extracted_data["age"] = int(age_gender_match.group(1))
            extracted_data["gender"] = age_gender_match.group(2)
            
        # 2. Glucose Extraction (Specific to the "130" or "139" result) 
        glucose_match = re.search(r'Random Blood Sugar\s*(\d+)', text)
        if glucose_match:
            extracted_data["glucose"] = int(glucose_match.group(1))
        
        # 3. Targeted Name Extraction
        # This matches "Name : " and stops BEFORE it hits "Patient ID" or a newline
        name_match = re.search(r'Name\s*:\s*(.*?)(?=Patient ID|Report ID|\n|$)', text)
        if name_match:
            # .strip() removes any trailing whitespace 
            extracted_data["name"] = name_match.group(1).strip()
            
    return extracted_data
