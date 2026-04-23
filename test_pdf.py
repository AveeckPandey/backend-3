from app.services.ocr_service import extract_from_pdf

data = extract_from_pdf(r"C:\Users\aveec\Desktop\Building\important assets\dummy-1.pdf")
print(data) 
# Expected Output: {'glucose': 130, 'age': 25, 'gender': 'Male'} [cite: 5, 9]