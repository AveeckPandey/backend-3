def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculates BMI and returns the value rounded to 2 decimal places."""
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)

def get_bp_category(systolic: int, diastolic: int) -> str:
    """Categorizes Blood Pressure based on standard medical ranges."""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif 120 <= systolic <= 129 and diastolic < 80:
        return "Elevated"
    elif 130 <= systolic <= 139 or 80 <= diastolic <= 89:
        return "Hypertension Stage 1"
    else:
        return "Hypertension Stage 2"