def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculates BMI and returns the value rounded to 2 decimal places."""
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)


def get_bp_category(systolic: int | float | None, diastolic: int | float | None) -> str:
    """Categorizes blood pressure based on standard ACC/AHA clinical ranges."""
    if systolic is None or diastolic is None:
        return "Unknown"

    if systolic > 180 or diastolic > 120:
        return "Hypertensive Crisis"
    if systolic >= 140 or diastolic >= 90:
        return "Hypertension Stage 2"
    if systolic >= 130 or diastolic >= 80:
        return "Hypertension Stage 1"
    if systolic >= 120 and diastolic < 80:
        return "Elevated"
    if systolic < 120 and diastolic < 80:
        return "Normal"
    return "Hypertension Stage 1"


def is_hypertensive(status: str) -> bool:
    return status in {"Hypertension Stage 1", "Hypertension Stage 2", "Hypertensive Crisis"}
