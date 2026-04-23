def generate_health_tips(results):
    # Logic: Look at the highest SHAP factor
    tips = []
    
    if results['diabetes']['probability'] > 0.4:
        tips.append("Your glucose is on the high side. Consider a low-glycemic diet and consulting a nutritionist.")
    
    if results['stroke']['probability'] > 0.2:
        tips.append("Consistent cardio (30 mins/day) can help manage your stroke risk markers.")
        
    return tips


def get_personalized_advice(results):
    advice = []
    
    # Check Diabetes Risk (from labs glucose)
    if results['diabetes']['probability'] > 0.4:
        advice.append({
            "category": "Metabolic",
            "text": "Your glucose is near the pre-diabetic threshold. Focus on high-fiber foods and limit refined sugars.",
            "urgency": "High"
        })

    # Check Stroke Risk (Age + Glucose impact)
    if results['stroke']['probability'] > 0.2:
        advice.append({
            "category": "Neurological",
            "text": "Monitor your blood pressure weekly. Reducing sodium intake can significantly lower stroke risk.",
            "urgency": "Medium"
        })
        
    return advice