import unittest

from app.services.analysis_processing import normalize_analysis_inputs
from app.services.health_utils import get_bp_category, is_hypertensive


class BloodPressureClassificationTests(unittest.TestCase):
    def test_bp_classification_thresholds(self):
        self.assertEqual(get_bp_category(118, 76), "Normal")
        self.assertEqual(get_bp_category(125, 78), "Elevated")
        self.assertEqual(get_bp_category(120, 80), "Hypertension Stage 1")
        self.assertEqual(get_bp_category(135, 79), "Hypertension Stage 1")
        self.assertEqual(get_bp_category(142, 88), "Hypertension Stage 2")
        self.assertEqual(get_bp_category(181, 90), "Hypertensive Crisis")
        self.assertEqual(get_bp_category(170, 121), "Hypertensive Crisis")

    def test_hypertension_flag(self):
        self.assertFalse(is_hypertensive("Normal"))
        self.assertFalse(is_hypertensive("Elevated"))
        self.assertTrue(is_hypertensive("Hypertension Stage 1"))
        self.assertTrue(is_hypertensive("Hypertension Stage 2"))
        self.assertTrue(is_hypertensive("Hypertensive Crisis"))


class NormalizationTests(unittest.TestCase):
    def test_missing_values_use_defaults_and_notes(self):
        normalized = normalize_analysis_inputs(
            {
                "name": "",
                "age": "",
                "gender": "",
                "glucose": "",
                "weight_kg": "",
                "height_cm": "",
                "systolic_bp": "",
                "diastolic_bp": "",
                "is_smoker": "",
                "sleep_duration": "",
                "stress_score": "",
            }
        )

        self.assertEqual(normalized["name"], "Patient")
        self.assertEqual(normalized["age"], 45)
        self.assertEqual(normalized["gender"], "Unknown")
        self.assertEqual(normalized["glucose"], 100.0)
        self.assertEqual(normalized["bmi"], 25.0)
        self.assertEqual(normalized["bp_status"], "Normal")
        self.assertFalse(normalized["is_smoker"])
        self.assertGreaterEqual(len(normalized["processing_notes"]), 6)

    def test_bmi_calculates_from_valid_inputs(self):
        normalized = normalize_analysis_inputs(
            {
                "name": "Ms Nitiksha Parvadia",
                "age": 23,
                "gender": "Female",
                "glucose": 139,
                "weight_kg": 70,
                "height_cm": 175,
                "systolic_bp": 120,
                "diastolic_bp": 80,
                "sleep_duration": 9,
                "stress_score": 2,
            }
        )

        self.assertAlmostEqual(normalized["bmi"], 22.86, places=2)
        self.assertEqual(normalized["bp_status"], "Hypertension Stage 1")
        self.assertEqual(normalized["input_summary"]["blood_pressure_reading"], "120/80 mmHg")


if __name__ == "__main__":
    unittest.main()
