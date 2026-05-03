import unittest

from app.schemas.analysis import ReportRequest
from app.services.report_generator import _flag_value


class ReportRequestFlagTests(unittest.TestCase):
    def test_report_request_keeps_top_level_condition_flags(self):
        payload = ReportRequest(
            bmi=30,
            results={},
            ai_recommendation="Follow up with a clinician.",
            family_history_diabetes=True,
            family_history_stroke=True,
            has_diabetes_history=True,
        ).model_dump()

        self.assertTrue(payload["family_history_diabetes"])
        self.assertTrue(payload["family_history_stroke"])
        self.assertTrue(payload["has_diabetes_history"])

    def test_flag_value_prefers_input_summary_then_falls_back_to_top_level(self):
        data = {"family_history_stroke": True}
        input_summary = {}
        self.assertTrue(_flag_value(data, input_summary, "family_history_stroke"))

        data = {"family_history_stroke": False}
        input_summary = {"family_history_stroke": True}
        self.assertTrue(_flag_value(data, input_summary, "family_history_stroke"))


if __name__ == "__main__":
    unittest.main()
