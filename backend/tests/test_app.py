import base64
import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app import app  # noqa: E402
from config import Config  # noqa: E402


class BackendApiTests(unittest.TestCase):
    def setUp(self):
        Config.USE_MOCK_MODEL = True
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "healthy")

    def test_ai_contract_endpoint_describes_detection_payload(self):
        response = self.client.get("/api/ai/contract")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("/detect", payload["detect"]["paths"])
        self.assertIn("/api/detect", payload["detect"]["paths"])
        self.assertEqual(
            payload["feature_extraction"]["output_csv"]["label_column"],
            "label",
        )

    def test_detect_accepts_base64_image_payload(self):
        image = base64.b64encode(b"not a real image, but valid base64").decode("ascii")

        response = self.client.post("/api/detect", json={"image": image})
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("insect_name", payload)
        self.assertIn("confidence", payload)
        self.assertEqual(payload["contract_version"], "1.0")

    def test_detect_rejects_invalid_base64(self):
        response = self.client.post("/api/detect", json={"image": "not-base64"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Invalid base64 image")


if __name__ == "__main__":
    unittest.main()
