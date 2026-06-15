import base64
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ai.inference import load_artifact
from ai.inference import predict_base64_image
from ai.inference import train_mlp_classifier_from_csv
from ai.inference.localization import locate_insect_crop


class InferencePipelineTests(unittest.TestCase):
    def test_train_save_load_and_predict_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "features.csv"
            artifact_path = temp_path / "insect_classifier.npz"

            rows = ["feat_0,feat_1,label"]
            for value in (0.0, 0.1, 0.2, 0.3):
                rows.append(f"{value},{value},0")
            for value in (1.0, 1.1, 1.2, 1.3):
                rows.append(f"{value},{value},1")
            csv_path.write_text("\n".join(rows), encoding="utf-8")

            result = train_mlp_classifier_from_csv(
                csv_path,
                artifact_path,
                hidden_dims=(4,),
                epochs=5,
                random_state=7,
                split_ratios=(0.5, 0.25, 0.25),
            )
            self.assertTrue(artifact_path.exists())
            artifact = load_artifact(artifact_path)
            prediction = artifact.predict_one(np.array([0.0, 0.0]))

        self.assertEqual(result.samples, 8)
        self.assertEqual(result.train_samples, 4)
        self.assertEqual(result.validation_samples, 2)
        self.assertEqual(result.test_samples, 2)
        self.assertEqual(result.input_dim, 2)
        self.assertIsNotNone(result.validation_accuracy)
        self.assertIsNotNone(result.test_accuracy)
        self.assertIn(prediction.label, {"ant", "bed-bug"})
        self.assertGreaterEqual(prediction.confidence, 0.0)
        self.assertLessEqual(prediction.confidence, 1.0)

    def test_localization_can_fallback_to_full_image(self):
        image = np.full((64, 64, 3), 255, dtype=np.uint8)

        crop = locate_insect_crop(image)

        self.assertTrue(crop.used_fallback)
        self.assertEqual(crop.bbox, (0, 0, 64, 64))

    def test_predict_base64_image_uses_feature_extraction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "features.csv"
            artifact_path = temp_path / "insect_classifier.npz"

            feature_columns = ",".join(f"feat_{index}" for index in range(461))
            zero_features = ",".join("0" for _ in range(461))
            one_features = ",".join("1" for _ in range(461))
            rows = [f"{feature_columns},label"]
            rows.extend(f"{zero_features},0" for _ in range(4))
            rows.extend(f"{one_features},1" for _ in range(4))
            csv_path.write_text("\n".join(rows), encoding="utf-8")
            train_mlp_classifier_from_csv(
                csv_path,
                artifact_path,
                hidden_dims=(3,),
                epochs=1,
                random_state=11,
                split_ratios=(0.5, 0.25, 0.25),
            )

            image = np.zeros((128, 128, 3), dtype=np.uint8)
            success, encoded = cv2.imencode(".png", image)
            self.assertTrue(success)
            payload = base64.b64encode(encoded.tobytes()).decode("ascii")

            prediction = predict_base64_image(
                payload,
                artifact_path,
                localization_mode="full",
            )

        self.assertIn("insect_name", prediction)
        self.assertIn("confidence", prediction)
        self.assertIn("crop_box", prediction)


if __name__ == "__main__":
    unittest.main()
