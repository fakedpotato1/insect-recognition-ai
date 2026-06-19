import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai.data import load_csv_dataset, load_csv_train_val_test, train_val_test_split


class CsvDatasetTests(unittest.TestCase):
    def test_load_csv_dataset_with_header_and_string_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "features.csv"
            csv_path.write_text(
                "area,perimeter,color,label\n"
                "1.0,2.0,3.0,bee\n"
                "4.0,5.0,6.0,ant\n"
                "7.0,8.0,9.0,bee\n",
                encoding="utf-8",
            )

            dataset = load_csv_dataset(csv_path)

        np.testing.assert_allclose(
            dataset.X,
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        )
        np.testing.assert_array_equal(dataset.y, np.array([1, 0, 1]))
        self.assertEqual(dataset.feature_names, ("area", "perimeter", "color"))
        self.assertEqual(dataset.label_name, "label")
        self.assertEqual(dataset.label_mapping, {"ant": 0, "bee": 1})
        self.assertEqual(dataset.inverse_label_mapping, {0: "ant", 1: "bee"})

    def test_load_csv_dataset_can_select_named_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "features.csv"
            csv_path.write_text(
                "image_id,width,height,label\n"
                "img_1,10,20,0\n"
                "img_2,30,40,1\n",
                encoding="utf-8",
            )

            dataset = load_csv_dataset(
                csv_path,
                label_column="label",
                feature_columns=["width", "height"],
            )

        np.testing.assert_allclose(dataset.X, np.array([[10.0, 20.0], [30.0, 40.0]]))
        np.testing.assert_array_equal(dataset.y, np.array([0, 1]))
        self.assertEqual(dataset.feature_names, ("width", "height"))

    def test_load_csv_dataset_rejects_non_contiguous_label_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "features.csv"
            csv_path.write_text(
                "feature,label\n"
                "1.0,ant\n"
                "2.0,bee\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_csv_dataset(csv_path, label_mapping={"ant": 1, "bee": 2})

    def test_load_csv_train_val_test_combines_loading_and_splitting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "features.csv"
            lines = ["f1,f2,label"]
            for index in range(10):
                lines.append(f"{index},{index + 0.5},{index % 2}")
            csv_path.write_text("\n".join(lines), encoding="utf-8")

            data = load_csv_train_val_test(
                csv_path,
                label_column="label",
                random_state=7,
                shuffle=False,
                stratify=False,
            )

        self.assertEqual(data.dataset.X.shape, (10, 2))
        self.assertEqual(data.splits.X_train.shape[0], 7)
        self.assertEqual(data.splits.X_val.shape[0], 2)
        self.assertEqual(data.splits.X_test.shape[0], 1)


class TrainValTestSplitTests(unittest.TestCase):
    def test_default_split_uses_seven_two_one_ratio(self):
        X = np.arange(40, dtype=float).reshape(10, 4)
        y = np.arange(10) % 2

        splits = train_val_test_split(
            X, y, shuffle=False, stratify=False, ratios=(0.7, 0.2, 0.1)
        )

        self.assertEqual(splits.X_train.shape[0], 7)
        self.assertEqual(splits.X_val.shape[0], 2)
        self.assertEqual(splits.X_test.shape[0], 1)
        np.testing.assert_array_equal(splits.train_indices, np.arange(7))
        np.testing.assert_array_equal(splits.val_indices, np.arange(7, 9))
        np.testing.assert_array_equal(splits.test_indices, np.arange(9, 10))

    def test_stratified_split_preserves_class_counts(self):
        X = np.arange(80, dtype=float).reshape(20, 4)
        y = np.array([0] * 10 + [1] * 10)

        splits = train_val_test_split(X, y, shuffle=False, stratify=True)

        self.assertEqual(splits.X_train.shape[0], 14)
        self.assertEqual(splits.X_val.shape[0], 4)
        self.assertEqual(splits.X_test.shape[0], 2)
        self.assertEqual(np.sum(splits.y_train == 0), 7)
        self.assertEqual(np.sum(splits.y_train == 1), 7)
        self.assertEqual(np.sum(splits.y_val == 0), 2)
        self.assertEqual(np.sum(splits.y_val == 1), 2)
        self.assertEqual(np.sum(splits.y_test == 0), 1)
        self.assertEqual(np.sum(splits.y_test == 1), 1)

    def test_stratified_split_keeps_global_ratio_with_singletons(self):
        X = np.arange(40, dtype=float).reshape(10, 4)
        y = np.arange(10)

        splits = train_val_test_split(X, y, shuffle=False, stratify=True)

        self.assertEqual(splits.X_train.shape[0], 7)
        self.assertEqual(splits.X_val.shape[0], 2)
        self.assertEqual(splits.X_test.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
