import unittest

import numpy as np

from ai.evaluation.accuracy import accuracy_score
from ai.evaluation.confusion_matrix import confusion_matrix
from ai.evaluation.precision_recall import classification_report_stats
from ai.utils.math import element_wise_op, matmul, relu, sigmoid


class EvaluationTests(unittest.TestCase):
    def test_accuracy_score(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 2, 0, 1, 1]

        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 4 / 6)

    def test_classification_report_stats(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 2, 0, 1, 1]

        stats = classification_report_stats(y_true, y_pred, num_classes=3)

        self.assertAlmostEqual(stats["macro_precision"], 2 / 3)
        self.assertAlmostEqual(stats["macro_recall"], 2 / 3)
        self.assertAlmostEqual(stats["macro_f1"], 2 / 3)

    def test_confusion_matrix(self):
        matrix = confusion_matrix(
            [0, 1, 2, 0, 1, 2],
            [0, 2, 2, 0, 1, 1],
            num_classes=3,
        )

        np.testing.assert_array_equal(
            matrix,
            np.array([[2, 0, 0], [0, 1, 1], [0, 1, 1]]),
        )


class MathUtilityTests(unittest.TestCase):
    def test_matrix_multiplication(self):
        result = matmul([[1, 2], [3, 4]], [[5], [6]])

        np.testing.assert_array_equal(result, np.array([[17], [39]]))

    def test_element_wise_operations(self):
        np.testing.assert_array_equal(
            element_wise_op([1, 2], [3, 4], "+"),
            np.array([4, 6]),
        )
        np.testing.assert_array_equal(
            element_wise_op([5, 6], [3, 4], "-"),
            np.array([2, 2]),
        )
        np.testing.assert_array_equal(
            element_wise_op([1, 2], [3, 4], "*"),
            np.array([3, 8]),
        )
        with self.assertRaises(ValueError):
            element_wise_op([1], [1], "/")

    def test_sigmoid_and_relu(self):
        np.testing.assert_allclose(sigmoid([0.0]), np.array([0.5]))
        np.testing.assert_array_equal(
            relu([-1.0, 0.0, 2.0]),
            np.array([0.0, 0.0, 2.0]),
        )


if __name__ == "__main__":
    unittest.main()
