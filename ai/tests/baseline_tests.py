import unittest

import numpy as np

from ai.baselines import (
    build_decision_tree_baseline,
    build_knn_baseline,
    build_naive_bayes_baseline,
    evaluate_baseline,
    evaluate_default_baselines,
)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 1.0],
                [1.1, 1.0],
                [2.0, 2.0],
                [2.0, 2.1],
            ]
        )
        self.y_train = np.array([0, 0, 1, 1, 2, 2])
        self.X_test = np.array([[0.05, 0.0], [1.0, 1.2], [2.2, 2.0]])
        self.y_test = np.array([0, 1, 2])

    def test_individual_baseline_builders_fit_and_predict(self):
        builders = (
            lambda: build_knn_baseline(n_neighbors=1),
            build_decision_tree_baseline,
            build_naive_bayes_baseline,
        )

        for build_estimator in builders:
            estimator = build_estimator()
            result = evaluate_baseline(
                estimator.__class__.__name__,
                estimator,
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
            )

            self.assertEqual(result.predictions.shape, (3,))
            self.assertGreaterEqual(result.accuracy, 0.0)
            self.assertLessEqual(result.accuracy, 1.0)
            self.assertGreaterEqual(result.macro_f1, 0.0)
            self.assertLessEqual(result.macro_f1, 1.0)

    def test_default_baseline_suite_returns_named_results(self):
        results = evaluate_default_baselines(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            knn_neighbors=1,
        )

        self.assertEqual(
            [result.name for result in results],
            ["KNN", "Decision Tree", "Naive Bayes"],
        )
        self.assertTrue(all(result.accuracy >= 0.66 for result in results))

    def test_evaluate_baseline_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            evaluate_baseline(
                "KNN",
                build_knn_baseline(n_neighbors=1),
                self.X_train,
                self.y_train[:-1],
                self.X_test,
                self.y_test,
            )


if __name__ == "__main__":
    unittest.main()
