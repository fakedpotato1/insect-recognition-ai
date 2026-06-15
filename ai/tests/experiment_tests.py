import unittest

import numpy as np

from ai.experiments import evaluate_classifier, run_model_comparison
from ai.neural_network import MLPClassifier


class ExperimentTests(unittest.TestCase):
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

    def test_evaluate_classifier_returns_project_metrics(self):
        classifier = MLPClassifier(
            2,
            [4],
            3,
            learning_rate=0.1,
            random_state=9,
        )

        result = evaluate_classifier(
            "BP",
            classifier,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            fit_kwargs={"epochs": 5, "batch_size": 3},
        )

        self.assertEqual(result.name, "BP")
        self.assertEqual(result.predictions.shape, (3,))
        self.assertIn("loss", result.history)
        self.assertGreaterEqual(result.macro_f1, 0.0)
        self.assertLessEqual(result.macro_f1, 1.0)

    def test_run_model_comparison_includes_ai_and_baseline_models(self):
        results = run_model_comparison(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            hidden_dims=(4,),
            bp_epochs=3,
            batch_size=3,
            random_state=11,
            pso_kwargs={"swarm_size": 4, "iterations": 2},
            ga_kwargs={"population_size": 4, "generations": 2, "elite_count": 1},
            knn_neighbors=1,
        )

        self.assertEqual(
            [result.name for result in results],
            ["BP", "PSO-BP", "GA-BP", "KNN", "Decision Tree", "Naive Bayes"],
        )
        self.assertTrue(all(result.predictions.shape == (3,) for result in results))
        self.assertTrue(all(0.0 <= result.accuracy <= 1.0 for result in results))


if __name__ == "__main__":
    unittest.main()
