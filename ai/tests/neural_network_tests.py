import unittest

import numpy as np

from ai.neural_network import DenseLayer, MLPClassifier, softmax, softmax_cross_entropy
from ai.optimization import GeneticAlgorithmBP, ParticleSwarmBP


class DenseLayerTests(unittest.TestCase):
    def test_forward_backward_shapes(self):
        layer = DenseLayer(3, 2, random_state=1)
        output = layer.forward(np.ones((4, 3)))
        grad_input = layer.backward(np.ones((4, 2)))

        self.assertEqual(output.shape, (4, 2))
        self.assertEqual(grad_input.shape, (4, 3))
        self.assertEqual(layer.grad_weights.shape, (3, 2))
        self.assertEqual(layer.grad_biases.shape, (2,))


class SoftmaxCrossEntropyTests(unittest.TestCase):
    def test_softmax_probabilities_sum_to_one(self):
        probabilities = softmax(np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]))
        np.testing.assert_allclose(np.sum(probabilities, axis=1), np.ones(2))

    def test_cross_entropy_gradient_shape_and_row_sum(self):
        logits = np.array([[2.0, 0.5], [0.1, 1.5]])
        loss, grad_logits, probabilities = softmax_cross_entropy(logits, np.array([0, 1]))

        self.assertGreater(loss, 0.0)
        self.assertEqual(grad_logits.shape, logits.shape)
        self.assertEqual(probabilities.shape, logits.shape)
        np.testing.assert_allclose(
            np.sum(grad_logits, axis=1), np.zeros(2), atol=1e-12
        )


class MLPClassifierTests(unittest.TestCase):
    def test_parameter_vector_round_trip(self):
        model = MLPClassifier(2, [3], 2, random_state=1)
        parameters = model.get_parameters_vector()
        replacement = np.linspace(-0.2, 0.2, parameters.size)

        model.set_parameters_vector(replacement)
        np.testing.assert_allclose(model.get_parameters_vector(), replacement)

    def test_training_reduces_loss(self):
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1, 1, 1])
        model = MLPClassifier(2, [4], 2, learning_rate=0.2, random_state=2)

        initial_loss = model.loss(X, y)
        history = model.fit(X, y, epochs=80, batch_size=4, shuffle=False)

        self.assertLess(history["loss"][-1], initial_loss)
        self.assertEqual(model.predict(X).shape, (4,))


class OptimizedBPTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.2, 0.8],
                [0.8, 0.2],
            ]
        )
        self.y = np.array([0, 1, 1, 1, 1, 1])

    def test_ga_bp_optimizes_initial_weights(self):
        model = MLPClassifier(2, [3], 2, learning_rate=0.1, random_state=3)
        ga_bp = GeneticAlgorithmBP(
            model,
            population_size=6,
            generations=4,
            elite_count=1,
            random_state=4,
        )

        history = ga_bp.optimize_initial_weights(self.X, self.y)

        self.assertTrue(np.isfinite(history["best_loss"]))
        self.assertEqual(len(history["optimizer_loss"]), 4)
        self.assertEqual(ga_bp.predict(self.X).shape, (6,))

    def test_pso_bp_optimizes_initial_weights(self):
        model = MLPClassifier(2, [3], 2, learning_rate=0.1, random_state=5)
        pso_bp = ParticleSwarmBP(model, swarm_size=6, iterations=4, random_state=6)

        history = pso_bp.optimize_initial_weights(self.X, self.y)

        self.assertTrue(np.isfinite(history["best_loss"]))
        self.assertEqual(len(history["optimizer_loss"]), 5)
        self.assertEqual(pso_bp.predict(self.X).shape, (6,))


if __name__ == "__main__":
    unittest.main()
