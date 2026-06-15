import numpy as np

from ai.neural_network.activations import relu, relu_derivative, softmax
from ai.neural_network.layers import DenseLayer
from ai.neural_network.losses import softmax_cross_entropy


class MLPClassifier:
    """Small multi-layer perceptron classifier with manual backpropagation."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        learning_rate=0.01,
        random_state=None,
    ):
        if output_dim <= 1:
            raise ValueError("output_dim must be greater than 1 for classification")

        self.input_dim = int(input_dim)
        self.hidden_dims = [int(dim) for dim in hidden_dims]
        self.output_dim = int(output_dim)
        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(random_state)

        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        self.layers = [
            DenseLayer(
                layer_dims[i],
                layer_dims[i + 1],
                random_state=self.rng.integers(0, np.iinfo(np.int32).max),
            )
            for i in range(len(layer_dims) - 1)
        ]
        self._hidden_z_cache = []

    def _forward_logits(self, X):
        activations = np.asarray(X, dtype=float)
        if activations.ndim != 2 or activations.shape[1] != self.input_dim:
            raise ValueError(
                f"expected X with shape (samples, {self.input_dim}), "
                f"got {activations.shape}"
            )

        self._hidden_z_cache = []
        for layer in self.layers[:-1]:
            z = layer.forward(activations)
            self._hidden_z_cache.append(z)
            activations = relu(z)
        return self.layers[-1].forward(activations)

    def predict_proba(self, X):
        return softmax(self._forward_logits(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def loss(self, X, y):
        logits = self._forward_logits(X)
        loss, _, _ = softmax_cross_entropy(logits, y)
        return loss

    def train_batch(self, X, y):
        logits = self._forward_logits(X)
        loss, grad_logits, _ = softmax_cross_entropy(logits, y)

        grad = self.layers[-1].backward(grad_logits)
        for layer_index in range(len(self.layers) - 2, -1, -1):
            grad = grad * relu_derivative(self._hidden_z_cache[layer_index])
            grad = self.layers[layer_index].backward(grad)

        for layer in self.layers:
            layer.update(self.learning_rate)

        return loss

    def fit(self, X, y, epochs=100, batch_size=None, shuffle=True):
        features = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=int)
        if features.ndim != 2 or features.shape[0] != labels.size:
            raise ValueError("X and y must contain the same number of samples")

        sample_count = features.shape[0]
        if batch_size is None or batch_size <= 0 or batch_size > sample_count:
            batch_size = sample_count

        history = {"loss": []}
        for _ in range(int(epochs)):
            indices = np.arange(sample_count)
            if shuffle:
                self.rng.shuffle(indices)

            epoch_losses = []
            for start in range(0, sample_count, batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_loss = self.train_batch(
                    features[batch_indices], labels[batch_indices]
                )
                epoch_losses.append(batch_loss)
            history["loss"].append(float(np.mean(epoch_losses)))

        return history

    def parameter_count(self):
        return sum(layer.parameter_count() for layer in self.layers)

    def get_parameters_vector(self):
        return np.concatenate([layer.get_parameters_vector() for layer in self.layers])

    def set_parameters_vector(self, values):
        flat = np.asarray(values, dtype=float).ravel()
        expected = self.parameter_count()
        if flat.size != expected:
            raise ValueError(f"expected {expected} parameters, got {flat.size}")

        cursor = 0
        for layer in self.layers:
            next_cursor = cursor + layer.parameter_count()
            layer.set_parameters_vector(flat[cursor:next_cursor])
            cursor = next_cursor
