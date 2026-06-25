import numpy as np

from ai.neural_network.activations import relu, relu_derivative, softmax
from ai.neural_network.layers import DenseLayer
from ai.neural_network.losses import softmax_cross_entropy


class MLPClassifier:
    """Small multi-layer perceptron classifier with manual backpropagation.

    The class intentionally avoids deep-learning frameworks so the forward
    pass, backpropagation, L2 penalty, and Adam update are all visible in NumPy.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        learning_rate=0.01,
        random_state=None,
        optimizer="sgd",
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        l2_penalty=0.0,
    ):
        if output_dim <= 1:
            raise ValueError("output_dim must be greater than 1 for classification")

        self.input_dim = int(input_dim)
        self.hidden_dims = [int(dim) for dim in hidden_dims]
        self.output_dim = int(output_dim)
        self.learning_rate = float(learning_rate)
        self.optimizer = optimizer
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.l2_penalty = float(l2_penalty)
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
        self._optimizer_step = 0
        self._adam_state = [
            {
                "mw": np.zeros_like(layer.weights),
                "vw": np.zeros_like(layer.weights),
                "mb": np.zeros_like(layer.biases),
                "vb": np.zeros_like(layer.biases),
            }
            for layer in self.layers
        ]

    def _regularized_grad_weights(self, layer):
        if self.l2_penalty <= 0:
            return layer.grad_weights
        return layer.grad_weights + self.l2_penalty * layer.weights

    def _apply_sgd(self):
        for layer in self.layers:
            grad_weights = self._regularized_grad_weights(layer)
            layer.apply_gradients(
                self.learning_rate * grad_weights,
                self.learning_rate * layer.grad_biases,
            )

    def _apply_adam(self):
        self._optimizer_step += 1
        for layer, state in zip(self.layers, self._adam_state):
            grad_weights = self._regularized_grad_weights(layer)
            state["mw"] = self.beta1 * state["mw"] + (1.0 - self.beta1) * grad_weights
            state["vw"] = (
                self.beta2 * state["vw"]
                + (1.0 - self.beta2) * np.square(grad_weights)
            )
            state["mb"] = self.beta1 * state["mb"] + (1.0 - self.beta1) * layer.grad_biases
            state["vb"] = (
                self.beta2 * state["vb"]
                + (1.0 - self.beta2) * np.square(layer.grad_biases)
            )

            # Bias correction keeps early Adam updates from being too small
            # because the moment vectors start from zeros.
            mw_hat = state["mw"] / (1.0 - self.beta1 ** self._optimizer_step)
            vw_hat = state["vw"] / (1.0 - self.beta2 ** self._optimizer_step)
            mb_hat = state["mb"] / (1.0 - self.beta1 ** self._optimizer_step)
            vb_hat = state["vb"] / (1.0 - self.beta2 ** self._optimizer_step)

            weight_step = self.learning_rate * mw_hat / (np.sqrt(vw_hat) + self.epsilon)
            bias_step = self.learning_rate * mb_hat / (np.sqrt(vb_hat) + self.epsilon)
            layer.apply_gradients(weight_step, bias_step)

    def _apply_optimizer(self):
        if self.optimizer == "sgd":
            self._apply_sgd()
        elif self.optimizer == "adam":
            self._apply_adam()
        else:
            raise ValueError("optimizer must be 'sgd' or 'adam'")

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
        if self.l2_penalty <= 0:
            return loss
        penalty = sum(np.sum(np.square(layer.weights)) for layer in self.layers)
        return loss + 0.5 * self.l2_penalty * penalty

    def train_batch(self, X, y):
        logits = self._forward_logits(X)
        loss, grad_logits, _ = softmax_cross_entropy(logits, y)

        # Softmax cross-entropy starts the backward pass at the output logits;
        # hidden layers then multiply by the ReLU derivative.
        grad = self.layers[-1].backward(grad_logits)
        for layer_index in range(len(self.layers) - 2, -1, -1):
            grad = grad * relu_derivative(self._hidden_z_cache[layer_index])
            grad = self.layers[layer_index].backward(grad)

        self._apply_optimizer()

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
