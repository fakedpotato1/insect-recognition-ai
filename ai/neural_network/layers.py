import numpy as np


class DenseLayer:
    """Fully connected layer implemented with NumPy matrix operations."""

    def __init__(self, input_dim, output_dim, random_state=None, weight_scale=None):
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rng = np.random.default_rng(random_state)

        if weight_scale is None:
            limit = np.sqrt(6.0 / (self.input_dim + self.output_dim))
            self.weights = self.rng.uniform(
                -limit, limit, size=(self.input_dim, self.output_dim)
            )
        else:
            self.weights = self.rng.normal(
                loc=0.0, scale=weight_scale, size=(self.input_dim, self.output_dim)
            )

        self.biases = np.zeros(self.output_dim, dtype=float)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)
        self._input_cache = None

    def forward(self, inputs):
        batch = np.asarray(inputs, dtype=float)
        if batch.ndim != 2 or batch.shape[1] != self.input_dim:
            raise ValueError(
                f"expected inputs with shape (samples, {self.input_dim}), "
                f"got {batch.shape}"
            )

        self._input_cache = batch
        return batch @ self.weights + self.biases

    def backward(self, grad_output):
        if self._input_cache is None:
            raise RuntimeError("forward must be called before backward")

        grad = np.asarray(grad_output, dtype=float)
        if grad.ndim != 2 or grad.shape[1] != self.output_dim:
            raise ValueError(
                f"expected grad_output with shape (samples, {self.output_dim}), "
                f"got {grad.shape}"
            )

        self.grad_weights = self._input_cache.T @ grad
        self.grad_biases = np.sum(grad, axis=0)
        return grad @ self.weights.T

    def update(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases

    def parameter_count(self):
        return self.weights.size + self.biases.size

    def get_parameters_vector(self):
        return np.concatenate([self.weights.ravel(), self.biases.ravel()])

    def set_parameters_vector(self, values):
        flat = np.asarray(values, dtype=float).ravel()
        expected = self.parameter_count()
        if flat.size != expected:
            raise ValueError(f"expected {expected} parameters, got {flat.size}")

        weight_count = self.weights.size
        self.weights = flat[:weight_count].reshape(self.weights.shape).copy()
        self.biases = flat[weight_count:].reshape(self.biases.shape).copy()
