import numpy as np

from ai.evolutionary import GeneticAlgorithm
from ai.neural_network.mlp import MLPClassifier


class GeneticAlgorithmBP(GeneticAlgorithm):
    """Use the project genetic algorithm to initialize a BP network."""

    def __init__(
        self,
        model,
        population_size=20,
        generations=30,
        elite_count=2,
        crossover_rate=0.8,
        mutation_rate=0.05,
        mutation_scale=0.1,
        initial_noise=0.5,
        random_state=None,
    ):
        self.model = model
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.initial_noise = float(initial_noise)
        self.history_ = None
        self._X = None
        self._y = None

        super().__init__(
            pop_size=population_size,
            chromosome_length=model.parameter_count(),
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_count=elite_count,
            encoding="real",
            mutation_scale=mutation_scale,
            maximize=False,
            random_state=random_state,
        )

    def fitness(self, chromosome):
        if self._X is None or self._y is None:
            raise RuntimeError("training data must be set before evaluating fitness")
        self.model.set_parameters_vector(chromosome)
        return self.model.loss(self._X, self._y)

    def optimize_initial_weights(self, X, y):
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=int)
        base_parameters = self.model.get_parameters_vector()
        result = self.evolve(
            generations=self.generations,
            center=base_parameters,
            noise_scale=self.initial_noise,
        )
        best_chromosome = result["best_chromosome"]

        self.model.set_parameters_vector(best_chromosome)
        self.history_ = {
            "best_loss": result["best_score"],
            "optimizer_loss": result["best_history"],
            "best_parameters": best_chromosome.copy(),
        }
        return self.history_

    def fit(self, X, y, bp_epochs=100, batch_size=None, shuffle=True):
        optimizer_history = self.optimize_initial_weights(X, y)
        bp_history = self.model.fit(
            X, y, epochs=bp_epochs, batch_size=batch_size, shuffle=shuffle
        )
        self.history_ = {
            **optimizer_history,
            "bp_loss": bp_history["loss"],
        }
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


def build_ga_bp_classifier(input_dim, hidden_dims, output_dim, **kwargs):
    model_kwargs = kwargs.pop("model_kwargs", {})
    optimizer_kwargs = kwargs
    model = MLPClassifier(input_dim, hidden_dims, output_dim, **model_kwargs)
    return GeneticAlgorithmBP(model, **optimizer_kwargs)
