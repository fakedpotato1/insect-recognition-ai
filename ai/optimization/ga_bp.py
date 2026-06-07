import numpy as np

from ai.neural_network.mlp import MLPClassifier


class GeneticAlgorithmBP:
    """Use a genetic algorithm to initialize a NumPy BP neural network."""

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
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if elite_count < 1 or elite_count >= population_size:
            raise ValueError("elite_count must be in [1, population_size)")

        self.model = model
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.elite_count = int(elite_count)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.mutation_scale = float(mutation_scale)
        self.initial_noise = float(initial_noise)
        self.rng = np.random.default_rng(random_state)
        self.history_ = None

    def _evaluate_population(self, population, X, y):
        losses = []
        for chromosome in population:
            self.model.set_parameters_vector(chromosome)
            losses.append(self.model.loss(X, y))
        return np.asarray(losses, dtype=float)

    def _select_parent(self, population, losses, tournament_size=3):
        sample_size = min(tournament_size, self.population_size)
        indices = self.rng.choice(self.population_size, size=sample_size, replace=False)
        best_index = indices[np.argmin(losses[indices])]
        return population[best_index]

    def _crossover(self, parent_a, parent_b):
        if self.rng.random() > self.crossover_rate:
            return parent_a.copy(), parent_b.copy()

        mask = self.rng.random(parent_a.size) < 0.5
        child_a = np.where(mask, parent_a, parent_b)
        child_b = np.where(mask, parent_b, parent_a)
        return child_a, child_b

    def _mutate(self, chromosome):
        mutation_mask = self.rng.random(chromosome.size) < self.mutation_rate
        noise = self.rng.normal(0.0, self.mutation_scale, size=chromosome.size)
        return chromosome + mutation_mask * noise

    def optimize_initial_weights(self, X, y):
        base_parameters = self.model.get_parameters_vector()
        population = base_parameters + self.rng.normal(
            0.0,
            self.initial_noise,
            size=(self.population_size, base_parameters.size),
        )

        best_loss = np.inf
        best_chromosome = base_parameters.copy()
        loss_history = []

        for _ in range(self.generations):
            losses = self._evaluate_population(population, X, y)
            best_index = int(np.argmin(losses))
            if losses[best_index] < best_loss:
                best_loss = float(losses[best_index])
                best_chromosome = population[best_index].copy()

            loss_history.append(best_loss)
            order = np.argsort(losses)
            next_population = [population[i].copy() for i in order[: self.elite_count]]

            while len(next_population) < self.population_size:
                parent_a = self._select_parent(population, losses)
                parent_b = self._select_parent(population, losses)
                child_a, child_b = self._crossover(parent_a, parent_b)
                next_population.append(self._mutate(child_a))
                if len(next_population) < self.population_size:
                    next_population.append(self._mutate(child_b))

            population = np.asarray(next_population, dtype=float)

        self.model.set_parameters_vector(best_chromosome)
        self.history_ = {
            "best_loss": best_loss,
            "optimizer_loss": loss_history,
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
