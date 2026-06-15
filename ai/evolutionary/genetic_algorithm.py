import numpy as np


class GeneticAlgorithm:
    """Reusable genetic algorithm implemented with NumPy."""

    def __init__(
        self,
        pop_size=50,
        chromosome_length=10,
        mutation_rate=0.1,
        crossover_rate=1.0,
        elite_count=0,
        encoding="binary",
        mutation_scale=0.1,
        maximize=True,
        random_state=None,
    ):
        if pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        if chromosome_length <= 0:
            raise ValueError("chromosome_length must be positive")
        if elite_count < 0 or elite_count >= pop_size:
            raise ValueError("elite_count must be in [0, pop_size)")
        if encoding not in {"binary", "real"}:
            raise ValueError("encoding must be 'binary' or 'real'")

        self.pop_size = int(pop_size)
        self.chromosome_length = int(chromosome_length)
        self.mutation_rate = float(mutation_rate)
        self.crossover_rate = float(crossover_rate)
        self.elite_count = int(elite_count)
        self.encoding = encoding
        self.mutation_scale = float(mutation_scale)
        self.maximize = bool(maximize)
        self.rng = np.random.default_rng(random_state)

    def initialize_population(self, center=None, noise_scale=0.5):
        if self.encoding == "binary":
            return self.rng.integers(0, 2, size=(self.pop_size, self.chromosome_length))

        if center is None:
            center = np.zeros(self.chromosome_length, dtype=float)
        center = np.asarray(center, dtype=float).ravel()
        if center.size != self.chromosome_length:
            raise ValueError("center must match chromosome_length")
        return center + self.rng.normal(
            0.0,
            float(noise_scale),
            size=(self.pop_size, self.chromosome_length),
        )

    def fitness(self, chromosome):
        return np.sum(chromosome)

    def selection(self, population, fitness_values=None, tournament_size=2):
        if fitness_values is None:
            fitness_values = self.evaluate_population(population)

        sample_size = min(int(tournament_size), len(population))
        indices = self.rng.choice(len(population), size=sample_size, replace=False)
        sampled_scores = fitness_values[indices]
        best_offset = (
            int(np.argmax(sampled_scores))
            if self.maximize
            else int(np.argmin(sampled_scores))
        )
        return population[indices[best_offset]]

    def crossover(self, parent1, parent2):
        if self.rng.random() > self.crossover_rate or self.chromosome_length == 1:
            return parent1.copy()

        point = self.rng.integers(1, self.chromosome_length)
        return np.concatenate([parent1[:point], parent2[point:]])

    def mutation(self, chromosome):
        mutated = chromosome.copy()
        mutation_mask = self.rng.random(self.chromosome_length) < self.mutation_rate
        if self.encoding == "binary":
            mutated[mutation_mask] = 1 - mutated[mutation_mask]
        else:
            mutated += mutation_mask * self.rng.normal(
                0.0,
                self.mutation_scale,
                size=self.chromosome_length,
            )
        return mutated

    def evaluate_population(self, population):
        return np.asarray([self.fitness(chromosome) for chromosome in population])

    def evolve(self, generations=30, population=None, center=None, noise_scale=0.5):
        if population is None:
            population = self.initialize_population(center=center, noise_scale=noise_scale)
        else:
            population = np.asarray(population, dtype=float)

        best_score = -np.inf if self.maximize else np.inf
        best_chromosome = population[0].copy()
        best_history = []

        for _ in range(int(generations)):
            fitness_values = self.evaluate_population(population)
            best_index = self._best_index(fitness_values)
            if self._is_better(fitness_values[best_index], best_score):
                best_score = float(fitness_values[best_index])
                best_chromosome = population[best_index].copy()

            best_history.append(best_score)
            order = (
                np.argsort(-fitness_values)
                if self.maximize
                else np.argsort(fitness_values)
            )
            next_population = [
                population[index].copy() for index in order[: self.elite_count]
            ]

            while len(next_population) < self.pop_size:
                parent1 = self.selection(population, fitness_values)
                parent2 = self.selection(population, fitness_values)
                child = self.crossover(parent1, parent2)
                next_population.append(self.mutation(child))

            population = np.asarray(next_population, dtype=float)

        return {
            "best_chromosome": best_chromosome,
            "best_score": best_score,
            "best_history": best_history,
            "population": population,
        }

    def _best_index(self, fitness_values):
        return int(np.argmax(fitness_values)) if self.maximize else int(np.argmin(fitness_values))

    def _is_better(self, score, best_score):
        return score > best_score if self.maximize else score < best_score
