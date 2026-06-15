import numpy as np


class PSO:
    """Reusable particle swarm optimizer implemented with NumPy."""

    def __init__(
        self,
        num_particles=30,
        dimensions=2,
        inertia=0.7,
        cognitive_weight=1.5,
        social_weight=1.5,
        initial_position_scale=10.0,
        initial_velocity_scale=0.1,
        velocity_clip=None,
        maximize=False,
        random_state=None,
    ):
        if num_particles < 2:
            raise ValueError("num_particles must be at least 2")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")

        self.num_particles = int(num_particles)
        self.dimensions = int(dimensions)
        self.inertia = float(inertia)
        self.cognitive_weight = float(cognitive_weight)
        self.social_weight = float(social_weight)
        self.initial_position_scale = float(initial_position_scale)
        self.initial_velocity_scale = float(initial_velocity_scale)
        self.velocity_clip = velocity_clip
        self.maximize = bool(maximize)
        self.rng = np.random.default_rng(random_state)

        self.positions = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_scores = None
        self.initialize_swarm()

    def initialize_swarm(self, center=None, position_scale=None, velocity_scale=None):
        if center is None:
            center = np.zeros(self.dimensions, dtype=float)
        center = np.asarray(center, dtype=float).ravel()
        if center.size != self.dimensions:
            raise ValueError("center must match dimensions")

        if position_scale is None:
            position_scale = self.initial_position_scale
        if velocity_scale is None:
            velocity_scale = self.initial_velocity_scale

        self.positions = center + self.rng.uniform(
            -float(position_scale),
            float(position_scale),
            size=(self.num_particles, self.dimensions),
        )
        self.velocities = self.rng.normal(
            0.0,
            float(velocity_scale),
            size=(self.num_particles, self.dimensions),
        )
        self.personal_best = self.positions.copy()
        self.personal_best_scores = None

    def objective(self, x):
        return np.sum(x**2)

    def update_velocity(self, particle, global_best, w=None, c1=None, c2=None):
        if w is None:
            w = self.inertia
        if c1 is None:
            c1 = self.cognitive_weight
        if c2 is None:
            c2 = self.social_weight

        r1 = self.rng.random(self.dimensions)
        r2 = self.rng.random(self.dimensions)
        cognitive = c1 * r1 * (self.personal_best[particle] - self.positions[particle])
        social = c2 * r2 * (global_best - self.positions[particle])

        self.velocities[particle] = w * self.velocities[particle] + cognitive + social
        if self.velocity_clip is not None:
            clip = abs(float(self.velocity_clip))
            self.velocities[particle] = np.clip(self.velocities[particle], -clip, clip)

    def update_position(self, particle):
        self.positions[particle] += self.velocities[particle]

    def evaluate_positions(self):
        return np.asarray([self.objective(position) for position in self.positions])

    def optimize(self, iterations=30, center=None, position_scale=None, velocity_scale=None):
        self.initialize_swarm(
            center=center,
            position_scale=position_scale,
            velocity_scale=velocity_scale,
        )

        scores = self.evaluate_positions()
        self.personal_best = self.positions.copy()
        self.personal_best_scores = scores.copy()
        global_best_index = self._best_index(self.personal_best_scores)
        global_best = self.personal_best[global_best_index].copy()
        global_best_score = float(self.personal_best_scores[global_best_index])
        best_history = [global_best_score]

        for _ in range(int(iterations)):
            for particle in range(self.num_particles):
                self.update_velocity(particle, global_best)
                self.update_position(particle)

            scores = self.evaluate_positions()
            improved = self._is_better_array(scores, self.personal_best_scores)
            self.personal_best[improved] = self.positions[improved]
            self.personal_best_scores[improved] = scores[improved]

            global_best_index = self._best_index(self.personal_best_scores)
            if self._is_better(self.personal_best_scores[global_best_index], global_best_score):
                global_best_score = float(self.personal_best_scores[global_best_index])
                global_best = self.personal_best[global_best_index].copy()
            best_history.append(global_best_score)

        return {
            "best_position": global_best,
            "best_score": global_best_score,
            "best_history": best_history,
        }

    def _best_index(self, scores):
        return int(np.argmax(scores)) if self.maximize else int(np.argmin(scores))

    def _is_better(self, score, best_score):
        return score > best_score if self.maximize else score < best_score

    def _is_better_array(self, scores, best_scores):
        return scores > best_scores if self.maximize else scores < best_scores
