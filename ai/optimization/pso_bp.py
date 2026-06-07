import numpy as np

from ai.neural_network.mlp import MLPClassifier


class ParticleSwarmBP:
    """Use particle swarm optimization to initialize a NumPy BP network."""

    def __init__(
        self,
        model,
        swarm_size=20,
        iterations=30,
        inertia=0.72,
        cognitive_weight=1.49,
        social_weight=1.49,
        initial_position_scale=0.5,
        initial_velocity_scale=0.1,
        velocity_clip=None,
        random_state=None,
    ):
        if swarm_size < 2:
            raise ValueError("swarm_size must be at least 2")

        self.model = model
        self.swarm_size = int(swarm_size)
        self.iterations = int(iterations)
        self.inertia = float(inertia)
        self.cognitive_weight = float(cognitive_weight)
        self.social_weight = float(social_weight)
        self.initial_position_scale = float(initial_position_scale)
        self.initial_velocity_scale = float(initial_velocity_scale)
        self.velocity_clip = velocity_clip
        self.rng = np.random.default_rng(random_state)
        self.history_ = None

    def _evaluate_positions(self, positions, X, y):
        losses = []
        for position in positions:
            self.model.set_parameters_vector(position)
            losses.append(self.model.loss(X, y))
        return np.asarray(losses, dtype=float)

    def optimize_initial_weights(self, X, y):
        base_parameters = self.model.get_parameters_vector()
        parameter_count = base_parameters.size
        positions = base_parameters + self.rng.normal(
            0.0,
            self.initial_position_scale,
            size=(self.swarm_size, parameter_count),
        )
        velocities = self.rng.normal(
            0.0,
            self.initial_velocity_scale,
            size=(self.swarm_size, parameter_count),
        )

        losses = self._evaluate_positions(positions, X, y)
        personal_best_positions = positions.copy()
        personal_best_losses = losses.copy()
        global_best_index = int(np.argmin(personal_best_losses))
        global_best_position = personal_best_positions[global_best_index].copy()
        global_best_loss = float(personal_best_losses[global_best_index])
        loss_history = [global_best_loss]

        for _ in range(self.iterations):
            random_personal = self.rng.random(size=(self.swarm_size, parameter_count))
            random_social = self.rng.random(size=(self.swarm_size, parameter_count))
            velocities = (
                self.inertia * velocities
                + self.cognitive_weight
                * random_personal
                * (personal_best_positions - positions)
                + self.social_weight * random_social * (global_best_position - positions)
            )
            if self.velocity_clip is not None:
                clip = abs(float(self.velocity_clip))
                velocities = np.clip(velocities, -clip, clip)

            positions = positions + velocities
            losses = self._evaluate_positions(positions, X, y)
            improved = losses < personal_best_losses
            personal_best_positions[improved] = positions[improved]
            personal_best_losses[improved] = losses[improved]

            global_best_index = int(np.argmin(personal_best_losses))
            if personal_best_losses[global_best_index] < global_best_loss:
                global_best_loss = float(personal_best_losses[global_best_index])
                global_best_position = personal_best_positions[global_best_index].copy()
            loss_history.append(global_best_loss)

        self.model.set_parameters_vector(global_best_position)
        self.history_ = {
            "best_loss": global_best_loss,
            "optimizer_loss": loss_history,
            "best_parameters": global_best_position.copy(),
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


def build_pso_bp_classifier(input_dim, hidden_dims, output_dim, **kwargs):
    model_kwargs = kwargs.pop("model_kwargs", {})
    optimizer_kwargs = kwargs
    model = MLPClassifier(input_dim, hidden_dims, output_dim, **model_kwargs)
    return ParticleSwarmBP(model, **optimizer_kwargs)
