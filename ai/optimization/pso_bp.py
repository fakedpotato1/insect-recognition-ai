import numpy as np

from ai.evolutionary import PSO
from ai.neural_network.mlp import MLPClassifier


class ParticleSwarmBP(PSO):
    """Adapt the evo-baselines particle swarm optimizer to initialize BP weights."""

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
        self.model = model
        self.swarm_size = int(swarm_size)
        self.iterations = int(iterations)
        self.history_ = None
        self._X = None
        self._y = None

        super().__init__(
            num_particles=swarm_size,
            dimensions=model.parameter_count(),
            inertia=inertia,
            cognitive_weight=cognitive_weight,
            social_weight=social_weight,
            initial_position_scale=initial_position_scale,
            initial_velocity_scale=initial_velocity_scale,
            velocity_clip=velocity_clip,
            maximize=False,
            random_state=random_state,
        )

    def objective(self, position):
        if self._X is None or self._y is None:
            raise RuntimeError("training data must be set before evaluating objective")
        self.model.set_parameters_vector(position)
        return self.model.loss(self._X, self._y)

    def optimize_initial_weights(self, X, y):
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=int)
        base_parameters = self.model.get_parameters_vector()
        result = self.optimize(
            iterations=self.iterations,
            center=base_parameters,
            position_scale=self.initial_position_scale,
            velocity_scale=self.initial_velocity_scale,
        )
        global_best_position = result["best_position"]

        self.model.set_parameters_vector(global_best_position)
        self.history_ = {
            "best_loss": result["best_score"],
            "optimizer_loss": result["best_history"],
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
