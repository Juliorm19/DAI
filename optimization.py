from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SearchResult:
    algorithm: str
    selected_features: list[str]
    regularization_c: float
    fitness: float
    validation_log_loss: float
    convergence: list[float]
    evaluations: int


class LogisticObjective:
    """Shared fitness function so GA and PSO receive a fair comparison."""

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_validation: pd.DataFrame,
        y_validation: pd.Series,
        feature_names: Sequence[str],
        feature_penalty: float = 0.002,
        random_seed: int = 42,
    ) -> None:
        self.feature_names = list(feature_names)
        self.y_train = np.asarray(y_train, dtype=int)
        self.y_validation = np.asarray(y_validation, dtype=int)
        self.feature_penalty = feature_penalty
        self.random_seed = random_seed
        # Do not add missing indicators here: their number depends on the subset. Instead,
        # median-impute all original columns once, then scale them using training data only.
        self.preprocessor = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        self.x_train = self.preprocessor.fit_transform(x_train[self.feature_names])
        self.x_validation = self.preprocessor.transform(x_validation[self.feature_names])
        self.cache: dict[tuple[tuple[bool, ...], float], tuple[float, float]] = {}

    def evaluate(self, mask: np.ndarray, log10_c: float) -> tuple[float, float]:
        mask = np.asarray(mask, dtype=bool).copy()
        if not mask.any():
            mask[0] = True
        log10_c = float(np.clip(log10_c, -3.0, 2.0))
        key = (tuple(bool(value) for value in mask), round(log10_c, 4))
        if key in self.cache:
            return self.cache[key]
        model = LogisticRegression(
            C=10**log10_c,
            max_iter=600,
            solver="lbfgs",
            random_state=self.random_seed,
        )
        model.fit(self.x_train[:, mask], self.y_train)
        probabilities = model.predict_proba(self.x_validation[:, mask])[:, 1]
        validation_loss = float(log_loss(self.y_validation, probabilities, labels=[0, 1]))
        fitness = validation_loss + self.feature_penalty * float(mask.mean())
        self.cache[key] = (fitness, validation_loss)
        return fitness, validation_loss

    @property
    def evaluations(self) -> int:
        return len(self.cache)


def genetic_search(
    objective: LogisticObjective,
    population_size: int,
    generations: int,
    random_seed: int = 42,
) -> SearchResult:
    rng = np.random.default_rng(random_seed)
    dimensions = len(objective.feature_names)
    population: list[tuple[np.ndarray, float]] = [(np.ones(dimensions, dtype=bool), 0.0)]
    while len(population) < population_size:
        mask = rng.random(dimensions) < rng.uniform(0.35, 0.85)
        if not mask.any():
            mask[rng.integers(dimensions)] = True
        population.append((mask, float(rng.uniform(-3.0, 2.0))))

    convergence: list[float] = []
    best_candidate: tuple[np.ndarray, float] | None = None
    best_score = np.inf
    best_validation_loss = np.inf

    def scored(candidate: tuple[np.ndarray, float]) -> tuple[float, float]:
        return objective.evaluate(candidate[0], candidate[1])

    for _ in range(generations):
        ranking = sorted(((scored(item)[0], item) for item in population), key=lambda pair: pair[0])
        if ranking[0][0] < best_score:
            best_score = ranking[0][0]
            best_candidate = (ranking[0][1][0].copy(), float(ranking[0][1][1]))
            best_validation_loss = scored(best_candidate)[1]
        convergence.append(float(best_score))
        elites = [
            (candidate[0].copy(), float(candidate[1]))
            for _, candidate in ranking[: max(2, population_size // 5)]
        ]

        def tournament() -> tuple[np.ndarray, float]:
            indices = rng.choice(len(population), size=min(3, len(population)), replace=False)
            candidates = [population[int(index)] for index in indices]
            winner = min(candidates, key=lambda item: scored(item)[0])
            return winner[0].copy(), float(winner[1])

        new_population = elites
        while len(new_population) < population_size:
            parent_a = tournament()
            parent_b = tournament()
            crossover = rng.random(dimensions) < 0.5
            child_mask = np.where(crossover, parent_a[0], parent_b[0]).astype(bool)
            child_log_c = float((parent_a[1] + parent_b[1]) / 2 + rng.normal(0, 0.25))
            mutation = rng.random(dimensions) < max(1 / dimensions, 0.04)
            child_mask ^= mutation
            if not child_mask.any():
                child_mask[rng.integers(dimensions)] = True
            if rng.random() < 0.25:
                child_log_c += float(rng.normal(0, 0.45))
            new_population.append((child_mask, float(np.clip(child_log_c, -3.0, 2.0))))
        population = new_population[:population_size]

    ranking = sorted(((scored(item)[0], item) for item in population), key=lambda pair: pair[0])
    if ranking[0][0] < best_score or best_candidate is None:
        best_score = ranking[0][0]
        best_candidate = (ranking[0][1][0].copy(), float(ranking[0][1][1]))
        best_validation_loss = scored(best_candidate)[1]
    selected = [name for name, keep in zip(objective.feature_names, best_candidate[0]) if keep]
    return SearchResult(
        algorithm="genetic_algorithm",
        selected_features=selected,
        regularization_c=float(10**best_candidate[1]),
        fitness=float(best_score),
        validation_log_loss=float(best_validation_loss),
        convergence=convergence,
        evaluations=objective.evaluations,
    )


def particle_swarm_search(
    objective: LogisticObjective,
    particles: int,
    iterations: int,
    random_seed: int = 42,
) -> SearchResult:
    rng = np.random.default_rng(random_seed + 101)
    feature_count = len(objective.feature_names)
    dimensions = feature_count + 1
    positions = np.empty((particles, dimensions), dtype=float)
    positions[:, :feature_count] = rng.uniform(0.0, 1.0, size=(particles, feature_count))
    positions[:, -1] = rng.uniform(-3.0, 2.0, size=particles)
    positions[0, :feature_count] = 0.9
    positions[0, -1] = 0.0
    velocities = rng.normal(0.0, 0.12, size=(particles, dimensions))

    personal_best = positions.copy()
    personal_scores = np.full(particles, np.inf)
    global_best = positions[0].copy()
    global_score = np.inf
    global_validation_loss = np.inf
    convergence: list[float] = []

    def decode(position: np.ndarray) -> tuple[np.ndarray, float]:
        mask = position[:feature_count] >= 0.5
        if not mask.any():
            mask[int(np.argmax(position[:feature_count]))] = True
        return mask, float(position[-1])

    for iteration in range(iterations):
        for index in range(particles):
            mask, log_c = decode(positions[index])
            score, validation_loss = objective.evaluate(mask, log_c)
            if score < personal_scores[index]:
                personal_scores[index] = score
                personal_best[index] = positions[index].copy()
            if score < global_score:
                global_score = score
                global_validation_loss = validation_loss
                global_best = positions[index].copy()
        convergence.append(float(global_score))
        inertia = 0.9 - 0.5 * (iteration / max(iterations - 1, 1))
        r1 = rng.random((particles, dimensions))
        r2 = rng.random((particles, dimensions))
        velocities = (
            inertia * velocities
            + 1.6 * r1 * (personal_best - positions)
            + 1.6 * r2 * (global_best - positions)
        )
        velocities[:, :feature_count] = np.clip(velocities[:, :feature_count], -0.25, 0.25)
        velocities[:, -1] = np.clip(velocities[:, -1], -0.6, 0.6)
        positions += velocities
        positions[:, :feature_count] = np.clip(positions[:, :feature_count], 0.0, 1.0)
        positions[:, -1] = np.clip(positions[:, -1], -3.0, 2.0)

    best_mask, best_log_c = decode(global_best)
    selected = [name for name, keep in zip(objective.feature_names, best_mask) if keep]
    return SearchResult(
        algorithm="particle_swarm_optimization",
        selected_features=selected,
        regularization_c=float(10**best_log_c),
        fitness=float(global_score),
        validation_log_loss=float(global_validation_loss),
        convergence=convergence,
        evaluations=objective.evaluations,
    )
