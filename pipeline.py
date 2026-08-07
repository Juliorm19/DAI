from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

from .config import MODEL_FEATURES, RANDOM_SEED, SEARCH_PROFILES, ProjectPaths
from .data import build_dataset, write_quality_json
from .eda import generate_eda, write_eda_report
from .modeling import (
    evaluate_probabilities,
    fit_probability_model,
    temporal_group_split,
)
from .optimization import LogisticObjective, SearchResult, genetic_search, particle_swarm_search


LOGGER = logging.getLogger(__name__)


def _serializable_search(result: SearchResult) -> dict[str, Any]:
    return asdict(result)


def _save_search_outputs(results: list[SearchResult], artifacts: Path, figures: Path) -> None:
    payload = [_serializable_search(result) for result in results]
    (artifacts / "optimizer_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = []
    for result in results:
        selected = set(result.selected_features)
        for feature in MODEL_FEATURES:
            rows.append(
                {
                    "algorithm": result.algorithm,
                    "feature": feature,
                    "selected": int(feature in selected),
                    "regularization_c": result.regularization_c,
                }
            )
    pd.DataFrame(rows).to_csv(artifacts / "feature_selection.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 4.8))
    for result in results:
        label = "Algoritmo genético" if result.algorithm == "genetic_algorithm" else "PSO"
        plt.plot(range(1, len(result.convergence) + 1), result.convergence, marker="o", label=label)
    plt.xlabel("Generación / iteración")
    plt.ylabel("Fitness (menor es mejor)")
    plt.title("Convergencia de los optimizadores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "optimizer_convergence.png", dpi=170, bbox_inches="tight")
    plt.close()


def _save_evaluation_plots(
    metrics: pd.DataFrame,
    truth: pd.Series,
    predictions: dict[str, np.ndarray],
    figures: Path,
) -> None:
    ordered = metrics.sort_values("log_loss")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.barplot(data=ordered, x="model", y="log_loss", hue="model", legend=False, ax=axes[0])
    axes[0].set_title("Log loss en prueba")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(data=ordered, x="model", y="brier_score", hue="model", legend=False, ax=axes[1])
    axes[1].set_title("Brier score en prueba")
    axes[1].tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(figures / "probability_metrics.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    plt.figure(figsize=(7, 6))
    for name, probabilities in predictions.items():
        observed, predicted = calibration_curve(truth, probabilities, n_bins=10, strategy="quantile")
        plt.plot(predicted, observed, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "--", color="black", label="Calibración perfecta")
    plt.xlabel("Probabilidad predicha")
    plt.ylabel("Frecuencia observada")
    plt.title("Curvas de calibración en el conjunto de prueba")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "calibration_curves.png", dpi=170, bbox_inches="tight")
    plt.close()


def _write_results_report(
    output: Path,
    quality: dict[str, Any],
    split_summary: dict[str, Any],
    metrics: pd.DataFrame,
    searches: list[SearchResult],
    deployed_name: str,
    deployed_method: str,
    profile: str,
) -> None:
    metric_lines = [
        "| Modelo | Accuracy | ROC-AUC | Log loss | Brier | ECE | Calibración |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics.itertuples(index=False):
        metric_lines.append(
            f"| {row.model} | {row.accuracy:.4f} | {row.roc_auc:.4f} | {row.log_loss:.4f} | "
            f"{row.brier_score:.4f} | {row.ece_10_bins:.4f} | {row.calibration_method} |"
        )
    optimizer_lines = []
    for result in searches:
        optimizer_lines.append(
            f"- **{result.algorithm}**: {len(result.selected_features)} variables, C={result.regularization_c:.5g}, "
            f"log loss de validación={result.validation_log_loss:.5f}, {result.evaluations} evaluaciones."
        )
    model_quality = quality["modeling_dataset"]
    content = [
        "# Informe técnico de resultados",
        "",
        "## Resumen ejecutivo",
        "",
        f"Se construyó un sistema reproducible de predicción NBA con {model_quality['rows']:,} observaciones "
        f"pareadas provenientes de {model_quality['unique_games']:,} encuentros únicos. Se comparó una regresión "
        "logística convencional con dos métodos de inteligencia distribuida: algoritmo genético y optimización "
        "por enjambre de partículas (PSO). Todas las variables se calculan con información anterior al partido.",
        "",
        f"El artefacto desplegado es **{deployed_name}**, calibrado mediante **{deployed_method}**. La elección se "
        "hizo dentro del bloque de calibración; el conjunto de prueba se reservó exclusivamente para la estimación "
        "final del rendimiento.",
        "",
        "## Protocolo experimental",
        "",
        f"Perfil de búsqueda ejecutado: `{profile}`.",
        "",
        "| Bloque | Filas | Partidos | Inicio | Fin |",
        "|---|---:|---:|---|---|",
    ]
    for name, values in split_summary.items():
        content.append(
            f"| {name} | {values['rows']:,} | {values['games']:,} | {values['date_min']} | {values['date_max']} |"
        )
    content.extend(
        [
            "",
            "Las particiones son cronológicas y se agrupan por `game_id`, por lo que las dos perspectivas de un "
            "partido nunca aparecen en bloques diferentes. Las medianas y escalas se aprenden exclusivamente con "
            "datos de desarrollo.",
            "",
            "## Optimización",
            "",
            *optimizer_lines,
            "",
            "Ambos optimizadores minimizaron exactamente la misma función: log loss de validación más una penalización "
            "pequeña por complejidad. El cromosoma/partícula codifica la máscara de variables y `log10(C)`.",
            "",
            "![Convergencia](figures/optimizer_convergence.png)",
            "",
            "## Resultados finales",
            "",
            *metric_lines,
            "",
            "![Métricas](figures/probability_metrics.png)",
            "",
            "![Calibración](figures/calibration_curves.png)",
            "",
            "Además de accuracy y ROC-AUC, se priorizan log loss, Brier y ECE porque el requisito principal es entregar "
            "probabilidades calibradas, no solo una clase ganadora.",
            "",
            "## Limitaciones",
            "",
            "- Los datos terminan el 22 de diciembre de 2022; una predicción de temporadas posteriores representa una "
            "extrapolación desde el último estado conocido.",
            "- Lesiones, alineaciones confirmadas, traspasos y contexto de playoffs no están modelados explícitamente.",
            "- Las dos perspectivas de cada encuentro son observaciones dependientes; por eso se agrupan durante todas "
            "las divisiones y se reporta también la cantidad de partidos únicos.",
            "- El modelo estima probabilidades, no garantiza resultados deportivos ni debe interpretarse como consejo de apuestas.",
        ]
    )
    output.write_text("\n".join(content) + "\n", encoding="utf-8")


def _split_metadata(parts: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, frame in parts.items():
        result[name] = {
            "rows": int(len(frame)),
            "games": int(frame["game_id"].nunique()),
            "date_min": frame["game_date"].min().date().isoformat(),
            "date_max": frame["game_date"].max().date().isoformat(),
        }
    return result


def run_pipeline(project_root: Path, profile: str = "standard", skip_eda: bool = False) -> dict[str, Any]:
    if profile not in SEARCH_PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(SEARCH_PROFILES)}")
    paths = ProjectPaths(project_root.resolve())
    paths.ensure_output_dirs()
    search_profile = SEARCH_PROFILES[profile]

    LOGGER.info("Auditing and building the historical dataset")
    bundle = build_dataset(paths.data, include_audit=True)
    write_quality_json(bundle.quality, paths.artifacts / "data_quality.json")
    bundle.modeling.to_csv(paths.processed / "modeling_dataset.csv", index=False, encoding="utf-8-sig")
    bundle.history.to_csv(paths.artifacts / "inference_history.csv", index=False, encoding="utf-8-sig")
    bundle.rankings.to_csv(paths.artifacts / "rankings_clean.csv", index=False, encoding="utf-8-sig")

    if not skip_eda:
        LOGGER.info("Generating EDA tables and figures")
        generate_eda(bundle.modeling, bundle.quality, paths.figures, paths.tables)
        write_eda_report(bundle.quality, paths.reports / "eda_report.md")

    splits = temporal_group_split(bundle.modeling)
    split_frames = {
        "Entrenamiento": splits.train,
        "Optimización": splits.optimization,
        "Calibración": splits.calibration,
        "Prueba": splits.test,
    }
    split_summary = _split_metadata(split_frames)
    (paths.artifacts / "split_summary.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    LOGGER.info("Running genetic search")
    ga_objective = LogisticObjective(
        splits.train[MODEL_FEATURES],
        splits.train["won"],
        splits.optimization[MODEL_FEATURES],
        splits.optimization["won"],
        MODEL_FEATURES,
        random_seed=RANDOM_SEED,
    )
    ga_result = genetic_search(
        ga_objective,
        population_size=search_profile.ga_population,
        generations=search_profile.ga_generations,
        random_seed=RANDOM_SEED,
    )

    LOGGER.info("Running particle swarm optimization")
    pso_objective = LogisticObjective(
        splits.train[MODEL_FEATURES],
        splits.train["won"],
        splits.optimization[MODEL_FEATURES],
        splits.optimization["won"],
        MODEL_FEATURES,
        random_seed=RANDOM_SEED,
    )
    pso_result = particle_swarm_search(
        pso_objective,
        particles=search_profile.pso_particles,
        iterations=search_profile.pso_iterations,
        random_seed=RANDOM_SEED,
    )
    searches = [ga_result, pso_result]
    _save_search_outputs(searches, paths.artifacts, paths.figures)

    development = pd.concat([splits.train, splits.optimization], ignore_index=True)
    candidate_specs = {
        "Regresión logística": (MODEL_FEATURES, 1.0),
        "GA-LR": (ga_result.selected_features, ga_result.regularization_c),
        "PSO-LR": (pso_result.selected_features, pso_result.regularization_c),
    }
    models: dict[str, Any] = {}
    selection_losses: dict[str, float] = {}
    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []

    home_rate = float(development.loc[development["is_home"] == 1, "won"].mean())
    home_probabilities = np.where(splits.test["is_home"].to_numpy() == 1, home_rate, 1 - home_rate)
    home_metrics = evaluate_probabilities(splits.test["won"], home_probabilities)
    metric_rows.append(
        {
            "model": "Ventaja local",
            **home_metrics,
            "calibration_method": "frecuencia empírica",
            "features": 1,
        }
    )
    predictions["Ventaja local"] = home_probabilities

    for name, (features, c_value) in candidate_specs.items():
        LOGGER.info("Fitting and calibrating %s", name)
        model, calibration_scores = fit_probability_model(
            name,
            development,
            splits.calibration,
            list(features),
            c_value,
            random_seed=RANDOM_SEED,
        )
        probabilities = model.predict_proba(splits.test[list(features)])
        metrics = evaluate_probabilities(splits.test["won"], probabilities)
        selection_loss = calibration_scores[model.calibrator.method]
        selection_losses[name] = selection_loss
        model.metadata.update(
            {
                "selection_log_loss": float(selection_loss),
                "data_date_min": bundle.modeling["game_date"].min().date().isoformat(),
                "data_date_max": bundle.modeling["game_date"].max().date().isoformat(),
                "search_profile": profile,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        models[name] = model
        predictions[name] = probabilities
        metric_rows.append(
            {
                "model": name,
                **metrics,
                "calibration_method": model.calibrator.method,
                "features": len(features),
            }
        )

    # The deployed model must incorporate a course optimization method. The ordinary
    # logistic model remains a benchmark and is never allowed to replace GA/PSO here.
    deployed_name = min(("GA-LR", "PSO-LR"), key=selection_losses.get)
    deployed_model = models[deployed_name]
    deployed_model.metadata["deployed_selection_rule"] = "lowest calibration-selection log loss among GA-LR and PSO-LR"
    joblib.dump(deployed_model, paths.artifacts / "best_model.joblib")

    metrics_frame = pd.DataFrame(metric_rows).sort_values("log_loss").reset_index(drop=True)
    metrics_frame.to_csv(paths.artifacts / "benchmark_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_payload = metrics_frame.to_dict(orient="records")
    (paths.artifacts / "benchmark_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    prediction_frame = splits.test[
        ["game_id", "game_date", "team_id", "opponent_id", "is_home", "won"]
    ].copy()
    for name, values in predictions.items():
        prediction_frame[f"prob_{name}"] = values
    prediction_frame.to_csv(paths.artifacts / "test_predictions.csv", index=False, encoding="utf-8-sig")
    _save_evaluation_plots(metrics_frame, splits.test["won"], predictions, paths.figures)
    _write_results_report(
        paths.reports / "technical_report.md",
        bundle.quality,
        split_summary,
        metrics_frame,
        searches,
        deployed_name,
        deployed_model.calibrator.method,
        profile,
    )

    summary = {
        "deployed_model": deployed_name,
        "calibration_method": deployed_model.calibrator.method,
        "selected_features": deployed_model.feature_names,
        "metrics": metrics_payload,
        "modeling_dataset": bundle.quality["modeling_dataset"],
        "split_summary": split_summary,
    }
    (paths.artifacts / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
