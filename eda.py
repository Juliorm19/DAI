from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import MODEL_FEATURES


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def generate_eda(modeling: pd.DataFrame, quality: dict[str, Any], figures_dir: Path, tables_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = modeling[MODEL_FEATURES].describe(percentiles=[0.25, 0.5, 0.75]).T
    summary["missing"] = modeling[MODEL_FEATURES].isna().sum()
    summary["missing_pct"] = summary["missing"] / len(modeling) * 100
    summary.to_csv(tables_dir / "feature_summary.csv", encoding="utf-8-sig")
    modeling[MODEL_FEATURES + ["won"]].corr(numeric_only=True).to_csv(
        tables_dir / "feature_correlations.csv", encoding="utf-8-sig"
    )

    plt.figure(figsize=(6.5, 4))
    counts = modeling["won"].value_counts().sort_index()
    ax = sns.barplot(x=["Derrota", "Victoria"], y=counts.values, hue=["Derrota", "Victoria"], legend=False)
    ax.set(title="Balance de la etiqueta", xlabel="Resultado del equipo A", ylabel="Observaciones")
    for index, value in enumerate(counts.values):
        ax.text(index, value + 250, f"{value:,}", ha="center")
    _save_figure(figures_dir / "class_balance.png")

    home = modeling[modeling["is_home"] == 1]
    season_rate = home.groupby("season", as_index=False)["won"].mean()
    plt.figure(figsize=(10, 4.5))
    ax = sns.lineplot(data=season_rate, x="season", y="won", marker="o")
    ax.axhline(home["won"].mean(), color="tab:red", linestyle="--", label="Promedio global")
    ax.set(title="Porcentaje de victorias del equipo local por temporada", xlabel="Temporada", ylabel="Proporción")
    ax.legend()
    _save_figure(figures_dir / "home_win_rate_by_season.png")

    correlations = modeling[MODEL_FEATURES + ["won"]].corr(numeric_only=True)
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlations, cmap="vlag", center=0, vmin=-1, vmax=1, square=False)
    plt.title("Matriz de correlación de variables históricas")
    _save_figure(figures_dir / "correlation_heatmap.png")

    selected = [
        "win_rate_10_diff",
        "point_margin_avg_10_diff",
        "pts_for_avg_5_diff",
        "pts_against_avg_5_diff",
        "reb_avg_5_diff",
        "ast_avg_5_diff",
        "rest_days_diff",
        "standings_win_pct_diff",
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    for column, axis in zip(selected, axes.flat):
        sns.histplot(modeling[column], bins=35, kde=True, ax=axis, color="steelblue")
        axis.set_title(column)
        axis.set_xlabel("")
    fig.suptitle("Distribuciones de variables seleccionadas", y=1.01, fontsize=15)
    _save_figure(figures_dir / "feature_distributions.png")

    for panel_index, start in enumerate(range(0, len(MODEL_FEATURES), 7), start=1):
        panel_features = MODEL_FEATURES[start : start + 7]
        fig, axes = plt.subplots(4, 2, figsize=(13, 14))
        for axis in axes.flat:
            axis.set_visible(False)
        for column, axis in zip(panel_features, axes.flat):
            axis.set_visible(True)
            values = modeling[column].dropna()
            sns.histplot(
                values,
                bins=min(35, max(5, int(values.nunique()))),
                kde=values.nunique() >= 15,
                ax=axis,
                color="steelblue",
            )
            axis.set_title(column)
            axis.set_xlabel("")
        fig.suptitle(f"Distribuciones de todas las variables - panel {panel_index}", y=1.01, fontsize=15)
        _save_figure(figures_dir / f"all_feature_distributions_{panel_index}.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, axis in zip(
        ["win_rate_10_diff", "point_margin_avg_10_diff", "standings_win_pct_diff", "rest_days_diff"],
        axes.flat,
    ):
        sample = modeling[[column, "won"]].dropna()
        if len(sample) > 20_000:
            sample = sample.sample(20_000, random_state=42)
        sns.boxplot(data=sample, x="won", y=column, hue="won", legend=False, showfliers=False, ax=axis)
        axis.set_xticks([0, 1], ["Derrota", "Victoria"])
    fig.suptitle("Variables históricas frente al resultado", y=1.01, fontsize=15)
    _save_figure(figures_dir / "features_vs_outcome.png")

    raw_names = [name for name in quality if name.endswith(".csv")]
    null_frame = pd.DataFrame(
        {
            "archivo": raw_names,
            "celdas_nulas": [quality[name]["null_cells"] for name in raw_names],
            "filas": [quality[name]["rows"] for name in raw_names],
            "columnas": [quality[name]["columns"] for name in raw_names],
        }
    )
    null_frame["porcentaje_nulo"] = (
        null_frame["celdas_nulas"] / (null_frame["filas"] * null_frame["columnas"]) * 100
    )
    null_frame.to_csv(tables_dir / "raw_quality_summary.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9, 4.5))
    ax = sns.barplot(data=null_frame, x="archivo", y="porcentaje_nulo", hue="archivo", legend=False)
    ax.set(title="Porcentaje de celdas nulas por archivo", xlabel="", ylabel="Porcentaje")
    ax.tick_params(axis="x", rotation=20)
    _save_figure(figures_dir / "raw_missingness.png")


def write_eda_report(quality: dict[str, Any], output: Path) -> None:
    lines = [
        "# Análisis exploratorio y calidad de datos",
        "",
        "## Inventario",
        "",
        "| Archivo | Filas | Columnas | Celdas nulas | Mal formateadas | Filas duplicadas |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, entry in quality.items():
        if not name.endswith(".csv"):
            continue
        lines.append(
            f"| {name} | {entry['rows']:,} | {entry['columns']} | {entry['null_cells']:,} | "
            f"{entry['malformed_numeric_cells']:,} | {entry['duplicate_rows']:,} |"
        )
    model = quality["modeling_dataset"]
    lines.extend(
        [
            "",
            "## Dataset de modelado",
            "",
            f"Después de remover partidos incompletos y consolidar identificadores repetidos, se obtuvieron "
            f"**{model['unique_games']:,} partidos únicos**. La representación pareada genera "
            f"**{model['rows']:,} filas**, con {model['features']} variables predictoras y una tasa positiva "
            f"de {model['positive_rate']:.1%}.",
            "",
            "Los nulos de `games_details.csv` corresponden principalmente a jugadores que no participaron. "
            "No se convierten en ceros a nivel de jugador: primero se agregan las estadísticas disponibles "
            "por equipo y partido. `RETURNTOPLAY` está casi completamente vacío en `ranking.csv` y no se usa.",
            "",
            "## Hallazgos visuales",
            "",
            "![Balance](figures/class_balance.png)",
            "",
            "![Localía](figures/home_win_rate_by_season.png)",
            "",
            "![Correlaciones](figures/correlation_heatmap.png)",
            "",
            "![Distribuciones](figures/feature_distributions.png)",
            "",
            "### Distribución de las 21 variables candidatas",
            "",
            "![Todas las distribuciones, panel 1](figures/all_feature_distributions_1.png)",
            "",
            "![Todas las distribuciones, panel 2](figures/all_feature_distributions_2.png)",
            "",
            "![Todas las distribuciones, panel 3](figures/all_feature_distributions_3.png)",
            "",
            "![Variables y resultado](figures/features_vs_outcome.png)",
            "",
            "![Nulos](figures/raw_missingness.png)",
            "",
            "Las estadísticas completas se encuentran en `reports/tables/feature_summary.csv` y la matriz "
            "numérica en `reports/tables/feature_correlations.csv`.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
