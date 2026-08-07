# Proyecto Capstone: predicción deportiva NBA

Sistema completo de predicción de resultados NBA con variables históricas, un algoritmo genético (GA),
optimización por enjambre de partículas (PSO), calibración explícita de probabilidades y una interfaz Streamlit.

## Cumplimiento del enunciado

- Baloncesto como deporte seleccionado.
- 668,628 filas en la fuente principal de detalle y 53,046 observaciones en el dataset pareado de modelado.
- 21 variables candidatas, todas calculadas con información anterior al partido.
- Auditoría de nulos, formatos, duplicados, estadísticas, distribuciones y correlaciones.
- Algoritmo genético e inteligencia de enjambre mediante PSO; no usa aprendizaje federado.
- Probabilidades calibradas con comparación entre sigmoid e isotónica.
- Benchmarks, partición cronológica, métricas de clasificación y de probabilidad.
- Interfaz para seleccionar un encuentro futuro.
- Documentación de infraestructura, metodología, resultados y guion de exposición.

## Instalación

Se recomienda Python 3.11 o posterior.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Ejecutar el proyecto

Entrenamiento recomendado:

```powershell
python run_pipeline.py --profile standard
```

Perfiles disponibles:

- `quick`: comprobación rápida del flujo.
- `standard`: equilibrio recomendado para entrega y demostración.
- `thorough`: búsqueda más extensa para el resultado final.

Luego se inicia la interfaz:

```powershell
streamlit run app.py
```

## Entregables generados

- `output/pdf/Informe_Tecnico_Infraestructura_NBA.pdf`: informe final listo para entrega.
- `reports/Informe_Tecnico_Infraestructura_NBA.docx`: versión editable del informe.
- `reports/technical_report.md`: resultados regenerables en formato abierto.
- `reports/eda_report.md`: análisis exploratorio y calidad de datos.
- `artifacts/best_model.joblib`: modelo GA-LR calibrado utilizado por Streamlit.
- `artifacts/benchmark_metrics.csv`: benchmark final de los cuatro enfoques.

Pruebas automatizadas:

```powershell
pytest -q
```

## Estructura

```text
.
├── app.py                         # interfaz Streamlit
├── run_pipeline.py                # entrada de entrenamiento
├── data/
│   ├── *.csv                      # fuentes originales
│   └── processed/                 # dataset histórico generado
├── src/nba_predictor/
│   ├── data.py                    # auditoría, limpieza y variables históricas
│   ├── optimization.py            # GA y PSO
│   ├── modeling.py                # particiones, calibración y métricas
│   ├── eda.py                     # tablas y visualizaciones
│   └── pipeline.py                # orquestación reproducible
├── artifacts/                     # modelo, métricas y predicciones
├── reports/                       # EDA, informe técnico, figuras y tablas
├── docs/                          # infraestructura, metodología y exposición
└── tests/                         # controles de fuga e integridad
```

## Regla temporal

Cada media móvil usa un desplazamiento de un partido antes de calcularse. Las clasificaciones se unen con
`allow_exact_matches=False`: una fila fechada el día del encuentro no puede entrar en la predicción. Las dos
perspectivas de un mismo partido permanecen siempre en la misma partición mediante agrupación por `game_id`.

Los datos terminan el 22 de diciembre de 2022. La aplicación puede construir encuentros posteriores, pero los
interpreta como una extrapolación basada en el último estado conocido.
