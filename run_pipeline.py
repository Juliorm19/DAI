from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nba_predictor.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena y evalúa el predictor NBA GA/PSO")
    parser.add_argument(
        "--profile",
        choices=["quick", "standard", "thorough"],
        default="standard",
        help="Presupuesto de búsqueda de los optimizadores",
    )
    parser.add_argument("--skip-eda", action="store_true", help="Omite regenerar tablas y figuras exploratorias")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = run_pipeline(ROOT, profile=arguments.profile, skip_eda=arguments.skip_eda)
    print(f"Modelo desplegado: {result['deployed_model']}")
    print(f"Calibración: {result['calibration_method']}")
    print(f"Variables seleccionadas: {len(result['selected_features'])}")
