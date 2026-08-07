from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nba_predictor.data import build_match_feature_row


st.set_page_config(page_title="Predictor NBA GA + PSO", page_icon="🏀", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "artifacts" / "best_model.joblib")


@st.cache_data
def load_supporting_data():
    history = pd.read_csv(ROOT / "artifacts" / "inference_history.csv", parse_dates=["game_date"])
    rankings = pd.read_csv(ROOT / "artifacts" / "rankings_clean.csv", parse_dates=["standing_date"])
    teams = pd.read_csv(ROOT / "data" / "teams.csv")
    teams["display_name"] = teams["CITY"].fillna("") + " " + teams["NICKNAME"].fillna("")
    teams["display_name"] = teams["display_name"].str.strip()
    return history, rankings, teams.sort_values("display_name")


st.title("🏀 Predicción de partidos NBA")
st.caption("Algoritmos genéticos + optimización por enjambre de partículas, con probabilidades calibradas")

required = [
    ROOT / "artifacts" / "best_model.joblib",
    ROOT / "artifacts" / "inference_history.csv",
    ROOT / "artifacts" / "rankings_clean.csv",
]
missing = [path.name for path in required if not path.exists()]
if missing:
    st.error("Faltan artefactos de entrenamiento: " + ", ".join(missing))
    st.code("python run_pipeline.py --profile standard", language="bash")
    st.stop()

model = load_model()
history, rankings, teams = load_supporting_data()
name_to_id = dict(zip(teams["display_name"], teams["TEAM_ID"].astype(int)))
team_names = list(name_to_id)
data_cutoff = history["game_date"].max().date()

left, right = st.columns(2)
with left:
    default_home = team_names.index("Los Angeles Lakers") if "Los Angeles Lakers" in team_names else 0
    home_name = st.selectbox("Equipo local", team_names, index=default_home)
with right:
    default_away = team_names.index("Boston Celtics") if "Boston Celtics" in team_names else min(1, len(team_names) - 1)
    away_name = st.selectbox("Equipo visitante", team_names, index=default_away)

date_column, season_column = st.columns(2)
with date_column:
    prediction_date = st.date_input(
        "Fecha del encuentro",
        value=data_cutoff + timedelta(days=1),
        min_value=history["game_date"].min().date(),
    )
with season_column:
    available_seasons = sorted(history["season"].unique().astype(int), reverse=True)
    season = st.selectbox("Temporada de referencia", available_seasons, index=0)

if prediction_date > data_cutoff:
    st.warning(
        f"Los datos terminan el {data_cutoff:%d/%m/%Y}. La predicción usa el último estado histórico disponible."
    )

if st.button("Calcular predicción", type="primary", width="stretch"):
    if home_name == away_name:
        st.error("Selecciona dos equipos diferentes.")
    else:
        feature_row = build_match_feature_row(
            history=history,
            rankings=rankings,
            home_team_id=name_to_id[home_name],
            away_team_id=name_to_id[away_name],
            prediction_date=pd.Timestamp(prediction_date),
            season=int(season),
        )
        probability_home = float(model.predict_proba(feature_row)[0])
        probability_away = 1.0 - probability_home
        winner = home_name if probability_home >= 0.5 else away_name

        first, second = st.columns(2)
        first.metric(f"Probabilidad {home_name}", f"{probability_home:.1%}")
        second.metric(f"Probabilidad {away_name}", f"{probability_away:.1%}")
        st.success(f"Predicción: **{winner}**")
        st.progress(probability_home, text=f"{home_name}: {probability_home:.1%}")

        with st.expander("Detalles técnicos"):
            st.write(
                {
                    "modelo": model.name,
                    "calibración": model.calibrator.method,
                    "variables seleccionadas": len(model.feature_names),
                    "fecha máxima de entrenamiento": model.metadata.get("data_date_max"),
                }
            )
            shown = feature_row[model.feature_names].T.rename(columns={0: "valor_local_menos_visitante"})
            st.dataframe(shown, width="stretch")

st.divider()
st.caption(
    "Uso académico. El modelo no incorpora lesiones o alineaciones en tiempo real y no constituye asesoría de apuestas."
)
