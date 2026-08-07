from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import HISTORICAL_FEATURES, MODEL_FEATURES


RAW_FILES = ["games.csv", "games_details.csv", "players.csv", "ranking.csv", "teams.csv"]
ROLLING_STATS_5 = [
    "pts_for",
    "pts_against",
    "point_margin",
    "reb",
    "ast",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
    "stl",
    "blk",
    "turnovers",
]


@dataclass
class DatasetBundle:
    modeling: pd.DataFrame
    history: pd.DataFrame
    rankings: pd.DataFrame
    quality: dict[str, Any]


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def audit_raw_files(data_dir: Path) -> dict[str, Any]:
    """Produce a reproducible quality audit for every provided CSV."""
    result: dict[str, Any] = {}
    date_columns = {"games.csv": "GAME_DATE_EST", "ranking.csv": "STANDINGSDATE"}
    numeric_columns = {
        "games.csv": [
            "GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "SEASON", "TEAM_ID_home", "PTS_home",
            "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home", "TEAM_ID_away",
            "PTS_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away",
            "HOME_TEAM_WINS",
        ],
        "games_details.csv": [
            "GAME_ID", "TEAM_ID", "PLAYER_ID", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "PTS",
            "PLUS_MINUS",
        ],
        "players.csv": ["TEAM_ID", "PLAYER_ID", "SEASON"],
        "ranking.csv": ["TEAM_ID", "LEAGUE_ID", "SEASON_ID", "G", "W", "L", "W_PCT", "RETURNTOPLAY"],
        "teams.csv": ["LEAGUE_ID", "TEAM_ID", "MIN_YEAR", "MAX_YEAR", "YEARFOUNDED", "ARENACAPACITY"],
    }

    for filename in RAW_FILES:
        path = data_dir / filename
        frame = pd.read_csv(path, low_memory=False)
        nulls = frame.isna().sum()
        entry: dict[str, Any] = {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "duplicate_rows": int(frame.duplicated().sum()),
            "null_cells": int(nulls.sum()),
            "nulls_by_column": {key: int(value) for key, value in nulls[nulls.gt(0)].items()},
        }
        malformed_by_column: dict[str, int] = {}
        for column in numeric_columns[filename]:
            malformed = int((frame[column].notna() & pd.to_numeric(frame[column], errors="coerce").isna()).sum())
            if malformed:
                malformed_by_column[column] = malformed
        entry["malformed_numeric_cells"] = int(sum(malformed_by_column.values()))
        entry["malformed_by_column"] = malformed_by_column
        if filename in date_columns:
            column = date_columns[filename]
            dates = pd.to_datetime(frame[column], errors="coerce")
            entry["date_min"] = dates.min().date().isoformat()
            entry["date_max"] = dates.max().date().isoformat()
            entry["invalid_dates"] = int(dates.isna().sum() - frame[column].isna().sum())
        if filename == "games.csv":
            entry["duplicate_game_ids"] = int(frame["GAME_ID"].duplicated().sum())
            entry["incomplete_games"] = int(
                (~frame[["PTS_home", "PTS_away"]].notna().all(axis=1)).sum()
            )
        result[filename] = entry
    return result


def write_quality_json(quality: dict[str, Any], output: Path) -> None:
    output.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, default=_json_value),
        encoding="utf-8",
    )


def load_clean_games(data_dir: Path) -> pd.DataFrame:
    games = pd.read_csv(data_dir / "games.csv", low_memory=False)
    games["GAME_DATE_EST"] = pd.to_datetime(games["GAME_DATE_EST"], errors="coerce")
    games = games.dropna(subset=["GAME_DATE_EST", "PTS_home", "PTS_away"]).copy()
    games = games.sort_values(["GAME_DATE_EST", "GAME_ID"])
    games = games.drop_duplicates(subset="GAME_ID", keep="last")
    games["HOME_TEAM_WINS"] = games["HOME_TEAM_WINS"].astype(int)
    return games.reset_index(drop=True)


def load_team_detail_totals(data_dir: Path) -> pd.DataFrame:
    columns = ["GAME_ID", "TEAM_ID", "STL", "BLK", "TO", "PF"]
    details = pd.read_csv(data_dir / "games_details.csv", usecols=columns, low_memory=False)
    for column in ["STL", "BLK", "TO", "PF"]:
        details[column] = pd.to_numeric(details[column], errors="coerce")
    totals = (
        details.groupby(["GAME_ID", "TEAM_ID"], as_index=False)[["STL", "BLK", "TO", "PF"]]
        .sum(min_count=1)
        .rename(
            columns={
                "GAME_ID": "game_id",
                "TEAM_ID": "team_id",
                "STL": "stl",
                "BLK": "blk",
                "TO": "turnovers",
                "PF": "personal_fouls",
            }
        )
    )
    return totals


def load_clean_rankings(data_dir: Path) -> pd.DataFrame:
    rankings = pd.read_csv(
        data_dir / "ranking.csv",
        usecols=["TEAM_ID", "STANDINGSDATE", "W_PCT"],
        low_memory=False,
    ).rename(
        columns={"TEAM_ID": "team_id", "STANDINGSDATE": "standing_date", "W_PCT": "standings_win_pct"}
    )
    rankings["standing_date"] = pd.to_datetime(rankings["standing_date"], errors="coerce")
    rankings["standings_win_pct"] = pd.to_numeric(rankings["standings_win_pct"], errors="coerce")
    rankings = rankings.dropna(subset=["team_id", "standing_date"])
    rankings = rankings.sort_values(["standing_date", "team_id"]).drop_duplicates(
        ["team_id", "standing_date"], keep="last"
    )
    rankings["team_id"] = rankings["team_id"].astype(np.int64)
    return rankings.reset_index(drop=True)


def build_long_history(games: pd.DataFrame, detail_totals: pd.DataFrame) -> pd.DataFrame:
    common = {
        "GAME_ID": "game_id",
        "GAME_DATE_EST": "game_date",
        "SEASON": "season",
    }
    home = games.rename(
        columns={
            **common,
            "HOME_TEAM_ID": "team_id",
            "VISITOR_TEAM_ID": "opponent_id",
            "PTS_home": "pts_for",
            "PTS_away": "pts_against",
            "FG_PCT_home": "fg_pct",
            "FT_PCT_home": "ft_pct",
            "FG3_PCT_home": "fg3_pct",
            "AST_home": "ast",
            "REB_home": "reb",
            "HOME_TEAM_WINS": "won",
        }
    )
    home["is_home"] = 1

    away = games.rename(
        columns={
            **common,
            "VISITOR_TEAM_ID": "team_id",
            "HOME_TEAM_ID": "opponent_id",
            "PTS_away": "pts_for",
            "PTS_home": "pts_against",
            "FG_PCT_away": "fg_pct",
            "FT_PCT_away": "ft_pct",
            "FG3_PCT_away": "fg3_pct",
            "AST_away": "ast",
            "REB_away": "reb",
        }
    )
    away["won"] = 1 - away["HOME_TEAM_WINS"].astype(int)
    away["is_home"] = 0

    columns = [
        "game_id",
        "game_date",
        "season",
        "team_id",
        "opponent_id",
        "is_home",
        "won",
        "pts_for",
        "pts_against",
        "fg_pct",
        "ft_pct",
        "fg3_pct",
        "ast",
        "reb",
    ]
    history = pd.concat([home[columns], away[columns]], ignore_index=True)
    history["point_margin"] = history["pts_for"] - history["pts_against"]
    history = history.merge(
        detail_totals,
        how="left",
        on=["game_id", "team_id"],
        validate="one_to_one",
    )
    history = history.sort_values(["game_date", "game_id", "is_home"], ascending=[True, True, False])
    integer_columns = ["game_id", "season", "team_id", "opponent_id", "is_home", "won"]
    for column in integer_columns:
        history[column] = history[column].astype(np.int64)
    return history.reset_index(drop=True)


def _rolling_prior(frame: pd.DataFrame, column: str, group_columns: list[str], window: int) -> pd.Series:
    return frame.groupby(group_columns, sort=False)[column].transform(
        lambda series: series.shift(1).rolling(window=window, min_periods=1).mean()
    )


def add_historical_features(history: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    featured = history.sort_values(["team_id", "game_date", "game_id"]).copy()
    featured["win_rate_5"] = _rolling_prior(featured, "won", ["team_id"], 5)
    featured["win_rate_10"] = _rolling_prior(featured, "won", ["team_id"], 10)
    for column in ROLLING_STATS_5:
        featured[f"{column}_avg_5"] = _rolling_prior(featured, column, ["team_id"], 5)
    featured["point_margin_avg_10"] = _rolling_prior(featured, "point_margin", ["team_id"], 10)

    team_group = featured.groupby("team_id", sort=False)
    featured["games_played"] = team_group.cumcount().astype(float)
    season_games = featured.groupby(["team_id", "season"], sort=False).cumcount()
    season_wins = featured.groupby(["team_id", "season"], sort=False)["won"].cumsum() - featured["won"]
    featured["season_win_pct"] = season_wins.div(season_games.replace(0, np.nan))
    featured["venue_win_rate_10"] = _rolling_prior(featured, "won", ["team_id", "is_home"], 10)
    featured["rest_days"] = team_group["game_date"].diff().dt.total_seconds().div(86400).clip(0, 14)
    featured["h2h_win_rate"] = featured.groupby(["team_id", "opponent_id"], sort=False)["won"].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )

    left = featured.sort_values(["game_date", "team_id"]).copy()
    right = rankings.sort_values(["standing_date", "team_id"]).copy()
    featured = pd.merge_asof(
        left,
        right,
        left_on="game_date",
        right_on="standing_date",
        by="team_id",
        direction="backward",
        allow_exact_matches=False,
    ).drop(columns="standing_date")
    return featured.sort_values(["game_date", "game_id", "is_home"], ascending=[True, True, False]).reset_index(
        drop=True
    )


def build_pairwise_modeling_table(featured: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["game_id", "game_date", "season", "team_id", "opponent_id", "is_home", "won"]
    own = featured[identifiers + HISTORICAL_FEATURES].copy()
    opponent = own[["game_id", "team_id", "opponent_id", *HISTORICAL_FEATURES]].copy()
    opponent = opponent.rename(columns={"team_id": "opponent_id", "opponent_id": "team_id"})
    opponent = opponent.rename(columns={name: f"opponent_{name}" for name in HISTORICAL_FEATURES})
    paired = own.merge(
        opponent,
        on=["game_id", "team_id", "opponent_id"],
        how="left",
        validate="one_to_one",
    )
    for name in HISTORICAL_FEATURES:
        paired[f"{name}_diff"] = paired[name] - paired[f"opponent_{name}"]
    keep = identifiers + [name for name in MODEL_FEATURES if name not in identifiers]
    result = paired[keep].sort_values(["game_date", "game_id", "is_home"], ascending=[True, True, False])
    return result.reset_index(drop=True)


def validate_modeling_dataset(modeling: pd.DataFrame) -> dict[str, Any]:
    required = {"game_id", "game_date", "team_id", "opponent_id", "won", *MODEL_FEATURES}
    missing = required - set(modeling.columns)
    if missing:
        raise ValueError(f"Missing modeling columns: {sorted(missing)}")
    if len(MODEL_FEATURES) < 10:
        raise ValueError("The assignment requires at least ten model features")
    if modeling.duplicated(["game_id", "team_id"]).any():
        raise ValueError("Duplicate team-game modeling rows detected")
    counts = modeling.groupby("game_id").size()
    if not counts.eq(2).all():
        raise ValueError("Every game must have exactly two team perspectives")
    label_sums = modeling.groupby("game_id")["won"].sum()
    if not label_sums.eq(1).all():
        raise ValueError("Every game must contain exactly one winner")
    if len(modeling) < 50_000:
        raise ValueError(f"Only {len(modeling):,} rows; assignment requires at least 50,000")
    if any(name in MODEL_FEATURES for name in ["pts_for", "pts_against", "point_margin"]):
        raise ValueError("Current-game statistics leaked into the model feature list")
    return {
        "rows": int(len(modeling)),
        "unique_games": int(modeling["game_id"].nunique()),
        "features": len(MODEL_FEATURES),
        "positive_rate": float(modeling["won"].mean()),
        "date_min": modeling["game_date"].min().date().isoformat(),
        "date_max": modeling["game_date"].max().date().isoformat(),
    }


def build_dataset(data_dir: Path, include_audit: bool = True) -> DatasetBundle:
    quality = audit_raw_files(data_dir) if include_audit else {}
    games = load_clean_games(data_dir)
    totals = load_team_detail_totals(data_dir)
    rankings = load_clean_rankings(data_dir)
    history = build_long_history(games, totals)
    featured = add_historical_features(history, rankings)
    modeling = build_pairwise_modeling_table(featured)
    quality["modeling_dataset"] = validate_modeling_dataset(modeling)
    return DatasetBundle(modeling=modeling, history=history, rankings=rankings, quality=quality)


def _team_snapshot(
    history: pd.DataFrame,
    rankings: pd.DataFrame,
    team_id: int,
    opponent_id: int,
    is_home: int,
    prediction_date: pd.Timestamp,
    season: int,
) -> dict[str, float]:
    prior = history[(history["team_id"] == team_id) & (history["game_date"] < prediction_date)].sort_values(
        ["game_date", "game_id"]
    )
    snapshot: dict[str, float] = {}
    snapshot["win_rate_5"] = prior["won"].tail(5).mean()
    snapshot["win_rate_10"] = prior["won"].tail(10).mean()
    for column in ROLLING_STATS_5:
        snapshot[f"{column}_avg_5"] = prior[column].tail(5).mean()
    snapshot["point_margin_avg_10"] = prior["point_margin"].tail(10).mean()
    season_rows = prior[prior["season"] == season]
    snapshot["season_win_pct"] = season_rows["won"].mean()
    venue_rows = prior[prior["is_home"] == is_home]
    snapshot["venue_win_rate_10"] = venue_rows["won"].tail(10).mean()
    snapshot["rest_days"] = (
        min(max((prediction_date - prior["game_date"].max()).total_seconds() / 86400, 0), 14)
        if len(prior)
        else np.nan
    )
    snapshot["h2h_win_rate"] = prior.loc[prior["opponent_id"] == opponent_id, "won"].mean()
    eligible_rankings = rankings[
        (rankings["team_id"] == team_id) & (rankings["standing_date"] < prediction_date)
    ]
    snapshot["standings_win_pct"] = (
        eligible_rankings.iloc[-1]["standings_win_pct"] if len(eligible_rankings) else np.nan
    )
    snapshot["games_played"] = float(len(prior))
    return snapshot


def build_match_feature_row(
    history: pd.DataFrame,
    rankings: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    prediction_date: str | pd.Timestamp,
    season: int,
) -> pd.DataFrame:
    date = pd.Timestamp(prediction_date)
    home = _team_snapshot(history, rankings, home_team_id, away_team_id, 1, date, season)
    away = _team_snapshot(history, rankings, away_team_id, home_team_id, 0, date, season)
    row: dict[str, float] = {"is_home": 1.0}
    for name in HISTORICAL_FEATURES:
        row[f"{name}_diff"] = home[name] - away[name]
    return pd.DataFrame([row], columns=MODEL_FEATURES)
