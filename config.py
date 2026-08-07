from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


HISTORICAL_FEATURES = [
    "win_rate_5",
    "win_rate_10",
    "pts_for_avg_5",
    "pts_against_avg_5",
    "point_margin_avg_5",
    "point_margin_avg_10",
    "reb_avg_5",
    "ast_avg_5",
    "fg_pct_avg_5",
    "fg3_pct_avg_5",
    "ft_pct_avg_5",
    "stl_avg_5",
    "blk_avg_5",
    "turnovers_avg_5",
    "season_win_pct",
    "venue_win_rate_10",
    "rest_days",
    "h2h_win_rate",
    "standings_win_pct",
    "games_played",
]

DIFFERENCE_FEATURES = [f"{name}_diff" for name in HISTORICAL_FEATURES]
MODEL_FEATURES = ["is_home", *DIFFERENCE_FEATURES]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "ProjectPaths":
        root = (start or Path(__file__).resolve().parents[2]).resolve()
        return cls(root=root)

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def tables(self) -> Path:
        return self.reports / "tables"

    def ensure_output_dirs(self) -> None:
        for path in (
            self.processed,
            self.artifacts,
            self.reports,
            self.figures,
            self.tables,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SearchProfile:
    ga_population: int
    ga_generations: int
    pso_particles: int
    pso_iterations: int


SEARCH_PROFILES = {
    "quick": SearchProfile(ga_population=10, ga_generations=5, pso_particles=10, pso_iterations=5),
    "standard": SearchProfile(ga_population=18, ga_generations=12, pso_particles=18, pso_iterations=12),
    "thorough": SearchProfile(ga_population=30, ga_generations=25, pso_particles=30, pso_iterations=25),
}

RANDOM_SEED = 42
