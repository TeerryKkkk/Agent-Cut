from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    launch_root: Path
    workspace_root: Path
    layout_mode: str = "standalone"

    @property
    def configs_dir(self) -> Path:
        return self.workspace_root / "configs"

    @property
    def data_dir(self) -> Path:
        return self.workspace_root / "data"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def openml_cache_dir(self) -> Path:
        return self.data_dir / "openml_cache"

    @property
    def runs_dir(self) -> Path:
        return self.workspace_root / "runs"

    @property
    def results_dir(self) -> Path:
        return self.workspace_root / "results"

    @property
    def reports_dir(self) -> Path:
        return self.workspace_root / "reports"

    @property
    def skills_dir(self) -> Path:
        return self.workspace_root / "skills"

    @property
    def compiled_skills_dir(self) -> Path:
        return self.skills_dir / "compiled"

    def relative_to_workspace(self, path: Path) -> str:
        return path.relative_to(self.workspace_root).as_posix()


def _candidate_roots(start: Path) -> list[Path]:
    return [start, *start.parents]


def detect_project_paths(start: Path | None = None) -> ProjectPaths:
    launch_root = (start or Path.cwd()).resolve()
    if launch_root.is_file():
        launch_root = launch_root.parent
    for candidate in _candidate_roots(launch_root):
        if (candidate / "configs" / "families.yaml").is_file() and (candidate / "cut").is_dir():
            return ProjectPaths(launch_root=launch_root, workspace_root=candidate)
    raise RuntimeError("Run from the Agent-Cut repository or pass its directory explicitly.")


def ensure_workspace_dirs(paths: ProjectPaths) -> None:
    required = [
        paths.configs_dir,
        paths.data_dir,
        paths.exports_dir,
        paths.openml_cache_dir,
        paths.runs_dir,
        paths.results_dir,
        paths.reports_dir,
        paths.skills_dir,
        paths.compiled_skills_dir,
    ]
    for directory in required:
        directory.mkdir(parents=True, exist_ok=True)
