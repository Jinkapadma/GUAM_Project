from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> str:
    """Return an absolute path inside the project root."""

    return str(PROJECT_ROOT.joinpath(*parts))


def resolve_project_path(path: str, *parts: str) -> str:
    """Resolve absolute-like project paths such as '/dataset/' from the repo root."""

    normalized = path.strip("/\\")

    return project_path(normalized, *parts)
