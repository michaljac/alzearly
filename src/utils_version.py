"""Utilities for run/version metadata.

This module is intentionally dependency‑light and safe to import from
training, serving, or data‑generation scripts. It helps you:

- capture Git information (short SHA, dirty state)
- generate a monotonic model version (semver‑ish) and run_id
- create a timestamped artifacts/run directory and a stable `artifacts/latest` link
- write/read simple JSON/YAML metadata files (JSON by default)

All functions are defensive: if Git or the filesystem is unavailable,
reasonable fallbacks are used so your pipeline does not crash.

Usage (typical):

>>> from src.utils_version import (
...     start_run, write_metadata, get_version_string,
... )
>>> ctx = start_run(base_artifacts_dir="artifacts")
>>> write_metadata(ctx["run_dir"] / "run_meta.json", ctx)
>>> print(get_version_string(ctx))

"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ----------------------------
# Dataclasses & core helpers
# ----------------------------

@dataclass
class RunContext:
    """Holds lightweight run metadata."""
    timestamp: str  # UTC ISO8601, e.g. 2025-08-31T15:22:11Z
    run_id: str     # e.g. 20250831_152211
    git_sha: str    # short SHA or "unknown"
    git_dirty: bool
    version: str    # semver‑ish string, e.g. 0.1.3
    base_artifacts_dir: Path
    run_dir: Path


ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts_str(dt: Optional[datetime] = None) -> str:
    dt = dt or _utc_now()
    return dt.strftime("%Y%m%d_%H%M%S")


# ----------------------------
# Git info (safe fallbacks)
# ----------------------------

def get_git_short_sha() -> str:
    """Return short SHA if repo exists; otherwise 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def get_git_dirty() -> bool:
    """Return True if repo has uncommitted changes, else False.
    Returns False if git is not available.
    """
    try:
        subprocess.check_output(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL)
        subprocess.check_output(["git", "diff", "--cached", "--quiet"], stderr=subprocess.DEVNULL)
        return False
    except subprocess.CalledProcessError:
        return True
    except Exception:
        return False


# ----------------------------
# Versioning helpers
# ----------------------------

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def _read_version_file(path: Path) -> Optional[str]:
    try:
        if path.exists():
            txt = path.read_text().strip()
            if SEMVER_RE.match(txt):
                return txt
    except Exception:
        pass
    return None


def _bump_patch(ver: str) -> str:
    m = SEMVER_RE.match(ver)
    if not m:
        return "0.1.0"
    major, minor, patch = map(int, m.groups())
    return f"{major}.{minor}.{patch + 1}"


def compute_model_version(base_artifacts_dir: Path) -> str:
    """Compute a semver‑ish version.

    Priority:
    1) ENV var MODEL_VERSION if provided and valid (or any string—used as‑is)
    2) artifacts/VERSION file (bump patch)
    3) default '0.1.0'
    """
    env_ver = os.getenv("MODEL_VERSION")
    if env_ver:
        return env_ver.strip()

    version_file = base_artifacts_dir / "VERSION"
    prev = _read_version_file(version_file)
    if prev:
        return _bump_patch(prev)
    return "0.1.0"


def persist_version(base_artifacts_dir: Path, version: str) -> None:
    try:
        (base_artifacts_dir).mkdir(parents=True, exist_ok=True)
        (base_artifacts_dir / "VERSION").write_text(version + "\n")
    except Exception:
        # non‑fatal
        pass


# ----------------------------
# Run bootstrap & metadata
# ----------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_symlink(target: Path, link_name: Path) -> None:
    try:
        if link_name.is_symlink() or link_name.exists():
            try:
                if link_name.is_dir() and not link_name.is_symlink():
                    # replace directory with symlink
                    for child in link_name.iterdir():
                        if child.is_file():
                            child.unlink()
                link_name.unlink(missing_ok=True)
            except Exception:
                pass
        link_name.symlink_to(target, target_is_directory=True)
    except Exception:
        # On Windows or restricted FS, symlinks may fail—ignore.
        pass


def start_run(base_artifacts_dir: str | Path = "artifacts",
              run_prefix: str = "run") -> RunContext:
    """Create a fresh run directory and return a RunContext.

    Structure:
      artifacts/
        VERSION
        latest -> artifacts/run_YYYYMMDD_HHMMSS
        run_YYYYMMDD_HHMMSS/
          (your artifacts live here)
    """
    base = Path(base_artifacts_dir).resolve()
    ensure_dir(base)

    ts = _utc_now()
    run_id = _ts_str(ts)
    run_name = f"{run_prefix}_{run_id}"
    run_dir = ensure_dir(base / run_name)

    version = compute_model_version(base)
    persist_version(base, version)

    git_sha = get_git_short_sha()
    git_dirty = get_git_dirty()

    # update latest symlink if possible
    safe_symlink(run_dir, base / "latest")

    ctx = RunContext(
        timestamp=ts.strftime(ISO_FMT),
        run_id=run_id,
        git_sha=git_sha,
        git_dirty=git_dirty,
        version=version,
        base_artifacts_dir=base,
        run_dir=run_dir,
    )
    return ctx


def get_version_string(ctx: RunContext | Dict[str, Any] | None = None) -> str:
    """Return a human‑readable version string for logging/filenames."""
    if ctx is None:
        # best‑effort standalone
        return f"v{compute_model_version(Path('artifacts'))}"
    d = asdict(ctx) if isinstance(ctx, RunContext) else dict(ctx)
    dirty = "+dirty" if d.get("git_dirty") else ""
    return f"v{d.get('version','0.1.0')}({d.get('git_sha','unknown')}{dirty})_{d.get('run_id','')}"


def write_metadata(path: Path | str, data: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        # Non‑fatal: avoid breaking training on I/O glitches
        pass


def load_metadata(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ----------------------------
# Convenience aliases used in various scripts
# ----------------------------

# Backwards‑compat names some projects expect
get_version_info = start_run  # alias for clarity in older code
get_git_sha = get_git_short_sha
is_repo_dirty = get_git_dirty


__all__ = [
    "RunContext",
    "start_run",
    "get_version_info",
    "get_version_string",
    "get_git_short_sha",
    "get_git_sha",
    "get_git_dirty",
    "is_repo_dirty",
    "compute_model_version",
    "persist_version",
    "write_metadata",
    "load_metadata",
]
