import os
from pathlib import Path

import yaml


#: The repository root: the directory that contains ``src/``. Relative
#: paths in config.yaml name files in the repository, so they are resolved
#: here when they do not exist from the current directory (C5). Before,
#: every one of them was read relative to wherever the user happened to
#: start Python: from another directory the default ``--config`` raised
#: FileNotFoundError, and with an explicit --config the run went ahead
#: with no skills, one-line agent prompts and a raw-data path under the
#: wrong folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


_REQUIRED_TOP_KEYS = {"models", "pipeline", "semantic_scholar", "paths"}
_REQUIRED_MODEL_KEYS = {"problem_formulator", "data_engineer", "analyst", "critic", "writer"}

_SANDBOX_DEFAULTS: dict = {
    "enabled": False,
    "image": "edm-ars-sandbox:latest",
    "memory_limit": "4g",
    "cpu_count": 2,
    "network_disabled": True,
    "auto_build": True,
}


def _validate_sandbox_config(config: dict) -> None:
    """Ensure config["sandbox"] exists and has all required keys with defaults."""
    config.setdefault("sandbox", dict(_SANDBOX_DEFAULTS))
    for key, val in _SANDBOX_DEFAULTS.items():
        config["sandbox"].setdefault(key, val)


def default_lsar_home() -> str:
    """Where LSAR (the separate review-gate repository) lives by default.

    A sibling checkout of this repository is the conventional layout, and
    ``git clone https://github.com/cgpan/LSAR-public`` names that
    directory ``LSAR-public``, not ``LSAR``. The first of the two that
    exists wins; when neither does, the ``LSAR`` sibling is returned so a
    gate that cannot find it names a concrete place. The old default was
    the literal ``../LSAR``, which also depended on the current directory.
    Set LSAR_HOME to point anywhere else.
    """
    parent = PROJECT_ROOT.parent
    for name in ("LSAR", "LSAR-public"):
        candidate = parent / name
        if candidate.is_dir():
            return str(candidate)
    return str(parent / "LSAR")


#: Kept for callers that import the name; computed once at import.
DEFAULT_LSAR_HOME = default_lsar_home()


def resolve_repo_path(path: str) -> str:
    """Return an absolute path for a path written in config or on the CLI.

    An absolute path is returned unchanged. A relative path that exists
    from the current directory keeps meaning that, so running from the
    repository root resolves every path exactly as before. Any other
    relative path is taken to name something in the repository and is
    resolved under PROJECT_ROOT, whether or not it exists yet (``output/``
    and ``data/raw/`` are created later). A trailing separator survives,
    because some callers build paths by string concatenation.
    """
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    trailing = expanded.endswith(("/", "\\"))
    if os.path.exists(expanded):
        resolved = os.path.abspath(expanded)
    else:
        resolved = os.path.normpath(str(PROJECT_ROOT / expanded))
    if trailing and not resolved.endswith(os.sep):
        resolved += os.sep
    return resolved


def resolve_config_path(path: str | None = None) -> str:
    """Resolve ``--config``: the repository's config.yaml by default."""
    if not path:
        return str(PROJECT_ROOT / "config.yaml")
    return resolve_repo_path(path)


def _anchor_config_paths(config: dict) -> None:
    """Make every relative ``paths.*`` entry and the findings memory path
    absolute (see :func:`resolve_repo_path`). Other relative strings in the
    config are left as written."""
    paths = config.get("paths")
    if isinstance(paths, dict):
        for key, value in list(paths.items()):
            if isinstance(value, str) and value:
                paths[key] = resolve_repo_path(value)
    memory = config.get("findings_memory")
    if isinstance(memory, dict):
        value = memory.get("path")
        if isinstance(value, str) and value:
            memory["path"] = resolve_repo_path(value)


def _expand_env(value):
    """Expand ``${VAR}`` in every string in a nested config structure.

    Paths to the companion LSAR repository differ per machine, so the
    shipped config refers to ``${LSAR_HOME}`` rather than hard-coding one
    checkout's layout. Without this pass those would stay literal and
    fail to resolve.
    """
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | None = "config.yaml") -> dict:
    """Load, validate and complete a pipeline config.

    ``path`` is resolved with :func:`resolve_config_path`, so the default
    finds the repository's config.yaml from any working directory.
    """
    path = resolve_config_path(path)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(
            f"config file {path} is empty or not a YAML mapping"
        )

    if not os.environ.get("LSAR_HOME"):
        os.environ["LSAR_HOME"] = default_lsar_home()
    config = _expand_env(config)

    missing_top = _REQUIRED_TOP_KEYS - set(config.keys())
    if missing_top:
        raise ValueError(f"config.yaml missing required top-level keys: {sorted(missing_top)}")

    missing_models = _REQUIRED_MODEL_KEYS - set(config["models"].keys())
    if missing_models:
        raise ValueError(f"config.yaml missing required model keys: {sorted(missing_models)}")

    _validate_sandbox_config(config)

    # Ensure task_type has a default for backward compatibility
    config["pipeline"].setdefault("task_type", "prediction")

    # Findings memory defaults (opt-in feature)
    config.setdefault("findings_memory", {})
    config["findings_memory"].setdefault("enabled", False)
    config["findings_memory"].setdefault("path", "findings_memory/memory.yaml")
    config["findings_memory"].setdefault("n_candidate_specs", 1)
    _anchor_config_paths(config)

    # LLM provider defaults
    config.setdefault("llm_provider", "anthropic")
    config.setdefault("minimax", {})
    config["minimax"].setdefault("base_url", "https://api.minimax.io/anthropic")
    _default_minimax_models = {
        "problem_formulator": "MiniMax-M2.5",
        "data_engineer": "MiniMax-M2.5",
        "analyst": "MiniMax-M2.5",
        "critic": "MiniMax-M2.5",
        "writer": "MiniMax-M2.5",
    }
    config["minimax"].setdefault("models", _default_minimax_models)

    return config
