import yaml


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


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)

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
