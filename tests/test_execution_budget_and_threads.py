"""The analysis stage must be able to finish on the host it is running on.

Both defects here were being held off by an external launcher script kept
outside the repo, so a fresh clone hit them immediately.

Thread oversubscription. The generated analysis code fits with n_jobs=-1,
so scikit-learn spawns one worker per core; with the inner BLAS pools
uncapped each worker then spawns its own core-count of threads. On a
28-core host, 32 processes contended for 28 cores and the battery ran
SLOWER than a capped run. The stage hit its timeout and burned all three
retries -- regenerating code that was never the problem.

Execution budget. 600s fits one pass of the battery. It does not fit the
journal-track case where the minority class is below the SMOTE threshold
and ablation_enabled trains the whole battery a second time.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from src.agents.analyst import Analyst
from src.sandbox import BLAS_THREAD_VARS, blas_thread_env


# --- thread capping ----------------------------------------------------

def test_every_blas_pool_is_capped() -> None:
    env = blas_thread_env({})
    for var in BLAS_THREAD_VARS:
        assert var in env, f"{var} left uncapped"
        assert int(env[var]) >= 1


def test_the_cap_is_small_enough_to_stop_nesting() -> None:
    """n_jobs=-1 already parallelises across models; the inner pool must not."""
    env = blas_thread_env({})
    assert int(env["OMP_NUM_THREADS"]) <= 4


def test_an_operators_own_setting_is_respected() -> None:
    """Someone who has tuned this for their host keeps their value."""
    env = blas_thread_env({"OMP_NUM_THREADS": "16"})
    assert env["OMP_NUM_THREADS"] == "16"
    assert "MKL_NUM_THREADS" in env  # the others are still filled in


def test_the_env_var_override_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDMARS_INNER_THREADS", "3")
    assert blas_thread_env({})["OMP_NUM_THREADS"] == "3"


def test_the_container_does_not_inherit_the_host_environment() -> None:
    """An explicit base must not be topped up with os.environ."""
    env = blas_thread_env({"OUTPUT_DIR": "/workspace"})
    assert env["OUTPUT_DIR"] == "/workspace"
    assert "PATH" not in env


def test_the_default_base_is_the_real_environment() -> None:
    """The subprocess still needs PATH to find python."""
    env = blas_thread_env()
    assert len(env) > len(BLAS_THREAD_VARS)


# --- execution budget --------------------------------------------------

def _analyst(config: dict) -> Analyst:
    agent = object.__new__(Analyst)
    agent.config = config
    return agent


def test_a_single_pass_keeps_the_documented_budget() -> None:
    agent = _analyst({"class_imbalance": {"ablation_enabled": False}})
    assert agent._exec_timeout_s() == 600


def test_training_the_battery_twice_doubles_the_budget() -> None:
    """The observed failure: SMOTE plus ablation, three timeouts in a row."""
    agent = _analyst({"class_imbalance": {"ablation_enabled": True}})
    assert agent._exec_timeout_s() == 1200


def test_config_overrides_the_derived_value() -> None:
    agent = _analyst(
        {
            "pipeline": {"analysis_exec_timeout_s": 2400},
            "class_imbalance": {"ablation_enabled": True},
        }
    )
    assert agent._exec_timeout_s() == 2400


def test_a_null_override_falls_back_to_deriving_it() -> None:
    """config.yaml ships the key as null; that must not mean zero."""
    agent = _analyst(
        {
            "pipeline": {"analysis_exec_timeout_s": None},
            "class_imbalance": {"ablation_enabled": True},
        }
    )
    assert agent._exec_timeout_s() == 1200


@pytest.mark.parametrize("bad", [0, -1, "2400", False])
def test_a_nonsense_override_is_ignored_rather_than_obeyed(bad: object) -> None:
    """A zero or negative timeout would kill every run instantly."""
    agent = _analyst(
        {
            "pipeline": {"analysis_exec_timeout_s": bad},
            "class_imbalance": {"ablation_enabled": False},
        }
    )
    assert agent._exec_timeout_s() == 600


def test_a_missing_config_still_yields_a_usable_budget() -> None:
    agent = object.__new__(Analyst)
    assert agent._exec_timeout_s() == 600


def test_the_class_attribute_is_still_patchable() -> None:
    """Existing callers and tests patch Analyst.EXEC_TIMEOUT_S directly."""
    agent = _analyst({"class_imbalance": {"ablation_enabled": False}})
    original = Analyst.EXEC_TIMEOUT_S
    try:
        Analyst.EXEC_TIMEOUT_S = 900
        assert agent._exec_timeout_s() == 900
    finally:
        Analyst.EXEC_TIMEOUT_S = original


def test_the_shipped_config_derives_rather_than_hardcodes() -> None:
    """Guards against someone pinning a number back into config.yaml."""
    import yaml

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    assert config["pipeline"]["analysis_exec_timeout_s"] is None
