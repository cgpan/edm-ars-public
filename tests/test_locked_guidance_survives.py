"""Free-text guidance in a locked spec must reach the agents that act on it.

The locked spec goes to the ProblemFormulator, which returns a
research_spec of its own, and that REPLACES the locked one. Anything the
PF does not echo back is gone before the DataEngineer runs.

A real study spec showed the cost. It carried an ENCODING CONTRACT --
"do NOT pass them to get_dummies or one-hot encoding under any
circumstance" -- naming five continuous predictors. The saved prompts
show it reached the ProblemFormulator (6 mentions of
continuous_predictors_do_not_encode) and nothing after it: DataEngineer,
Analyst and Critic all zero. X1SES became 5,514 dummy columns and
X1TXMTSCOR 9,350 -- 15,008 features for 12,918 students -- and SHAP for
X1TXMTSCOR summed to 0.0 across its dummies.

The unrecognised-key warning tells authors to put such guidance in
`additional_constraints`. That advice only became true when this carry
existed: before it, the recommended field died at exactly the same
boundary as the keys it was recommending a retreat from.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.orchestrator import Orchestrator

CONTRACT = "Keep X1SES and X1TXMTSCOR numeric. Never one-hot encode them."


def _orchestrator(locked: dict | None) -> Orchestrator:
    orch = object.__new__(Orchestrator)
    orch.ctx = SimpleNamespace(locked_research_spec=locked, log=[])
    orch._log = lambda *a, **k: None  # type: ignore[method-assign]
    return orch


def test_guidance_is_carried_into_the_emitted_spec() -> None:
    orch = _orchestrator({"additional_constraints": CONTRACT})
    out = orch._carry_locked_guidance({"outcome_variable": "X4ATPRLVLA"})
    assert out["additional_constraints"] == CONTRACT


def test_the_formulators_own_wording_is_not_overwritten() -> None:
    """If the PF echoed it back, that version stands."""
    orch = _orchestrator({"additional_constraints": CONTRACT})
    out = orch._carry_locked_guidance(
        {"additional_constraints": "PF refined wording"}
    )
    assert out["additional_constraints"] == "PF refined wording"


def test_an_empty_echo_is_treated_as_missing() -> None:
    """An empty string is not guidance; the locked text should win."""
    orch = _orchestrator({"additional_constraints": CONTRACT})
    assert orch._carry_locked_guidance(
        {"additional_constraints": ""}
    )["additional_constraints"] == CONTRACT


def test_no_locked_spec_changes_nothing() -> None:
    orch = _orchestrator(None)
    spec = {"outcome_variable": "X4ATPRLVLA"}
    assert orch._carry_locked_guidance(spec) == spec


def test_a_locked_spec_without_guidance_adds_no_key() -> None:
    """Never invent an empty field for the agents to read."""
    orch = _orchestrator({"task_type": "prediction"})
    out = orch._carry_locked_guidance({"outcome_variable": "X"})
    assert "additional_constraints" not in out


@pytest.mark.parametrize("bad", [None, "not a dict", 42, []])
def test_a_non_dict_spec_passes_through(bad: object) -> None:
    """A failed PF must not become an AttributeError here."""
    orch = _orchestrator({"additional_constraints": CONTRACT})
    assert orch._carry_locked_guidance(bad) is bad  # type: ignore[arg-type]


def test_a_non_dict_locked_spec_is_ignored() -> None:
    orch = _orchestrator("not a dict")  # type: ignore[arg-type]
    spec = {"outcome_variable": "X"}
    assert orch._carry_locked_guidance(spec) == spec


def test_the_carry_is_wired_into_both_assignment_sites() -> None:
    """It is useless unless the FORMULATING and revision paths both use it.

    The revision path re-runs the ProblemFormulator and reassigns
    ctx.research_spec, so a carry applied only on the first pass would be
    undone by the first REVISE cycle -- which is exactly when a spec
    violation is being corrected.
    """
    import inspect

    source = inspect.getsource(Orchestrator)
    assignments = source.count("self.ctx.research_spec = ")
    carried = source.count("_carry_locked_guidance(")
    # One definition plus one call per PF assignment site.
    assert carried - 1 >= 2, (
        f"{assignments} research_spec assignments but only {carried - 1} carries"
    )
