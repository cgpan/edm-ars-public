"""Every declared protected attribute must actually be reported.

pcc_05 checked only that subgroup_performance was non-empty, so a run
reporting two of three declared attributes passed. That happened live:
the spec declared [X1SEX, X1RACE, X1SES], test_protected.csv carried only
X1RACE and X1SES, and the gender analysis was skipped with a warning
nobody had to act on -- while the paper could still claim subgroup
analysis was conducted for protected attributes.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.pre_critic_checks import PreCriticResult, _check_subgroup_performance_present


def _run(declared, reported):
    ctx = SimpleNamespace(
        research_spec={"subgroup_analyses": declared},
        results_object={"subgroup_performance": reported},
    )
    result = PreCriticResult(failures=[])
    _check_subgroup_performance_present(ctx, result)
    return result


def test_the_live_case_is_caught():
    r = _run(["X1SEX", "X1RACE", "X1SES"], {"X1RACE": {}, "X1SES": {}})
    assert r.failures
    assert "X1SEX" in r.failures[0].message
    assert r.failures[0].severity == "major"


def test_the_fix_is_aimed_at_who_writes_the_file():
    r = _run(["X1SEX", "X1RACE"], {"X1RACE": {}})
    assert r.failures[0].target_agent == "DataEngineer"


def test_a_complete_report_passes():
    assert not _run(["X1SEX", "X1RACE"], {"X1SEX": {}, "X1RACE": {}}).failures


def test_extra_reported_attributes_are_fine():
    assert not _run(["X1SEX"], {"X1SEX": {}, "X1RACE": {}}).failures


def test_an_empty_report_still_fails_the_old_way():
    r = _run(["X1SEX"], {})
    assert r.failures and "did not run" in r.failures[0].message
    assert r.failures[0].target_agent == "Analyst"


def test_no_declared_attributes_is_not_a_failure():
    assert not _run([], {"X1RACE": {}}).failures


def test_a_missing_spec_does_not_raise():
    ctx = SimpleNamespace(results_object={"subgroup_performance": {"X1RACE": {}}})
    result = PreCriticResult(failures=[])
    _check_subgroup_performance_present(ctx, result)
    assert not result.failures
