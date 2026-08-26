"""A locked spec must say which of its keys the pipeline will ignore.

A8. `--research-spec` accepts a JSON spec, but the ProblemFormulator
rewrites it and preserves only:

    outcome_variable, outcome_type, predictor_set, research_question,
    subgroup_analyses, target_population

Everything else was discarded silently — including `known_concerns_to_flag`
and `required_reporting`. Grepping the rendered DataEngineer prompt for
those custom keys returned zero occurrences.

Silence is the trap. A spec that is read, structurally validated and then
quietly stripped is indistinguishable, from the outside, from one that
was honoured — so a user writes careful constraints and gets a run that
behaves as though they had never written them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.main import (
    SPEC_KEYS_HONOURED,
    SPEC_KEYS_METADATA,
    _warn_on_unrecognised_spec_keys,
)


def _spec(**extra) -> dict:
    base = {
        "task_type": "prediction",
        "dataset": "els_2002",
        "research_question": "Does X predict Y?",
        "outcome_variable": "F2EVRATT",
        "outcome_type": "binary",
    }
    base.update(extra)
    return base


def test_an_unrecognised_key_warns(capsys: pytest.CaptureFixture) -> None:
    _warn_on_unrecognised_spec_keys(_spec(my_custom_rule="do the thing"), "s.json")
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "my_custom_rule" in err


def test_the_warning_points_at_the_supported_route(
    capsys: pytest.CaptureFixture,
) -> None:
    """A warning that does not say what to do instead just annoys."""
    _warn_on_unrecognised_spec_keys(_spec(whatever=1), "s.json")
    assert "additional_constraints" in capsys.readouterr().err


def test_documented_guidance_keys_are_reported_as_non_steering(
    capsys: pytest.CaptureFixture,
) -> None:
    """`required_reporting` and `known_concerns_to_flag` LOOK operative.

    They are the two keys most likely to be written in good faith and
    silently ignored, so they get an explicit note rather than silence.
    """
    _warn_on_unrecognised_spec_keys(
        _spec(required_reporting=["report AUC"], known_concerns_to_flag=["x"]),
        "s.json",
    )
    err = capsys.readouterr().err
    assert "NOTE:" in err
    assert "required_reporting" in err
    assert "known_concerns_to_flag" in err
    assert "do NOT steer the run" in err


def test_a_clean_spec_is_silent(capsys: pytest.CaptureFixture) -> None:
    """Noise on every run is how a warning gets ignored."""
    _warn_on_unrecognised_spec_keys(_spec(predictor_set=[], subgroup_analyses=[]), "s.json")
    assert capsys.readouterr().err == ""


def test_additional_constraints_is_honoured_not_warned(
    capsys: pytest.CaptureFixture,
) -> None:
    """The escape hatch must not warn about itself."""
    _warn_on_unrecognised_spec_keys(
        _spec(additional_constraints="Fit an ordinal model."), "s.json"
    )
    assert capsys.readouterr().err == ""
    assert "additional_constraints" in SPEC_KEYS_HONOURED


def test_the_two_key_groups_do_not_overlap() -> None:
    """A key in both groups would produce contradictory advice."""
    assert not (SPEC_KEYS_HONOURED & SPEC_KEYS_METADATA)


@pytest.mark.parametrize(
    "fixture",
    sorted((Path(__file__).resolve().parents[1] / "runs" / "fixtures").glob("*.json")),
    ids=lambda p: p.stem,
)
def test_shipped_fixtures_declare_no_unrecognised_keys(
    fixture: Path, capsys: pytest.CaptureFixture
) -> None:
    """The specs we ship should not themselves trip the warning.

    A NOTE about provenance keys is fine; a WARNING means we shipped a
    spec whose author expected something the pipeline never reads.
    """
    spec = json.loads(fixture.read_text(encoding="utf-8"))
    _warn_on_unrecognised_spec_keys(spec, str(fixture))
    assert "WARNING" not in capsys.readouterr().err
