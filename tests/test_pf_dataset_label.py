"""The ProblemFormulator's generation task names the run's dataset (C1).

With no locked spec, the user message always asked for "a prediction
research question using the HSLS:09 dataset", whatever dataset the run
had loaded (ELS:2002, ASSISTments). HSLS:09 runs keep that exact wording.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.agents.problem_formulator import ProblemFormulator, _dataset_label


def _message(registry: dict, dataset_name: str) -> str:
    pf = MagicMock(spec=ProblemFormulator)
    pf.ctx = MagicMock()
    pf.ctx.task_type = "prediction"
    pf.ctx.dataset_name = dataset_name
    return ProblemFormulator._build_user_message(
        pf, registry=registry, task_template={}, s2_context={"papers": []},
        user_prompt=None, revision_instructions=None,
    )


def test_hsls_wording_is_unchanged() -> None:
    msg = _message({"name": "hsls09_public"}, "hsls09_public")
    assert "Design a prediction research question using the HSLS:09 dataset. " in msg


@pytest.mark.parametrize(
    "name, label",
    [("els_2002", "ELS:2002"), ("assistments_0910", "ASSISTments 2009-10")],
)
def test_other_datasets_are_named(name: str, label: str) -> None:
    msg = _message({"name": name}, name)
    assert f"using the {label} dataset." in msg
    assert "HSLS:09 dataset" not in msg


def test_label_falls_back_to_the_registry_then_the_name() -> None:
    assert _dataset_label({"name": "new_ds", "full_name": "A New Study"}, None) == "A New Study"
    assert _dataset_label({}, "new_ds") == "new_ds"
    assert _dataset_label(None, None) == "HSLS:09"
    assert _dataset_label({}, MagicMock()) == "HSLS:09"  # a non-string name is ignored
