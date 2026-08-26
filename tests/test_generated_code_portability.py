"""Generated code must not use APIs this host does not have.

An Analyst wrote `signal.SIGALRM` to impose its own timeout. That
attribute is Unix-only, so on Windows every model in the battery raised
AttributeError, `results.all_models` came back empty, the pre-Critic
guard aborted the run, and ~17 minutes and $0.11 were spent producing
nothing. No skill suggested it — the model reached for a Unix idiom
unprompted, which is why fixing this by editing a prompt would not be
enough.

The executor already imposes a hard timeout, so generated code never
needs its own: the construct was unnecessary as well as unportable.

The check lives at the executor because that is the single chokepoint all
generated code passes through, and because the failure is otherwise
silent — the Analyst catches each model's exception into results.errors
and the pipeline carries on to write a paper about an analysis that never
ran.
"""

from __future__ import annotations

import pytest

from src.sandbox import UNPORTABLE_CONSTRUCTS, SubprocessExecutor, check_portability


@pytest.mark.parametrize("construct", sorted(UNPORTABLE_CONSTRUCTS))
def test_every_registered_construct_is_detected(construct: str) -> None:
    findings = check_portability(f"import os\nx = {construct}\n")
    assert findings, f"{construct} is registered but not detected"
    assert construct in findings[0]


def test_the_construct_that_broke_a_real_run() -> None:
    code = (
        "import signal\n"
        "def handler(signum, frame):\n"
        "    raise TimeoutError()\n"
        "signal.signal(signal.SIGALRM, handler)\n"
        "signal.alarm(300)\n"
    )
    findings = check_portability(code)
    assert any("SIGALRM" in f for f in findings)
    assert any("executor already enforces" in f for f in findings)


@pytest.mark.parametrize(
    "code",
    [
        "import pandas as pd\nX = pd.read_csv('train_X.csv')\n",
        "from sklearn.linear_model import LogisticRegression\nm = LogisticRegression()\n",
        "import numpy as np\nnp.random.default_rng(42)\n",
        # 'signal' as an ordinary word must not trip the check
        "signal_strength = 0.5  # not the signal module\n",
    ],
    ids=["pandas", "sklearn", "numpy", "word-signal"],
)
def test_ordinary_analysis_code_is_not_flagged(code: str) -> None:
    """A check that fires on normal code gets switched off."""
    assert check_portability(code) == []


def test_executor_refuses_unportable_code_before_running_it(tmp_path) -> None:
    """The whole point is failing BEFORE the battery dies one model at a time."""
    executor = SubprocessExecutor()
    result = executor.run(
        code="import signal\nsignal.alarm(5)\nprint('should never run')\n",
        output_dir=str(tmp_path),
        timeout_s=30,
    )
    assert result["returncode"] == 2
    assert "PORTABILITY CHECK FAILED" in result["stderr"]
    assert "signal.alarm" in result["stderr"]
    assert result["stdout"] == ""
    assert not (tmp_path / "_generated_script.py").exists(), (
        "code should be rejected before it is written to disk"
    )


def test_executor_still_runs_clean_code(tmp_path) -> None:
    """Guarding against a check that blocks everything."""
    executor = SubprocessExecutor()
    result = executor.run(
        code="print('portable code ran')\n",
        output_dir=str(tmp_path),
        timeout_s=60,
    )
    assert result["returncode"] == 0
    assert "portable code ran" in result["stdout"]
