import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.context import PipelineContext, PipelineState, allocate_run_dir


def test_initial_state() -> None:
    ctx = PipelineContext(
        dataset_name="test",
        raw_data_path="data/raw/test.csv",
        output_dir="output/test",
    )
    assert ctx.current_state == "INITIALIZED"
    assert ctx.current_state == PipelineState.INITIALIZED
    assert ctx.revision_cycle == 0
    assert ctx.completed_stages == []
    assert ctx.errors == []
    assert ctx.log == []
    assert ctx.research_spec is None
    assert ctx.literature_context is None
    assert ctx.data_report is None
    assert ctx.results_object is None
    assert ctx.review_report is None
    assert ctx.paper_text is None
    assert ctx.max_revision_cycles == 2


def test_to_dict_from_dict() -> None:
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/test.csv",
        output_dir="output/test",
        max_revision_cycles=2,
    )
    ctx.current_state = PipelineState.ANALYZING
    ctx.completed_stages = ["FORMULATING", "ENGINEERING"]
    ctx.revision_cycle = 1
    ctx.research_spec = {"research_question": "test question"}
    ctx.errors = ["some error"]

    d = ctx.to_dict()

    assert d["schema_version"] == "1.0"
    assert "timestamp" in d
    assert d["current_state"] == "ANALYZING"
    assert "FORMULATING" in d["completed_stages"]
    assert "ENGINEERING" in d["completed_stages"]
    assert d["revision_cycle"] == 1
    assert d["research_spec"]["research_question"] == "test question"
    assert d["errors"] == ["some error"]

    restored = PipelineContext.from_dict(d)
    assert restored.current_state == "ANALYZING"
    assert restored.current_state == PipelineState.ANALYZING
    assert restored.completed_stages == ["FORMULATING", "ENGINEERING"]
    assert restored.revision_cycle == 1
    assert restored.research_spec == {"research_question": "test question"}
    assert restored.dataset_name == "hsls09_public"
    assert restored.max_revision_cycles == 2


def test_completed_stages_tracking() -> None:
    ctx = PipelineContext(
        dataset_name="test",
        raw_data_path="data/raw/test.csv",
        output_dir="output/test",
    )
    ctx.completed_stages.append("FORMULATING")
    ctx.completed_stages.append("ENGINEERING")
    assert len(ctx.completed_stages) == 2
    assert ctx.completed_stages[0] == "FORMULATING"
    assert "ENGINEERING" in ctx.completed_stages
    assert "ANALYZING" not in ctx.completed_stages


def test_abort_info_round_trips() -> None:
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/test.csv",
        output_dir="output/test",
    )
    assert ctx.abort_info is None
    ctx.abort_info = {
        "stage": "ENGINEERING", "code": "NETWORK", "message": "reset",
        "resumable": True, "at": "2026-09-25T00:00:00+00:00",
    }
    restored = PipelineContext.from_dict(ctx.to_dict())
    assert restored.abort_info == ctx.abort_info


def test_run_start_time_is_timezone_aware() -> None:
    """A naive stamp could not be subtracted from an aware 'now'."""
    from datetime import datetime

    ctx = PipelineContext(
        dataset_name="d", raw_data_path="x", output_dir="y",
    )
    assert datetime.fromisoformat(ctx.run_start_time).tzinfo is not None


def test_allocate_run_dir_never_shares_a_directory(tmp_path: Path) -> None:
    """Two launches in the same second used to share one run directory."""
    import os
    from datetime import datetime

    now = datetime(2026, 9, 25, 12, 0, 0)
    a = allocate_run_dir(str(tmp_path), now=now)
    b = allocate_run_dir(str(tmp_path), now=now)
    assert a != b
    assert os.path.basename(a) == "run_20260925_120000"
    assert os.path.basename(b) == "run_20260925_120000_2"
    assert os.path.isdir(a) and os.path.isdir(b)
    assert os.path.isabs(a)
