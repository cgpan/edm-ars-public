import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.context import PipelineContext, PipelineState


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
