from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class PipelineState(str, Enum):
    INITIALIZED = "INITIALIZED"
    FORMULATING = "FORMULATING"
    ENGINEERING = "ENGINEERING"
    ANALYZING = "ANALYZING"
    CRITIQUING = "CRITIQUING"
    REVISING = "REVISING"
    WRITING = "WRITING"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"


@dataclass
class PipelineContext:
    # Configuration
    dataset_name: str
    raw_data_path: str
    output_dir: str
    task_type: str = "prediction"
    max_revision_cycles: int = 2

    # Agent outputs
    research_spec: Optional[dict] = None
    literature_context: Optional[dict] = None
    data_report: Optional[dict] = None
    results_object: Optional[dict] = None
    review_report: Optional[dict] = None
    paper_text: Optional[str] = None

    # Pipeline metadata
    current_state: str = PipelineState.INITIALIZED
    completed_stages: list = field(default_factory=list)
    revision_cycle: int = 0
    errors: list = field(default_factory=list)
    log: list = field(default_factory=list)
    run_start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "schema_version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_name": self.dataset_name,
            "raw_data_path": self.raw_data_path,
            "output_dir": self.output_dir,
            "task_type": self.task_type,
            "max_revision_cycles": self.max_revision_cycles,
            "current_state": self.current_state.value if isinstance(self.current_state, PipelineState) else str(self.current_state),
            "completed_stages": self.completed_stages,
            "revision_cycle": self.revision_cycle,
            "research_spec": self.research_spec,
            "literature_context": self.literature_context,
            "data_report": self.data_report,
            "results_object": self.results_object,
            "review_report": self.review_report,
            "paper_text": self.paper_text,
            "errors": self.errors,
            "log": self.log,
            "run_start_time": self.run_start_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PipelineContext:
        ctx = cls(
            dataset_name=data["dataset_name"],
            raw_data_path=data["raw_data_path"],
            output_dir=data["output_dir"],
            task_type=data.get("task_type", "prediction"),
            max_revision_cycles=data.get("max_revision_cycles", 2),
        )
        ctx.current_state = PipelineState(data.get("current_state", "INITIALIZED"))
        ctx.completed_stages = data.get("completed_stages", [])
        ctx.revision_cycle = data.get("revision_cycle", 0)
        ctx.research_spec = data.get("research_spec")
        ctx.literature_context = data.get("literature_context")
        ctx.data_report = data.get("data_report")
        ctx.results_object = data.get("results_object")
        ctx.review_report = data.get("review_report")
        ctx.paper_text = data.get("paper_text")
        ctx.errors = data.get("errors", [])
        ctx.log = data.get("log", [])
        ctx.run_start_time = data.get("run_start_time", "")
        return ctx
