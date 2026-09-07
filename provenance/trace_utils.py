from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utils.io_utils import append_jsonl, write_json


@dataclass
class TraceRecord:
    step_id: str
    role_in: str
    role_out: str
    input_artifacts: list[str]
    output_artifacts: list[str]
    function: str
    status: str
    wall_clock_s: float
    details: dict[str, Any] = field(default_factory=dict)


class TraceRecorder:
    def __init__(self, trace_path: Path) -> None:
        self.trace_path = trace_path
        self.records: list[TraceRecord] = []

    def record(
        self,
        *,
        step_id: str,
        role_in: str,
        role_out: str,
        input_artifacts: list[str],
        output_artifacts: list[str],
        function: str,
        status: str,
        wall_clock_s: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        record = TraceRecord(
            step_id=step_id,
            role_in=role_in,
            role_out=role_out,
            input_artifacts=input_artifacts,
            output_artifacts=output_artifacts,
            function=function,
            status=status,
            wall_clock_s=round(wall_clock_s, 6),
            details=details or {},
        )
        self.records.append(record)
        append_jsonl(self.trace_path, asdict(record))

    def write_summary(self, path: Path, payload: dict[str, Any]) -> None:
        summary = dict(payload)
        summary["trace_length"] = len(self.records)
        summary["steps"] = [asdict(record) for record in self.records]
        write_json(path, summary)
