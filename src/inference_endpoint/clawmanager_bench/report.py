# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aggregates per-instance run results and CPU samples into a benchmark report."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .cpu_sampler import CPUSampleSeries

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstanceRunResult:
    instance_id: int
    conversation_id: str
    tool_calls_executed: int
    wall_time_s: float
    turns_completed: int
    failed: bool
    failure_reason: str | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    num_instances: int
    num_failed_instances: int
    total_wall_time_s: float
    total_tool_calls_executed: int
    throughput_tool_calls_per_s: float
    cpu_seconds_per_instance: dict[int, float] = field(default_factory=dict)
    per_instance: list[InstanceRunResult] = field(default_factory=list)


def aggregate(
    results: list[InstanceRunResult], cpu_series: CPUSampleSeries
) -> BenchmarkReport:
    total_wall_time_s = max((r.wall_time_s for r in results), default=0.0)
    total_tool_calls = sum(r.tool_calls_executed for r in results)
    num_failed = sum(1 for r in results if r.failed)
    throughput = total_tool_calls / total_wall_time_s if total_wall_time_s > 0 else 0.0
    cpu_seconds_per_instance = {
        r.instance_id: cpu_series.cpu_seconds(r.instance_id) for r in results
    }
    return BenchmarkReport(
        num_instances=len(results),
        num_failed_instances=num_failed,
        total_wall_time_s=total_wall_time_s,
        total_tool_calls_executed=total_tool_calls,
        throughput_tool_calls_per_s=throughput,
        cpu_seconds_per_instance=cpu_seconds_per_instance,
        per_instance=list(results),
    )


def write_json(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    # dict keys must be strings for JSON; instance ids are ints in-memory.
    payload["cpu_seconds_per_instance"] = {
        str(k): v for k, v in report.cpu_seconds_per_instance.items()
    }
    path.write_text(json.dumps(payload, indent=2))


def print_summary(report: BenchmarkReport) -> None:
    logger.info(
        "ClawManager tool-exec benchmark: %d instances (%d failed), "
        "%d tool calls in %.1fs (%.2f tool-calls/s)",
        report.num_instances,
        report.num_failed_instances,
        report.total_tool_calls_executed,
        report.total_wall_time_s,
        report.throughput_tool_calls_per_s,
    )
    for instance_id, cpu_seconds in sorted(report.cpu_seconds_per_instance.items()):
        logger.info("  instance %d: ~%.2f CPU-seconds", instance_id, cpu_seconds)
