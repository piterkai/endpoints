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

"""Unit tests for report.aggregate/write_json — pure functions on canned fixtures.

Has no live-cluster counterpart: aggregation is pure math over already-collected
InstanceRunResult/CPUSampleSeries values.
"""

import json
from pathlib import Path

import pytest

from inference_endpoint.clawmanager_bench.cpu_sampler import CPUSample, CPUSampleSeries
from inference_endpoint.clawmanager_bench.report import (
    InstanceRunResult,
    aggregate,
    write_json,
)


@pytest.mark.unit
def test_aggregate_computes_throughput_and_failure_count() -> None:
    results = [
        InstanceRunResult(
            instance_id=1,
            conversation_id="c1",
            tool_calls_executed=10,
            wall_time_s=5.0,
            turns_completed=4,
            failed=False,
        ),
        InstanceRunResult(
            instance_id=2,
            conversation_id="c2",
            tool_calls_executed=0,
            wall_time_s=2.0,
            turns_completed=0,
            failed=True,
            failure_reason="timeout",
        ),
    ]
    series = CPUSampleSeries(
        samples=[
            CPUSample(
                instance_id=1,
                timestamp=0.0,
                cpu_cores=1.0,
                load_1m=1.0,
                load_5m=None,
                load_15m=None,
            ),
            CPUSample(
                instance_id=1,
                timestamp=5.0,
                cpu_cores=1.0,
                load_1m=1.0,
                load_5m=None,
                load_15m=None,
            ),
        ]
    )

    report = aggregate(results, series)

    assert report.num_instances == 2
    assert report.num_failed_instances == 1
    assert report.total_tool_calls_executed == 10
    assert report.total_wall_time_s == 5.0
    assert report.throughput_tool_calls_per_s == pytest.approx(2.0)
    assert report.cpu_seconds_per_instance[1] == pytest.approx(5.0)
    assert report.cpu_seconds_per_instance[2] == 0.0


@pytest.mark.unit
def test_aggregate_handles_empty_results() -> None:
    report = aggregate([], CPUSampleSeries())
    assert report.num_instances == 0
    assert report.throughput_tool_calls_per_s == 0.0


@pytest.mark.unit
def test_write_json_round_trips(tmp_path: Path) -> None:
    results = [
        InstanceRunResult(
            instance_id=1,
            conversation_id="c1",
            tool_calls_executed=3,
            wall_time_s=1.5,
            turns_completed=2,
            failed=False,
        )
    ]
    report = aggregate(results, CPUSampleSeries())
    out_path = tmp_path / "nested" / "report.json"

    write_json(report, out_path)

    payload = json.loads(out_path.read_text())
    assert payload["num_instances"] == 1
    assert payload["cpu_seconds_per_instance"] == {"1": 0.0}
