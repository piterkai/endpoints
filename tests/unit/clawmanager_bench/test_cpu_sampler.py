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

"""Unit tests for cpu_sampler's parsing helpers and CPUSampleSeries aggregation.

A live-cluster @pytest.mark.integration counterpart (polling a real ClawManager
runtime endpoint, or shelling out to a real `kubectl top pod`) could later be
added at tests/integration/clawmanager_bench/ — out of scope here.
"""

import pytest

from inference_endpoint.clawmanager_bench.cpu_sampler import (
    CPUSample,
    CPUSampleSeries,
    extract_cpu_metrics,
    parse_kubectl_top_output,
)


@pytest.mark.unit
def test_extract_cpu_metrics_happy_path() -> None:
    system_info = {"cpu": {"cores": 4, "load": {"1m": 0.5, "5m": 0.4, "15m": 0.3}}}
    assert extract_cpu_metrics(system_info) == (4, 0.5, 0.4, 0.3)


@pytest.mark.unit
@pytest.mark.parametrize(
    "system_info",
    [
        None,
        {},
        {"cpu": "not-a-dict"},
        {"cpu": {"cores": 4}},  # missing "load"
        {"cpu": {"cores": "not-a-number", "load": {"1m": 0.5}}},
    ],
)
def test_extract_cpu_metrics_defensive_on_missing_or_malformed_fields(
    system_info,
) -> None:
    cores, load_1m, load_5m, load_15m = extract_cpu_metrics(system_info)
    # Must never raise — every field independently falls back to None.
    assert load_5m is None or isinstance(load_5m, (int, float))
    assert load_15m is None or isinstance(load_15m, (int, float))


@pytest.mark.unit
def test_parse_kubectl_top_output() -> None:
    output = "NAME       CPU(cores)   MEMORY(bytes)\ncmbench-0  120m         256Mi\ncmbench-1  80m          128Mi\n"
    parsed = parse_kubectl_top_output(output)
    assert parsed == {"cmbench-0": 120.0, "cmbench-1": 80.0}


@pytest.mark.unit
def test_parse_kubectl_top_output_ignores_header_and_blank_lines() -> None:
    output = "NAME  CPU(cores)  MEMORY(bytes)\n\ncmbench-0  50m  10Mi\n"
    assert parse_kubectl_top_output(output) == {"cmbench-0": 50.0}


@pytest.mark.unit
def test_cpu_sample_series_integrates_load_over_time() -> None:
    series = CPUSampleSeries(
        samples=[
            CPUSample(
                instance_id=1,
                timestamp=0.0,
                cpu_cores=2.0,
                load_1m=1.0,
                load_5m=None,
                load_15m=None,
            ),
            CPUSample(
                instance_id=1,
                timestamp=10.0,
                cpu_cores=2.0,
                load_1m=1.0,
                load_5m=None,
                load_15m=None,
            ),
        ]
    )
    # Constant load of 1.0 * 2.0 cores over 10s -> 20 CPU-seconds.
    assert series.cpu_seconds(1) == pytest.approx(20.0)


@pytest.mark.unit
def test_cpu_sample_series_returns_zero_with_fewer_than_two_points() -> None:
    series = CPUSampleSeries(
        samples=[
            CPUSample(
                instance_id=1,
                timestamp=0.0,
                cpu_cores=2.0,
                load_1m=1.0,
                load_5m=None,
                load_15m=None,
            )
        ]
    )
    assert series.cpu_seconds(1) == 0.0
    assert series.cpu_seconds(999) == 0.0
