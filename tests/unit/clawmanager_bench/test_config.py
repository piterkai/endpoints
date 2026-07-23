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

"""Unit tests for ToolExecBenchConfig validation."""

from pathlib import Path

import pydantic
import pytest

from inference_endpoint.clawmanager_bench.config import ToolExecBenchConfig
from inference_endpoint.exceptions import InputValidationError


def _base_kwargs(tmp_path: Path, **overrides: object) -> dict:
    kwargs = {
        "clawmanager_base_url": "https://clawmanager.example.internal",
        "admin_username": "admin",
        "admin_password": "secret",
        "mock_llm_cluster_url": "http://cmbench-mock.default.svc.cluster.local:8080",
        "trajectory_dataset_path": tmp_path / "trajectories.jsonl",
        "cli_launch_template": "openclaw chat --conversation {conversation_id} '{message}'",
        "confirm_only_active_model": True,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.unit
def test_valid_config_constructs(tmp_path: Path) -> None:
    config = ToolExecBenchConfig(**_base_kwargs(tmp_path))
    assert config.num_instances == 1
    assert config.teardown_after_run is True
    assert config.cpu_sampler == "runtime-poll"


@pytest.mark.unit
def test_missing_required_field_raises() -> None:
    with pytest.raises(pydantic.ValidationError):
        ToolExecBenchConfig(admin_username="admin")  # type: ignore[call-arg]


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_url", ["http://localhost:8080", "http://127.0.0.1:8080/v1"]
)
def test_localhost_mock_llm_url_rejected(tmp_path: Path, bad_url: str) -> None:
    with pytest.raises(InputValidationError, match="loopback"):
        ToolExecBenchConfig(**_base_kwargs(tmp_path, mock_llm_cluster_url=bad_url))


@pytest.mark.unit
def test_unconfirmed_model_registration_rejected(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="confirm_only_active_model"):
        ToolExecBenchConfig(**_base_kwargs(tmp_path, confirm_only_active_model=False))


@pytest.mark.unit
def test_num_instances_bounds_enforced(tmp_path: Path) -> None:
    with pytest.raises(pydantic.ValidationError):
        ToolExecBenchConfig(**_base_kwargs(tmp_path, num_instances=0))
    with pytest.raises(pydantic.ValidationError):
        ToolExecBenchConfig(**_base_kwargs(tmp_path, num_instances=501))


@pytest.mark.unit
def test_config_is_frozen(tmp_path: Path) -> None:
    config = ToolExecBenchConfig(**_base_kwargs(tmp_path))
    with pytest.raises(pydantic.ValidationError):
        config.num_instances = 5  # type: ignore[misc]
