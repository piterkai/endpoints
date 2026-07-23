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

"""Configuration for the ClawManager tool-execution benchmark.

A few fields have no safe default and must be supplied by the operator after
inspecting their own ClawManager deployment (see
``examples/12_ClawManager_ToolExec_Benchmark/README.md`` "Step 0"):
``cli_launch_template`` and ``turn_marker_regex`` are specific to whatever
OpenClaw/Hermes CLI the deployed runtime image actually runs, which is not
observable from this repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

import cyclopts
from pydantic import BaseModel, ConfigDict, Field, model_validator

from inference_endpoint.exceptions import InputValidationError


@cyclopts.Parameter(name="*")
class ToolExecBenchConfig(BaseModel):
    """CLI-exposed configuration for a single benchmark run.

    Every field is its own CLI flag (cyclopts flattens the model, see
    ``cli.py``). There is no YAML loader yet;
    ``examples/12_ClawManager_ToolExec_Benchmark/config.yaml`` documents field
    values as a reference for the equivalent flags.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    clawmanager_base_url: str
    admin_username: str
    admin_password: str

    num_instances: Annotated[int, Field(ge=1, le=500)] = 1

    mock_llm_bind_host: str = "0.0.0.0"
    mock_llm_bind_port: int = 0
    mock_llm_cluster_url: str = Field(
        description=(
            "Address the mock LLM server is reachable at from inside the K8s "
            "cluster (DNS name, NodePort, or Ingress route) — never localhost."
        )
    )

    trajectory_dataset_path: Path

    cli_launch_template: str = Field(
        description=(
            "Terminal input typed to start a conversation, with "
            "'{conversation_id}' and '{message}' placeholders. Discover this "
            "by hand first — see README Step 0."
        )
    )
    turn_marker_regex: str = Field(
        default=r"\$\s*$",
        description=(
            "Regex matched against ANSI-stripped terminal output to detect a "
            "turn boundary. The default (a bare shell prompt) is almost "
            "certainly wrong for a real agent CLI and must be overridden."
        ),
    )
    shell_auth_mode: Annotated[
        Literal["header", "query-param"],
        cyclopts.Parameter(alias="--shell-auth-mode"),
    ] = "header"
    shell_io_timeout_s: float = Field(60.0, gt=0)

    runtime_poll_interval_s: float = Field(5.0, gt=0)
    cpu_sampler: Annotated[
        Literal["runtime-poll", "kubectl"],
        cyclopts.Parameter(alias="--cpu-sampler"),
    ] = "runtime-poll"
    kubectl_namespace: str | None = None

    duration_s: float | None = Field(None, gt=0)
    teardown_after_run: bool = True
    confirm_only_active_model: Annotated[
        bool,
        cyclopts.Parameter(
            help=(
                "Must be explicitly set: acknowledges that the mock LLM model "
                "will be registered as active, which affects every other "
                "instance on this ClawManager deployment using model=auto."
            )
        ),
    ] = False

    log_raw_requests_path: Path | None = None
    report_output_path: Path | None = None

    @model_validator(mode="after")
    def _check_reachability_hint(self) -> Self:
        lowered = self.mock_llm_cluster_url.lower()
        if "localhost" in lowered or "127.0.0.1" in lowered:
            raise InputValidationError(
                "mock_llm_cluster_url looks like a loopback address "
                f"({self.mock_llm_cluster_url!r}), which is almost certainly "
                "unreachable from inside the K8s cluster. Use a cluster-internal "
                "DNS name, NodePort, or Ingress route instead."
            )
        return self

    @model_validator(mode="after")
    def _check_model_confirmation(self) -> Self:
        if not self.confirm_only_active_model:
            raise InputValidationError(
                "confirm_only_active_model must be set to true: registering the "
                "mock LLM model affects every instance on this ClawManager "
                "deployment resolving model=auto. Set it only once you have "
                "confirmed this is safe to do."
            )
        return self
