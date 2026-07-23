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

"""Exceptions for the ClawManager tool-execution benchmark."""

from __future__ import annotations

from inference_endpoint.exceptions import ExecutionError, InputValidationError


class ClawManagerBenchError(ExecutionError):
    """Base exception for this benchmark's own execution failures."""


class ClawManagerAPIError(ClawManagerBenchError):
    """A ClawManager REST API call returned a non-2xx response."""

    def __init__(self, status: int, body: str, url: str) -> None:
        self.status = status
        self.body = body
        self.url = url
        super().__init__(f"ClawManager API error {status} for {url}: {body[:500]}")


class TrajectoryValidationError(InputValidationError):
    """A trajectory JSONL file is malformed or uses an unsupported tool."""


class ShellDriverTimeoutError(ClawManagerBenchError):
    """The WebSocket shell driver timed out waiting for a turn boundary."""

    def __init__(self, instance_id: int, timeout_s: float) -> None:
        self.instance_id = instance_id
        self.timeout_s = timeout_s
        super().__init__(
            f"Instance {instance_id}: no turn-boundary marker observed within "
            f"{timeout_s}s (see --turn-marker-regex)"
        )
