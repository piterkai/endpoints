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

"""Shared fixtures for clawmanager_bench unit tests.

None of these tests exercise a live ClawManager cluster — see the module
docstrings for where an ``@pytest.mark.integration`` test against a real
deployment could later be added, gated behind an env var.
"""

import json
from pathlib import Path

import pytest

# "c1": a single tool call per assistant turn.
# "c2": two tool calls in one assistant turn, to verify turn counting does not
# conflate "number of tool-role messages" with "number of dataset turns".
_ROWS = [
    {
        "conversation_id": "c1",
        "turn": 1,
        "role": "user",
        "content": "List files in the repo.",
        "system": "You are a coding agent.",
    },
    {
        "conversation_id": "c1",
        "turn": 2,
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "shell_exec",
                    "arguments": json.dumps({"command": ["ls"]}),
                },
            }
        ],
    },
    {
        "conversation_id": "c1",
        "turn": 3,
        "role": "tool",
        "tool_results": [
            {"tool_call_id": "call_1", "content": "expected: file listing"}
        ],
    },
    {
        "conversation_id": "c1",
        "turn": 4,
        "role": "assistant",
        "content": "Done listing files.",
    },
    {
        "conversation_id": "c2",
        "turn": 1,
        "role": "user",
        "content": "Read two files.",
        "system": "You are a coding agent.",
    },
    {
        "conversation_id": "c2",
        "turn": 2,
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_a",
                "type": "function",
                "function": {
                    "name": "file_read",
                    "arguments": json.dumps({"path": "a.txt"}),
                },
            },
            {
                "id": "call_b",
                "type": "function",
                "function": {
                    "name": "file_read",
                    "arguments": json.dumps({"path": "b.txt"}),
                },
            },
        ],
    },
    {
        "conversation_id": "c2",
        "turn": 3,
        "role": "tool",
        "tool_results": [
            {"tool_call_id": "call_a", "content": "expected a"},
            {"tool_call_id": "call_b", "content": "expected b"},
        ],
    },
    {
        "conversation_id": "c2",
        "turn": 4,
        "role": "assistant",
        "content": "Read both files.",
    },
]


@pytest.fixture
def trajectory_jsonl_path(tmp_path: Path) -> Path:
    path = tmp_path / "trajectories.jsonl"
    with path.open("w") as f:
        for row in _ROWS:
            f.write(json.dumps(row) + "\n")
    return path
