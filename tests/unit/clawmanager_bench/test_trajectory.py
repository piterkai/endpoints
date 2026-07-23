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

"""Unit tests for trajectory.load_trajectory_set.

A live-cluster @pytest.mark.integration counterpart could later be added at
tests/integration/clawmanager_bench/, gated behind an env var — out of scope here.
"""

import json
from pathlib import Path

import pytest

from inference_endpoint.clawmanager_bench.exceptions import TrajectoryValidationError
from inference_endpoint.clawmanager_bench.trajectory import load_trajectory_set


@pytest.mark.unit
def test_loads_two_conversations(trajectory_jsonl_path: Path) -> None:
    trajectories = load_trajectory_set(trajectory_jsonl_path)

    assert trajectories.conversation_ids == ["c1", "c2"]
    assert trajectories.system_prompt_by_conversation["c1"] == "You are a coding agent."
    assert (
        trajectories.initial_user_message_by_conversation["c1"]
        == "List files in the repo."
    )


@pytest.mark.unit
def test_assistant_turns_ordered_and_normalized(trajectory_jsonl_path: Path) -> None:
    trajectories = load_trajectory_set(trajectory_jsonl_path)

    c1_turns = trajectories.assistant_turns_by_conversation["c1"]
    assert len(c1_turns) == 2
    assert c1_turns[0].turn == 2
    assert len(c1_turns[0].tool_calls) == 1
    assert c1_turns[0].tool_calls[0]["function"]["name"] == "shell_exec"
    # arguments must be a JSON-encoded string on the wire, not a dict.
    assert isinstance(c1_turns[0].tool_calls[0]["function"]["arguments"], str)
    assert c1_turns[1].turn == 4
    assert c1_turns[1].content == "Done listing files."
    assert c1_turns[1].tool_calls == []


@pytest.mark.unit
def test_multi_tool_call_turn_preserved(trajectory_jsonl_path: Path) -> None:
    trajectories = load_trajectory_set(trajectory_jsonl_path)

    c2_turns = trajectories.assistant_turns_by_conversation["c2"]
    assert len(c2_turns[0].tool_calls) == 2


@pytest.mark.unit
def test_rejects_unsupported_tool_name(tmp_path: Path) -> None:
    rows = [
        {
            "conversation_id": "c1",
            "turn": 1,
            "role": "user",
            "content": "hi",
            "system": "sys",
        },
        {
            "conversation_id": "c1",
            "turn": 2,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "delete_prod_database", "arguments": "{}"},
                }
            ],
        },
        {
            "conversation_id": "c1",
            "turn": 3,
            "role": "tool",
            "tool_results": [{"tool_call_id": "call_1", "content": "n/a"}],
        },
        {"conversation_id": "c1", "turn": 4, "role": "assistant", "content": "done"},
    ]
    path = tmp_path / "bad.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    with pytest.raises(TrajectoryValidationError, match="unsupported"):
        load_trajectory_set(path)


@pytest.mark.unit
def test_rejects_malformed_role_sequence(tmp_path: Path) -> None:
    rows = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "hi"},
        {"conversation_id": "c1", "turn": 2, "role": "user", "content": "again"},
    ]
    path = tmp_path / "malformed.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    with pytest.raises(TrajectoryValidationError):
        load_trajectory_set(path)
