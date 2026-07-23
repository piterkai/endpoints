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

"""Unit tests for conversation_matcher.

A live-cluster @pytest.mark.integration counterpart could later be added at
tests/integration/clawmanager_bench/ — out of scope here.
"""

from pathlib import Path

import pytest

from inference_endpoint.clawmanager_bench.conversation_matcher import (
    count_client_turn_groups,
    embed_conversation_marker,
    extract_conversation_id,
    match_turn,
)
from inference_endpoint.clawmanager_bench.trajectory import load_trajectory_set


@pytest.mark.unit
def test_embed_and_extract_round_trip() -> None:
    prompt = embed_conversation_marker("You are a coding agent.", "c1")
    assert extract_conversation_id([{"role": "system", "content": prompt}]) == "c1"


@pytest.mark.unit
def test_extract_returns_none_without_marker() -> None:
    assert (
        extract_conversation_id([{"role": "system", "content": "no marker here"}])
        is None
    )
    assert extract_conversation_id([]) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ([], 0),
        ([{"role": "user", "content": "hi"}], 1),
        (
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "again"},
            ],
            2,
        ),
        (
            # A single scripted turn with 2 tool calls -> 2 tool-role messages,
            # but still ONE completed client-turn group.
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": []},
                {"role": "tool", "tool_call_id": "a", "content": "..."},
                {"role": "tool", "tool_call_id": "b", "content": "..."},
            ],
            2,
        ),
    ],
)
def test_count_client_turn_groups(messages: list[dict], expected: int) -> None:
    assert count_client_turn_groups(messages) == expected


@pytest.mark.unit
def test_match_turn_advances_with_multi_tool_call_turn(
    trajectory_jsonl_path: Path,
) -> None:
    trajectories = load_trajectory_set(trajectory_jsonl_path)

    # After only the initial user message (count=1), c2's first assistant turn
    # (the one issuing 2 tool calls) is due.
    first = match_turn(
        trajectories, "c2", [{"role": "user", "content": "Read two files."}]
    )
    assert first is not None
    assert len(first.tool_calls) == 2

    # After the user turn plus a run of 2 real tool-role messages (count=2),
    # the SECOND assistant turn is due — not skipped or repeated.
    second = match_turn(
        trajectories,
        "c2",
        [
            {"role": "user", "content": "Read two files."},
            {"role": "tool", "tool_call_id": "call_a", "content": "real output a"},
            {"role": "tool", "tool_call_id": "call_b", "content": "real output b"},
        ],
    )
    assert second is not None
    assert second.content == "Read both files."


@pytest.mark.unit
def test_match_turn_returns_none_when_exhausted(trajectory_jsonl_path: Path) -> None:
    trajectories = load_trajectory_set(trajectory_jsonl_path)

    messages = [
        {"role": "user", "content": "List files in the repo."},
        {"role": "tool", "tool_call_id": "call_1", "content": "real ls output"},
        {"role": "user", "content": "anything else?"},
    ]
    assert match_turn(trajectories, "c1", messages) is None


@pytest.mark.unit
def test_match_turn_returns_none_for_unknown_conversation(
    trajectory_jsonl_path: Path,
) -> None:
    trajectories = load_trajectory_set(trajectory_jsonl_path)
    assert (
        match_turn(trajectories, "does-not-exist", [{"role": "user", "content": "hi"}])
        is None
    )
