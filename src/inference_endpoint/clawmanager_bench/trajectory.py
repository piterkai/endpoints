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

"""Loads scripted tool-calling trajectories for the mock LLM server.

Reuses ``AgenticInferenceDataset``'s JSONL schema and structural validators
(conversation grouping, role-sequence FSM, turn numbering) directly, since the
row format is identical. That dataset is designed to *drive* client (user/tool)
turns against a real endpoint, so it discards assistant rows once loaded; this
module instead reads the assistant rows straight from ``dataset.dataframe``
(populated by the constructor, before any ``.load()`` call), since those are
exactly the scripted responses this benchmark's mock LLM server needs to serve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from inference_endpoint.dataset_manager.agentic_inference_dataset import (
    AgenticInferenceDataset,
)
from inference_endpoint.exceptions import InputValidationError

from .exceptions import TrajectoryValidationError

SUPPORTED_TOOL_NAMES = frozenset(
    {"shell_exec", "file_read", "file_write", "file_search", "code_exec"}
)


@dataclass(frozen=True)
class ScriptedAssistantTurn:
    """One scripted "LLM reply" — content and/or tool calls for the runtime to execute."""

    turn: int
    content: str | None
    tool_calls: list[dict[str, Any]]


@dataclass(frozen=True)
class ToolExecTrajectorySet:
    """All conversations loaded from a trajectory JSONL file.

    ``assistant_turns_by_conversation[conv_id]`` is ordered by turn number. Because
    the dataset's role-sequence FSM guarantees an assistant row is always both
    preceded and followed by exactly one client (user/tool) turn group, the K-th
    entry (1-indexed) is always the correct reply to the K-th client turn group
    observed in a request's message history — see ``conversation_matcher.match_turn``.
    """

    conversation_ids: list[str]
    assistant_turns_by_conversation: dict[str, list[ScriptedAssistantTurn]]
    system_prompt_by_conversation: dict[str, str]
    initial_user_message_by_conversation: dict[str, str]


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
    """Normalize a row's ``tool_calls`` into wire-ready OpenAI tool_calls.

    ``function.arguments`` must be a JSON-encoded string on the wire; the dataset
    schema also allows a dict (validated by ``AgenticInferenceDataset``), which is
    encoded here.
    """
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for call in raw:
        function = dict(call["function"])
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            function["arguments"] = json.dumps(arguments)
        normalized.append(
            {
                "id": call["id"],
                "type": call.get("type", "function"),
                "function": function,
            }
        )
    return normalized


def _validate_tool_names(
    conversation_id: str, turn: int, tool_calls: list[dict[str, Any]]
) -> None:
    for call in tool_calls:
        name = call["function"]["name"]
        if name not in SUPPORTED_TOOL_NAMES:
            raise TrajectoryValidationError(
                f"conversation {conversation_id!r} turn {turn}: unsupported "
                f"tool_calls function name {name!r}; supported names are "
                f"{sorted(SUPPORTED_TOOL_NAMES)}"
            )


def load_trajectory_set(path: Path) -> ToolExecTrajectorySet:
    """Load and validate a trajectory JSONL file.

    Raises:
        TrajectoryValidationError: if the file is malformed, or an ``assistant``
            row's ``tool_calls`` names a tool this benchmark does not support.
    """
    df = pd.read_json(path, lines=True)
    try:
        dataset = AgenticInferenceDataset(df)
    except (ValueError, InputValidationError) as e:
        raise TrajectoryValidationError(str(e)) from e

    conversation_ids: list[str] = []
    assistant_turns_by_conversation: dict[str, list[ScriptedAssistantTurn]] = {}
    system_prompt_by_conversation: dict[str, str] = {}
    initial_user_message_by_conversation: dict[str, str] = {}

    assert dataset.dataframe is not None
    conv_groups = dict(
        list(dataset.dataframe.groupby("conversation_id", sort=False, dropna=False))
    )
    for conv_id, group in conv_groups.items():
        str_conv_id = str(conv_id)
        conversation_ids.append(str_conv_id)
        sorted_group = group.sort_values("turn")

        turns: list[ScriptedAssistantTurn] = []
        system_prompt: str | None = None
        initial_user_message: str | None = None

        for _, row in sorted_group.iterrows():
            role = row.get("role")
            if role == "assistant":
                content = row.get("content")
                tool_calls = _normalize_tool_calls(row.get("tool_calls"))
                turn_number = int(row["turn"])
                _validate_tool_names(str_conv_id, turn_number, tool_calls)
                turns.append(
                    ScriptedAssistantTurn(
                        turn=turn_number,
                        content=None if _is_missing(content) else str(content),
                        tool_calls=tool_calls,
                    )
                )
            elif role == "user":
                if system_prompt is None and not _is_missing(row.get("system")):
                    system_prompt = str(row["system"])
                if initial_user_message is None and not _is_missing(row.get("content")):
                    initial_user_message = str(row["content"])

        assistant_turns_by_conversation[str_conv_id] = turns
        system_prompt_by_conversation[str_conv_id] = system_prompt or ""
        initial_user_message_by_conversation[str_conv_id] = initial_user_message or ""

    return ToolExecTrajectorySet(
        conversation_ids=conversation_ids,
        assistant_turns_by_conversation=assistant_turns_by_conversation,
        system_prompt_by_conversation=system_prompt_by_conversation,
        initial_user_message_by_conversation=initial_user_message_by_conversation,
    )
