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

"""Matches an incoming mock-LLM request to a trajectory turn.

Unlike the Edge-Agentic replay (which never reads back live model output), here a
real agent runtime executes tool calls for real and feeds the *real* tool results
back into the next request's ``messages``. Matching by message content is therefore
not viable — real tool output never equals any canned reference — so requests are
matched by (1) a conversation-id marker embedded in the system prompt and (2) a
count of completed client-turn groups, which is robust to real tool output content.
"""

from __future__ import annotations

import re
from typing import Any

from .trajectory import ScriptedAssistantTurn, ToolExecTrajectorySet

CONVERSATION_MARKER_PREFIX = "[cmbench:conv="
CONVERSATION_MARKER_SUFFIX = "]"

_MARKER_RE = re.compile(
    re.escape(CONVERSATION_MARKER_PREFIX)
    + r"([^\]]+)"
    + re.escape(CONVERSATION_MARKER_SUFFIX)
)


def embed_conversation_marker(system_prompt: str, conversation_id: str) -> str:
    """Append a conversation-id marker to a system prompt.

    Called once by the orchestrator when it assembles the launch prompt for an
    instance <-> conversation_id assignment (see ``shell_driver.py``).
    """
    marker = (
        f"{CONVERSATION_MARKER_PREFIX}{conversation_id}{CONVERSATION_MARKER_SUFFIX}"
    )
    if not system_prompt:
        return marker
    return f"{system_prompt}\n\n{marker}"


def extract_conversation_id(messages: list[dict[str, Any]]) -> str | None:
    """Find the conversation-id marker in the request's system message, if any."""
    for message in messages:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = _MARKER_RE.search(content)
        if match:
            return match.group(1)
    return None


def count_client_turn_groups(messages: list[dict[str, Any]]) -> int:
    """Count completed client-turn groups: one per ``user`` message, one per
    maximal contiguous run of ``tool`` messages.

    A single scripted assistant turn with N tool calls produces one dataset
    "turn" but N real tool-role messages (one per ``tool_call_id``) — counting
    role-run boundaries, not raw message count, keeps this aligned with
    ``ToolExecTrajectorySet.assistant_turns_by_conversation``'s per-turn indexing
    regardless of how many tool calls a given turn made.
    """
    count = 0
    prev_role: str | None = None
    for message in messages:
        role = message.get("role")
        if role == "user":
            count += 1
        elif role == "tool":
            if prev_role != "tool":
                count += 1
        prev_role = role
    return count


def match_turn(
    trajectories: ToolExecTrajectorySet,
    conversation_id: str,
    incoming_messages: list[dict[str, Any]],
) -> ScriptedAssistantTurn | None:
    """Return the scripted assistant turn due for this request, or ``None``.

    ``None`` means either the conversation id is unknown or its trajectory is
    exhausted (every scripted assistant turn has already been served) — callers
    should serve a terminal "conversation complete" response in that case rather
    than erroring, so the runtime under test stops looping naturally.
    """
    turns = trajectories.assistant_turns_by_conversation.get(conversation_id)
    if not turns:
        return None
    completed = count_client_turn_groups(incoming_messages)
    if completed < 1 or completed > len(turns):
        return None
    return turns[completed - 1]
